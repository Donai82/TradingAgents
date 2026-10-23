"""Run lifecycle management — extracted from the original gui/app.py.

Key changes from the old in-app.py implementation:

  * **Real stop / cancel.** A ``threading.Event`` per run, polled between
    graph chunks. The old Stop button on the UI hit a function that just
    posted a toast.
  * **Centralised agent / node mapping** via gui.agent_map — kills the
    fragile string-matching on node names.
  * **Multi-run support.** ``RunManager`` keys runs by id, lets clients
    target a specific run's event stream, and preserves recent runs for
    history. Backward-compat: when callers don't specify a run id we fall
    through to the most-recent run.
  * **Live cost tracking.** Tokens-in/out plus a USD estimate updated in
    real time via the price table in gui.stats.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from queue import Queue

from .agent_map import REPORT_SECTIONS, build_initial_roster
from .run_worker import run_worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run:
    """One analysis run, in-flight or completed."""

    def __init__(self, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.ticker: str = params.get("ticker", "?").upper()
        self.date:   str = params.get("date", datetime.date.today().isoformat())
        self.selected: list[str] = list(params.get("analysts") or [])
        self.status: str = "queued"
        self.started_at: float = time.time()
        self.ended_at: float | None = None
        self.error: str | None = None
        self.decision: str | None = None

        self.roster: dict[str, str] = build_initial_roster(self.selected)
        self.reports: dict[str, str] = dict.fromkeys(REPORT_SECTIONS, "")
        self.stats = {
            "llm_calls": 0, "tool_calls": 0,
            "tokens_in": 0, "tokens_out": 0,
            "cost_usd": 0.0, "elapsed_s": 0.0,
        }

        # Event fan-out
        self.queue: Queue = Queue()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None

    def emit(self, event: dict) -> None:
        event.setdefault("run_id", self.id)
        self.queue.put(event)

    def stop(self) -> None:
        self.cancel_event.set()
        self.emit({"type": "status", "message": "🛑 Stop requested — finishing current step then aborting."})

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def snapshot(self) -> dict:
        return {
            "run_id":   self.id,
            "ticker":   self.ticker,
            "date":     self.date,
            "status":   self.status,
            "started":  self.started_at,
            "ended":    self.ended_at,
            "error":    self.error,
            "decision": self.decision,
            "selected": self.selected,
            "agents":   dict(self.roster),
            "reports":  {k: v for k, v in self.reports.items() if v},
            "stats":    dict(self.stats),
        }


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------


_DEFAULT_RUNS_PATH = Path.home() / ".tradingagents" / "runs.json"


class RunManager:
    """Tracks active and recent runs.

    Run metadata (status, ticker, date, decision, stats) is persisted to a
    JSON file under ``~/.tradingagents/runs.json`` so the history survives
    server restarts. The persisted entries are a small subset of ``Run`` —
    not the live queue / threading.Event, which are obviously not
    serializable. On startup we load the file and skip anything that was
    in-flight (those threads are gone; we mark them ``error``).
    """

    def __init__(self, max_recent: int = 25,
                 persist_path: Path | None = None):
        self.runs: dict[str, Run] = {}
        self.order: list[str] = []
        self.max_recent = max_recent
        self._lock = threading.Lock()
        # Most recent run id — the SSE stream falls back to this when the
        # client doesn't specify a run.
        self.current_id: str | None = None
        self._persist_path = persist_path or _DEFAULT_RUNS_PATH
        self._load_persisted()

    def get(self, run_id: str | None) -> Run | None:
        if not run_id:
            return self.runs.get(self.current_id) if self.current_id else None
        return self.runs.get(run_id)

    def list(self) -> list[Run]:
        return [self.runs[i] for i in reversed(self.order) if i in self.runs]

    def is_anything_running(self) -> bool:
        return any(r.status == "running" for r in self.runs.values())

    def start(self, params: dict) -> Run:
        run = Run(params)
        with self._lock:
            self.runs[run.id] = run
            self.order.append(run.id)
            self.current_id = run.id
            # Evict oldest completed runs once we exceed the cap.
            while len(self.order) > self.max_recent:
                oldest_id = self.order[0]
                oldest = self.runs.get(oldest_id)
                if oldest and oldest.status in ("completed", "stopped", "error"):
                    self.order.pop(0)
                    self.runs.pop(oldest_id, None)
                else:
                    break

        run.thread = threading.Thread(
            target=run_worker, args=(run, self.persist),
            name=f"run-{run.id}", daemon=True,
        )
        run.thread.start()
        # Persist immediately so a crash during the run still leaves a
        # 'queued' record visible after restart (which will be flipped to
        # 'error' on next load).
        self.persist()
        return run

    def stop(self, run_id: str | None = None) -> bool:
        run = self.get(run_id)
        if not run or run.status not in ("running", "queued"):
            return False
        run.stop()
        return True

    # --------------------------------------------------------------- persistence

    def persist(self) -> None:
        """Write the current run list to disk. Best-effort — failures are
        logged but never raise into the agent loop or HTTP handlers."""
        snapshot = {
            "version": 1,
            "current_id": self.current_id,
            "order": self.order,
            "runs": {rid: _run_to_dict(r) for rid, r in self.runs.items()},
        }
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            import os
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix=".runs-", suffix=".tmp",
                                       dir=str(self._persist_path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self._persist_path)
            except Exception:
                with suppress(OSError):
                    os.unlink(tmp)
                raise
        except Exception:
            logger.exception("RunManager.persist failed")

    def reset(self) -> None:
        """Wipe all in-memory runs and remove the persisted ``runs.json``.

        Used by the Danger Zone wipe in the UI. Refuses nothing here — the
        caller is expected to gate on ``is_anything_running()`` first.
        """
        with self._lock:
            self.runs.clear()
            self.order.clear()
            self.current_id = None
        try:
            self._persist_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("RunManager.reset: failed to unlink %s", self._persist_path)

    def _load_persisted(self) -> None:
        """Restore runs from disk. ``running``/``queued`` states from a
        previous server lifetime become ``error`` (those threads are gone)."""
        if not self._persist_path.exists():
            return
        try:
            with self._persist_path.open("r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        order = snapshot.get("order") or []
        runs  = snapshot.get("runs")  or {}
        for rid in order:
            data = runs.get(rid)
            if not data:
                continue
            run = _run_from_dict(data)
            if run is not None:
                self.runs[rid] = run
                self.order.append(rid)
        # current_id is informational — clear it so a fresh SSE client doesn't
        # try to subscribe to a non-running run.
        self.current_id = None


# ---------------------------------------------------------------------------
# Snapshot helpers — Run <-> JSON
# ---------------------------------------------------------------------------


_SERIALIZABLE_RUN_FIELDS = (
    "id", "ticker", "date", "selected", "status",
    "started_at", "ended_at", "error", "decision",
    "roster", "reports", "stats", "params",
)


def _run_to_dict(run: Run) -> dict:
    """Pluck only the JSON-safe bits off ``Run`` for on-disk persistence."""
    return {k: getattr(run, k, None) for k in _SERIALIZABLE_RUN_FIELDS}


def _run_from_dict(data: dict) -> Run | None:
    """Reconstruct a Run from its on-disk dict. Returns ``None`` if the dict
    is malformed. In-flight states (``running``/``queued``) are rewritten to
    ``error`` since the worker thread is obviously gone."""
    try:
        params = data.get("params") or {}
        run = Run(params)
        run.id      = data.get("id", run.id)
        run.ticker  = data.get("ticker", run.ticker)
        run.date    = data.get("date",   run.date)
        run.selected = list(data.get("selected") or [])
        # Restore final fields first, then apply the "running -> error" flip
        # so the synthetic error message isn't overwritten by the disk value
        # (which would be None for a run that was still in flight).
        run.started_at = float(data.get("started_at") or time.time())
        run.ended_at   = data.get("ended_at")
        run.error      = data.get("error")
        run.decision   = data.get("decision")
        run.status     = data.get("status", "error")
        if run.status in ("running", "queued"):
            run.status = "error"
            run.error  = run.error or "Run did not complete (server restart)."
            run.ended_at = run.ended_at or time.time()
        run.roster     = dict(data.get("roster") or {})
        run.reports    = dict(data.get("reports") or {})
        run.stats.update(data.get("stats") or {})
        return run
    except Exception:
        logger.exception("RunManager: failed to rehydrate run from %r", data)
        return None
