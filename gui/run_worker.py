"""Run one analysis for the web GUI, emitting events for the frontend.

This is the GUI's replacement for the v0.2.5 worker.  It drives the
TradingAgents v0.5.2 graph the same way the CLI does:

  * ``create_run_state`` builds the initial state (settles the memory log,
    resolves the instrument and injects past context);
  * ``stream_run`` yields the agents' messages plus the cumulative run state
    (analysts run as subgraphs in 0.5.x, so a plain ``graph.stream`` call
    would miss their live messages);
  * ``begin_checkpoint`` / ``checkpoint_input`` / ``end_checkpoint`` make the
    GUI's checkpoint toggle actually save and resume;
  * ``record_decision`` and ``write_report_tree`` give the GUI the same
    memory-log and on-disk reports the CLI produces.

Events are plain dicts pushed onto ``run.queue`` and consumed by the SSE
endpoint in ``gui.app``.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from tradingagents.agents.rating import parse_rating
from tradingagents.dataflows.symbols import safe_ticker_component

from .agent_map import ANALYSTS, REPORT_SECTIONS
from .stats import estimate_cost

logger = logging.getLogger(__name__)


# Which report section is produced by which roster agent.
_SECTION_AGENT = {
    "market_report": "Market Analyst",
    "sentiment_report": "Sentiment Analyst",
    "news_report": "News Analyst",
    "fundamentals_report": "Fundamentals Analyst",
    "investment_plan": "Research Manager",
    "trader_investment_plan": "Trader",
    "final_trade_decision": "Portfolio Manager",
}

_RESEARCH = ("Bull Researcher", "Bear Researcher", "Research Manager")
_RISK = ("Aggressive Analyst", "Neutral Analyst", "Conservative Analyst")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_worker(run, persist_cb=None) -> None:
    """Execute the graph for ``run`` and emit progress to its queue."""
    try:
        run.status = "running"
        run.emit({"type": "status", "message": f"🚀 Starting analysis: {run.ticker} on {run.date}"})

        # The analysts run in parallel; light them all up at once.
        for key in run.selected:
            meta = ANALYSTS.get(key)
            if meta:
                _set_status(run, meta["display"], "in_progress")
        run.emit({"type": "agents_init", "agents": dict(run.roster)})

        # Late imports keep the module light and avoid importing the ML stack
        # when the GUI is only serving its settings pages.
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.reporting import write_report_tree

        config = _build_config(run.params, DEFAULT_CONFIG)

        if run.params.get("clear_checkpoints"):
            try:
                n = clear_all_checkpoints(config["data_cache_dir"])
                run.emit({"type": "status", "message": f"🗑 Cleared {n} checkpoint(s)"})
            except Exception:
                logger.exception("clear_checkpoints failed")

        graph = TradingAgentsGraph(run.selected, config=config, debug=False)

        # Same layout as the CLI: <results>/<ticker>/<date>/reports/.
        report_dir = (
            Path(config["results_dir"])
            / safe_ticker_component(run.ticker)
            / run.date
            / "reports"
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        # v0.5.2 owns the initial state (memory settlement, instrument context,
        # asset type).  Hard-coding a dict here is what breaks older GUIs.
        init_state = graph.create_run_state(run.ticker, run.date, "stock")
        args = graph.propagator.get_graph_args()

        checkpoint_tid = graph.begin_checkpoint(run.ticker, run.date, "stock")
        if checkpoint_tid is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid

        final_state: dict[str, Any] = {}
        debate_state: dict[str, str] = {}
        risk_state: dict[str, str] = {}
        # stream_run() replays the cumulative state on every step, so the same
        # message arrives repeatedly. Keep the feed and token counters honest.
        seen_messages: set = set()

        try:
            for messages, state in graph.stream_run(
                graph.checkpoint_input(init_state), **args
            ):
                if run.is_cancelled():
                    break

                _handle_messages(run, messages, config, seen_messages)

                if state is None:
                    continue
                if isinstance(state, dict):
                    final_state.update(state)
                    _handle_state(run, state, debate_state, risk_state)
        finally:
            graph.end_checkpoint()

        if run.is_cancelled():
            run.status = "stopped"
            run.emit({"type": "status", "message": "🛑 Run stopped."})
        else:
            _finish(
                run, graph, final_state, debate_state, risk_state,
                report_dir, write_report_tree,
            )
            run.status = "completed"
            run.emit({
                "type": "final_report",
                "sections": {k: v for k, v in run.reports.items() if v},
                "ticker": run.ticker,
                "date": run.date,
                "decision": run.decision,
            })

    except Exception as exc:
        logger.exception("run %s failed", getattr(run, "id", "?"))
        run.error = f"{type(exc).__name__}: {exc}"
        run.status = "error"
        for agent, status in run.roster.items():
            if status == "in_progress":
                _set_status(run, agent, "error")
                break
        run.emit({
            "type": "error",
            "message": run.error,
            "traceback": traceback.format_exc(),
        })
    finally:
        run.ended_at = time.time()
        run.stats["elapsed_s"] = round(run.ended_at - run.started_at, 2)
        run.emit({"type": "stats", "stats": dict(run.stats)})
        run.emit({
            "type": "done",
            "stopped": run.status == "stopped",
            "status": run.status,
            "error": run.error,
        })
        if persist_cb is not None:
            try:
                persist_cb()
            except Exception:
                logger.exception("RunManager persist hook failed at run end")


# ---------------------------------------------------------------------------
# Finishing: decision, reports, memory
# ---------------------------------------------------------------------------


def _finish(run, graph, final_state, debate_state, risk_state,
            report_dir: Path, write_report_tree) -> None:
    for agent in run.roster:
        _set_status(run, agent, "completed")

    # Keep the live report buffer and the final event in sync with the state.
    for section in REPORT_SECTIONS:
        if final_state.get(section):
            run.reports[section] = str(final_state[section])

    if not run.reports.get("investment_plan") and debate_state:
        run.reports["investment_plan"] = _format_debate(debate_state)
    if not run.reports.get("final_trade_decision") and risk_state:
        run.reports["final_trade_decision"] = _format_risk(risk_state)

    decision_text = (
        final_state.get("final_trade_decision")
        or run.reports.get("final_trade_decision")
        or ""
    )
    run.decision = parse_rating(decision_text) if decision_text.strip() else None

    # CLI parity: write the state log + memory entry, then the report tree.
    try:
        graph.record_decision(run.ticker, run.date, final_state)
    except Exception:
        logger.exception("record_decision failed (continuing)")

    try:
        write_report_tree(
            final_state, run.ticker, report_dir, settings=graph.run_settings()
        )
    except Exception:
        logger.exception("write_report_tree failed (continuing)")

    # The GUI's report index reads run.json next to complete_report.md.
    meta = {
        "run_id": run.id,
        "ticker": run.ticker,
        "date": run.date,
        "decision": run.decision,
        "stats": run.stats,
        "saved_at": time.time(),
    }
    try:
        (report_dir / "run.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    except OSError:
        logger.exception("failed to write run.json")

    try:
        graph.clear_checkpoint_on_success(run.ticker, run.date, "stock")
    except Exception:
        logger.exception("clear_checkpoint_on_success failed (continuing)")


# ---------------------------------------------------------------------------
# Streaming handlers
# ---------------------------------------------------------------------------


def _handle_messages(run, messages, config: dict, seen: set) -> None:
    """Turn a batch of LangChain messages into feed/stats events."""
    for msg in messages or []:
        if msg is None:
            continue

        key = getattr(msg, "id", None) or id(msg)
        if key in seen:
            continue
        seen.add(key)

        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        msg_type = type(msg).__name__.replace("Message", "")

        if content:
            run.emit({
                "type": "message",
                "msg_type": msg_type,
                "content": str(content)[:2000],
                "node": "",
            })

        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
            run.stats["tool_calls"] += 1
            run.emit({"type": "tool_call", "tool": str(name), "node": ""})

        if msg_type == "AI":
            run.stats["llm_calls"] += 1
            usage = getattr(msg, "usage_metadata", None) or {}
            if usage:
                tokens_in = int(usage.get("input_tokens") or 0)
                tokens_out = int(usage.get("output_tokens") or 0)
                run.stats["tokens_in"] += tokens_in
                run.stats["tokens_out"] += tokens_out
                model = _message_model(msg, config)
                run.stats["cost_usd"] = round(
                    run.stats["cost_usd"]
                    + estimate_cost(tokens_in, tokens_out, model),
                    4,
                )

    run.stats["elapsed_s"] = round(time.time() - run.started_at, 2)
    run.emit({"type": "stats", "stats": dict(run.stats)})


def _message_model(msg, config: dict) -> str:
    meta = getattr(msg, "response_metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("model_name", "model", "model_id"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    return config.get("quick_think_llm") or ""


def _handle_state(run, state: dict, debate_state: dict, risk_state: dict) -> None:
    """Update roster/report state from one cumulative graph state."""
    event_state: dict = {}

    for section in REPORT_SECTIONS:
        value = state.get(section)
        if not value:
            continue
        text = str(value)
        if text == run.reports.get(section):
            continue
        run.reports[section] = text
        event_state[section] = text
        producer = _SECTION_AGENT.get(section)
        if producer:
            _set_status(run, producer, "completed")
        _start_after(run)

    debate = state.get("investment_debate_state")
    if isinstance(debate, dict):
        changed = False
        for key in ("bull_history", "bear_history", "judge_decision"):
            value = debate.get(key)
            if value and value != debate_state.get(key):
                debate_state[key] = value
                changed = True
        if debate_state.get("bull_history"):
            _set_status(run, "Bull Researcher", "completed")
        if debate_state.get("bear_history"):
            _set_status(run, "Bear Researcher", "completed")
        if debate_state.get("judge_decision"):
            _set_status(run, "Research Manager", "completed")
        if changed:
            event_state["investment_debate_state"] = dict(debate_state)

    risk = state.get("risk_debate_state")
    if isinstance(risk, dict):
        changed = False
        for key, agent in (
            ("aggressive_history", "Aggressive Analyst"),
            ("neutral_history", "Neutral Analyst"),
            ("conservative_history", "Conservative Analyst"),
        ):
            value = risk.get(key)
            if value and value != risk_state.get(key):
                risk_state[key] = value
                changed = True
            if risk_state.get(key):
                _set_status(run, agent, "completed")
        judge = risk.get("judge_decision")
        if judge and judge != risk_state.get("judge_decision"):
            risk_state["judge_decision"] = judge
            changed = True
        if risk_state.get("judge_decision"):
            _set_status(run, "Portfolio Manager", "completed")
        if changed:
            event_state["risk_debate_state"] = dict(risk_state)

    _start_after(run)

    if event_state:
        run.emit({"type": "chunk", "node": "", "state": event_state})


def _start_after(run) -> None:
    """Advance pending agents once the stage in front of them is done."""
    analysts = [
        ANALYSTS[key]["display"]
        for key in run.selected
        if key in ANALYSTS and ANALYSTS[key]["display"] in run.roster
    ]
    if analysts and all(run.roster.get(a) == "completed" for a in analysts):
        for agent in _RESEARCH:
            if run.roster.get(agent) == "pending":
                _set_status(run, agent, "in_progress")

    if run.reports.get("investment_plan"):
        for agent in _RESEARCH:
            _set_status(run, agent, "completed")
        if run.roster.get("Trader") == "pending":
            _set_status(run, "Trader", "in_progress")

    if run.reports.get("trader_investment_plan"):
        _set_status(run, "Trader", "completed")
        for agent in _RISK:
            if run.roster.get(agent) == "pending":
                _set_status(run, agent, "in_progress")

    if run.reports.get("final_trade_decision"):
        for agent in _RISK:
            _set_status(run, agent, "completed")
        _set_status(run, "Portfolio Manager", "completed")


def _set_status(run, display: str, status: str) -> None:
    if run.roster.get(display) == status:
        return
    run.roster[display] = status
    run.emit({
        "type": "agent_update",
        "agent": display,
        "status": status,
        "agents": dict(run.roster),
    })


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _coerce_int(value, default=None):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_config(params: dict, default_config: dict) -> dict:
    """Translate the web form payload into a framework config dict."""
    from copy import deepcopy

    cfg = deepcopy(default_config)
    depth = _coerce_int(params.get("research_depth"), default=3)

    overrides = {
        "llm_provider": params.get("provider"),
        "quick_think_llm": params.get("quick_model"),
        "deep_think_llm": params.get("deep_model"),
        "max_debate_rounds": depth,
        "max_risk_discuss_rounds": depth,
        "output_language": params.get("output_language"),
        "checkpoint_enabled": params.get("checkpoint"),
        "backend_url": params.get("backend_url"),
        "google_thinking_level": params.get("google_thinking_level"),
        "openai_reasoning_effort": params.get("openai_reasoning_effort"),
        "anthropic_effort": params.get("anthropic_effort"),
    }
    for key, value in overrides.items():
        if value is not None and value != "":
            cfg[key] = value

    vendors = params.get("data_vendors") or {}
    if isinstance(vendors, dict) and vendors:
        cfg.setdefault("data_vendors", {}).update(vendors)

    return cfg


# ---------------------------------------------------------------------------
# Report formatting fallbacks
# ---------------------------------------------------------------------------


def _format_debate(debate: dict) -> str:
    parts = []
    if debate.get("bull_history"):
        parts.append(f"### Bull Researcher\n{debate['bull_history']}")
    if debate.get("bear_history"):
        parts.append(f"### Bear Researcher\n{debate['bear_history']}")
    if debate.get("judge_decision"):
        parts.append(f"### Research Manager Decision\n{debate['judge_decision']}")
    return "\n\n".join(parts)


def _format_risk(risk: dict) -> str:
    parts = []
    if risk.get("aggressive_history"):
        parts.append(f"### Aggressive Analyst\n{risk['aggressive_history']}")
    if risk.get("neutral_history"):
        parts.append(f"### Neutral Analyst\n{risk['neutral_history']}")
    if risk.get("conservative_history"):
        parts.append(f"### Conservative Analyst\n{risk['conservative_history']}")
    if risk.get("judge_decision"):
        parts.append(f"### Portfolio Manager Decision\n{risk['judge_decision']}")
    return "\n\n".join(parts)
