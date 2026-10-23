"""On-disk credential storage that is atomic and owner-only.

Shared by the ChatGPT credential file. Kept free of any schema: the shape of
``~/.codex/auth.json`` belongs to :mod:`tradingagents.auth.codex_auth`, which
also has to interoperate with the Codex CLI and the ``openai-oauth`` project,
so the format must not be invented here.

Writes are atomic (temp file + ``os.replace``) so an interrupted run cannot
leave a truncated credential file that locks the user out of their own account.
``os.replace`` carries the temp file's mode, so the 0600 has to be applied to
the temp file before the rename rather than to the destination afterwards.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
from pathlib import Path

__all__ = ["read_json", "restrict_dir", "write_json"]

# chmod is advisory on Windows and would raise on odd filesystems; the file is
# still written, just without the tightened mode. A directory needs the execute
# bit to be searchable at all, so it gets 0700 while files get 0600 — applying
# the file mode to a directory would make it unwritable.
FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
DIR_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR


def read_json(path: Path) -> dict:
    """Parse ``path`` as a JSON object. Missing or unusable -> ``{}``.

    A hand-edited or truncated credential file must not break every provider,
    so a bad file reads as signed out and can be re-authenticated.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def restrict_dir(path: Path) -> None:
    """Make a directory owner-only. Best-effort: some filesystems reject chmod."""
    with contextlib.suppress(OSError):  # pragma: no cover - platform dependent
        path.chmod(DIR_MODE)


def write_json(path: Path, data: dict) -> Path:
    """Write ``data`` to ``path``, atomically and owner-only. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Restrict the directory too: it holds the file, and a readable listing is
    # a needless hint that credentials exist on this account.
    restrict_dir(path.parent)

    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".auth-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp:
            json.dump(data, temp, indent=2)
            temp.write("\n")
        with contextlib.suppress(OSError):  # pragma: no cover - platform dependent
            Path(temp_name).chmod(FILE_MODE)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path
