#!/usr/bin/env bash
#
# start.sh - bootstrap and run TradingAgents.
#
# Creates a local virtual environment in .venv on first use, installs the
# package (and its dependencies) into it, then runs the program from that
# environment. Subsequent runs skip straight to launching.
#
# Usage:
#   ./start.sh                       # launch the interactive CLI
#   ./start.sh --web                 # launch the local web dashboard
#   ./start.sh --web --port 5000     # ... on another port
#   ./start.sh --ticker NVDA --date 2026-09-23 --save --no-show
#   ./start.sh --reinstall           # rebuild the environment first
#   PYTHON=python3.13 ./start.sh     # pick the interpreter used to build .venv
#
# Anything after the flags below is forwarded to the CLI (or to the web
# dashboard with --web) unchanged, so the commands from the README work as-is.

set -euo pipefail

# Resolve this script's directory so it works from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.installed"
INSTALL_SPEC='.[gui]'

REINSTALL=0
WEB=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --reinstall) REINSTALL=1 ;;
        --web)       WEB=1 ;;
        *)           ARGS+=("$arg") ;;
    esac
done

find_python() {
    # Prefer an explicit override, then versioned interpreters meeting the
    # project's requirement (>=3.11), then plain python3.
    if [[ -n "${PYTHON:-}" ]]; then
        command -v "$PYTHON"
        return
    fi
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    return 1
}

if [[ "$REINSTALL" -eq 1 && -d "$VENV_DIR" ]]; then
    echo ">> Removing existing environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_PY" ]]; then
    PYTHON_BIN="$(find_python)" || {
        echo "!! No suitable Python found. TradingAgents needs Python 3.11 or later." >&2
        exit 1
    }
    echo ">> Creating virtual environment with $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Install/refresh the package when it is missing, the sources changed, the
# install spec changed, or a reinstall was requested. The stamp records the
# spec of a successful install so adding extras (e.g. the GUI) triggers it.
needs_install=0
if [[ ! -f "$STAMP" ]]; then
    needs_install=1
elif [[ "$REINSTALL" -eq 1 ]]; then
    needs_install=1
elif [[ "$(cat "$STAMP" 2>/dev/null)" != "$INSTALL_SPEC" ]]; then
    needs_install=1
elif [[ "$SCRIPT_DIR/pyproject.toml" -nt "$STAMP" ]]; then
    needs_install=1
fi

if [[ "$needs_install" -eq 1 ]]; then
    echo ">> Installing TradingAgents into $VENV_DIR (first run may take a while)"
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
    # The gui extra also covers the CLI, so one install serves both modes.
    "$VENV_PY" -m pip install -e "$INSTALL_SPEC"
    printf '%s' "$INSTALL_SPEC" > "$STAMP"
fi

# Run the program from the virtual environment.
if [[ "$WEB" -eq 1 ]]; then
    if [[ -x "$VENV_DIR/bin/tradingagents-gui" ]]; then
        exec "$VENV_DIR/bin/tradingagents-gui" "${ARGS[@]}"
    else
        exec "$VENV_PY" -m gui.app "${ARGS[@]}"
    fi
elif [[ -x "$VENV_DIR/bin/tradingagents" ]]; then
    exec "$VENV_DIR/bin/tradingagents" "${ARGS[@]}"
else
    # Fallback for source checkouts where the console script is unavailable.
    exec "$VENV_PY" -m cli.main "${ARGS[@]}"
fi
