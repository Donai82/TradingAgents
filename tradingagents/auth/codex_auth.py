"""Reading and writing ``~/.codex/auth.json``, the Codex CLI's credential file.

ChatGPT sign-in uses the same file, and the same shape, as the Codex CLI and the
``openai-oauth`` project::

    {"auth_mode": "chatgpt",
     "OPENAI_API_KEY": "...",              # present only if the user has one
     "tokens": {"id_token": "...", "access_token": "...",
                "refresh_token": "...", "account_id": "..."},
     "last_refresh": "2026-01-09T12:00:00.000Z"}

Sharing the file rather than keeping a private one is the point: a user who has
already run ``codex login`` or ``npx openai-oauth login`` is signed in here too,
and a sign-in in either place signs in the other. So the format is read and
written as given, and unknown fields are preserved — the file belongs to the
Codex CLI as much as to us, and it stores an ``OPENAI_API_KEY`` that a refresh
here must not drop.

$CODEX_HOME moves it, as it does for Codex. ``TRADINGAGENTS_AUTH_FILE`` is the
equivalent override for people who do not want their Codex home touched.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .store import read_json, write_json

__all__ = [
    "CHATGPT_ACCOUNT_CLAIM",
    "auth_file_path",
    "clear_tokens",
    "jwt_claims",
    "load_tokens",
    "read_token",
    "save_tokens",
    "token_expires_at",
]

# The namespace OpenAI puts ChatGPT account facts in, inside both tokens.
CHATGPT_ACCOUNT_CLAIM = "https://api.openai.com/auth"

_AUTH_FILENAME = "auth.json"


def auth_file_path() -> Path:
    """Where the ChatGPT credential lives.

    ``TRADINGAGENTS_AUTH_FILE`` wins, then ``$CODEX_HOME/auth.json``, then
    ``~/.codex/auth.json`` — the same order the Codex CLI uses.
    """
    override = os.environ.get("TRADINGAGENTS_AUTH_FILE")
    if override:
        return Path(override).expanduser()
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / _AUTH_FILENAME
    return Path(os.path.expanduser("~")) / ".codex" / _AUTH_FILENAME


# --- JWT claims ------------------------------------------------------------

def jwt_claims(token: str | None) -> dict:
    """The payload of a JWT, unverified, or ``{}``.

    The token arrived from the token endpoint over TLS and is used only to read
    the expiry and the account claim that must be echoed on every request, so
    there is nothing to verify it against.
    """
    if not token or not isinstance(token, str):
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padded = payload + "=" * (-len(payload) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, TypeError):
        return {}


def _chatgpt_claim(token: str | None) -> dict:
    claim = jwt_claims(token).get(CHATGPT_ACCOUNT_CLAIM)
    return claim if isinstance(claim, dict) else {}


def token_expires_at(token: str | None) -> float | None:
    """The token's ``exp`` claim as a POSIX timestamp, or None if it has none.

    Read from the token rather than from a stored expiry: the JWT is the
    authority, and a stored value written by another tool can be stale.
    """
    exp = jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def account_id_from_tokens(*tokens: str | None) -> str | None:
    """The ChatGPT account id, from the first token that carries one.

    Looked for in the namespaced auth claim, then as a top-level claim, then as
    the first organization — the same fallbacks the reference implementation
    uses, so a token minted by any of these tools resolves.
    """
    for token in tokens:
        claim = _chatgpt_claim(token)
        account_id = claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
        claims = jwt_claims(token)
        account_id = claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
        organizations = claims.get("organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, dict) and isinstance(first.get("id"), str) and first["id"]:
                return first["id"]
    return None


def is_fedramp(*tokens: str | None) -> bool:
    """Whether the account is a FedRAMP one, which needs its own request header."""
    return any(_chatgpt_claim(token).get("chatgpt_account_is_fedramp") is True for token in tokens)


def plan_from_tokens(*tokens: str | None) -> str | None:
    """The ChatGPT plan name, when the token carries one."""
    for token in tokens:
        plan = _chatgpt_claim(token).get("chatgpt_plan_type")
        if isinstance(plan, str) and plan:
            return plan
    return None


# --- the file --------------------------------------------------------------

def load_tokens() -> dict:
    """The stored ``tokens`` object, or ``{}`` when signed out."""
    tokens = read_json(auth_file_path()).get("tokens")
    if not isinstance(tokens, dict):
        return {}
    return {
        key: value
        for key, value in tokens.items()
        if isinstance(value, str) and value
    }


def read_token(key: str) -> str | None:
    return load_tokens().get(key)


def save_tokens(
    access_token: str,
    *,
    refresh_token: str | None = None,
    id_token: str | None = None,
    account_id: str | None = None,
) -> Path:
    """Store a token set, preserving everything else in the file.

    ``OPENAI_API_KEY`` and any field another tool wrote stay put: this file is
    shared with the Codex CLI, so a refresh must not clobber a credential it
    did not write.
    """
    path = auth_file_path()
    auth = read_json(path)
    tokens = {
        "id_token": id_token,
        "access_token": access_token,
        # Keep the previous refresh token when the response omits a new one;
        # some responses do, and losing it means signing in again.
        "refresh_token": refresh_token or auth.get("tokens", {}).get("refresh_token"),
        "account_id": account_id,
    }
    auth["auth_mode"] = "chatgpt"
    auth["tokens"] = {k: v for k, v in tokens.items() if v}
    auth["last_refresh"] = _utcnow()
    return write_json(path, auth)


def clear_tokens() -> bool:
    """Sign out: drop the ChatGPT credential, keeping the rest of the file.

    The whole file is only removed when the ChatGPT credential was the only
    thing in it, so signing out here never deletes a Codex CLI API key.
    """
    path = auth_file_path()
    auth = read_json(path)
    if "tokens" not in auth and "auth_mode" not in auth:
        return False
    auth.pop("tokens", None)
    auth.pop("auth_mode", None)
    auth.pop("last_refresh", None)
    if not auth:
        path.unlink(missing_ok=True)
        return True
    write_json(path, auth)
    return True


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now() -> float:
    """Wall clock, as a seam for tests that must not depend on real time."""
    return time.time()
