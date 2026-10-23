"""ChatGPT sign-in: OAuth 2.0 + PKCE against OpenAI, Codex-file credentials.

This follows the ``openai-oauth`` implementation (EvanZhouDev/openai-oauth,
Apache-2.0), which in turn uses the same tokens as the Codex CLI. Sign in with
a ChatGPT account and requests go to ``chatgpt.com/backend-api/codex`` on the
subscription you already pay for, instead of billing a platform API key.

Two ways in, both ending in the same credential in ``~/.codex/auth.json``:

* ``browser`` — a loopback redirect on port 1455 catches the authorization
  code. What a developer at a terminal wants.
* ``device``  — show a short code to enter on another machine, for headless
  boxes where no browser can reach the callback.

The access token is short-lived, so callers hold a :class:`ChatGPTSession`
rather than a string: it re-mints from the refresh token on demand, under a
lock, so the many concurrent agent calls in a run share one refresh instead of
racing twenty.

Two details worth keeping in step with the reference, because the token
endpoint is not forgiving about them: the authorization-code exchange is
form-encoded and the refresh is JSON, and the authorize URL asks for the
``codex_cli_simplified_flow`` the Codex backend expects.

Not every ChatGPT plan exposes the Codex backend. :func:`models` lists what the
signed-in account can actually reach, so that is checkable before a run starts
rather than discovered as a 401.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from .codex_auth import (
    account_id_from_tokens,
    auth_file_path,
    clear_tokens,
    is_fedramp,
    load_tokens,
    now,
    plan_from_tokens,
    save_tokens,
    token_expires_at,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHATGPT_BASE_URL",
    "ChatGPTSession",
    "NotSignedIn",
    "codex_client_version",
    "login",
    "logout",
    "models",
    "status",
]

ISSUER = "https://auth.openai.com"
AUTHORIZE_ENDPOINT = f"{ISSUER}/oauth/authorize"
TOKEN_ENDPOINT = f"{ISSUER}/oauth/token"
# Public client id of the OpenAI Codex CLI. It is a public identifier (it ships
# in the codex binary and in every open-source integration of this flow), not a
# secret; what protects the account is the user's own consent in the browser.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
SCOPES = "openid profile email offline_access"

CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"

# The Codex backend, not platform OpenAI: the Responses API only, authenticated
# with a ChatGPT token.
CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"

_TIMEOUT = 30.0
# The reference implementation refreshes 5 minutes before expiry, and also
# after 55 minutes even on a token that claims to be live.
_REFRESH_MARGIN = 5 * 60
_REFRESH_INTERVAL = 55 * 60
_AUTH_WAIT_SECONDS = 300
# The Codex version sent when asking the backend which models this account may
# use. The endpoint gates some models on a minimum client version.
_DEFAULT_CODEX_VERSION = "0.144.1"
_CODEX_REGISTRY_URL = "https://registry.npmjs.org/@openai/codex/latest"
_SEMVER = re.compile(r"\d+\.\d+\.\d+")


class NotSignedIn(RuntimeError):
    """No stored ChatGPT credential, or it could not be refreshed."""


# --- PKCE ------------------------------------------------------------------

def _pkce_pair() -> tuple[str, str]:
    """A ``(code_verifier, code_challenge)`` pair for S256."""
    verifier = secrets.token_urlsafe(48)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _post_token(form: dict, *, as_json: bool = False) -> dict:
    """POST to the token endpoint and return the JSON body.

    The two grants are encoded differently by OpenAI: the authorization-code
    exchange is form-encoded, the refresh is JSON. Sending the wrong one is
    rejected, so ``as_json`` is part of the call rather than a detail.
    """
    if as_json:
        response = requests.post(
            TOKEN_ENDPOINT,
            json={"client_id": CLIENT_ID, **form},
            headers={"Content-Type": "application/json"},
            timeout=_TIMEOUT,
        )
    else:
        response = requests.post(
            TOKEN_ENDPOINT,
            data={"client_id": CLIENT_ID, **form},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_TIMEOUT,
        )
    if response.status_code != 200:
        # The error body names the exact problem (invalid_grant, unauthorized
        # client, ...), so surface it instead of a bare status code.
        raise NotSignedIn(f"ChatGPT sign-in failed (HTTP {response.status_code}): {_detail(response)}")
    return response.json()


def _detail(response) -> str:
    try:
        body = response.json()
        return body.get("error_description") or body.get("error") or response.text[:200]
    except ValueError:
        return response.text[:200]


# --- browser callback ------------------------------------------------------

class _CallbackHandler(BaseHTTPRequestHandler):
    """One-shot handler: capture the authorization code and answer the browser."""

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        query = urllib.parse.urlparse(self.path)
        if query.path != CALLBACK_PATH:
            self._respond(404, "Not found")
            return
        params = urllib.parse.parse_qs(query.query)
        error = (params.get("error") or [None])[0]
        if error:
            self.server.auth_error = error
            self._respond(400, "Sign-in failed. Return to your terminal.")
            return
        if (params.get("state") or [""])[0] != self.server.auth_state:
            self.server.auth_error = "state mismatch"
            self._respond(400, "Sign-in failed (state mismatch). Return to your terminal.")
            return
        code = (params.get("code") or [None])[0]
        if not code:
            self.server.auth_error = "no authorization code"
            self._respond(400, "Sign-in failed (no code). Return to your terminal.")
            return
        self.server.auth_code = code
        self._respond(200, "Signed in. You can close this tab and return to your terminal.")

    def _respond(self, status: int, message: str) -> None:
        body = (
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>TradingAgents</title></head>"
            f"<body style='font-family:system-ui;padding:3rem'>"
            f"<h2>TradingAgents</h2><p>{message}</p></body></html>"
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D102 - silence the default stderr access log
        return


def _await_callback(server: HTTPServer, timeout: int) -> str:
    """Serve until the callback lands, then return the authorization code."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + timeout
        while server.auth_code is None and server.auth_error is None:
            if time.monotonic() > deadline:
                raise NotSignedIn("timed out waiting for the browser to complete sign-in")
            time.sleep(0.2)
    finally:
        server.shutdown()
        server.server_close()
    if server.auth_error:
        raise NotSignedIn(f"ChatGPT sign-in failed: {server.auth_error}")
    return server.auth_code


def _authorize_url(state: str, challenge: str) -> str:
    query = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        # The Codex backend's simplified flow; without it the account id claim
        # may not come back in the id token.
        "codex_cli_simplified_flow": "true",
    })
    return f"{AUTHORIZE_ENDPOINT}?{query}"


def _login_browser() -> dict:
    """Sign in through the browser and a loopback redirect."""
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)
    url = _authorize_url(state, challenge)

    try:
        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    except OSError as exc:
        raise NotSignedIn(
            f"port {CALLBACK_PORT} is unavailable ({exc}); "
            "sign in with the device-code flow instead: tradingagents auth login chatgpt --device"
        ) from exc
    server.auth_code = server.auth_error = None
    server.auth_state = state

    print(f"Opening {url}")
    if not webbrowser.open(url):
        print("No browser could be opened. Visit that URL to continue.")
    return _post_token({
        "grant_type": "authorization_code",
        "code": _await_callback(server, _AUTH_WAIT_SECONDS),
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    })


def _login_device() -> dict:
    """Sign in with a code entered on another machine (headless-friendly)."""
    device = _post_token({"grant_type": "device_code", "scope": SCOPES})
    device_code = device.get("device_code")
    user_code = device.get("user_code")
    verify_url = device.get("verification_uri_complete") or device.get("verification_uri")
    if not device_code or not user_code:
        raise NotSignedIn("the device-code flow is not available for this account")

    print(f"Go to {verify_url}\nEnter code: {user_code}")
    interval = max(int(device.get("interval") or 5), 1)
    deadline = time.monotonic() + int(device.get("expires_in") or 900)
    # The user may take a minute to switch devices, so the endpoint's polling
    # interval is a floor on how often we check, not a deadline for the sign-in.
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            return _post_token({"grant_type": "device_code", "device_code": device_code})
        except NotSignedIn as exc:
            message = str(exc)
            if "authorization_pending" in message:
                continue
            if "slow_down" in message:
                interval += 5
                continue
            raise
    raise NotSignedIn("the device code expired before sign-in completed")


def _store(response: dict) -> dict:
    """Persist a token response and return the record that was written."""
    access_token = response.get("access_token")
    if not access_token:
        raise NotSignedIn("the sign-in response carried no access token")
    id_token = response.get("id_token")
    record = {
        "access_token": access_token,
        "refresh_token": response.get("refresh_token"),
        "id_token": id_token,
        # Fall back to the access token: either one can carry the claim.
        "account_id": response.get("account_id") or account_id_from_tokens(id_token, access_token),
    }
    save_tokens(**record)
    return record


def login(mode: str = "browser") -> dict:
    """Sign in to ChatGPT and store the credential. Returns the token record."""
    if mode not in ("browser", "device"):
        raise ValueError(f"unknown sign-in mode: {mode!r} (use 'browser' or 'device')")
    return _store(_login_browser() if mode == "browser" else _login_device())


def logout() -> bool:
    """Remove the stored ChatGPT credential. True when one was there."""
    return clear_tokens()


def status() -> dict | None:
    """The stored credential's account and plan, or None when signed out.

    Refreshes first when needed, so a plan that lapsed since the last run is
    reported from a current token rather than a stale one.
    """
    if not load_tokens():
        return None
    try:
        session = ChatGPTSession()
    except NotSignedIn:
        return None
    details = session.describe()
    details["path"] = str(auth_file_path())
    return details


# --- the account's model catalog -------------------------------------------

def codex_client_version() -> str:
    """The Codex version to report when asking the backend for models.

    The endpoint gates some models on a minimum client version, so this asks npm
    for the current release and falls back to a known-good one when the network
    says nothing useful. ``TRADINGAGENTS_CODEX_VERSION`` pins it.
    """
    pinned = os.environ.get("TRADINGAGENTS_CODEX_VERSION")
    if pinned:
        return pinned
    try:
        response = requests.get(
            _CODEX_REGISTRY_URL, headers={"accept": "application/json"}, timeout=_TIMEOUT
        )
        if response.ok:
            version = response.json().get("version")
            if isinstance(version, str) and (match := _SEMVER.search(version)):
                return match.group(0)
    except Exception as exc:  # noqa: BLE001 - any failure just means "use the default"
        logger.debug("could not resolve the Codex version: %s", exc)
    return _DEFAULT_CODEX_VERSION


def models(session: ChatGPTSession | None = None) -> list[str]:
    """The model IDs this account can reach, or ``[]`` if the list is unavailable.

    Which models a subscription exposes is account- and plan-dependent, and it
    changes; the backend is the only authority. A failure here is not fatal —
    the user can still type a model ID — so it returns an empty list.
    """
    session = session or ChatGPTSession()
    try:
        response = requests.get(
            f"{CHATGPT_BASE_URL}/models",
            params={"client_version": codex_client_version()},
            headers=session.auth_headers(),
            timeout=_TIMEOUT,
        )
        if not response.ok:
            logger.debug("model catalog returned HTTP %s", response.status_code)
            return []
        entries = response.json().get("models")
    except Exception as exc:  # noqa: BLE001 - the catalog is a convenience
        logger.debug("could not load the ChatGPT model catalog: %s", exc)
        return []
    if not isinstance(entries, list):
        return []
    return [
        slug for slug in (_public_slug(entry) for entry in entries) if slug
    ]


def _public_slug(entry) -> str | None:
    """A catalog entry's slug, if the model is reachable by an API client."""
    if not isinstance(entry, dict):
        return None
    slug = entry.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    if entry.get("supported_in_api") is False:
        return None
    visibility = entry.get("visibility")
    if visibility is not None and visibility != "list":
        return None
    return slug


# --- session ---------------------------------------------------------------

class ChatGPTSession:
    """A live ChatGPT credential: reads the file, refreshes when it is stale.

    Hold one per run and call :attr:`access_token` per request — the token
    behind it may be re-minted between calls, so caching the string is what
    makes long multi-agent runs fail partway through with a 401.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens = load_tokens()
        if not self._tokens:
            raise NotSignedIn("not signed in to ChatGPT; run: tradingagents auth login chatgpt")

    @property
    def account_id(self) -> str | None:
        return self._tokens.get("account_id") or account_id_from_tokens(
            self._tokens.get("id_token"), self._tokens.get("access_token")
        )

    @property
    def is_fedramp(self) -> bool:
        return is_fedramp(self._tokens.get("id_token"), self._tokens.get("access_token"))

    def auth_headers(self) -> dict[str, str]:
        """The headers the Codex backend authenticates with.

        Only the bearer token, the account id, and — for a FedRAMP account, which
        the backend routes differently — the FedRAMP marker. Deliberately no
        ``OpenAI-Beta`` header: the endpoint has not needed one for this flow,
        and sending a stale beta flag is a common way to get a 400.
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "chatgpt-account-id": self.account_id or "",
        }
        if self.is_fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers

    def describe(self) -> dict:
        """Account id, plan and token expiry, after making the token current."""
        _ = self.access_token  # refreshes a stale token, so the expiry is honest
        return {
            "account_id": self.account_id,
            "plan": plan_from_tokens(self._tokens.get("id_token"), self._tokens.get("access_token")),
            "expires_at": token_expires_at(self._tokens.get("access_token")),
        }

    def _stale(self) -> bool:
        """Whether the access token should be re-minted before use."""
        expires_at = token_expires_at(self._tokens.get("access_token"))
        if expires_at is not None and expires_at <= now() + _REFRESH_MARGIN:
            return True
        # A token with no readable expiry is trusted only until it is an hour
        # old, so a long run cannot keep replaying one the backend will reject.
        age = self._age()
        return age is None or age >= _REFRESH_INTERVAL

    def _age(self) -> float | None:
        """Seconds since the credential file was last written, or None if unreadable."""
        try:
            return max(0.0, now() - auth_file_path().stat().st_mtime)
        except OSError:
            return None

    def _refresh(self) -> None:
        refresh_token = self._tokens.get("refresh_token")
        if not refresh_token:
            raise NotSignedIn(
                "the stored ChatGPT credential cannot be refreshed (no refresh token); "
                "sign in again: tradingagents auth login chatgpt"
            )
        # JSON-encoded, unlike the authorization-code exchange.
        payload = _post_token(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}, as_json=True
        )
        access_token = payload.get("access_token") or self._tokens["access_token"]
        id_token = payload.get("id_token") or self._tokens.get("id_token")
        self._tokens = {
            "access_token": access_token,
            # The endpoint may rotate the refresh token; keep the new one and
            # only fall back to the old when it omits the field.
            "refresh_token": payload.get("refresh_token") or refresh_token,
            "id_token": id_token,
            "account_id": payload.get("account_id") or self.account_id or account_id_from_tokens(
                id_token, access_token
            ),
        }
        # Persisted through save_tokens so a second process reading the file
        # does not refresh the same stale token again.
        save_tokens(**self._tokens)

    @property
    def access_token(self) -> str:
        """A token valid now, re-minted from the refresh token if it is stale."""
        with self._lock:
            if self._stale():
                self._refresh()
            return self._tokens["access_token"]
