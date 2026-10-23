"""ChatGPT sign-in and the Codex credential file.

The file, its location and its shape all come from EvanZhouDev/openai-oauth and
the Codex CLI, because that is what makes a sign-in in either place a sign-in
here. These tests pin that contract, not just the happy path: the schema has to
stay readable by the other tools, and a refresh must not drop a credential this
module did not write.

The network is stubbed at ``requests``; nothing here reaches auth.openai.com.
"""

import base64
import json
import os
import stat
import time
from unittest import mock

import pytest

from tradingagents.auth import chatgpt, codex_auth


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{payload}.signature"


def _id_token(account="acct_1", plan="plus", fedramp=False, exp=None):
    auth = {"chatgpt_account_id": account, "chatgpt_plan_type": plan}
    if fedramp:
        auth["chatgpt_account_is_fedramp"] = True
    claims = {codex_auth.CHATGPT_ACCOUNT_CLAIM: auth}
    if exp is not None:
        claims["exp"] = exp
    return _jwt(claims)


@pytest.fixture
def auth_file(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    monkeypatch.setenv("TRADINGAGENTS_AUTH_FILE", str(path))
    return path


# --- where the credential lives --------------------------------------------

def test_the_default_location_is_the_codex_cli_auth_file(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_AUTH_FILE", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    expected = os.path.join(os.path.expanduser("~"), ".codex", "auth.json")
    assert codex_auth.auth_file_path() == codex_auth.Path(expected)


def test_codex_home_moves_the_credential(monkeypatch, tmp_path):
    monkeypatch.delenv("TRADINGAGENTS_AUTH_FILE", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    assert codex_auth.auth_file_path() == tmp_path / "codex" / "auth.json"


def test_an_explicit_path_wins_over_codex_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_AUTH_FILE", str(tmp_path / "mine.json"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    assert codex_auth.auth_file_path() == tmp_path / "mine.json"


# --- the file's shape ------------------------------------------------------

def test_the_file_matches_the_codex_schema(auth_file):
    codex_auth.save_tokens(
        "at", refresh_token="rt", id_token=_id_token("acct_9", "pro"), account_id="acct_9"
    )
    written = json.loads(auth_file.read_text())

    assert written["auth_mode"] == "chatgpt"
    assert set(written["tokens"]) == {"id_token", "access_token", "refresh_token", "account_id"}
    assert written["tokens"]["access_token"] == "at"
    assert written["tokens"]["account_id"] == "acct_9"
    assert written["last_refresh"].endswith("Z")


def test_the_credential_is_owner_only(auth_file):
    codex_auth.save_tokens("at", refresh_token="rt")
    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
    # Searchable, so a later write can land in it.
    assert stat.S_IMODE(auth_file.parent.stat().st_mode) == 0o700


def test_writing_leaves_no_temp_file_behind(auth_file):
    codex_auth.save_tokens("first", refresh_token="rt")
    codex_auth.save_tokens("second", refresh_token="rt")
    assert [p.name for p in auth_file.parent.iterdir()] == [auth_file.name]


def test_an_api_key_written_by_the_codex_cli_survives_a_refresh(auth_file):
    # This file is shared: a refresh must not drop a credential it did not write.
    auth_file.write_text(json.dumps({
        "OPENAI_API_KEY": "sk-codex-cli",
        "tokens": {"access_token": "old", "refresh_token": "rt"},
        "last_refresh": "2026-01-01T00:00:00.000Z",
    }))

    codex_auth.save_tokens("new", refresh_token="rt2")

    written = json.loads(auth_file.read_text())
    assert written["OPENAI_API_KEY"] == "sk-codex-cli"
    assert written["tokens"]["access_token"] == "new"


def test_an_unrelated_field_survives(auth_file):
    auth_file.write_text(json.dumps({"tokens": {"access_token": "old"}, "future": {"x": 1}}))
    codex_auth.save_tokens("new")
    assert json.loads(auth_file.read_text())["future"] == {"x": 1}


def test_logout_keeps_a_codex_cli_api_key(auth_file):
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": "sk-codex-cli", "tokens": {"access_token": "a"}}))

    assert codex_auth.clear_tokens() is True

    written = json.loads(auth_file.read_text())
    assert written["OPENAI_API_KEY"] == "sk-codex-cli"
    assert "tokens" not in written


def test_logout_removes_the_file_when_nothing_else_is_there(auth_file):
    codex_auth.save_tokens("at", refresh_token="rt")
    assert codex_auth.clear_tokens() is True
    assert not auth_file.exists()


def test_logout_when_signed_out_reports_false(auth_file):
    assert codex_auth.clear_tokens() is False


def test_a_corrupt_file_reads_as_signed_out(auth_file):
    auth_file.write_text("{not json", encoding="utf-8")
    assert codex_auth.load_tokens() == {}


def test_reading_does_not_create_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_AUTH_FILE", str(tmp_path / "never.json"))
    assert codex_auth.load_tokens() == {}
    assert not (tmp_path / "never.json").exists()


# --- token claims ----------------------------------------------------------

def test_the_expiry_comes_from_the_token_not_a_stored_field():
    # The JWT is the authority; a value written by another tool can be stale.
    expires = codex_auth.token_expires_at(_id_token(exp=1_800_000_000))
    assert expires == 1_800_000_000
    assert codex_auth.token_expires_at(_id_token()) is None
    assert codex_auth.token_expires_at("nonsense") is None
    assert codex_auth.token_expires_at(None) is None


def test_the_account_id_is_read_from_the_auth_claim():
    assert codex_auth.account_id_from_tokens(_id_token("acct_42")) == "acct_42"


def test_the_account_id_falls_back_to_a_top_level_claim():
    assert codex_auth.account_id_from_tokens(_jwt({"chatgpt_account_id": "acct_t"})) == "acct_t"


def test_the_account_id_falls_back_to_the_first_organization():
    token = _jwt({"organizations": [{"id": "org-1"}, {"id": "org-2"}]})
    assert codex_auth.account_id_from_tokens(token) == "org-1"


def test_the_account_id_is_taken_from_whichever_token_carries_it():
    assert codex_auth.account_id_from_tokens(None, _id_token("acct_9")) == "acct_9"


def test_a_fedramp_account_is_recognized():
    assert codex_auth.is_fedramp(_id_token(fedramp=True)) is True
    assert codex_auth.is_fedramp(_id_token()) is False
    assert codex_auth.is_fedramp(None) is False


def test_the_plan_is_read_from_the_auth_claim():
    assert codex_auth.plan_from_tokens(_id_token(plan="team")) == "team"
    assert codex_auth.plan_from_tokens(None) is None


# --- session ---------------------------------------------------------------

def test_session_without_a_credential_raises(auth_file):
    with pytest.raises(chatgpt.NotSignedIn, match="tradingagents auth login chatgpt"):
        chatgpt.ChatGPTSession()


def test_a_token_expiring_soon_is_refreshed(auth_file, monkeypatch):
    # Inside the 5-minute margin: a run must not start a request with a token
    # that lapses while it is in flight.
    stale = _id_token(exp=time.time() + 30)
    codex_auth.save_tokens(stale, refresh_token="r1", account_id="acct-1")
    monkeypatch.setattr(
        chatgpt, "_post_token", lambda form, as_json=False: {"access_token": "fresh"}
    )

    assert chatgpt.ChatGPTSession().access_token == "fresh"


def test_a_live_token_is_not_refreshed(auth_file, monkeypatch):
    live = _id_token(exp=time.time() + 3600)
    codex_auth.save_tokens(live, refresh_token="r1", account_id="acct-1")

    def explode(form, as_json=False):
        raise AssertionError("a live token must not be refreshed")

    monkeypatch.setattr(chatgpt, "_post_token", explode)
    assert chatgpt.ChatGPTSession().access_token == live


def test_the_refresh_is_json_encoded(auth_file, monkeypatch):
    # OpenAI accepts the authorization-code exchange as a form and the refresh
    # as JSON; sending the wrong one is rejected.
    codex_auth.save_tokens("stale", refresh_token="r1", account_id="acct-1")
    _expire(auth_file, within_seconds=-1)
    seen = {}

    def fake(form, as_json=False):
        seen.update(form=form, as_json=as_json)
        return {"access_token": "fresh"}

    monkeypatch.setattr(chatgpt, "_post_token", fake)
    _ = chatgpt.ChatGPTSession().access_token

    assert seen["as_json"] is True
    assert seen["form"]["grant_type"] == "refresh_token"


def test_the_code_exchange_is_form_encoded(auth_file, monkeypatch):
    monkeypatch.setattr(chatgpt.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(chatgpt, "_await_callback", lambda server, timeout: "code-123")
    seen = {}

    def fake(form, as_json=False):
        seen.update(form=form, as_json=as_json)
        return {"access_token": "at", "id_token": _id_token("acct-1")}

    monkeypatch.setattr(chatgpt, "_post_token", fake)
    chatgpt.login()

    assert seen["as_json"] is False
    assert seen["form"]["grant_type"] == "authorization_code"


def test_a_refresh_persists_a_rotated_refresh_token(auth_file, monkeypatch):
    codex_auth.save_tokens("stale", refresh_token="r1", account_id="acct-1")
    _expire(auth_file, within_seconds=-1)
    monkeypatch.setattr(
        chatgpt,
        "_post_token",
        lambda form, as_json=False: {"access_token": "fresh", "refresh_token": "r2"},
    )

    _ = chatgpt.ChatGPTSession().access_token
    assert codex_auth.read_token("refresh_token") == "r2"


def test_a_refresh_without_a_refresh_token_explains_itself(auth_file):
    codex_auth.save_tokens("stale", account_id="acct-1")
    _expire(auth_file, within_seconds=-1)
    with pytest.raises(chatgpt.NotSignedIn, match="sign in again"):
        _ = chatgpt.ChatGPTSession().access_token


def test_status_is_none_when_signed_out(auth_file):
    assert chatgpt.status() is None


def test_status_reports_the_plan_account_and_path(auth_file):
    codex_auth.save_tokens(
        "at",
        refresh_token="rt",
        id_token=_id_token("acct_7", "team", exp=time.time() + 3600),
        account_id="acct_7",
    )
    details = chatgpt.status()
    assert details["plan"] == "team"
    assert details["account_id"] == "acct_7"
    assert details["path"] == str(auth_file)


# --- request headers -------------------------------------------------------

def test_the_auth_headers_carry_the_token_and_account(auth_file):
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="acct-7")

    headers = chatgpt.ChatGPTSession().auth_headers()

    assert headers["Authorization"] == "Bearer at"
    assert headers["chatgpt-account-id"] == "acct-7"


def test_a_fedramp_account_is_marked(auth_file):
    codex_auth.save_tokens(
        "at", refresh_token="rt", id_token=_id_token(fedramp=True, exp=time.time() + 3600), account_id="a"
    )
    assert chatgpt.ChatGPTSession().auth_headers()["X-OpenAI-Fedramp"] == "true"


def test_a_normal_account_is_not_marked(auth_file):
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="a")
    assert "X-OpenAI-Fedramp" not in chatgpt.ChatGPTSession().auth_headers()


def test_no_beta_header_is_sent(auth_file):
    # The reference implementation deletes any openai-beta header rather than
    # setting one; a stale beta flag is a common way to get a 400.
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="a")
    assert not any("beta" in name.lower() for name in chatgpt.ChatGPTSession().auth_headers())


# --- login -----------------------------------------------------------------

def test_login_stores_the_credential(auth_file, monkeypatch):
    monkeypatch.setattr(chatgpt.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(chatgpt, "_await_callback", lambda server, timeout: "code-123")
    monkeypatch.setattr(chatgpt, "_post_token", lambda form, as_json=False: {
        "access_token": "at", "refresh_token": "rt", "id_token": _id_token("acct_3", "plus"),
        "expires_in": 3600,
    })

    chatgpt.login()

    assert codex_auth.load_tokens()["account_id"] == "acct_3"


def test_the_authorize_url_carries_the_simplified_flow_flag():
    # Without it the account id claim may not come back in the id token.
    url = chatgpt._authorize_url("st8", "ch4llenge")
    assert "codex_cli_simplified_flow=true" in url
    assert "id_token_add_organizations=true" in url
    assert "code_challenge_method=S256" in url


def test_login_device_polls_until_a_token_arrives(auth_file, monkeypatch):
    def fake(form, as_json=False):
        if "device_code" not in form:
            return {"device_code": "dc", "user_code": "ABCD-1234",
                    "verification_uri": "https://x/verify", "interval": 1}
        return {"access_token": "at", "refresh_token": "rt", "id_token": _id_token()}

    monkeypatch.setattr(chatgpt, "_post_token", fake)
    monkeypatch.setattr(chatgpt.time, "sleep", lambda s: None)

    chatgpt.login(mode="device")
    assert codex_auth.read_token("access_token") == "at"


def test_login_device_keeps_polling_while_authorization_is_pending(auth_file, monkeypatch):
    state = {"issued": False, "polls": 0}

    def fake(form, as_json=False):
        if not state["issued"]:
            state["issued"] = True
            return {"device_code": "dc", "user_code": "ABCD",
                    "verification_uri": "https://x/v", "interval": 1}
        state["polls"] += 1
        if state["polls"] < 2:
            # The expected "not yet" answer; sign-in must wait, not abort.
            raise chatgpt.NotSignedIn("HTTP 400: authorization_pending")
        return {"access_token": "at", "refresh_token": "rt", "id_token": _id_token()}

    monkeypatch.setattr(chatgpt, "_post_token", fake)
    monkeypatch.setattr(chatgpt.time, "sleep", lambda s: None)

    chatgpt.login(mode="device")
    assert state["polls"] == 2


def test_login_device_aborts_on_a_real_rejection(auth_file, monkeypatch):
    def fake(form, as_json=False):
        if "device_code" not in form:
            return {"device_code": "dc", "user_code": "ABCD", "verification_uri": "https://x/v"}
        raise chatgpt.NotSignedIn("HTTP 400: access_denied")

    monkeypatch.setattr(chatgpt, "_post_token", fake)
    monkeypatch.setattr(chatgpt.time, "sleep", lambda s: None)

    with pytest.raises(chatgpt.NotSignedIn, match="access_denied"):
        chatgpt.login(mode="device")


def test_login_rejects_an_unknown_mode(auth_file):
    with pytest.raises(ValueError, match="unknown sign-in mode"):
        chatgpt.login(mode="carrier-pigeon")


def test_login_without_an_access_token_raises(auth_file, monkeypatch):
    monkeypatch.setattr(chatgpt.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(chatgpt, "_await_callback", lambda server, timeout: "code-123")
    monkeypatch.setattr(chatgpt, "_post_token", lambda form, as_json=False: {"refresh_token": "rt"})
    with pytest.raises(chatgpt.NotSignedIn, match="no access token"):
        chatgpt.login()


def test_the_callback_handler_captures_the_code():
    handler = _callback_handler(state="expected", path=f"{chatgpt.CALLBACK_PATH}?state=expected&code=abc123")
    assert handler.server.auth_code == "abc123"
    assert handler.server.auth_error is None


def test_the_callback_handler_rejects_a_state_mismatch():
    handler = _callback_handler(state="expected", path=f"{chatgpt.CALLBACK_PATH}?state=forged&code=x")
    assert handler.server.auth_error == "state mismatch"
    assert handler.server.auth_code is None


def test_the_callback_handler_reports_a_provider_error():
    handler = _callback_handler(state="expected", path=f"{chatgpt.CALLBACK_PATH}?error=access_denied")
    assert handler.server.auth_error == "access_denied"


def test_pkce_pair_is_a_verifier_and_matching_challenge():
    import hashlib

    verifier, challenge = chatgpt._pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected
    assert verifier != challenge  # fresh random verifier each call


# --- the model catalog -----------------------------------------------------

def test_the_catalog_lists_what_the_account_can_reach(auth_file, monkeypatch):
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="a")
    monkeypatch.setattr(chatgpt, "codex_client_version", lambda: "1.2.3")
    monkeypatch.setattr(chatgpt.requests, "get", lambda *a, **k: _response(200, {"models": [
        {"slug": "gpt-5.6-sol"}, {"slug": "gpt-5.6-terra"},
    ]}))

    assert chatgpt.models() == ["gpt-5.6-sol", "gpt-5.6-terra"]


def test_the_catalog_filters_models_an_api_client_cannot_use(auth_file, monkeypatch):
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="a")
    monkeypatch.setattr(chatgpt.requests, "get", lambda *a, **k: _response(200, {"models": [
        {"slug": "visible"},
        {"slug": "hidden", "visibility": "internal"},
        {"slug": "not-in-api", "supported_in_api": False},
        {"no_slug": True},
        "junk",
    ]}))

    assert chatgpt.models() == ["visible"]


def test_the_catalog_failure_is_not_fatal(auth_file, monkeypatch):
    # The model is typed by hand in the CLI, so an unavailable list is a
    # nuisance rather than a failure.
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="a")
    monkeypatch.setattr(chatgpt.requests, "get", lambda *a, **k: _response(500, {}))
    assert chatgpt.models() == []


def test_the_catalog_sends_the_client_version_and_auth(auth_file, monkeypatch):
    codex_auth.save_tokens("at", refresh_token="rt", id_token=_id_token(exp=time.time() + 3600), account_id="acct-5")
    monkeypatch.setattr(chatgpt, "codex_client_version", lambda: "9.9.9")
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return _response(200, {"models": [{"slug": "m"}]})

    monkeypatch.setattr(chatgpt.requests, "get", fake_get)
    chatgpt.models()

    # Some models are gated on a minimum client version, so the version matters.
    assert seen["params"] == {"client_version": "9.9.9"}
    assert seen["headers"]["Authorization"] == "Bearer at"
    assert seen["headers"]["chatgpt-account-id"] == "acct-5"


def test_the_codex_version_falls_back_when_npm_is_unreachable(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_CODEX_VERSION", raising=False)
    monkeypatch.setattr(chatgpt.requests, "get", mock.Mock(side_effect=OSError("no network")))
    assert chatgpt.codex_client_version() == chatgpt._DEFAULT_CODEX_VERSION


def test_the_codex_version_can_be_pinned(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CODEX_VERSION", "0.0.1")
    assert chatgpt.codex_client_version() == "0.0.1"


# --- test doubles ----------------------------------------------------------

def _expire(path, within_seconds: int) -> None:
    """Store an access token whose ``exp`` is ``within_seconds`` away."""
    data = json.loads(path.read_text())
    data["tokens"]["access_token"] = _id_token(exp=time.time() + within_seconds)
    path.write_text(json.dumps(data))


def _response(status, payload):
    class _Response:
        status_code = status
        ok = status < 400

        def json(self):
            if self._payload is None:
                raise ValueError("no JSON")
            return self._payload

        _payload = payload

    return _Response()


def _callback_handler(*, state, path):
    """Drive the real callback handler with a recorded request."""
    handler = chatgpt._CallbackHandler.__new__(chatgpt._CallbackHandler)
    handler.path = path
    handler.server = type("S", (), {"auth_state": state, "auth_code": None, "auth_error": None})()
    handler.send_response = lambda *a, **k: None
    handler.send_header = lambda *a, **k: None
    handler.end_headers = lambda *a, **k: None
    handler.wfile = type("W", (), {"write": lambda self, b: None})()
    handler.log_message = lambda *a, **k: None
    chatgpt._CallbackHandler.do_GET(handler)
    return handler
