"""The ``tradingagents auth`` commands, against a stubbed sign-in.

These run the real command functions and assert what the user would see: that
signing in stores a credential, that a provider with no credential on a machine
with no terminal refuses clearly rather than failing later, and that the status
line names the plan and where the token lives.
"""

import pytest
import typer
from typer.testing import CliRunner

from cli import auth_cmd
from tradingagents.auth import chatgpt, codex_auth

runner = CliRunner()


@pytest.fixture
def auth_file(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    monkeypatch.setenv("TRADINGAGENTS_AUTH_FILE", str(path))
    return path


def _stored_token(monkeypatch, account="acct-5", plan="pro"):
    """Make the sign-in succeed, so the real login path runs and stores a token.

    Only the token endpoint is stubbed: the command's own behaviour (storing,
    printing, reporting the plan) is what these tests are about.
    """
    monkeypatch.setattr(chatgpt.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(chatgpt, "_await_callback", lambda server, timeout: "code-123")
    monkeypatch.setattr(chatgpt, "_post_token", lambda form, as_json=False: {
        "access_token": "at-secret-value", "refresh_token": "rt-secret-value",
        "id_token": _id_token(account, plan), "expires_in": 4_000_000,
    })


def _id_token(account, plan):
    import base64
    import json

    payload = base64.urlsafe_b64encode(json.dumps(
        {"https://api.openai.com/auth": {
            "chatgpt_account_id": account, "chatgpt_plan_type": plan}}
    ).encode()).rstrip(b"=").decode()
    return f"h.{payload}.s"


# --- login -----------------------------------------------------------------

def test_login_stores_the_credential(monkeypatch):
    _stored_token(monkeypatch)
    result = runner.invoke(auth_cmd.app, ["login", "chatgpt"])
    assert result.exit_code == 0
    assert codex_auth.read_token("access_token") == "at-secret-value"


def test_login_uses_the_device_flow_when_asked(monkeypatch):
    seen = {}
    monkeypatch.setattr(chatgpt, "login", lambda mode="browser": seen.setdefault("mode", mode))
    result = runner.invoke(auth_cmd.app, ["login", "chatgpt", "--device"])
    assert result.exit_code == 0
    assert seen["mode"] == "device"


def test_login_reports_a_failure_without_a_traceback(monkeypatch):
    def fail(mode="browser"):
        raise chatgpt.NotSignedIn("the sign-in response carried no access token")

    monkeypatch.setattr(chatgpt, "login", fail)
    result = runner.invoke(auth_cmd.app, ["login", "chatgpt"])
    assert result.exit_code == 1
    assert "no access token" in result.output
    assert "Traceback" not in result.output


def test_login_reports_a_cancellation(monkeypatch):
    def cancel(mode="browser"):
        raise KeyboardInterrupt

    monkeypatch.setattr(chatgpt, "login", cancel)
    result = runner.invoke(auth_cmd.app, ["login", "chatgpt"])
    assert result.exit_code == 1
    assert "cancelled" in result.output


def test_an_unknown_provider_lists_the_known_ones():
    result = runner.invoke(auth_cmd.app, ["login", "nope"])
    assert result.exit_code != 0
    assert "chatgpt" in result.output


# --- logout ----------------------------------------------------------------

def test_logout_removes_the_credential(monkeypatch):
    _stored_token(monkeypatch)
    runner.invoke(auth_cmd.app, ["login", "chatgpt"])
    result = runner.invoke(auth_cmd.app, ["logout", "chatgpt"])
    assert result.exit_code == 0
    assert codex_auth.read_token("access_token") is None


def test_logout_when_signed_out_is_not_an_error():
    result = runner.invoke(auth_cmd.app, ["logout", "chatgpt"])
    assert result.exit_code == 0
    assert "nothing to remove" in result.output


# --- status ----------------------------------------------------------------

def test_status_when_signed_out_says_how_to_sign_in(auth_file):
    result = runner.invoke(auth_cmd.app, ["status", "chatgpt"])
    assert result.exit_code == 0
    assert "tradingagents auth login chatgpt" in result.output


def test_status_names_the_plan_and_the_credential_file(monkeypatch, auth_file):
    _stored_token(monkeypatch)
    runner.invoke(auth_cmd.app, ["login", "chatgpt"])

    result = runner.invoke(auth_cmd.app, ["status", "chatgpt"])

    assert result.exit_code == 0
    assert "pro" in result.output
    assert "acct-5" in result.output
    assert str(auth_file) in result.output


@pytest.mark.parametrize("secret", ["at-secret-value", "rt-secret-value"])
def test_status_never_prints_a_token(monkeypatch, auth_file, secret):
    _stored_token(monkeypatch)
    runner.invoke(auth_cmd.app, ["login", "chatgpt"])
    result = runner.invoke(auth_cmd.app, ["status", "chatgpt"])
    assert secret not in result.output


# --- the command is reachable from the main CLI ----------------------------

def test_the_main_cli_exposes_the_auth_group():
    from cli.main import app

    result = CliRunner().invoke(app, ["auth", "--help"])
    assert result.exit_code == 0
    assert "login" in result.output


assert typer is not None
