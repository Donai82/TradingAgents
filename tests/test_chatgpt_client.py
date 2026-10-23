"""The ChatGPT (OAuth) provider client.

The two things worth guarding are the endpoint and the auth: the request has to
go to the Codex backend on the Responses API (not the platform API), and it has
to carry a bearer token that is re-read per request rather than captured when
the client was built. A run that outlives the token is the normal case, not an
edge case, so the second point is asserted on a real request.
"""

import base64
import json
import time

import pytest

from tradingagents.auth import chatgpt as chatgpt_auth, codex_auth
from tradingagents.llm_clients.chatgpt_client import ChatGPTClient
from tradingagents.llm_clients.factory import create_llm_client


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{payload}.signature"


@pytest.fixture
def signed_in(tmp_path, monkeypatch):
    """A stored ChatGPT credential, valid for an hour."""
    monkeypatch.setenv("TRADINGAGENTS_AUTH_FILE", str(tmp_path / "auth.json"))
    codex_auth.save_tokens(
        _jwt({"exp": time.time() + 3600}),
        refresh_token="r1",
        account_id="acct-77",
    )
    return "acct-77"


# --- endpoint and API ------------------------------------------------------

def test_the_client_targets_the_codex_backend(signed_in):
    llm = ChatGPTClient("gpt-5.1-codex").get_llm()
    assert str(llm.openai_api_base).rstrip("/") == chatgpt_auth.CHATGPT_BASE_URL
    assert "chatgpt.com" in str(llm.openai_api_base)


def test_the_codex_backend_speaks_the_responses_api(signed_in):
    # The backend has no /chat/completions; only the Responses API is routed.
    assert ChatGPTClient("gpt-5.1-codex").get_llm().use_responses_api is True


def test_a_base_url_override_is_honored(signed_in):
    llm = ChatGPTClient("gpt-5.1-codex", base_url="https://proxy.example/codex").get_llm()
    assert "proxy.example" in str(llm.openai_api_base)


# --- authentication --------------------------------------------------------

def test_a_missing_credential_says_how_to_sign_in(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_AUTH_FILE", str(tmp_path / "auth.json"))
    with pytest.raises(ValueError, match="tradingagents auth login chatgpt"):
        ChatGPTClient("gpt-5.1-codex")


def test_the_placeholder_key_is_never_what_gets_sent(signed_in):
    # The openai SDK needs *some* key to construct; the request hook then
    # replaces the Authorization header with the real OAuth token, so the
    # placeholder is never what goes on the wire.
    llm = ChatGPTClient("gpt-5.1-codex").get_llm()
    assert not llm.openai_api_key.get_secret_value().startswith("sk-")

    sent = _invoke("gpt-5.1-codex")
    assert "chatgpt-oauth" not in sent["headers"]["authorization"]


def test_requests_carry_the_stored_token_and_account(signed_in):
    sent = _invoke("gpt-5.1-codex")

    assert sent["headers"]["authorization"].startswith("Bearer ")
    assert sent["headers"]["chatgpt-account-id"] == "acct-77"


def test_requests_identify_the_client(signed_in):
    sent = _invoke("gpt-5.1-codex")
    assert sent["headers"]["session_id"]
    # The SDK's own User-Agent names the library, not this program.
    assert sent["headers"]["user-agent"].startswith("tradingagents/")


def test_no_originator_header_is_claimed(signed_in):
    # The official Codex client identifies itself with originator=codex_cli_rs.
    # This program is not that client, and the header the reference
    # implementation sends is only the bearer token and the account id.
    sent = _invoke("gpt-5.1-codex")
    assert "originator" not in sent["headers"]


def test_a_refreshed_token_reaches_the_next_request(signed_in, monkeypatch):
    # The token behind the client is re-read on every request, so a refresh
    # mid-run is picked up instead of replaying a stale string.
    monkeypatch.setattr(
        chatgpt_auth.ChatGPTSession,
        "auth_headers",
        lambda self: {"Authorization": "Bearer refreshed-token", "chatgpt-account-id": "acct-77"},
    )
    sent = _invoke("gpt-5.1-codex")
    assert sent["headers"]["authorization"] == "Bearer refreshed-token"


def test_requests_are_stateless(signed_in):
    # The Codex Responses endpoint keeps no server-side state and rejects
    # previous_response_id, so the conversation travels in full each time.
    sent = _invoke("gpt-5.1-codex")
    assert sent["body"].get("store") is False
    assert "previous_response_id" not in sent["body"]


def test_the_session_id_is_stable_within_a_client(signed_in):
    # One conversation identifier for the client's life, so the backend keeps a
    # single response store per run rather than a new one per call.
    client = ChatGPTClient("gpt-5.1-codex")
    assert client.session_id == client.session_id

    sent = _invoke(client)
    assert sent["headers"]["session_id"] == client.session_id


def test_the_session_id_can_be_pinned_by_the_environment(signed_in, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CHATGPT_SESSION", "pinned")
    assert ChatGPTClient("gpt-5.1-codex").session_id == "pinned"


# --- validation and registration -------------------------------------------

def test_any_model_name_is_accepted(signed_in):
    # Which models a subscription exposes is the account's business, and the
    # list changes; a stale local list would only produce false warnings.
    assert ChatGPTClient("some-future-model").validate_model() is True


def test_the_factory_builds_the_chatgpt_client(signed_in):
    assert isinstance(create_llm_client("chatgpt", "gpt-5.1-codex"), ChatGPTClient)


def test_the_provider_is_not_in_the_openai_compatible_registry():
    from tradingagents.llm_clients.openai_client import is_openai_compatible

    assert is_openai_compatible("chatgpt") is False


# --- test double -----------------------------------------------------------

def _invoke(model) -> dict:
    """Send one real request through a mock transport; return what went out.

    ``model`` is either a model name (a client is built for it) or an already
    built client.
    """
    client = model.get_llm() if isinstance(model, ChatGPTClient) else ChatGPTClient(model).get_llm()
    http = _http_client(client)
    model_name = client.model_name
    captured: dict = {}

    response_type = _httpx_module(http).Response

    def handler(request):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return response_type(200, json=_responses_body(model_name))

    _mock_transport(http, handler)
    client.invoke("hi")
    return captured


def _responses_body(model: str) -> dict:
    return {
        "id": "resp_x", "object": "response", "created_at": 1, "model": model,
        "status": "completed",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _http_client(client):
    """The httpx client a built langchain client actually sends through."""
    for start in (client, getattr(client, "http_client", None), getattr(client, "client", None)):
        node = start
        for _ in range(6):
            if node is None:
                break
            if hasattr(node, "_transport"):
                return node
            node = getattr(node, "client", None) or getattr(node, "_client", None)
    raise AssertionError("no httpx client found on the built client")


def _httpx_module(http):
    """The httpx package ``http`` belongs to.

    Two httpx packages are installed here and they are not interchangeable: a
    response built from the wrong one is rejected as a non-sync stream.
    """
    import importlib

    return importlib.import_module(type(http).__module__.split(".")[0])


def _mock_transport(http, handler) -> None:
    """Route ``http``'s calls to ``handler``."""
    http._transport = _httpx_module(http).MockTransport(handler)
