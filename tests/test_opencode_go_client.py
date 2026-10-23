"""The OpenCode Go client: wire-format routing, key handling, Go's headers.

Go is one endpoint serving three different APIs, so the model decides which
langchain client is built. That routing is the whole risk of this provider —
a model on the wrong format fails at request time, not at construction — and it
is checked here against Go's published model list.
"""

import pytest

from tradingagents.llm_clients import opencode_go_client as opencode_go
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import is_openai_compatible
from tradingagents.llm_clients.opencode_go_client import (
    GO_BASE_URL,
    OpenCodeGoClient,
    _wire_format,
    go_session_id,
    user_agent,
)

# Every model on Go's published list, with the API that list says it speaks.
GO_MODELS = {
    # Responses API
    "grok-4.7": "responses",
    "grok-4.6": "responses",
    "gpt-6-luna": "responses",
    "gpt-5.6-luna": "responses",
    "muse-spark-1.3-contributor": "responses",
    "muse-spark-1.2-contributor": "responses",
    # Anthropic Messages API
    "minimax-m3": "messages",
    "minimax-m2.7": "messages",
    "qwen3.8-max": "messages",
    "qwen3.8-flash": "messages",
    "qwen3.7-plus": "messages",
    # OpenAI-compatible chat completions
    "glm-5.3-flash": "chat",
    "glm-5.3": "chat",
    "glm-5.2": "chat",
    "kimi-k3": "chat",
    "kimi-k2.7-code": "chat",
    "kimi-k2.6": "chat",
    "longcat-2.0": "chat",
    "longcat-2.5-preview-free": "chat",
    "mimo-v2.6-flash": "chat",
    "mimo-v2.6-pro": "chat",
    "mimo-v2.5": "chat",
    "mimo-v2.5-pro": "chat",
    "deepseek-v4.1-flash": "chat",
    "deepseek-v4-pro": "chat",
    "deepseek-v4-flash": "chat",
    "deepseek-v4-flash-vision-exp": "chat",
    "hy4-preview": "chat",
    "hy3": "chat",
    "space-bunny-free": "chat",
}


@pytest.fixture
def go_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-test-key")
    return "go-test-key"


# --- routing ---------------------------------------------------------------

@pytest.mark.parametrize(("model", "expected"), sorted(GO_MODELS.items()))
def test_every_go_model_routes_to_its_documented_api(model, expected):
    assert _wire_format(model) == expected


def test_an_unknown_model_defaults_to_chat_completions():
    # Go adds models continuously; a new one should work without a code change.
    assert _wire_format("some-future-model") == "chat"


def test_routing_ignores_case_and_surrounding_space():
    assert _wire_format("  GLM-5.3  ") == "chat"
    assert _wire_format("GROK-4.7") == "responses"


# --- client construction ---------------------------------------------------

def test_a_responses_model_gets_the_responses_api(go_key):
    llm = OpenCodeGoClient("grok-4.7").get_llm()
    assert llm.use_responses_api is True
    assert llm.model_name == "grok-4.7"
    assert str(llm.openai_api_base).rstrip("/") == GO_BASE_URL


def test_a_chat_model_does_not_use_the_responses_api(go_key):
    llm = OpenCodeGoClient("glm-5.3").get_llm()
    assert llm.use_responses_api is not True


def test_a_messages_model_gets_the_anthropic_client(go_key):
    from tradingagents.llm_clients.anthropic_client import NormalizedChatAnthropic

    llm = OpenCodeGoClient("qwen3.8-max").get_llm()
    assert isinstance(llm, NormalizedChatAnthropic)
    assert llm.model == "qwen3.8-max"


def test_a_messages_model_is_given_the_api_root_not_the_version(go_key):
    # The Anthropic SDK appends /v1/messages to the base URL it is given.
    llm = OpenCodeGoClient("qwen3.8-max").get_llm()
    assert str(llm.anthropic_api_url).rstrip("/") == GO_BASE_URL.removesuffix("/v1")


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://opencode.ai/zen/go/v1", "https://opencode.ai/zen/go"),
        ("https://opencode.ai/zen/go/v1/", "https://opencode.ai/zen/go"),
        ("https://proxy.example/go", "https://proxy.example/go"),
        ("https://proxy.example/go/", "https://proxy.example/go"),
    ],
)
def test_only_a_trailing_version_is_stripped(base, expected):
    from tradingagents.llm_clients.opencode_go_client import _messages_root

    assert _messages_root(base) == expected


def test_the_key_reaches_the_client(go_key):
    llm = OpenCodeGoClient("glm-5.3").get_llm()
    assert llm.openai_api_key.get_secret_value() == "go-test-key"


def test_a_base_url_override_is_honored(go_key):
    llm = OpenCodeGoClient("glm-5.3", base_url="https://proxy.example/go/v1").get_llm()
    assert "proxy.example" in str(llm.openai_api_base)


def test_an_env_base_url_is_used_when_no_override(go_key, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_BASE_URL", "https://mirror.example/v1")
    llm = OpenCodeGoClient("glm-5.3").get_llm()
    assert "mirror.example" in str(llm.openai_api_base)


# --- the key ---------------------------------------------------------------

def test_a_missing_key_names_the_variable(go_key, monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY")
    with pytest.raises(ValueError, match="OPENCODE_GO_API_KEY"):
        OpenCodeGoClient("glm-5.3").get_llm()


def test_the_error_points_at_the_console(go_key, monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY")
    with pytest.raises(ValueError, match="opencode.ai/console"):
        OpenCodeGoClient("glm-5.3").get_llm()


# --- Go's session and identification headers -------------------------------

def test_the_session_id_is_stable_within_a_client(go_key):
    client = OpenCodeGoClient("glm-5.3")
    assert client.session_id == client.session_id


def test_the_request_id_is_stable_within_a_client(go_key):
    client = OpenCodeGoClient("glm-5.3")
    assert client.request_id == client.request_id


def test_the_request_id_can_be_pinned_by_the_environment(go_key, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_OPENCODE_REQUEST", "pinned-request")
    assert OpenCodeGoClient("glm-5.3").request_id == "pinned-request"


def test_the_session_id_can_be_pinned_by_the_environment(monkeypatch):
    # go_session_id caches per process, so a pin has to reach the cache
    # through its reset rather than through the environment at first use.
    monkeypatch.setattr(opencode_go, "_SESSION_ID", None)
    monkeypatch.setenv("TRADINGAGENTS_OPENCODE_SESSION", "pinned-session")
    assert go_session_id() == "pinned-session"
    assert OpenCodeGoClient("glm-5.3").session_id == "pinned-session"


def test_two_clients_in_a_process_share_one_session_by_default():
    # Go routes and caches per session, so the deep and quick models of one
    # run must agree on it.
    assert go_session_id() == go_session_id()
    assert OpenCodeGoClient("glm-5.3").session_id == OpenCodeGoClient("kimi-k3").session_id


def test_the_user_agent_names_this_client():
    from tradingagents import __version__

    assert user_agent() == f"tradingagents/{__version__}"


@pytest.mark.parametrize(
    ("model", "attribute"),
    [("glm-5.3", "openai"), ("grok-4.7", "openai"), ("qwen3.8-max", "anthropic")],
)
def test_requests_carry_the_session_and_user_agent(go_key, model, attribute):
    # Go asks integrations to identify themselves and to send a stable session
    # id, and uses both. Asserted on a real request, because the provider SDKs
    # set their own User-Agent and would quietly win over a default set anywhere
    # less direct.
    sent = _capture_openai_request(model) if attribute == "openai" else _capture_anthropic_request(model)
    headers = sent["headers"]

    assert headers["x-opencode-session"] == OpenCodeGoClient(model).session_id
    assert headers["user-agent"].startswith("tradingagents/")


@pytest.mark.parametrize("model", ["glm-5.3", "grok-4.7", "qwen3.8-max"])
def test_requests_carry_the_opencode_client_and_request_headers(go_key, model):
    # The OpenCode client sends both; Go uses them in abuse monitoring.
    sent = _capture_openai_request(model) if _wire_format(model) != "messages" else _capture_anthropic_request(model)

    assert sent["headers"]["x-opencode-client"] == "tradingagents"
    assert sent["headers"]["x-opencode-request"]


@pytest.mark.parametrize("model", ["glm-5.3", "grok-4.7", "qwen3.8-max"])
def test_no_opencode_project_header_is_claimed(go_key, model):
    # OpenCode's project id identifies an OpenCode project. This program has no
    # equivalent, and sending a wrong one would be worse than sending none.
    sent = _capture_openai_request(model) if _wire_format(model) != "messages" else _capture_anthropic_request(model)
    assert "x-opencode-project" not in sent["headers"]


def test_the_messages_path_is_not_version_prefixed_twice(go_key):
    # The Anthropic SDK appends /v1/messages itself. Handing it a base URL that
    # already ends in /v1 produces /v1/v1/messages, which is a 404 — and that
    # would hit every Qwen and MiniMax model on Go.
    sent = _capture_anthropic_request("qwen3.8-max")
    assert sent["url"].endswith("/v1/messages")
    assert "/v1/v1/" not in sent["url"]


def test_the_openai_paths_keep_their_version_prefix(go_key):
    # Their SDKs append only /chat/completions or /responses, so the /v1 has
    # to stay.
    for model, suffix in (("glm-5.3", "/v1/chat/completions"), ("grok-4.7", "/v1/responses")):
        assert _capture_openai_request(model)["url"].endswith(suffix)


def test_the_messages_path_authenticates_with_both_header_forms(go_key):
    # The Anthropic SDK sends x-api-key. Go's Messages endpoint is
    # Anthropic-shaped, but sending the bearer form too means the request does
    # not depend on which of the two it checks.
    headers = _capture_anthropic_request("qwen3.8-max")["headers"]
    assert headers["x-api-key"] == "go-test-key"
    assert headers["authorization"] == "Bearer go-test-key"


def _mock_transport(client, handler) -> None:
    """Route a built langchain client's HTTP calls to ``handler``.

    Both langchain clients wrap their provider SDK's httpx client, and both put
    it two attributes down. The exact depth differs between the openai and
    anthropic SDKs, so walk the chain rather than hard-coding a path that one
    of them will not match.
    """
    import httpx2

    transport = httpx2.MockTransport(handler)
    node = client
    for _ in range(6):
        node = getattr(node, "client", None) or getattr(node, "_client", None)
        if node is None:
            break
        if hasattr(node, "_transport"):
            node._transport = transport
            return
    raise AssertionError("no httpx transport found on the built client")


def _capture_openai_request(model: str) -> dict:
    """Send a chat request through a mock transport and record the headers."""
    import httpx2

    captured: dict = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx2.Response(200, json=_completion_body(model))

    client = OpenCodeGoClient(model).get_llm()
    _mock_transport(client, handler)
    client.invoke("hi")
    return captured


def _completion_body(model: str) -> dict:
    """A response body matching whichever API this model's client speaks."""
    if _wire_format(model) == "responses":
        return {
            "id": "resp_x", "object": "response", "created_at": 1, "model": model,
            "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    return {
        "id": "chatcmpl_x", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _capture_anthropic_request(model: str) -> dict:
    """Send a messages request through a mock transport and record the headers."""
    import httpx2

    captured: dict = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx2.Response(200, json={
            "id": "msg_x", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })


    client = OpenCodeGoClient(model).get_llm()
    _mock_transport(client, handler)
    client.invoke("hi")
    return captured


# --- validation and registration -------------------------------------------

def test_any_model_name_is_accepted(go_key):
    # Go's catalog changes and an account sees only its own plans.
    assert OpenCodeGoClient("some-future-model").validate_model() is True


def test_the_provider_is_not_in_the_openai_compatible_registry():
    # It has its own client; a registry row would misroute it to ChatOpenAI.
    assert is_openai_compatible("opencode-go") is False


def test_the_factory_builds_the_go_client(go_key):
    client = create_llm_client("opencode-go", "grok-4.7")
    assert isinstance(client, OpenCodeGoClient)
