"""OpenCode Go: a subscription that serves many vendors' open models.

One key, one endpoint, but three different wire formats behind it — Go routes a
model to whichever API that vendor actually speaks:

* ``/chat/completions`` — OpenAI-compatible (GLM, Kimi, DeepSeek, MiMo, LongCat, Hy)
* ``/responses``        — the OpenAI Responses API (Grok, GPT Luna, Muse Spark)
* ``/messages``         — the Anthropic Messages API (MiniMax, Qwen)

So the client is a router, not one fixed class: it picks the langchain client to
match the model. ``x-opencode-session`` and a real User-Agent go on every
request; Go asks for both, and uses the session id for routing and prompt-cache
reuse across an agent run.
"""

import os
import uuid
from typing import Any

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient
from .http_headers import user_agent
from .validators import validate_model

__all__ = ["OpenCodeGoClient", "GO_BASE_URL", "go_request_id", "go_session_id"]

GO_BASE_URL = "https://opencode.ai/zen/go/v1"
_GO_MODELS_ENV = "OPENCODE_GO_BASE_URL"

# Wire format per model, from the Go model list. Anything unlisted speaks
# OpenAI-compatible chat completions, which is the common case; the sets below
# are the exceptions that need the Responses or Messages API.
_RESPONSES_MODELS = frozenset({
    "grok-4.7", "grok-4.6",
    "gpt-6-luna", "gpt-5.6-luna",
    "muse-spark-1.3-contributor", "muse-spark-1.2-contributor",
})
_MESSAGES_MODELS = frozenset({
    "minimax-m3", "minimax-m2.7",
    "qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus",
})

_TIMEOUT = 600.0

# How Go's docs ask a client to describe itself, and the client name the
# OpenCode client reports. Both are free-form; they exist so Go's abuse
# monitoring can tell integrations apart.
_CLIENT = "tradingagents"

# Resolved once per process by go_session_id().
_SESSION_ID: str | None = None


def go_request_id() -> str:
    """An identifier for this client's run, sent as ``x-opencode-request``.

    The OpenCode client sends its current user-message id here, and a program
    that has no message ids has nothing more specific to offer, so this is one
    id per client rather than per request — stable for the run, like the session
    id. Pin it with ``TRADINGAGENTS_OPENCODE_REQUEST``.
    """
    return os.environ.get("TRADINGAGENTS_OPENCODE_REQUEST") or str(uuid.uuid4())


def go_session_id() -> str:
    """A stable per-process session id, or one pinned by the environment.

    Go routes on ``x-opencode-session`` and reuses its prompt cache per session,
    so every call in a run must carry the same value. Cached at module level
    because the two models of a run are two separate clients, and two ids
    would mean two cache entries for one conversation. Pin it with
    ``TRADINGAGENTS_OPENCODE_SESSION`` to carry a session across processes.
    """
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = os.environ.get("TRADINGAGENTS_OPENCODE_SESSION") or str(uuid.uuid4())
    return _SESSION_ID


def _wire_format(model: str) -> str:
    """Which API ``model`` speaks on Go: 'chat', 'responses' or 'messages'."""
    name = model.lower().strip()
    if name in _RESPONSES_MODELS:
        return "responses"
    if name in _MESSAGES_MODELS:
        return "messages"
    return "chat"


def _messages_root(base_url: str) -> str:
    """The API root for the Anthropic-shaped models.

    The Anthropic SDK appends ``/v1/messages`` to whatever base URL it is given,
    so it wants the API root, not the versioned prefix. Go publishes the
    Messages endpoint as ``/zen/go/v1/messages``; handing the SDK
    ``/zen/go/v1`` would produce ``/zen/go/v1/v1/messages`` and 404, so a
    trailing ``/v1`` is dropped. The OpenAI-shaped paths keep it — their SDKs
    append only ``/chat/completions`` or ``/responses``.
    """
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


class OpenCodeGoClient(BaseLLMClient):
    """Client for the OpenCode Go endpoint."""

    provider = "opencode-go"

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.session_id = go_session_id()
        self.request_id = go_request_id()

    def get_llm(self) -> Any:
        """Return a langchain client for this model's wire format."""
        self.warn_if_unknown_model()

        api_key_env = get_api_key_env(self.provider)
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not api_key:
            raise ValueError(
                f"API key for provider '{self.provider}' is not set. "
                f"Please set the {api_key_env} environment variable "
                f"(e.g. add {api_key_env}=your_key to your .env file). "
                "Get one from https://opencode.ai/console."
            )

        base_url = self.base_url or os.environ.get(_GO_MODELS_ENV) or GO_BASE_URL
        headers = self._headers()

        wire = _wire_format(self.model)
        if wire == "messages":
            return self._messages_client(api_key, _messages_root(base_url), headers)
        if wire == "responses":
            return self._openai_client(api_key, base_url, headers, use_responses=True)
        return self._openai_client(api_key, base_url, headers, use_responses=False)

    def _headers(self) -> dict[str, str]:
        """The headers Go expects, matching what the OpenCode client sends.

        Go's docs require two: a User-Agent that names this program rather than
        an SDK or HTTP library, and a stable ``x-opencode-session`` so requests
        in one conversation share a route and a prompt cache. The OpenCode
        client also sends ``x-opencode-client`` and ``x-opencode-request``, which
        it uses in abuse monitoring; those are cheap and worth sending too.

        ``x-opencode-project`` and ``x-parent-session-id`` are deliberately
        absent: the first is OpenCode's own project id, which this program has
        no equivalent for, and the second describes a session tree this program
        does not have.
        """
        return {
            "User-Agent": user_agent(),
            "x-opencode-session": self.session_id,
            "x-opencode-client": _CLIENT,
            "x-opencode-request": self.request_id,
        }

    def _openai_client(self, api_key, base_url, headers, *, use_responses: bool) -> Any:
        from .openai_client import NormalizedChatOpenAI

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": base_url,
            "api_key": api_key,
            "default_headers": headers,
            "use_responses_api": use_responses,
        }
        for key in ("timeout", "max_retries", "temperature", "max_tokens", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatOpenAI(**llm_kwargs)

    def _messages_client(self, api_key, base_url, headers) -> Any:
        from .anthropic_client import NormalizedChatAnthropic

        # The Anthropic SDK authenticates with x-api-key, which Go's
        # Anthropic-shaped endpoint reads. Bearer is sent as well: gateways of
        # this shape conventionally accept both, and it costs nothing to not
        # depend on which one Go checks.
        headers = {**headers, "Authorization": f"Bearer {api_key}"}
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": base_url,
            "api_key": api_key,
            "default_headers": headers,
        }
        for key in ("timeout", "max_retries", "temperature", "max_tokens", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate the model for OpenCode Go.

        Go's catalog changes and an account sees only its own plans, so any
        model string is accepted without warning.
        """
        return validate_model(self.provider, self.model)
