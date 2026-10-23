"""ChatGPT (OAuth) provider: models on your ChatGPT subscription, not the API.

The ``openai`` provider bills a platform API key. This one is the alternative
the Codex CLI offers: sign in to ChatGPT, and requests go to
``chatgpt.com/backend-api/codex`` on the subscription you already pay for.
The flow follows EvanZhouDev/openai-oauth, so a user already signed in to the
Codex CLI is signed in here too.

Three things this client gets right that are easy to get wrong:

* **The backend speaks the Responses API only**, so that is always on.
* **The credential is read per request**, from a session that refreshes when
  the token is stale. A run outlives a one-hour token, and a client holding a
  snapshot of it starts failing with 401 partway through.
* **Requests are stateless.** The Codex endpoint does not implement
  ``previous_response_id`` or ``item_reference``, so ``store`` is off and every
  request carries the full conversation.
"""

import os
import uuid
from typing import Any

from .base_client import BaseLLMClient
from .http_headers import auth_header_hook, client_with_hook, user_agent
from .openai_client import NormalizedChatOpenAI
from .validators import validate_model

__all__ = ["ChatGPTClient"]

_TIMEOUT = 600.0

# The openai SDK refuses to construct without a key and derives its own
# Authorization header from one. The real credential comes from the request
# hook below, which runs last and overwrites it, so this is never sent.
_PLACEHOLDER_KEY = "chatgpt-oauth"

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "temperature", "max_tokens", "reasoning_effort",
    "callbacks", "http_async_client",
)


class ChatGPTClient(BaseLLMClient):
    """Client for the ChatGPT subscription's Codex endpoint."""

    provider = "chatgpt"

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.session = self._sign_in()
        # One conversation identifier for the life of the client, so the backend
        # can keep a single response store per run. Pin it to carry a session
        # across processes.
        self.session_id = os.environ.get("TRADINGAGENTS_CHATGPT_SESSION") or uuid.uuid4().hex
    @staticmethod
    def _sign_in():
        """The stored ChatGPT credential, or a message saying how to get one."""
        from tradingagents.auth.chatgpt import ChatGPTSession, NotSignedIn

        try:
            return ChatGPTSession()
        except NotSignedIn as exc:
            # Match the shape of the other clients' misconfiguration errors:
            # a setup problem, raised at construction rather than as a 401
            # halfway through a run.
            raise ValueError(str(exc)) from None

    def get_llm(self) -> Any:
        """Return a ChatOpenAI pointed at the Codex backend, on a live token."""
        from tradingagents.auth.chatgpt import CHATGPT_BASE_URL

        base_url = self.base_url or CHATGPT_BASE_URL
        # The bearer token, account id and FedRAMP marker are resolved per
        # request, because the token behind them is refreshed as the run goes on.
        http_client = client_with_hook(
            auth_header_hook(self.session.auth_headers),
            timeout=_TIMEOUT,
        )
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": base_url,
            "use_responses_api": True,
            "http_client": http_client,
            "api_key": _PLACEHOLDER_KEY,
            "default_headers": {
                # No originator header: that is how the official Codex client
                # identifies itself, and it is not this program. The
                # User-Agent below names us instead, without claiming to be
                # something we are not.
                "session_id": self.session_id,
                # Left alone the SDK sends "OpenAI/Python <version>", which names
                # the library rather than the application.
                "User-Agent": user_agent(),
            },
            # The Codex Responses endpoint keeps no server-side state and rejects
            # previous_response_id, so the conversation is sent in full each time.
            "store": False,
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Whether the model is one this ChatGPT account can serve.

        The catalogue behind a subscription is account- and plan-dependent and
        changes, so this accepts anything and the backend is the authority;
        ``tradingagents auth models`` lists what the account can reach.
        """
        return validate_model("chatgpt", self.model)
