"""Per-request HTTP headers applied through an httpx event hook.

A langchain chat client ultimately hands its request to the openai or anthropic
SDK, and both set their own headers — the openai SDK pins its own
``Authorization`` from its api key, and pins a ``User-Agent`` that identifies
the SDK rather than the application. Headers set anywhere else lose to those.

A *request* event hook on the httpx client runs after the SDK has finished
building the request, which makes it the one place a caller-supplied header
reliably survives. That is what an OAuth provider needs: the hook is called on
every request, so it can read a freshly minted access token instead of the one
captured when the client was built. A long multi-agent run outlives a
short-lived token, and a client holding a snapshot of it starts failing with
401 partway through.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["auth_header_hook", "client_with_hook", "user_agent"]


def user_agent() -> str:
    """This application's User-Agent, as a provider should be told.

    Both provider SDKs would otherwise send their own ("openai-python/x.y",
    "Anthropic/Python x.y"), which identifies the library rather than the
    program making the request. OpenCode Go asks integrations to send their own
    name and uses it in abuse monitoring, and OpenAI's Codex backend is shared
    with the official Codex client, so it is better to identify honestly.
    """
    from tradingagents import __version__

    return f"tradingagents/{__version__}"


def auth_header_hook(headers_for: Callable[[], dict[str, str]]) -> Callable[[Any], None]:
    """A request hook that sets auth headers resolved per request.

    ``headers_for`` is called for every request, so a token refreshed mid-run is
    the one that goes on the wire. Empty values are skipped, so an optional
    header (an account id, a FedRAMP marker) is simply absent rather than sent
    blank. The SDK's own Authorization header is overwritten, so a placeholder
    key never ships.
    """

    def apply(request) -> None:
        for name, value in headers_for().items():
            if value:
                request.headers[name] = value

    return apply


def client_with_hook(hook: Callable[[Any], None], **client_kwargs: Any):
    """An ``httpx.Client`` that runs ``hook`` before sending each request.

    httpx is imported lazily: it reaches this process through whichever
    provider SDK is in use, and importing this module should not pull one in.
    """
    import httpx

    return httpx.Client(event_hooks={"request": [hook]}, **client_kwargs)
