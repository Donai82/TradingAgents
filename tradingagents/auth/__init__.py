"""Credential sign-in for providers that authenticate with OAuth.

Currently just ChatGPT (see :mod:`tradingagents.auth.chatgpt`). The credential
lives in ``~/.codex/auth.json``, the Codex CLI's own file and the one the
``openai-oauth`` project writes, so a sign-in in any of the three is a sign-in
in all of them.
"""

from .chatgpt import ChatGPTSession, NotSignedIn, login, logout, models, status
from .codex_auth import auth_file_path

__all__ = [
    "ChatGPTSession",
    "NotSignedIn",
    "auth_file_path",
    "login",
    "logout",
    "models",
    "status",
]
