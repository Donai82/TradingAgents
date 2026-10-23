"""``tradingagents auth`` — sign in to providers that use OAuth instead of a key.

The interactive provider picker reaches this: choosing ChatGPT with no
credential on hand starts the sign-in rather than failing on the first API call.
The same commands are available non-interactively for scripted setups, where a
terminal may not exist at all (the browser flow then needs ``--device``).
"""

import typer

from cli.display import console
from tradingagents.auth import chatgpt
from tradingagents.auth.codex_auth import auth_file_path

app = typer.Typer(
    name="auth",
    help="Sign in to providers that authenticate with OAuth instead of an API key.",
    no_args_is_help=True,
)

_PROVIDERS = {"chatgpt": "ChatGPT"}


def _known(provider: str) -> str:
    """The display name of ``provider``, or a usage error."""
    name = _PROVIDERS.get(provider.lower())
    if name is None:
        listed = ", ".join(_PROVIDERS)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose from {listed}")
    return name


def _fail(exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(code=1) from None


@app.command("login")
def login(
    provider: str = typer.Argument("chatgpt", help="Provider to sign in to"),
    device: bool = typer.Option(
        False, "--device",
        help="Use the device-code flow: for headless machines, where no browser "
             "can reach the localhost sign-in callback.",
    ),
) -> None:
    """Sign in and store the credential for later runs."""
    name = _known(provider)
    mode = "device" if device else "browser"
    console.print(f"[cyan]Signing in to {name} ({mode} flow)...[/cyan]")
    try:
        chatgpt.login(mode=mode)
    except (chatgpt.NotSignedIn, OSError) as exc:
        _fail(exc)
    except KeyboardInterrupt:
        _fail("sign-in cancelled")
    _print_status()


@app.command("logout")
def logout(
    provider: str = typer.Argument("chatgpt", help="Provider to sign out of"),
) -> None:
    """Remove the stored credential."""
    name = _known(provider)
    if not chatgpt.logout():
        console.print(f"[yellow]Not signed in to {name}; nothing to remove.[/yellow]")
        return
    console.print(f"[green]Signed out of {name}.[/green]")


@app.command("status")
def status(
    provider: str = typer.Argument("chatgpt", help="Provider to report on"),
) -> None:
    """Show the stored credential's account and expiry."""
    name = _known(provider)
    details = chatgpt.status()
    if details is None:
        console.print(
            f"[yellow]Not signed in to {name}.[/yellow]\n"
            f"Sign in with: [cyan]tradingagents auth login {provider.lower()}[/cyan]"
        )
        return
    _print_status(details)


@app.command("models")
def models(
    provider: str = typer.Argument("chatgpt", help="Provider to list models for"),
) -> None:
    """List the models this signed-in account can actually use.

    A subscription's catalogue depends on the plan and changes, so the backend is
    the only authority on it. The CLI's model menu offers a free-text field
    instead; this is for finding out what to type in it.
    """
    _known(provider)
    try:
        found = chatgpt.models()
    except chatgpt.NotSignedIn as exc:
        _fail(exc)
    if not found:
        console.print(
            "[yellow]The model list is unavailable.[/yellow]\n"
            "That is not fatal — the model menu's \"Custom model ID\" still "
            "accepts any ID. The backend refuses unknown ones."
        )
        return
    console.print(f"[green]{len(found)} model(s) available:[/green]")
    for slug in found:
        console.print(f"  {slug}")


def _print_status(details: dict | None = None) -> None:
    """Report the signed-in account, the plan, and where the token lives."""
    details = details or chatgpt.status() or {}
    account = details.get("account_id")
    plan = details.get("plan")
    expiry = details.get("expires_at")
    if expiry:
        from datetime import datetime

        expiry = datetime.fromtimestamp(float(expiry)).strftime("%Y-%m-%d %H:%M")
    console.print(
        f"[green]Signed in to ChatGPT[/green]"
        f" (plan: {plan or 'unknown'}"
        + (f", account: {account}" if account else "")
        + (f", token valid until {expiry}" if expiry else "")
        + ")"
    )
    path = details.get("path") or auth_file_path()
    console.print(f"[dim]Credential: {path}[/dim]")
    console.print(
        "[dim]Shared with the Codex CLI; sign in there with `codex login`.[/dim]"
    )
