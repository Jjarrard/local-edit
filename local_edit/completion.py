"""Tab completion for the REPL.

Wires up readline so `/model` (then Tab) cycles through whatever models Jan
currently has loaded, and any other `/` command completes/cycles by name.
"""

from __future__ import annotations

from collections.abc import Callable

SLASH_COMMANDS = [
    "/help", "/status", "/todo", "/add", "/work", "/model", "/fast", "/run",
    "/auto", "/review", "/diff", "/undo", "/reset", "/quit", "/exit",
]
# Commands whose argument is a model id, so Tab offers the loaded models.
MODEL_COMMANDS = ("/model", "/fast")


def candidates(buffer: str, text: str, models: list[str]) -> list[str]:
    """Given the full input line so far and the word currently being
    completed, return sorted completion candidates."""
    if buffer.startswith(MODEL_COMMANDS):
        options = models
    elif buffer.startswith("/") and " " not in buffer:
        options = SLASH_COMMANDS
    else:
        options = []
    return sorted(o for o in options if o.startswith(text))


def install(get_models: Callable[[], list[str]]) -> None:
    """Wire up readline tab completion. `get_models` is called fresh each
    time completion runs, so it can reflect changes made via `/model`.
    No-op if readline isn't available (e.g. some Windows Python builds)."""
    try:
        import readline
    except ImportError:
        return

    # Model ids often contain "/" (e.g. "bartowski/DeepSeek-R1-..."), which
    # is a default word-delimiter and would otherwise chop them into just
    # the tail. Restrict delimiters to whitespace so `text` is the whole word.
    readline.set_completer_delims(" \t\n")

    def completer(text: str, state: int) -> str | None:
        buffer = readline.get_line_buffer()
        options = candidates(buffer, text, get_models())
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    is_libedit = "libedit" in (readline.__doc__ or "")
    bind = "bind ^I rl_complete" if is_libedit else "tab: menu-complete"
    try:
        readline.parse_and_bind(bind)
    except Exception:  # noqa: BLE001 - tab completion is a nicety, never fatal
        pass
