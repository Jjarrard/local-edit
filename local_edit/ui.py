"""Terminal UI primitives: a prompt readline can't corrupt, an arrow-key
picker, live status, and input hygiene.

The rules this module exists to enforce:
  - The prompt is never editable. rich's Console.input() prints the prompt
    itself and then calls bare input(), so readline believes the line starts
    at column 0 and happily lets Backspace eat "you> ". We pass the prompt
    *into* input() with \\001/\\002 width markers so readline knows how wide
    the non-editable part is.
  - Nothing typed while the model is busy is ever consumed as an answer to a
    later question. Every interactive prompt drains buffered stdin first.
  - Everything degrades cleanly when stdin/stdout isn't a terminal (pipes,
    tests, CI) instead of raising or hanging.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Sequence

from rich.console import Console
from rich.text import Text

# readline width markers: "the bytes between these take up no screen columns"
RL_START = "\001"
RL_END = "\002"

ESC = "\x1b"
CTRL_C = "\x03"
CTRL_D = "\x04"


def _plain_prompt_requested() -> bool:
    return bool(os.environ.get("LOCAL_EDIT_PLAIN_PROMPT"))


def is_interactive() -> bool:
    """True only when we can actually drive a terminal in both directions."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def drain_stdin() -> None:
    """Discard anything typed while we were busy.

    Without this, keystrokes hammered during a long generation sit in the tty
    buffer and get eaten by the next approval prompt - so an edit could be
    'approved' by a stray keypress the user aimed at nothing in particular.
    """
    if not is_interactive():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError):
        pass


def build_prompt(label: str, color: str = "1;34") -> str:
    """Build a prompt string that readline measures correctly."""
    if _plain_prompt_requested() or not is_interactive():
        return label
    return f"{RL_START}\x1b[{color}m{RL_END}{label}{RL_START}\x1b[0m{RL_END}"


def ask_line(label: str = "you> ", color: str = "1;34") -> str:
    """Read one line. The prompt itself can never be edited or erased.

    Raises EOFError / KeyboardInterrupt to the caller, same as input().
    """
    drain_stdin()
    return input(build_prompt(label, color))


ARROW_CODES = {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}


def decode_escape(tail: bytes) -> str:
    """Map the bytes following ESC to a key name.

    `tail` is everything that arrived in the same burst after \\x1b - empty
    means the user really did press Escape. Both CSI ("[A") and application
    cursor mode ("OA") are handled, since terminals emit either depending on
    the mode the program left them in.
    """
    if not tail:
        return "esc"
    if tail[:1] in (b"[", b"O"):
        return ARROW_CODES.get(tail[1:2], "other")
    return "other"


def decode_key(data: bytes) -> str:
    """Map one keypress worth of bytes to a key name.

    Pure and total, so the terminal handling below stays testable: every
    branch here is exercised by tests rather than by a person mashing keys.
    """
    if not data:
        return "eof"
    if data[:1] == b"\x1b":
        return decode_escape(data[1:])
    if data in (b"\r", b"\n"):
        return "enter"
    if data == b"\x03":
        return "ctrl-c"
    if data == b"\x04":
        return "esc"
    if data in (b"\x7f", b"\b"):
        return "backspace"
    try:
        ch = data.decode("utf-8")
    except UnicodeDecodeError:
        return "other"
    return ch if ch.isprintable() else "other"


def _utf8_length(first_byte: int) -> int:
    if first_byte < 0x80:
        return 1
    if first_byte >> 5 == 0b110:
        return 2
    if first_byte >> 4 == 0b1110:
        return 3
    if first_byte >> 3 == 0b11110:
        return 4
    return 1


def split_key(buf: bytes) -> tuple[str, int] | None:
    """Parse exactly one key off the front of `buf`.

    Returns (key_name, bytes_consumed), or None when `buf` holds only part
    of a sequence and more input is needed. Consuming precisely one key's
    worth of bytes is what keeps a burst of held-down arrow presses from
    being swallowed by an over-eager read.
    """
    if not buf:
        return None
    if buf[:1] != b"\x1b":
        n = _utf8_length(buf[0])
        if len(buf) < n:
            return None
        return decode_key(buf[:n]), n
    if len(buf) == 1:
        return None  # bare ESC so far - caller decides via timeout
    if buf[1:2] in (b"[", b"O"):
        for i in range(2, len(buf)):
            if 0x40 <= buf[i] <= 0x7E:  # CSI final byte
                return decode_escape(buf[1 : i + 1]), i + 1
        return None  # incomplete sequence
    return "other", 2


class KeyReader:
    """stdin in cbreak mode for the lifetime of a picker.

    Two things this gets right that the naive version did not:

    - It reads the raw file descriptor with os.read() instead of the
      buffered sys.stdin. Mixing select() (which polls the OS fd) with a
      buffered reader loses the tail of every escape sequence, so Down
      (\\x1b[B) decoded as bare-Escape plus a stray "B".
    - It sets the terminal mode once for the whole session rather than per
      keypress, so keys pressed between reads aren't echoed into the display.

    cbreak rather than raw, deliberately: it leaves output post-processing
    alone, so rich can keep rendering normally while we read.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved = None
        # Bytes read but not yet consumed. A single read() can return several
        # keypresses at once (holding an arrow key, or a fast typist), and
        # every one of them must be delivered.
        self._pending = bytearray()

    def __enter__(self) -> "KeyReader":
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc) -> None:
        import termios

        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read_key(self, escape_window: float = 0.05) -> str:
        """Block until one complete keypress is available, and return it."""
        import select

        fd = self._fd
        assert fd is not None, "read_key() used outside the context manager"
        while True:
            parsed = split_key(bytes(self._pending))
            if parsed is not None:
                key, consumed = parsed
                del self._pending[:consumed]
                return key

            # An ESC with nothing after it is ambiguous: either the Escape
            # key, or the start of a sequence still in flight. Give the rest
            # of the sequence a brief window to arrive before deciding.
            if bytes(self._pending) == b"\x1b" and not select.select([fd], [], [], escape_window)[0]:
                self._pending.clear()
                return "esc"

            try:
                chunk = os.read(fd, 64)
            except (OSError, InterruptedError):
                return "eof"
            if not chunk:
                self._pending.clear()
                return "eof"
            self._pending += chunk


def _render_menu(title: str, options: Sequence[str], index: int, current: str | None, filter_text: str) -> Text:
    body = Text()
    body.append(title + "\n", style="bold")
    hint = "up/down to move, Enter to select, Esc to cancel, type to filter"
    body.append(hint + "\n", style="dim")
    if filter_text:
        body.append(f"filter: {filter_text}\n", style="yellow")
    if not options:
        body.append("(no matches)\n", style="red")
        return body
    for i, opt in enumerate(options):
        selected = i == index
        marker = ">" if selected else " "
        style = "bold reverse" if selected else ""
        suffix = "  (current)" if current is not None and opt == current else ""
        body.append(f" {marker} {opt}{suffix}\n", style=style)
    return body


def select(
    console: Console,
    title: str,
    options: Sequence[str],
    current: str | None = None,
) -> str | None:
    """Arrow-key picker. Returns the chosen option, or None if cancelled.

    Falls back to a numbered prompt when there's no usable terminal, so this
    is safe to call from scripts and tests.
    """
    options = list(options)
    if not options:
        return None
    if not is_interactive():
        return _select_fallback(console, title, options)

    drain_stdin()
    index = options.index(current) if current in options else 0
    filter_text = ""
    visible = list(options)

    from rich.live import Live

    try:
        with KeyReader() as keys, Live(console=console, auto_refresh=False, transient=True) as live:
            while True:
                live.update(_render_menu(title, visible, index, current, filter_text), refresh=True)
                try:
                    key = keys.read_key()
                except KeyboardInterrupt:  # cbreak leaves Ctrl-C as a signal
                    return None

                if key == "up" and visible:
                    index = (index - 1) % len(visible)
                elif key == "down" and visible:
                    index = (index + 1) % len(visible)
                elif key == "enter":
                    if visible:
                        return visible[index]
                elif key in ("esc", "ctrl-c", "eof"):
                    return None
                elif key == "backspace":
                    filter_text = filter_text[:-1]
                    visible = [o for o in options if filter_text.lower() in o.lower()]
                    index = 0
                elif len(key) == 1 and key.isprintable():
                    candidate = filter_text + key
                    narrowed = [o for o in options if candidate.lower() in o.lower()]
                    # Ignore a keystroke that would match nothing, rather than
                    # stranding the user on an empty list they must backspace out of.
                    if narrowed:
                        filter_text, visible, index = candidate, narrowed, 0
    except (OSError, ValueError, ImportError):
        # No usable terminal (or termios missing) - fall back rather than fail.
        return _select_fallback(console, title, options)


def _select_fallback(console: Console, title: str, options: Sequence[str]) -> str | None:
    console.print(title)
    for i, opt in enumerate(options, 1):
        console.print(f"  {i}. {opt}")
    try:
        raw = input("number (blank to cancel)> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw.isdigit():
        return None
    idx = int(raw)
    return options[idx - 1] if 1 <= idx <= len(options) else None


# (key, returned value, spelled-out word) - the key is what gets displayed.
YES_NO_STOP = (("y", "yes", "yes"), ("n", "no", "no"), ("s", "stop", "stop"))


def confirm(
    console: Console,
    question: str,
    options: Sequence[tuple[str, str, str]] = YES_NO_STOP,
    default: str = "yes",
) -> str:
    """Ask a single-key question. Enter takes the default.

    `options` is ordered (key, value, word). The keys are what's displayed,
    and typing the key, the word, or the value all work - the prompt must
    never advertise an answer it then rejects.

    Buffered keystrokes are dropped first, so an answer always reflects a
    deliberate keypress made after the question appeared.
    """
    lookup: dict[str, str] = {}
    for key, value, word in options:
        lookup[key.lower()] = value
        lookup[word.lower()] = value
        lookup[value.lower()] = value

    # Capitalise the default so it's obvious what Enter does: [Y/n/s]
    hint = "/".join(key.upper() if value == default else key for key, value, _ in options)
    accepted = ", ".join(key for key, _, _ in options)

    while True:
        drain_stdin()
        try:
            raw = input(build_prompt(f"{question} [{hint}] ", color="1")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return default
        if not raw:
            return default
        if raw in lookup:
            return lookup[raw]
        console.print(f"[dim]answer {accepted} - or press Enter for {default}[/dim]")


@contextlib.contextmanager
def status(console: Console, message: str) -> Iterator[object]:
    """Spinner while something slow happens.

    Silent when output isn't a terminal: a spinner is a live-display
    affordance, and echoing its text into a pipe or log just duplicates
    whatever the caller already printed.
    """
    if not console.is_terminal:
        yield None
        return
    with console.status(f"[dim]{message}[/dim]", spinner="dots") as st:
        yield st
