import io
import re

from rich.console import Console

from local_edit import ui


def test_prompt_wraps_ansi_in_readline_width_markers(monkeypatch):
    # The whole point: readline must be told which bytes take up no screen
    # columns, otherwise Backspace walks left over the prompt and erases it.
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.delenv("LOCAL_EDIT_PLAIN_PROMPT", raising=False)

    prompt = ui.build_prompt("you> ")

    assert ui.RL_START in prompt and ui.RL_END in prompt
    assert "you> " in prompt
    # every escape sequence must sit inside a \001..\002 pair
    for chunk in prompt.split(ui.RL_START)[1:]:
        assert ui.RL_END in chunk, "an escape sequence was left unwrapped"


def test_visible_prompt_width_is_just_the_label(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.delenv("LOCAL_EDIT_PLAIN_PROMPT", raising=False)

    prompt = ui.build_prompt("you> ")

    # strip everything readline is told to ignore; what remains is what the
    # terminal actually draws, and readline's column math depends on it
    visible = ""
    in_ignored = False
    for ch in prompt:
        if ch == ui.RL_START:
            in_ignored = True
        elif ch == ui.RL_END:
            in_ignored = False
        elif not in_ignored:
            visible += ch
    assert visible == "you> "


def test_plain_prompt_env_var_disables_escapes(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setenv("LOCAL_EDIT_PLAIN_PROMPT", "1")
    assert ui.build_prompt("you> ") == "you> "


def test_non_interactive_prompt_is_plain(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    assert ui.build_prompt("you> ") == "you> "


def test_drain_stdin_is_safe_when_not_a_terminal(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    ui.drain_stdin()  # must not raise


def test_select_falls_back_to_numbers_without_a_tty(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: "2")
    console = Console(file=io.StringIO())
    assert ui.select(console, "pick", ["a", "b", "c"]) == "b"


def test_select_fallback_cancels_on_blank(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    console = Console(file=io.StringIO())
    assert ui.select(console, "pick", ["a", "b"]) is None


def test_select_fallback_rejects_out_of_range(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: "99")
    console = Console(file=io.StringIO())
    assert ui.select(console, "pick", ["a", "b"]) is None


def test_select_with_no_options_returns_none():
    console = Console(file=io.StringIO())
    assert ui.select(console, "pick", []) is None


def test_confirm_maps_input_to_choice(monkeypatch):
    monkeypatch.setattr(ui, "drain_stdin", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    console = Console(file=io.StringIO())
    assert ui.confirm(console, "ok?", default="yes") == "no"


def test_confirm_accepts_the_full_word_too(monkeypatch):
    # the prompt shows single letters, but the spelled-out word must also work
    monkeypatch.setattr(ui, "drain_stdin", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    console = Console(file=io.StringIO())
    assert ui.confirm(console, "ok?", default="yes") == "no"


def test_confirm_blank_input_uses_default(monkeypatch):
    monkeypatch.setattr(ui, "drain_stdin", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    console = Console(file=io.StringIO())
    assert ui.confirm(console, "ok?", default="yes") == "yes"


def test_confirm_drains_stdin_before_asking(monkeypatch):
    # Keystrokes typed while the model was streaming must never be consumed
    # as the answer to an approval prompt.
    drained = []
    monkeypatch.setattr(ui, "drain_stdin", lambda: drained.append(True))
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    console = Console(file=io.StringIO())
    ui.confirm(console, "ok?", default="yes")
    assert drained, "confirm() must drain buffered input first"


def test_confirm_eof_returns_default(monkeypatch):
    monkeypatch.setattr(ui, "drain_stdin", lambda: None)

    def raise_eof(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    console = Console(file=io.StringIO())
    assert ui.confirm(console, "ok?", default="yes") == "yes"


def test_confirm_prompt_only_advertises_answers_it_accepts(monkeypatch):
    # The exact bug reported: the prompt displayed "[quit/run/skip]" but
    # only "y"/"n"/"q" were actually accepted - typing the displayed word
    # did nothing.
    monkeypatch.setattr(ui, "drain_stdin", lambda: None)
    captured_prompt = {}

    def fake_input(prompt=""):
        captured_prompt["text"] = prompt
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)
    console = Console(file=io.StringIO())

    result = ui.confirm(console, "Run it?", default="yes")
    assert result == "yes"

    bracket = re.search(r"\[([a-zA-Z/]+)\]", captured_prompt["text"])
    assert bracket, f"no [choices] shown in {captured_prompt['text']!r}"

    # Every letter shown in the prompt must, fed back in on its own, produce
    # a real answer rather than being rejected.
    for letter in bracket.group(1).lower().split("/"):
        monkeypatch.setattr("builtins.input", lambda prompt="", _l=letter: _l)
        assert ui.confirm(console, "Run it?", default="yes") in ("yes", "no", "stop")
