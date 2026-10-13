"""Commands wrapped in decoration must still parse.

Origin: DeepSeek-R1-Distill-Qwen-14B emitted

    COMMAND: INSERT index.html 0
    ```html
    ...
    ```

The parser required the keyword at the start of a line, so "COMMAND:" made
it fall through to SAY. The SAY happened to contain a code block, which
triggered the "code in a chat message" nudge, and the model - being asked to
resend as INSERT - produced the identical thing again. It never created the
file and looped until it ran out of steps.
"""

import io

import pytest
from rich.console import Console

from local_edit.agent import Agent
from local_edit.protocol import parse_action
from local_edit.state import Session

BLOCK = "```html\n<!DOCTYPE html>\n<html></html>\n```"


@pytest.mark.parametrize(
    "text,expected",
    [
        (f"COMMAND: INSERT index.html 0\n{BLOCK}", "INSERT"),
        (f"Command: INSERT index.html 0\n{BLOCK}", "INSERT"),
        (f"ACTION: INSERT index.html 0\n{BLOCK}", "INSERT"),
        (f"STEP: INSERT index.html 0\n{BLOCK}", "INSERT"),
        (f"TOOL: INSERT index.html 0\n{BLOCK}", "INSERT"),
        ("COMMAND - READ src/app.py", "READ"),
        ("- READ src/app.py", "READ"),
        ("* READ src/app.py", "READ"),
        ("• READ src/app.py", "READ"),
        ("1. READ src/app.py", "READ"),
        ("2) READ src/app.py", "READ"),
        ("> RUN pytest -q", "RUN"),
        ("`READ src/app.py`", "READ"),
        ("**READ src/app.py**", "READ"),
        ("**COMMAND:** LIST .", "LIST"),
        ("ACTION: DONE", "DONE"),
    ],
)
def test_decorated_commands_parse(text, expected):
    assert parse_action(text).kind == expected


def test_arguments_survive_the_decoration():
    action = parse_action("**EDIT src/app.py 12-14**\n```\nnew line\n```")
    assert action.kind == "EDIT"
    assert action.path == "src/app.py"
    assert (action.start, action.end) == (12, 14)
    assert action.payload == "new line"


def test_backticked_path_is_clean():
    assert parse_action("`READ src/app.py`").path == "src/app.py"


def test_run_command_keeps_its_full_argument():
    assert parse_action("COMMAND: RUN pytest -q --tb=short").text == "pytest -q --tb=short"


def test_plain_prose_is_still_chat():
    action = parse_action("I'll help you create a hello world file.")
    assert action.kind == "SAY"


def test_prose_after_a_label_is_not_run_as_a_command():
    # a labelled line that reads like a sentence must stay chat
    assert parse_action("COMMAND: is a word I sometimes use in a sentence.").kind == "SAY"


def test_labelled_shell_command_becomes_a_run():
    # "COMMAND: <shell>" is an instruction to execute, even without RUN;
    # it still passes through the normal approval gate
    action = parse_action("COMMAND: mkdir assets")
    assert action.kind == "RUN"
    assert action.text == "mkdir assets"


class DeepSeekStyleClient:
    """Reproduces the observed reply shape, verbatim."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        yield (
            "\nI'll help create a simple hello world HTML file.\n\n"
            "COMMAND: INSERT index.html 0\n"
            "```html\n<!DOCTYPE html>\n<html>\n<head><title>Hello World</title></head>\n"
            "<body><h1>Hello, World!</h1></body>\n</html>\n```\n"
        )


def test_the_file_actually_gets_created(tmp_path):
    session = Session(tmp_path)
    agent = Agent(
        DeepSeekStyleClient(), tmp_path, session, Console(file=io.StringIO()),
        auto_apply=True, self_review_enabled=False, max_steps=4,
    )

    agent.chat_turn("create a hello world html file")

    created = tmp_path / "index.html"
    assert created.exists(), "the INSERT must actually run"
    assert "Hello, World!" in created.read_text()


class StubbornCodeInSayClient:
    """Never issues a real command - only chat with a code block."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        yield f"Here is the file you asked for, attempt {self.calls}:\n```html\n<html></html>\n```\n"


def test_code_in_chat_stops_instead_of_spinning(tmp_path):
    session = Session(tmp_path)
    agent = Agent(
        StubbornCodeInSayClient(), tmp_path, session, Console(file=io.StringIO()),
        auto_apply=True, self_review_enabled=False, max_steps=12,
    )

    agent.chat_turn("create a hello world html file")

    assert agent.client.calls <= 3, f"looped {agent.client.calls} times instead of bailing out"
