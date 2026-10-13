"""Creating folders, and recognising shell commands written without RUN.

Origin: asked to "create a new folder", Jan-v3.5-4B replied

    COMMAND: cd /Users/.../src && mkdir assets && ls -la

Neither `cd` nor `mkdir` is a local-edit keyword, so after the COMMAND: label
was stripped there was nothing to match - it fell through to chat and the
turn ended silently with nothing created.
"""

import io

import pytest
from rich.console import Console

from local_edit import files, ui
from local_edit.agent import Agent
from local_edit.protocol import find_bare_shell_command, parse_action
from local_edit.state import Session


# -- recognising an unlabelled shell command --------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("COMMAND: mkdir assets", "mkdir assets"),
        ("COMMAND: cd src && mkdir assets && ls -la", "cd src && mkdir assets && ls -la"),
        ("SHELL: ls -la", "ls -la"),
        ("BASH: pytest -q", "pytest -q"),
        ("TERMINAL: git status", "git status"),
        ("$ mkdir -p src/assets", "mkdir -p src/assets"),
        ("Here you go:\n```bash\nmkdir assets\n```", "mkdir assets"),
        ("```sh\n$ mkdir assets\n```", "mkdir assets"),
        ("```console\n# a comment\nls -la\n```", "ls -la"),
    ],
)
def test_bare_shell_commands_are_found(text, expected):
    assert find_bare_shell_command(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I'll help you create a folder for your project.",
        "COMMAND: is a word I sometimes use in a sentence.",
        "Let me think about how to structure this.",
        "",
    ],
)
def test_prose_is_never_treated_as_a_command(text):
    assert find_bare_shell_command(text) is None


def test_bare_shell_parses_as_a_run_action():
    action = parse_action("COMMAND: cd src && mkdir assets && ls -la")
    assert action.kind == "RUN"
    assert action.text == "cd src && mkdir assets && ls -la"


def test_a_real_local_edit_keyword_still_wins():
    # an explicit command must not be reinterpreted as shell
    assert parse_action("COMMAND: LIST src").kind == "LIST"
    assert parse_action("COMMAND: READ a.py").kind == "READ"


# -- folders actually getting created ---------------------------------------


def test_insert_creates_missing_parent_folders(tmp_path):
    result = files.plan_insert(tmp_path, "assets/css/style.css", 0, "body { color: red; }")
    files.write(tmp_path, result.rel, result.new_text)
    created = tmp_path / "assets" / "css" / "style.css"
    assert created.exists()
    assert created.parent.is_dir()


class MkdirClient:
    """Replies the way the real model did - a labelled bare shell command."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        if self.calls == 1:
            yield "COMMAND: mkdir -p assets\n"
        else:
            yield "DONE\n"


def test_model_can_create_a_folder_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: "yes")
    session = Session(tmp_path)
    agent = Agent(
        MkdirClient(), tmp_path, session, Console(file=io.StringIO()),
        auto_apply=False, self_review_enabled=False, max_steps=4,
    )

    agent.chat_turn("create a new folder called assets")

    assert (tmp_path / "assets").is_dir(), "the folder should actually exist"


def test_folder_creation_still_asks_first(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: asked.append(True) or "no")
    session = Session(tmp_path)
    agent = Agent(
        MkdirClient(), tmp_path, session, Console(file=io.StringIO()),
        auto_apply=False, self_review_enabled=False, max_steps=2,
    )

    agent.chat_turn("create a new folder called assets")

    assert asked, "mkdir mutates the project, so it must go through approval"
    assert not (tmp_path / "assets").exists(), "declining must not create it"


def test_system_prompt_explains_folder_creation():
    from local_edit.protocol import build_system_prompt

    prompt = build_system_prompt("/project", allow_run=True)
    assert "folder" in prompt.lower()
    assert "mkdir" in prompt.lower()
