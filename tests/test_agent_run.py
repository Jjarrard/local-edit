"""Agent-level behaviour for RUN commands and the pre-write validation gate."""

import io

import pytest
from rich.console import Console

from local_edit import ui
from local_edit.agent import Agent, ReviewAborted
from local_edit.protocol import parse_action
from local_edit.state import Session


class DummyClient:
    model = "fake"

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield "DONE\n"


def _agent(tmp_path, **kwargs):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    kwargs.setdefault("auto_apply", True)
    kwargs.setdefault("self_review_enabled", False)
    return Agent(DummyClient(), tmp_path, session, console, **kwargs), session


# -- RUN safety -----------------------------------------------------


def test_blocked_command_is_refused_and_never_executed(tmp_path):
    agent, _ = _agent(tmp_path)
    canary = tmp_path / "canary.txt"
    canary.write_text("still here")

    result_text, finished = agent._handle_run(parse_action(f"RUN rm -rf {tmp_path}"))

    assert finished is False
    assert "refused" in result_text.lower()
    assert canary.exists(), "a blocked command must not run"


def test_blocked_command_tells_model_not_to_retry(tmp_path):
    agent, _ = _agent(tmp_path)
    result_text, _ = agent._handle_run(parse_action("RUN sudo reboot"))
    assert "never be allowed" in result_text


def test_run_disabled_returns_error(tmp_path):
    agent, _ = _agent(tmp_path, allow_run=False)
    result_text, _ = agent._handle_run(parse_action("RUN ls"))
    assert "disabled" in result_text.lower()


def test_safe_command_runs_and_output_goes_back_to_model(tmp_path):
    (tmp_path / "hello.txt").write_text("x")
    agent, _ = _agent(tmp_path)

    result_text, finished = agent._handle_run(parse_action("RUN ls"))

    assert finished is False
    assert "hello.txt" in result_text
    assert "exit code 0" in result_text


def test_failing_command_reports_exit_code_to_model(tmp_path):
    agent, _ = _agent(tmp_path)
    # relative and nonexistent, so it fails on its own merits rather than
    # tripping the (correct, separate) outside-project approval requirement
    result_text, _ = agent._handle_run(parse_action("RUN ls definitely/not/here"))
    assert "exit code" in result_text
    assert "exit code 0" not in result_text


def test_command_needing_approval_is_skipped_when_declined(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, auto_apply=False)
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: "no")

    result_text, _ = agent._handle_run(parse_action("RUN touch created.txt"))

    assert not (tmp_path / "created.txt").exists()
    assert "declined" in result_text.lower()


def test_command_needing_approval_runs_when_accepted(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, auto_apply=False)
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: "yes")

    agent._handle_run(parse_action("RUN touch created.txt"))

    assert (tmp_path / "created.txt").exists()


def test_quitting_at_the_command_prompt_aborts(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, auto_apply=False)
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: "stop")
    with pytest.raises(ReviewAborted):
        agent._handle_run(parse_action("RUN touch x.txt"))


def test_confirm_all_commands_asks_even_for_safe_ones(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, auto_apply=False, auto_run_safe=False)
    asked = []
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: asked.append(True) or "no")

    agent._handle_run(parse_action("RUN ls"))

    assert asked, "with --confirm-all-commands even a safe command should prompt"


# -- validation gate -----------------------------------------------------


def test_edit_that_breaks_python_syntax_is_rejected(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("def f():\n    return 1\n")
    agent, _ = _agent(tmp_path)

    action = parse_action("EDIT app.py 2-2\n```\n    return (1\n```\n")
    result_text, _ = agent._handle_action(action, None)

    assert "ERROR" in result_text
    assert "NOT changed" in result_text
    assert target.read_text() == "def f():\n    return 1\n", "file must be untouched"


def test_valid_edit_still_applies(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("def f():\n    return 1\n")
    agent, _ = _agent(tmp_path)

    action = parse_action("EDIT app.py 2-2\n```\n    return 2\n```\n")
    result_text, _ = agent._handle_action(action, None)

    assert "Applied" in result_text
    assert "return 2" in target.read_text()


def test_new_file_with_broken_syntax_is_not_created(tmp_path):
    agent, _ = _agent(tmp_path)
    action = parse_action("INSERT broken.py 0\n```\ndef f(:\n```\n")

    result_text, _ = agent._handle_action(action, None)

    assert "ERROR" in result_text
    assert not (tmp_path / "broken.py").exists()


def test_broken_json_edit_is_rejected(tmp_path):
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n')
    agent, _ = _agent(tmp_path)

    action = parse_action('EDIT data.json 1-1\n```\n{"a": 1,}\n```\n')
    result_text, _ = agent._handle_action(action, None)

    assert "ERROR" in result_text
    assert target.read_text() == '{"a": 1}\n'


def test_non_code_file_is_not_blocked_by_validation(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("hello\n")
    agent, _ = _agent(tmp_path)

    action = parse_action("EDIT notes.md 1-1\n```\n# heading { unbalanced\n```\n")
    result_text, _ = agent._handle_action(action, None)

    assert "Applied" in result_text
