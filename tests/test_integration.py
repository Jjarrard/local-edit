"""End-to-end pipeline tests: a model response goes through validate,
self-review, and write (or gets stopped by one of them), using a stub client
in place of a real Jan server.

Unlike the per-module unit tests elsewhere, these exercise the full path a
real EDIT action takes through Agent._handle_action so a regression in how
the pieces are wired together (not just in one module) gets caught.
"""

import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.protocol import parse_action
from local_edit.state import Session


class StubReviewClient:
    """Streams a fixed self-review verdict, standing in for the fast model."""

    model = "fake"

    def __init__(self, verdict_text):
        self.verdict_text = verdict_text

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield self.verdict_text


def _agent(tmp_path, verdict="YES - looks correct.", **kwargs):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    kwargs.setdefault("auto_apply", True)
    agent = Agent(StubReviewClient(verdict), tmp_path, session, console, self_review_enabled=True, **kwargs)
    return agent, session


ORIGINAL = "def add(a, b):\n    return a - b\n"


def test_good_edit_passes_validate_review_and_lands_on_disk(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, session = _agent(tmp_path, "YES - correctly fixes the bug.")

    result_text, finished = agent._handle_action(
        parse_action("EDIT calc.py 2-2\n```\n    return a + b\n```\n"), None
    )

    assert finished is False
    assert "Applied" in result_text
    assert target.read_text() == "def add(a, b):\n    return a + b\n"
    assert agent.edits_applied == 1
    assert agent.validation_failures == 0
    assert agent.review_rejections == 0
    # the journal (used by /diff, /undo, /history) recorded the change
    assert len(session.applied_edits) == 1
    assert session.applied_edits[0]["rel"] == "calc.py"


def test_syntactically_broken_edit_never_reaches_review_or_disk(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, session = _agent(tmp_path)

    result_text, finished = agent._handle_action(
        parse_action("EDIT calc.py 2-2\n```\n    return a +\n```\n"), None
    )

    assert finished is False
    assert "would leave" in result_text
    assert target.read_text() == ORIGINAL
    assert agent.validation_failures == 1
    assert agent.edits_applied == 0
    assert session.applied_edits == []


def test_review_rejected_edit_never_reaches_disk_but_can_be_forced_through(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, session = _agent(tmp_path, "NO - this doesn't fix the bug.")
    action_text = "EDIT calc.py 2-2\n```\n    return a + b\n```\n"

    # First attempt: review objects, nothing is written.
    result_text, finished = agent._handle_action(parse_action(action_text), None)
    assert "REJECTED by review" in result_text
    assert target.read_text() == ORIGINAL
    assert agent.review_rejections == 1
    assert session.applied_edits == []

    # Model resends the identical edit: the harness applies it rather than
    # deadlocking on one wrong verdict.
    result_text, finished = agent._handle_action(parse_action(action_text), None)
    assert "Applied" in result_text
    assert target.read_text() == "def add(a, b):\n    return a + b\n"
    assert len(session.applied_edits) == 1
