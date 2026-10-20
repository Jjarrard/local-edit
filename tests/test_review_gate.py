"""A FLAGGED self-review must block the write.

Seen live: the reviewer correctly rejected an edit that nested a function
inside itself ("introduces a nested function... does not correctly implement
the intended task") and the edit was applied anyway, because --auto-apply
skipped straight past the verdict. The review was decorative.
"""

import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.protocol import parse_action
from local_edit.state import Session


class VerdictClient:
    """Streams a fixed self-review verdict."""

    model = "fake"

    def __init__(self, verdict_text):
        self.verdict_text = verdict_text

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield self.verdict_text


def _agent(tmp_path, verdict, **kwargs):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    kwargs.setdefault("auto_apply", True)
    agent = Agent(VerdictClient(verdict), tmp_path, session, console, self_review_enabled=True, **kwargs)
    return agent, session


ORIGINAL = "def add(a, b):\n    return a - b\n"
GOOD_EDIT = "EDIT calc.py 2-2\n```\n    return a + b\n```\n"


def test_flagged_review_blocks_the_write_even_under_auto_apply(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "NO - this nests the function inside itself and breaks the task.")

    result_text, finished = agent._handle_action(parse_action(GOOD_EDIT), None)

    assert target.read_text() == ORIGINAL, "a rejected diff must not reach disk"
    assert "REJECTED by review" in result_text
    assert finished is False


def test_rejection_feeds_the_reviewer_reason_back_to_the_model(tmp_path):
    (tmp_path / "calc.py").write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "NO - it nests add() inside itself.")

    result_text, _ = agent._handle_action(parse_action(GOOD_EDIT), None)

    assert "nests add() inside itself" in result_text
    assert "corrected edit" in result_text


def test_rejected_edit_is_not_journaled_so_undo_is_unaffected(tmp_path):
    (tmp_path / "calc.py").write_text(ORIGINAL)
    agent, session = _agent(tmp_path, "NO, wrong.")

    agent._handle_action(parse_action(GOOD_EDIT), None)

    assert session.applied_edits == []


def test_approved_review_still_applies(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, session = _agent(tmp_path, "YES - correct and minimal.")

    result_text, _ = agent._handle_action(parse_action(GOOD_EDIT), None)

    assert "return a + b" in target.read_text()
    assert "Applied" in result_text
    assert len(session.applied_edits) == 1


def test_review_disabled_applies_without_a_verdict(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    agent = Agent(
        VerdictClient("NO, terrible"), tmp_path, session, console,
        auto_apply=True, self_review_enabled=False,
    )

    agent._handle_action(parse_action(GOOD_EDIT), None)

    assert "return a + b" in target.read_text(), "with review off, the verdict is never consulted"


def test_identical_edit_resent_after_a_flag_is_applied(tmp_path):
    # A small reviewer produces false negatives too - it flagged a perfectly
    # correct fix in testing. If the model reconsiders and stands by the same
    # edit, apply it rather than deadlocking the session forever.
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "NO - I misread this, but I object anyway.")

    first, _ = agent._handle_action(parse_action(GOOD_EDIT), None)
    assert target.read_text() == ORIGINAL, "first attempt is held back"
    assert "REJECTED by review" in first

    second, _ = agent._handle_action(parse_action(GOOD_EDIT), None)
    assert "return a + b" in target.read_text(), "an unchanged retry must get through"
    assert "Applied" in second


def test_a_different_edit_after_a_flag_is_still_reviewed(tmp_path):
    # standing by the *same* edit earns a pass; a new one gets its own review
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "NO, wrong.")

    agent._handle_action(parse_action(GOOD_EDIT), None)
    other = "EDIT calc.py 2-2\n```\n    return b - a\n```\n"
    result_text, _ = agent._handle_action(parse_action(other), None)

    assert target.read_text() == ORIGINAL
    assert "REJECTED by review" in result_text


def test_unparseable_verdict_defaults_to_allowing_the_edit(tmp_path):
    # fail-open: an unusable review must not wedge the agent
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "hmm, hard to say either way")

    agent._handle_action(parse_action(GOOD_EDIT), None)

    assert "return a + b" in target.read_text()


def test_lowercase_no_in_the_reason_does_not_flip_a_yes(tmp_path):
    # Seen live: "YES\nThe file is valid HTML/CSS with no syntax errors" was
    # read as NO because the last yes/no word anywhere in the reply won.
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "YES  \nThe change is valid, with no syntax errors.")

    result_text, _ = agent._handle_action(parse_action(GOOD_EDIT), None)

    assert target.read_text() == "def add(a, b):\n    return a + b\n"
    assert result_text.startswith("Applied.")


def test_no_op_edit_is_neither_reviewed_nor_applied(tmp_path):
    # Seen live: a model resent the unchanged file; the review objected, the
    # identical retry slipped through the "apply unchanged retries" escape
    # hatch, and the user was asked to approve "(no textual change)" forever.
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL)
    agent, _ = _agent(tmp_path, "NO - nothing changed.")
    same = "EDIT calc.py 2-2\n```\n    return a - b\n```\n"

    for _ in range(2):
        result_text, _ = agent._handle_action(parse_action(same), None)
        assert result_text.startswith("No change")
    assert agent.review_rejections == 0
    assert agent.edits_applied == 0


def test_unfinished_html_page_is_reported_back_to_the_model(tmp_path):
    # Seen live: an INSERT ended mid-<style>, was applied, and the model
    # declared success while the browser showed a blank page.
    agent, _ = _agent(tmp_path, "YES - fine.")
    partial = "INSERT index.html 0\n```html\n<html><head><style>\nbody { color: red; }\n```\n"

    result_text, _ = agent._handle_action(parse_action(partial), None)

    assert (tmp_path / "index.html").exists()
    assert "unfinished" in result_text and "<style>" in result_text
