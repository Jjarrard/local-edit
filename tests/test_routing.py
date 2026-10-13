"""Cheap phases (planning, diff review) must go to the small model, while
code generation stays on the capable one."""

import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.review import REVIEW_MAX_TOKENS
from local_edit.state import Session


class RecordingClient:
    model = "big-code-model"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls.append({"model": model, "max_tokens": max_tokens})
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        yield reply


def _agent(tmp_path, client, **kwargs):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    kwargs.setdefault("auto_apply", True)
    kwargs.setdefault("fast_model", "small-fast-model")
    return Agent(client, tmp_path, session, console, **kwargs), session


def test_main_turn_uses_the_code_model(tmp_path):
    client = RecordingClient(["DONE\n"])
    agent, _ = _agent(tmp_path, client, self_review_enabled=False)

    agent.chat_turn("hi")

    # model=None means "use the client's own model", i.e. the code model
    assert client.calls[0]["model"] is None


def test_planning_is_routed_to_the_fast_model(tmp_path):
    client = RecordingClient(["TODO ADD do a thing\n"])
    agent, _ = _agent(tmp_path, client, self_review_enabled=False)

    agent.plan("some goal")

    assert client.calls[0]["model"] == "small-fast-model"


def test_self_review_is_routed_to_the_fast_model_with_a_small_budget(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    client = RecordingClient(["YES fine\n"])
    agent, _ = _agent(tmp_path, client, self_review_enabled=True)

    ok, _reason = agent._run_self_review("task", "a.py", "before", "after")

    assert ok is True
    assert client.calls[0]["model"] == "small-fast-model"
    assert client.calls[0]["max_tokens"] == REVIEW_MAX_TOKENS


def test_review_budget_is_much_smaller_than_a_full_generation():
    # the whole point of the cap: a verdict shouldn't cost as much as the edit
    assert REVIEW_MAX_TOKENS <= 512


def test_no_fast_model_means_everything_uses_the_code_model(tmp_path):
    client = RecordingClient(["TODO ADD thing\n"])
    agent, _ = _agent(tmp_path, client, fast_model=None, self_review_enabled=False)

    agent.plan("goal")

    assert client.calls[0]["model"] is None
