"""Context-window budgeting and recovery.

Origin: "create a hello world html" failed instantly with
  request (5294 tokens) exceeds the available context size (4096 tokens)
because the system prompt had grown to ~889 tokens and the history budget
was a hardcoded 12000 chars (~3000 tokens), with no awareness of the actual
window. llama.cpp's --parallel 2 also halves the context per slot, so a
server started with --ctx-size 8192 only offers 4096.
"""

import io

from rich.console import Console

from local_edit.agent import CHARS_PER_TOKEN, DEFAULT_CONTEXT_SIZE, Agent
from local_edit.client import ContextTooLargeError, _parse_context_error
from local_edit.protocol import build_system_prompt
from local_edit.state import Session


# -- parsing the server's complaint -----------------------------------------


def test_parses_the_real_llamacpp_error():
    body = (
        '{"error":{"code":400,"message":"request (5294 tokens) exceeds the available '
        'context size (4096 tokens), try increasing it","type":"exceed_context_size_error",'
        '"n_prompt_tokens":5294,"n_ctx":4096}}'
    )
    assert _parse_context_error(body) == (4096, 5294)


def test_parses_message_text_when_fields_are_absent():
    body = (
        '{"error":{"message":"request (900 tokens) exceeds the available context size '
        '(512 tokens)","type":"exceed_context_size_error"}}'
    )
    assert _parse_context_error(body) == (512, 900)


def test_unrelated_errors_are_not_mistaken_for_context_overflow():
    assert _parse_context_error('{"error":{"message":"template broken"}}') == (None, None)
    assert _parse_context_error("not json at all") == (None, None)


# -- the system prompt has to leave room to work ----------------------------


def test_system_prompt_fits_comfortably_in_a_small_window():
    prompt = build_system_prompt("/project", allow_run=True)
    tokens = len(prompt) // CHARS_PER_TOKEN
    assert tokens < 600, f"system prompt is {tokens} tokens - too big for a 4096 window"


# -- budgeting ---------------------------------------------------------------


class FakeClient:
    model = "fake"
    max_tokens = 512

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield "DONE\n"


def _agent(tmp_path, context_size=DEFAULT_CONTEXT_SIZE):
    session = Session(tmp_path)
    agent = Agent(
        FakeClient(), tmp_path, session, Console(file=io.StringIO()),
        auto_apply=True, self_review_enabled=False, context_size=context_size,
    )
    return agent, session


def test_total_prompt_stays_within_the_window(tmp_path):
    agent, session = _agent(tmp_path, context_size=4096)
    for i in range(40):
        session.add_message("user", "u" * 2000)
        session.add_message("assistant", "a" * 2000)

    messages = agent._messages()
    total_tokens = sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN
    assert total_tokens + agent.client.max_tokens < 4096


def test_smaller_window_keeps_less_history(tmp_path):
    big, session_big = _agent(tmp_path, context_size=8192)
    for i in range(40):
        session_big.add_message("user", "u" * 500)
        session_big.add_message("assistant", "a" * 500)
    small, session_small = _agent(tmp_path, context_size=2048)
    small.session = session_big

    assert sum(len(m["content"]) for m in small._messages()) < sum(
        len(m["content"]) for m in big._messages()
    )


def test_one_oversized_message_is_trimmed_rather_than_failing(tmp_path):
    agent, session = _agent(tmp_path, context_size=2048)
    session.add_message("user", "x" * 100_000)  # e.g. a huge READ or command dump

    messages = agent._messages()
    total = sum(len(m["content"]) for m in messages)
    assert total < 2048 * CHARS_PER_TOKEN
    assert "trimmed to fit" in messages[-1]["content"]


def test_history_is_never_squeezed_to_nothing(tmp_path):
    agent, session = _agent(tmp_path, context_size=512)  # absurdly small
    session.add_message("user", "please help")
    messages = agent._messages()
    assert len(messages) >= 2, "the current request must always survive"


# -- self-healing ------------------------------------------------------------


def test_learns_the_real_window_from_the_error(tmp_path):
    agent, _ = _agent(tmp_path, context_size=32768)
    changed = agent._adapt_to_context(ContextTooLargeError("too big", n_ctx=4096, n_prompt_tokens=9000))
    assert changed
    assert agent.context_size == 4096


def test_reply_budget_is_clawed_back_on_a_small_window(tmp_path):
    agent, _ = _agent(tmp_path, context_size=4096)
    agent.client.max_tokens = 4000  # leaves no room for any prompt
    agent._adapt_to_context(ContextTooLargeError("too big", n_ctx=4096))
    assert agent.client.max_tokens <= 4096 // 3


def test_gives_up_when_there_is_nothing_left_to_trim(tmp_path):
    agent, session = _agent(tmp_path, context_size=4096)
    agent.client.max_tokens = 256
    session.history.clear()
    assert agent._adapt_to_context(ContextTooLargeError("too big", n_ctx=4096)) is False


class OverflowThenSucceedClient:
    """Rejects the first request the way llama.cpp does, then accepts."""

    model = "fake"
    max_tokens = 512

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        if self.calls == 1:
            raise ContextTooLargeError("prompt was 5294 tokens", n_ctx=4096, n_prompt_tokens=5294)
        yield "DONE\n"


def test_model_turn_retries_after_learning_the_window(tmp_path):
    session = Session(tmp_path)
    client = OverflowThenSucceedClient()
    agent = Agent(
        client, tmp_path, session, Console(file=io.StringIO()),
        auto_apply=True, self_review_enabled=False, context_size=32768,
    )
    session.add_message("user", "create a hello world html")

    result = agent._model_turn()

    assert result == "DONE"
    assert client.calls == 2, "should retry once after shrinking, not surface the error"
    assert agent.context_size == 4096
