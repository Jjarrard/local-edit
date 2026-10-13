"""Guard against the model role-playing the whole conversation.

Seen live: asked to fix a bug, a 4B model replied with a fabricated
transcript - "You: INSERT src/calc.py 0 ... Result: 3 + 4 = 7 ... DONE" -
inventing both the tool output and its own success. It then reported the
task complete while the file was never touched.
"""

import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.protocol import parse_action, strip_hallucinated_continuation
from local_edit.state import Session

HALLUCINATED = """LIST .
Result:
project/
└── src/main.py

You: INSERT src/calc.py 0
```
def add(a, b):
    return a + b
```
You: RUN python3 src/calc.py
Result:
    3 + 4 = 7
You: DONE
"""


def test_only_the_real_command_survives():
    cleaned = strip_hallucinated_continuation(HALLUCINATED)
    assert cleaned.strip() == "LIST ."
    assert "INSERT" not in cleaned
    assert "3 + 4 = 7" not in cleaned


def test_parses_to_the_first_real_action():
    action = parse_action(strip_hallucinated_continuation(HALLUCINATED))
    assert action.kind == "LIST"


def test_clean_reply_is_left_alone():
    text = "EDIT a.py 1-1\n```\nx = 1\n```"
    assert strip_hallucinated_continuation(text) == text


def test_markers_inside_a_fenced_block_are_preserved():
    # a legitimate code payload may contain these words; only text outside
    # the fence indicates fabrication
    text = 'INSERT log.py 0\n```\nprint("Result: done")\nprint("You: hi")\n```'
    assert strip_hallucinated_continuation(text) == text


def test_various_roleplay_markers_are_cut():
    for marker in ("Result:", "Output:", "You:", "User:", "Assistant:", "Human:", "System:"):
        text = f"READ a.py\n{marker} something invented"
        assert strip_hallucinated_continuation(text).strip() == "READ a.py", marker


def test_case_insensitive_markers():
    assert strip_hallucinated_continuation("READ a.py\nRESULT: fake").strip() == "READ a.py"


class HallucinatingClient:
    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        yield HALLUCINATED


def test_agent_does_not_store_invented_output_in_history(tmp_path):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    agent = Agent(
        HallucinatingClient(), tmp_path, session, console,
        auto_apply=True, self_review_enabled=False, max_steps=1,
    )

    agent.step()

    assistant = [m["content"] for m in session.history if m["role"] == "assistant"]
    assert assistant, "the real command should still be recorded"
    joined = "\n".join(assistant)
    assert "3 + 4 = 7" not in joined, "invented tool output must never enter history"
    assert "DONE" not in joined, "invented completion must never enter history"


class PureFabricationClient:
    """Replies with nothing but invented conversation - no real command."""

    model = "fake"

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield "Result:\n    all tests passed\nYou: DONE\n"


def test_pure_fabrication_is_corrected_not_accepted(tmp_path):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    agent = Agent(
        PureFabricationClient(), tmp_path, session, console,
        auto_apply=True, self_review_enabled=False, max_steps=1,
    )

    finished, kind, _raw = agent.step()

    assert kind == "parse_error"
    assert finished is False, "a fabricated DONE must not end the task"
    last = session.history[-1]
    assert last["role"] == "user"
    assert "imagined results" in last["content"]
