"""A reply cut off by max_tokens mid-block is continued, not thrown away.

Seen live: a 4B model with a 512-token reply budget could never write a whole
web page - every INSERT was cut off, rejected, and resent from the top, so
nothing past the first ~70 lines was ever written.
"""

import io

from rich.console import Console

from local_edit.agent import Agent, _drop_echo
from local_edit.state import Session


class ScriptedClient:
    """Plays back one reply per call, reporting each one's finish reason."""

    model = "fake"
    max_tokens = 512

    def __init__(self, replies, echo=False):
        self.replies = list(replies)  # (text, finish_reason)
        self.echo = echo
        self.calls = []
        self.last_finish_reason = None

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls.append(messages)
        text, self.last_finish_reason = self.replies.pop(0)
        if self.echo and messages[-1]["role"] == "assistant":
            yield messages[-1]["content"]
        yield text


def _agent(tmp_path, client):
    session = Session(tmp_path)
    session.add_message("user", "make index.html")
    return Agent(client, tmp_path, session, Console(file=io.StringIO()), self_review_enabled=False)


def test_cut_off_block_is_continued_and_stitched(tmp_path):
    client = ScriptedClient([
        ("INSERT index.html 0\n```html\n<html>\n<bo", "length"),
        ("dy>hi</body>\n</html>\n```", "stop"),
    ])
    agent = _agent(tmp_path, client)

    reply = agent._model_turn()

    assert reply == "INSERT index.html 0\n```html\n<html>\n<body>hi</body>\n</html>\n```"
    continued = client.calls[1]
    assert continued[-1] == {"role": "assistant", "content": "INSERT index.html 0\n```html\n<html>\n<bo"}


def test_server_echo_of_the_prefill_is_dropped(tmp_path):
    client = ScriptedClient(
        [("INSERT a.css 0\n```\nbody {", "length"), ("\n  color: red;\n}\n```", "stop")], echo=True
    )
    agent = _agent(tmp_path, client)

    assert agent._model_turn() == "INSERT a.css 0\n```\nbody {\n  color: red;\n}\n```"


def test_cut_off_chat_is_not_continued(tmp_path):
    client = ScriptedClient([("SAY here is a long explanation that ran o", "length")])
    agent = _agent(tmp_path, client)

    agent._model_turn()

    assert len(client.calls) == 1


def test_continuations_are_capped(tmp_path):
    client = ScriptedClient([("INSERT a.txt 0\n```\nx", "length")] + [("x", "length")] * 20)
    agent = _agent(tmp_path, client)

    agent._model_turn()

    assert len(client.calls) == 5  # the first reply plus MAX_CONTINUATIONS


def test_drop_echo_handles_split_and_absent_echoes():
    assert "".join(_drop_echo(["ab", "cd", "EF"], "abcd")) == "EF"
    assert "".join(_drop_echo(["EF"], "abcd")) == "EF"
    assert "".join(_drop_echo(["ab"], "abcd")) == ""
