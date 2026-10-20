"""Regressions from a live session that looped instead of writing a page.

The model narrated ("SAY I'll create...") before a full INSERT, wrote a whole
new page as INSERT-at-0 into a file that already existed, used line 0, and
wrote five identical card lines that tripped the repetition guard.
"""

import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.protocol import parse_action
from local_edit.state import Session

PAGE = "<!DOCTYPE html>\n<html>\n<body>\n<h1>Stalls</h1>\n</body>\n</html>"


def test_narration_before_an_insert_is_the_insert():
    action = parse_action(f"SAY I'll create the page.\n\nINSERT index.html 0\n```html\n{PAGE}\n```")
    assert action.kind == "INSERT"
    assert action.payload == PAGE
    assert action.preamble.startswith("SAY I'll create")


def test_plain_say_with_a_sql_snippet_stays_chat():
    action = parse_action("SAY run this:\n```sql\nDELETE FROM users WHERE id = 1;\n```")
    assert action.kind == "SAY"


def test_line_zero_means_line_one():
    assert (parse_action("READ index.html 0-0").start, parse_action("READ index.html 0-0").end) == (1, 1)
    edit = parse_action("EDIT index.html 0-53\n```\nx\n```")
    assert (edit.start, edit.end) == (1, 53)


class OneReply:
    model = "fake"
    max_tokens = 512
    last_finish_reason = "stop"

    def __init__(self, text):
        self.text = text

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield self.text


def _agent(tmp_path, reply="YES"):
    return Agent(OneReply(reply), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                 auto_apply=True, self_review_enabled=False)


def test_whole_page_insert_over_an_existing_page_replaces_it(tmp_path):
    (tmp_path / "index.html").write_text("<!DOCTYPE html>\n<html>\n<body>Hell world</body>\n</html>\n")
    agent = _agent(tmp_path)

    agent._handle_action(parse_action(f"INSERT index.html 0\n```html\n{PAGE}\n```"), None)

    written = (tmp_path / "index.html").read_text()
    assert written.count("<html>") == 1
    assert "Stalls" in written and "Hell world" not in written


def test_insert_at_top_of_unrelated_content_still_inserts(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n")
    agent = _agent(tmp_path)

    agent._handle_action(parse_action("INSERT app.py 0\n```\nimport os\n```"), None)

    assert (tmp_path / "app.py").read_text() == "import os\nprint('hi')\n"


def test_repeated_card_markup_is_not_mistaken_for_a_loop(tmp_path):
    cards = "\n".join('<div class="stall-card">\n  <h2>Stall %d</h2>\n</div>' % i for i in range(6))
    agent = _agent(tmp_path, reply=cards)

    out = agent._stream_and_print([{"role": "user", "content": "x"}])

    assert out.count('<div class="stall-card">') == 6
    assert agent._last_stream_aborted is False


def test_a_genuine_run_of_one_line_is_still_cut_off(tmp_path):
    agent = _agent(tmp_path, reply="\n".join(["the same line again"] * 20))

    agent._stream_and_print([{"role": "user", "content": "x"}])

    assert agent._last_stream_aborted is True


class Replies:
    model = "fake"
    max_tokens = 512
    last_finish_reason = "stop"

    def __init__(self, *replies):
        self.replies = list(replies)

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield self.replies.pop(0)


def test_code_written_as_chat_is_held_and_saved_by_a_bare_insert(tmp_path):
    # Seen live: the model wrote a full page in chat, was told to resend it
    # as INSERT, rewrote all of it in chat again (filling the context), then
    # finally sent "INSERT index.html 0" with no code - which failed to parse.
    client = Replies(f"SAY here is the page\n```html\n{PAGE}\n```", "INSERT index.html 0")
    session = Session(tmp_path)
    session.add_message("user", "make a stalls page")
    agent = Agent(client, tmp_path, session, Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)

    agent.step()
    assert "<h1>Stalls</h1>" not in session.history[-2]["content"], "held code is shrunk out of history"
    agent.step()

    assert (tmp_path / "index.html").read_text().rstrip("\n") == PAGE
    assert agent._held_code is None
