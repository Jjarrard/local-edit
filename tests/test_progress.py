"""The progress note and old-turn trimming keep a small model on task in a
small window: the request stays visible however long the session runs, and
the prompt stops growing with every file the model reads."""

import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.progress import MAX_NOTE_CHARS, MAX_STEPS, Progress, trim_old_message
from local_edit.protocol import parse_action
from local_edit.state import Session


def test_note_is_capped_however_long_the_session_runs():
    p = Progress()
    p.start("build a page listing bookable stalls in the city centre " * 20)
    for i in range(50):
        p.note_file(f"src/file_{i}.html", i)
        p.record(f"EDIT src/file_{i}.html 1-{i}", "ERROR: " + "x" * 500)
    note = p.render()
    assert len(note) <= MAX_NOTE_CHARS
    assert note.count("\n  ") <= MAX_STEPS
    assert "Request:" in note


def test_follow_up_keeps_the_request_it_follows():
    p = Progress()
    p.start("make a stall booking page")
    p.start("it's still white")
    note = p.render()
    assert "Request: it's still white" in note
    assert "Before that: make a stall booking page" in note


def test_failures_are_remembered_until_an_edit_lands():
    p = Progress()
    p.start("fix the header")
    p.record("EDIT index.html 3-4", "Your EDIT to index.html was REJECTED by review and NOT applied.\n"
             "Reviewer said: NO - nests the rule inside itself.")
    assert "nests the rule" in p.render()
    p.record("EDIT index.html 3-5", "Applied. index.html now reflects the change.")
    assert "Unresolved problem" not in p.render()


def test_old_file_dumps_are_trimmed_but_the_users_words_are_not():
    dump = "--- index.html (lines 1-80 of 80) ---\n" + "\n".join(f"{i:>4}: <p>line</p>" for i in range(80))
    trimmed = trim_old_message({"role": "user", "content": dump})["content"]
    assert trimmed.startswith("--- index.html (lines 1-80 of 80) ---")
    assert len(trimmed) < 200

    request = {"role": "user", "content": "please build " + "a really detailed page " * 30}
    assert trim_old_message(request) == request


def test_old_code_blocks_are_stubbed():
    code = "\n".join(f"line {i}" for i in range(40))
    msg = {"role": "assistant", "content": f"INSERT index.html 0\n```html\n{code}\n```"}
    assert trim_old_message(msg)["content"] == "INSERT index.html 0\n[40-line code block - handled, see progress note]"


class Silent:
    model = "fake"
    max_tokens = 512
    last_finish_reason = "stop"

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield "DONE"


def _agent(tmp_path):
    return Agent(Silent(), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                 auto_apply=True, self_review_enabled=False)


def test_note_rides_on_the_newest_message_not_the_system_prompt(tmp_path):
    agent = _agent(tmp_path)
    agent.session.add_message("user", "make index.html")
    agent.progress.start("make index.html")
    system_before = agent._messages()[0]["content"]

    agent.progress.record("READ index.html", "--- index.html ---")
    messages = agent._messages()

    assert messages[0]["content"] == system_before, "system prompt must stay byte-identical for prompt caching"
    assert "[progress note" in messages[-1]["content"]
    assert "[progress note" not in agent.session.history[-1]["content"], "the note is never stored"


def test_prompt_stops_growing_as_the_model_reads_files(tmp_path):
    (tmp_path / "big.html").write_text("\n".join(f"<p>paragraph {i} of the page</p>" for i in range(60)))
    agent = _agent(tmp_path)
    agent.session.add_message("user", "tidy big.html")
    agent.progress.start("tidy big.html")
    sizes = []
    for _ in range(6):
        action = parse_action("READ big.html")
        agent.session.add_message("assistant", "READ big.html")
        result, _ = agent._handle_action(action, None)
        agent.session.add_message("user", result)
        sizes.append(sum(len(m["content"]) for m in agent._messages()))
    assert sizes[-1] < sizes[1] * 1.5, sizes


class Checker:
    """Answers the DONE check with a fixed verdict."""

    model = "fake"
    max_tokens = 512
    last_finish_reason = "stop"

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        yield self.verdict


def test_premature_done_is_sent_back_once_with_whats_missing(tmp_path):
    # Seen live: asked for a green header AND a footer, the model changed the
    # header and said DONE.
    client = Checker("NO - the footer with contact info was never added.")
    agent = Agent(client, tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=True)
    agent.progress.start("make the header green and add a footer")
    agent.progress.record("EDIT index.html 15-17", "Applied.")

    result, finished = agent._handle_action(parse_action("DONE"), None)
    assert finished is False
    assert "footer" in result

    result, finished = agent._handle_action(parse_action("DONE"), None)
    assert finished is True, "only checked once per request - a stubborn NO can't trap the model"
    assert client.calls == 1


def test_done_is_accepted_when_the_check_agrees(tmp_path):
    agent = Agent(Checker("YES"), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=True)
    agent.progress.start("make the header green")

    assert agent._handle_action(parse_action("DONE"), None) == (None, True)
