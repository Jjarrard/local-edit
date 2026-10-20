import io

from rich.console import Console

from local_edit.agent import Agent
from local_edit.client import JanConnectionError
from local_edit.protocol import parse_action
from local_edit.state import STATUS_BLOCKED, STATUS_PENDING, Session, TodoItem


class BrokenClient:
    model = "fake"

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        raise JanConnectionError("connection refused")
        yield  # pragma: no cover - makes this a generator function


def _agent(tmp_path):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    agent = Agent(BrokenClient(), tmp_path, session, console, auto_apply=True)
    return agent, session


def test_say_with_code_block_gets_pushed_to_redo_as_real_edit(tmp_path):
    # Reproduces a real incident: asked to "recreate the game Pong" in an
    # empty project, the model wrote the whole file as a SAY message with a
    # fenced code block instead of an INSERT - the code was never written
    # anywhere, and the chat turn just ended as if it had finished.
    agent, session = _agent(tmp_path)
    action = parse_action("SAY here's the code\n```python\nprint('hi')\n```\n")

    result_text, finished = agent._handle_action(action, None)

    assert finished is False
    assert result_text is not None
    assert "NOT saved" in result_text
    assert agent._held_code == "print('hi')"


def test_say_without_code_block_still_ends_chat_turn(tmp_path):
    agent, session = _agent(tmp_path)
    action = parse_action("SAY just chatting, no code here")

    result_text, finished = agent._handle_action(action, None)

    assert finished is True
    assert result_text is None


def test_chat_turn_failure_leaves_empty_history_untouched(tmp_path):
    agent, session = _agent(tmp_path)
    agent.chat_turn("hello?")
    assert session.history == []


def test_validation_failure_increments_counter_and_leaves_file_untouched(tmp_path):
    (tmp_path / "bad.py").write_text("def f():\n    return 1\n")
    agent, session = _agent(tmp_path)
    action = parse_action("EDIT bad.py 2-2\n```\n    return (\n```\n")

    result_text, finished = agent._handle_action(action, None)

    assert finished is False
    assert "would leave" in result_text
    assert agent.validation_failures == 1
    assert agent.edits_applied == 0
    assert (tmp_path / "bad.py").read_text() == "def f():\n    return 1\n"


def test_applied_edit_increments_edits_applied_counter(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    agent, session = _agent(tmp_path)
    action = parse_action("EDIT ok.py 1-1\n```\nx = 2\n```\n")

    agent._handle_action(action, None)

    assert agent.edits_applied == 1
    assert agent.validation_failures == 0


class ScriptedClient:
    model = "fake"

    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        yield reply


def test_chat_turn_continues_through_tool_actions_to_a_final_reply(tmp_path):
    # Reproduces a real bug: a chat message that needs a couple of LIST/GREP
    # calls before there's anything to say (e.g. "recreate the game Pong" in
    # an empty project) used to just stop after the first tool call, leaving
    # the user staring at a bare prompt with nothing having happened.
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    client = ScriptedClient(["LIST .\n", "GREP foo\n", "SAY here's my plan\n"])
    agent = Agent(client, tmp_path, session, console, auto_apply=True, self_review_enabled=False, max_steps=10)

    agent.chat_turn("recreate the game Pong")

    assert client.calls == 3
    assistant_msgs = [m for m in session.history if m["role"] == "assistant"]
    assert len(assistant_msgs) == 3
    assert session.history[-1]["role"] == "assistant"
    assert "here's my plan" in session.history[-1]["content"]


class CyclingListClient:
    """Never says anything final - always issues a distinct LIST, so the
    repeat-detector never trips and the loop only ends via the step cap."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        yield f"LIST nonexistent_{self.calls}\n"


def test_chat_turn_gives_up_after_max_steps_without_a_reply(tmp_path):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    client = CyclingListClient()
    agent = Agent(client, tmp_path, session, console, auto_apply=True, self_review_enabled=False, max_steps=4)

    agent.chat_turn("do something open-ended")

    assert client.calls == 4  # stopped at max_steps, not looping forever
    assistant_msgs = [m for m in session.history if m["role"] == "assistant"]
    assert len(assistant_msgs) == 4


class SucceedsThenFailsClient:
    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        if self.calls == 1:
            yield "LIST .\n"
        elif self.calls == 2:
            yield "GREP foo\n"
        else:
            raise JanConnectionError("dropped mid-conversation")


def test_chat_turn_keeps_partial_progress_when_it_fails_partway(tmp_path):
    # Only the very first step's dangling user turn is safe to roll back on
    # failure. If earlier steps in this same chat_turn already succeeded,
    # their exchanges are legitimate and must survive a later connection drop.
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    client = SucceedsThenFailsClient()
    agent = Agent(client, tmp_path, session, console, auto_apply=True, self_review_enabled=False, max_steps=10)

    agent.chat_turn("recreate the game Pong")

    assistant_msgs = [m for m in session.history if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2
    assert session.history[-1]["role"] == "user"


def test_chat_turn_failure_restores_prior_message_instead_of_deleting_it(tmp_path):
    # Reproduces a real bug: if history already ends on a "user" turn (e.g. a
    # todo's closing tool-result), add_message() merges the new user text
    # into it rather than appending. A naive pop()-on-failure then deleted
    # that legitimate prior content instead of just undoing the new text.
    agent, session = _agent(tmp_path)
    session.add_message("assistant", "previous reply")
    session.add_message("user", "existing tool result that must survive")

    agent.chat_turn("new question")

    assert len(session.history) == 2
    assert session.history[-1]["content"] == "existing tool result that must survive"
    assert "new question" not in session.history[-1]["content"]


def test_messages_never_start_with_assistant_after_trimming(tmp_path):
    # Reproduces a real bug seen against Gemma's chat template, which hard-
    # rejects a conversation that doesn't open with a user turn. A history
    # that's an odd number of messages long (always ends on "user" here,
    # since every session starts with "user") loses its leading "user" entry
    # when trimmed to an even MAX_HISTORY_MESSAGES window, leaving an
    # orphaned "assistant" reply at the front.
    from local_edit.agent import MAX_HISTORY_MESSAGES

    agent, session = _agent(tmp_path)
    for i in range(MAX_HISTORY_MESSAGES + 5):
        session.add_message("user", f"user msg {i}")
        session.add_message("assistant", f"assistant msg {i}")
    session.add_message("user", "final dangling user prompt")  # makes history length odd

    messages = agent._messages()
    non_system = messages[1:]
    assert non_system, "should have kept some history"
    assert non_system[0]["role"] == "user", non_system[0]

    roles = [m["role"] for m in non_system]
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], f"non-alternating roles at index {i}: {roles}"


class SpammyTodoAddClient:
    """A model that, instead of working the assigned task, keeps adding new
    (never-repeating) todos. Reproduces a real incident: this ran for 200+
    todos and never terminated, because TODO_ADD counted as "progress" and
    reset the stall counter every time."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.calls += 1
        yield f"TODO ADD unrelated item {self.calls}\n"


def test_run_todo_blocks_on_off_task_todo_add_spam(tmp_path):
    session = Session(tmp_path)
    console = Console(file=io.StringIO())
    client = SpammyTodoAddClient()
    agent = Agent(client, tmp_path, session, console, auto_apply=True, self_review_enabled=False, max_steps=10)
    todo = session.todos.add("do the actual assigned task")

    agent.run_todo(todo)

    assert todo.status == STATUS_BLOCKED
    assert "instead of working" in todo.note
    # blocked after 3 non-progress turns, not all 10 max_steps - each of those
    # turns also queued a new todo, so this bounds the runaway growth too
    assert client.calls == 3


class AlwaysSayClient:
    """A model that never acts - every todo it touches stalls out fast."""

    model = "fake"

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        yield "SAY still thinking about it\n"


def test_work_stops_at_hard_todo_cap_even_if_queue_outpaces_it(tmp_path):
    session = Session(tmp_path)
    # Bypass TodoStore's own cap directly, to prove work()'s circuit breaker
    # is an independent safety net, not just a rename of the same limit.
    for i in range(Agent.MAX_TODOS_PER_WORK_CALL + 5):
        session.todos.items.append(TodoItem(id=i + 1, text=f"todo {i}"))
    session.todos._next_id = len(session.todos.items) + 1

    console = Console(file=io.StringIO())
    agent = Agent(
        AlwaysSayClient(), tmp_path, session, console, auto_apply=True, self_review_enabled=False, max_steps=10
    )

    agent.work()

    touched = [t for t in session.todos.items if t.status != STATUS_PENDING]
    still_pending = [t for t in session.todos.items if t.status == STATUS_PENDING]
    assert len(touched) == Agent.MAX_TODOS_PER_WORK_CALL
    assert len(still_pending) == 5
