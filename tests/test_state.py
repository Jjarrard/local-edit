from local_edit.state import MAX_TODOS, Session


def test_add_message_alternates_roles(tmp_path):
    session = Session(tmp_path)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    session.add_message("user", "one")
    session.add_message("user", "two")  # e.g. a todo's closing tool-result, then the next todo's prompt
    session.add_message("assistant", "ok")

    roles = [m["role"] for m in session.history]
    assert roles == ["user", "assistant", "user", "assistant"], roles
    assert session.history[2]["content"] == "one\n\ntwo"


def test_add_message_never_leaves_consecutive_same_role(tmp_path):
    session = Session(tmp_path)
    for i in range(6):
        session.add_message("user" if i % 2 == 0 else "user", f"msg{i}")  # all same role, worst case
    roles = [m["role"] for m in session.history]
    assert roles == ["user"], "all same-role adds should collapse into a single turn"


def test_todostore_caps_at_max_todos(tmp_path):
    # A misbehaving model spamming TODO ADD must not be able to grow the
    # queue without bound - this is the circuit breaker for that.
    session = Session(tmp_path)
    for i in range(MAX_TODOS + 10):
        session.todos.add(f"todo {i}")
    assert len(session.todos.items) == MAX_TODOS
    assert session.todos.add("one more") is None
