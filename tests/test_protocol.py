import pytest

from local_edit.protocol import Action, ParseError, parse_action, parse_todo_adds


def test_say_is_the_default_when_no_command_found():
    a = parse_action("hey there, how can I help you today?")
    assert a.kind == "SAY"
    assert "hey there" in a.text


def test_empty_response_is_a_say():
    a = parse_action("   ")
    assert a.kind == "SAY"


def test_read_with_range():
    a = parse_action("READ src/app.py 10 20")
    assert a.kind == "READ"
    assert a.path == "src/app.py"
    assert a.start == 10 and a.end == 20


def test_read_bare_path():
    a = parse_action("READ src/app.py")
    assert a.kind == "READ"
    assert a.start is None and a.end is None


def test_read_with_hyphen_range():
    # models used to EDIT/DELETE's "start-end" range syntax often reuse it for READ too
    a = parse_action("READ src/app.py 2-2")
    assert a.kind == "READ"
    assert a.start == 2 and a.end == 2


def test_read_single_line_number():
    a = parse_action("READ src/app.py 7")
    assert a.kind == "READ"
    assert a.start == 7 and a.end is None


def test_list_defaults_to_dot():
    a = parse_action("LIST")
    assert a.kind == "LIST"
    assert a.path == "."


def test_grep_with_in_clause():
    a = parse_action("GREP TODO in src")
    assert a.kind == "GREP"
    assert a.pattern == "TODO"
    assert a.path == "src"


def test_grep_without_path():
    a = parse_action("GREP some phrase")
    assert a.kind == "GREP"
    assert a.pattern == "some phrase"
    assert a.path == "."


def test_todo_add():
    a = parse_action("TODO ADD write the parser tests")
    assert a.kind == "TODO_ADD"
    assert a.text == "write the parser tests"


def test_todo_done():
    a = parse_action("TODO DONE 3")
    assert a.kind == "TODO_DONE"
    assert a.todo_id == 3


def test_todo_done_requires_numeric_id():
    with pytest.raises(ParseError):
        parse_action("TODO DONE abc")


def test_todo_list():
    a = parse_action("TODO LIST")
    assert a.kind == "TODO_LIST"


def test_edit_with_fenced_block():
    text = 'EDIT src/app.py 12-14\n```python\n    return x + 1\n```\n'
    a = parse_action(text)
    assert a.kind == "EDIT"
    assert a.path == "src/app.py"
    assert (a.start, a.end) == (12, 14)
    assert a.payload == "    return x + 1"
    assert not a.payload_truncated


def test_edit_single_line_range():
    a = parse_action("EDIT f.py 5\n```\nx = 1\n```\n")
    assert (a.start, a.end) == (5, 5)


def test_edit_missing_block_raises():
    with pytest.raises(ParseError):
        parse_action("EDIT src/app.py 12-14")


def test_edit_truncated_block_detected():
    text = "EDIT src/app.py 12-14\n```python\n    return x + 1\n"
    a = parse_action(text)
    assert a.payload_truncated is True
    assert a.payload == "    return x + 1"


def test_insert():
    text = "INSERT src/app.py 0\n```\nimport os\n```\n"
    a = parse_action(text)
    assert a.kind == "INSERT"
    assert a.path == "src/app.py"
    assert a.start == 0
    assert a.payload == "import os"


def test_delete():
    a = parse_action("DELETE src/app.py 3-3")
    assert a.kind == "DELETE"
    assert (a.start, a.end) == (3, 3)


def test_done():
    a = parse_action("DONE")
    assert a.kind == "DONE"


def test_run_command():
    a = parse_action("RUN pytest -q")
    assert a.kind == "RUN"
    assert a.text == "pytest -q"


def test_run_keeps_full_command_with_flags_and_quotes():
    a = parse_action("RUN python -c \"print('hi')\"")
    assert a.kind == "RUN"
    assert a.text == "python -c \"print('hi')\""


def test_run_in_fenced_block_is_accepted():
    # models habitually wrap shell commands in backticks
    a = parse_action("RUN\n```bash\nls -la\n```\n")
    assert a.kind == "RUN"
    assert a.text == "ls -la"


def test_run_without_command_raises():
    with pytest.raises(ParseError):
        parse_action("RUN")


def test_say_explicit_keyword():
    a = parse_action("SAY I fixed the bug for you.")
    assert a.kind == "SAY"
    assert a.text == "I fixed the bug for you."


def test_preamble_before_command_is_captured():
    a = parse_action("Sure, let me look.\nREAD src/app.py")
    assert a.kind == "READ"
    assert a.preamble == "Sure, let me look."


def test_case_insensitive_command():
    a = parse_action("edit src/app.py 1-1\n```\nx\n```")
    assert a.kind == "EDIT"


def test_parse_todo_adds_multiple_lines():
    text = (
        "TODO ADD write tests\n"
        "TODO ADD fix the bug\n"
        "some other line\n"
        "TODO ADD update docs\n"
    )
    items = parse_todo_adds(text)
    assert items == ["write tests", "fix the bug", "update docs"]


def test_bad_range_raises():
    with pytest.raises(ParseError):
        parse_action("EDIT f.py notarange\n```\nx\n```")
