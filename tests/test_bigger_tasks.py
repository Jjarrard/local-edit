"""Multi-step and multi-file work: what broke when a 4B model ran /work on
bigger tasks (bench/big.py), each failure now handled by code."""

import io

from rich.console import Console

from local_edit import focus
from local_edit.agent import Agent
from local_edit.protocol import parse_action
from local_edit.state import Session
from local_edit.validate import validate


class ByPrompt:
    """Answers by what the prompt asks, so a test doesn't depend on how
    many questions the harness happens to ask first."""

    model = "fake"
    max_tokens = 512
    last_finish_reason = "stop"

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        yield self.answer(prompt)


def _agent(tmp_path, answer):
    client = ByPrompt(answer)
    agent = Agent(client, tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)
    return agent, client


def test_return_outside_a_function_is_rejected():
    # ast.parse accepts this; it only fails at compile time
    assert not validate("calc.py", "def f():\n    pass\nreturn 1\n").ok


def test_create_parses_with_and_without_content():
    a = parse_action("CREATE app.js add, tick off and delete tasks")
    assert (a.kind, a.path, a.text) == ("CREATE", "app.js", "add, tick off and delete tasks")
    b = parse_action("CREATE style.css\n```css\nbody { margin: 0; }\n```")
    assert (b.kind, b.path, b.start, b.payload) == ("INSERT", "style.css", 0, "body { margin: 0; }")


def test_body_code_is_told_apart_from_top_level_code():
    assert focus.is_body_code("if not numbers:\n    return 0")
    assert focus.is_body_code("raise ValueError('empty')")
    assert not focus.is_body_code("def median(numbers):\n    pass")
    assert not focus.is_body_code("import statistics")
    assert not focus.is_body_code("LIMIT = 3")


CALC = '''def average(numbers):
    return sum(numbers) / len(numbers)


def median(numbers):
    ordered = sorted(numbers)
    return ordered[len(ordered) // 2]
'''


def test_body_code_after_a_function_becomes_a_rewrite_of_it(tmp_path):
    # Real run: "handle an empty list in median" came back as a bare
    # `if not numbers:` guard, placed after the function at the top level.
    (tmp_path / "calc.py").write_text(CALC)

    def answer(prompt):
        if "Rewrite ONLY" in prompt:
            return ("```python\ndef median(numbers):\n    if not numbers:\n        return 0\n"
                    "    ordered = sorted(numbers)\n    return ordered[len(ordered) // 2]\n```")
        if "Write ONLY the new code" in prompt:
            return "```python\nif not numbers:\n    return 0\n```"
        if "NONE of the above" in prompt:
            return str(prompt.count("] lines ") + 1)  # the "none of the above" number
        return "1"

    agent, client = _agent(tmp_path, answer)
    result, _ = agent._handle_action(parse_action("FIX calc.py add a check for an empty list to median"), None)

    after = (tmp_path / "calc.py").read_text()
    assert result.startswith("Applied ("), result
    assert validate("calc.py", after).ok
    ns = {}
    exec(after, ns)
    assert ns["median"]([]) == 0 and ns["median"]([3, 1, 2]) == 2
    assert any("Rewrite ONLY" in p for p in client.prompts)


PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>Todos</title>
</head>
<body>
    <form id="add-form">
        <input id="task-input" type="text">
        <button type="submit">Add</button>
    </form>
    <ul id="task-list"></ul>
</body>
</html>
"""


def test_create_shows_the_page_ids_and_links_the_new_script(tmp_path):
    (tmp_path / "index.html").write_text(PAGE)
    agent, client = _agent(tmp_path, lambda p: "```js\ndocument.getElementById('add-form');\n```")
    agent.progress.start("build a todo app with index.html and app.js")

    result, _ = agent._handle_action(parse_action("CREATE app.js add and delete tasks"), None)

    assert result.startswith("Created app.js"), result
    assert 'id="task-list"' in client.prompts[0], "the page's ids are in the prompt"
    assert "<!DOCTYPE" not in client.prompts[0], "but not the whole page"
    page = (tmp_path / "index.html").read_text()
    assert '<script src="app.js"></script>\n</body>' in page
    assert "Linked app.js from index.html" in result


def test_a_new_page_picks_up_the_stylesheet_beside_it(tmp_path):
    (tmp_path / "style.css").write_text("body { margin: 0; }\n")
    agent, client = _agent(tmp_path, lambda p: "```html\n" + PAGE + "```")
    agent.progress.start("build a todo app with index.html and style.css")

    result, _ = agent._handle_action(parse_action("CREATE index.html the page"), None)

    assert "style.css" in client.prompts[0]
    assert '<link rel="stylesheet" href="style.css">' in (tmp_path / "index.html").read_text()
    assert result.startswith("Created index.html"), result


def test_create_on_an_existing_file_points_at_fix(tmp_path):
    (tmp_path / "index.html").write_text(PAGE)
    agent, _ = _agent(tmp_path, lambda p: "")
    result, _ = agent._handle_action(parse_action("CREATE index.html a page"), None)
    assert result.startswith("ERROR") and "FIX index.html" in result


def test_planner_padding_and_invented_files_are_dropped(tmp_path):
    for name in ("index.html", "about.html", "style.css"):
        (tmp_path / name).write_text("x\n")
    agent, _ = _agent(tmp_path, lambda p: "")

    kept = agent._usable_todos([
        "Change header background to green in styles.css",
        "Test header responsiveness on mobile",
        "Ensure the green doesn't clash",
        "add about link to nav in /src/index.html",
        "add about link to nav in /src/blog.html",
        "create app.js to handle clicks",
    ], goal="make the header green and add app.js")

    assert kept == [
        "Change header background to green in style.css",
        "add about link to nav in index.html",
        "create app.js to handle clicks",
    ]


def test_planner_keeps_testing_items_when_the_goal_asks_for_tests(tmp_path):
    agent, _ = _agent(tmp_path, lambda p: "")
    kept = agent._usable_todos(["test median in test_calc.py"], goal="add median and a test in test_calc.py")
    assert kept == ["test median in test_calc.py"]


def test_a_name_the_edit_uses_is_added_to_the_existing_import(tmp_path):
    # Real run: test_median() was added, `from calc import average, total`
    # was left alone, and the test died on a NameError.
    (tmp_path / "calc.py").write_text(CALC)
    (tmp_path / "test_calc.py").write_text("from calc import average\n\n\ndef test_average():\n"
                                           "    assert average([2, 4]) == 3\n")
    agent, _ = _agent(tmp_path, lambda p: "")
    agent.progress.start("add a test for median")
    agent.progress.record("READ test_calc.py", "1: from calc import average")

    result, _ = agent._handle_action(parse_action(
        "INSERT test_calc.py 5\n```python\n\n\ndef test_median():\n    assert median([3, 1, 2]) == 2\n```"), None)

    assert "Imported median" in result, result
    assert (tmp_path / "test_calc.py").read_text().startswith("from calc import average, median\n")


def test_say_ending_in_todo_done_marks_the_todo_done():
    a = parse_action("SAY The script tag is already in place. TODO DONE 4")
    assert (a.kind, a.todo_id) == ("TODO_DONE", 4)
    assert parse_action("SAY I will do TODO DONE 4 later, first the header").kind == "SAY"


DISCOUNT = '''def apply_discount(price, percent):
    """Price after taking `percent` off."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be 0-100")
    discount = price * percent / 100
    return round(discount, 2)
'''


def test_a_rewrite_that_invents_a_variable_is_rejected_with_a_hint(tmp_path):
    # Real run: "return discount_amount" - a name the function never sets.
    (tmp_path / "inventory.py").write_text(DISCOUNT)
    agent, _ = _agent(tmp_path, lambda p: "")
    agent.progress.start("fix apply_discount")
    agent.progress.record("READ inventory.py", "")

    result, _ = agent._handle_action(parse_action(
        "EDIT inventory.py 6-6\n```python\n    return round(price - discount_amount, 2)\n```"), None)

    assert result.startswith("ERROR") and "discount_amount" in result and "did you mean discount" in result
    assert (tmp_path / "inventory.py").read_text() == DISCOUNT


def test_existing_undefined_names_dont_block_an_edit(tmp_path):
    messy = "def f():\n    return helper_from_nowhere()\n\n\ndef g():\n    return 1\n"
    (tmp_path / "m.py").write_text(messy)
    agent, _ = _agent(tmp_path, lambda p: "")
    agent.progress.start("make g return 2")
    agent.progress.record("READ m.py", "")
    result, _ = agent._handle_action(parse_action("EDIT m.py 6-6\n```python\n    return 2\n```"), None)
    assert result.startswith("Applied"), result


def test_a_line_chosen_inside_a_function_rewrites_the_whole_function(tmp_path):
    (tmp_path / "inventory.py").write_text(DISCOUNT)
    blocks = focus.split_blocks(DISCOUNT)
    line = next(b for b in blocks if b.head.startswith("return round"))
    widened = focus.enclosing_function(blocks, line)
    assert widened is not None and widened.head.startswith("def apply_discount")
    assert "discount = price" in widened.text, "the rewrite sees what the variable is called"


def test_looks_changes_go_to_the_linked_stylesheet():
    page = '<head><link rel="stylesheet" href="style.css"></head><body><header>Hi</header></body>'
    assert focus.stylesheet_for(page, "make the header background green") == "style.css"
    assert focus.stylesheet_for(page, "add an About link to the nav") is None
    inline = page.replace("</head>", "<style>header { color: red; }</style></head>")
    assert focus.stylesheet_for(inline, "make the header green") is None


def test_fix_on_a_page_styled_from_a_stylesheet_edits_the_stylesheet(tmp_path):
    (tmp_path / "index.html").write_text(
        '<!DOCTYPE html>\n<html>\n<head>\n    <link rel="stylesheet" href="style.css">\n</head>\n'
        "<body>\n    <header>\n        <h1>Hi</h1>\n    </header>\n</body>\n</html>\n")
    (tmp_path / "style.css").write_text("header {\n    background: #336699;\n}\n\nfooter {\n    padding: 4px;\n}\n")

    def answer(prompt):
        if "Rewrite ONLY" in prompt:
            return "```css\nheader {\n    background: green;\n}\n```"
        return "1"

    agent, _ = _agent(tmp_path, answer)
    page_before = (tmp_path / "index.html").read_text()
    result, _ = agent._handle_action(parse_action("FIX index.html make the header background green"), None)

    assert result.startswith("Applied"), result
    assert "background: green" in (tmp_path / "style.css").read_text()
    assert (tmp_path / "index.html").read_text() == page_before
