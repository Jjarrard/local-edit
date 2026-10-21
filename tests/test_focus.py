"""FIX: code finds the block, the model only picks from a list and rewrites it."""

import io
import re

from rich.console import Console

from local_edit import focus
from local_edit.agent import Agent
from local_edit.protocol import parse_action
from local_edit.state import Session

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Stalls</title>
    <style>
        body {
            margin: 0;
        }
        header {
            background-color: #3498db;
            color: white;
        }
        .card {
            padding: 1rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>City Stalls</h1>
    </header>
    <main>
        <div class="card">Fruit</div>
    </main>
</body>
</html>
"""


def _heads(blocks):
    return [(b.start, b.end, b.head) for b in blocks]


def test_blocks_follow_indentation_and_closing_lines():
    blocks = focus.split_blocks(PAGE)
    assert (9, 12, "header {") in _heads(blocks)
    assert (19, 21, "<header>") in _heads(blocks)
    assert (22, 24, "<main>") in _heads(blocks)


def test_python_functions_are_blocks():
    src = "import os\n\ndef a():\n    return 1\n\ndef b(x):\n    if x:\n        return 2\n    return 3\n"
    assert [(b.start, b.end) for b in focus.split_blocks(src) if b.head.startswith("def")] == [(3, 4), (6, 9)]


def test_the_named_rule_ranks_among_the_candidates():
    heads = [b.head for b in focus.candidates(focus.split_blocks(PAGE), "make the header background green")]
    assert "header {" in heads and "<header>" in heads


def test_insert_spots_never_offer_after_a_document_wrapper():
    heads = [b.head for b in focus.places(focus.split_blocks(PAGE))]
    assert not any(re.match(r"<(body|style|head|html)\b|</", h) for h in heads)
    assert "<main>" in heads


def test_parse_choice():
    assert focus.parse_choice("2", 3) == 2
    assert focus.parse_choice("[3] header", 3) == 3
    assert focus.parse_choice("NEW", 3) == "NEW"
    assert focus.parse_choice("9", 3) is None
    assert focus.parse_choice("<think>maybe 1</think>\nNEW", 3) == "NEW"


def test_whole_file_rewrites_are_spotted():
    header = focus.split_blocks(PAGE)[0]
    assert focus.looks_like_whole_file("<!DOCTYPE html>\n<html>", None)
    assert not focus.looks_like_whole_file("header {\n  color: red;\n}", None)
    assert not focus.looks_like_whole_file("<!DOCTYPE html>", header)


def test_match_indent_keeps_relative_indentation():
    assert focus.match_indent("header {\n    color: red;\n}", 8) == "        header {\n            color: red;\n        }"


class Scripted:
    model = "fake"
    max_tokens = 512
    last_finish_reason = "stop"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def stream_chat(self, messages, stop=None, max_tokens=None, model=None):
        self.prompts.append(messages[-1]["content"])
        yield self.replies.pop(0)


def _run_fix(tmp_path, command, *replies):
    (tmp_path / "index.html").write_text(PAGE)
    client = Scripted(*replies)
    agent = Agent(client, tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)
    result, _ = agent._handle_action(parse_action(command), None)
    return result, (tmp_path / "index.html").read_text(), client


def test_fix_rewrites_only_the_chosen_block(tmp_path):
    choices = focus.candidates(focus.split_blocks(PAGE), "make the header background green")
    pick = 1 + [b.head for b in choices].index("header {")
    result, after, client = _run_fix(
        tmp_path, "FIX index.html make the header background green",
        str(pick), "```css\nheader {\n    background-color: green;\n    color: white;\n}\n```",
    )
    assert result.startswith("Applied (")
    assert after == PAGE.replace("#3498db", "green")
    assert "<!DOCTYPE" not in client.prompts[1], "the rewrite prompt shows the block, not the file"


def test_fix_new_code_goes_after_the_chosen_spot(tmp_path):
    # Nothing in the page is a footer, so code skips "which block?" and the
    # model is only asked where the new part goes.
    blocks = focus.split_blocks(PAGE)
    after_main = 1 + [b.head for b in focus.places(blocks)].index("<main>")
    result, after, _ = _run_fix(
        tmp_path, "FIX index.html add a footer with contact info",
        str(after_main), "```html\n<footer>\n    <p>info@stalls.example</p>\n</footer>\n```",
    )
    assert result.startswith("Applied (")
    lines = after.splitlines()
    assert lines[24:27] == ["    <footer>", "        <p>info@stalls.example</p>", "    </footer>"]
    assert lines[27] == "</body>"


def test_fix_retries_once_when_given_the_whole_file(tmp_path):
    result, after, _ = _run_fix(
        tmp_path, "FIX index.html make the header background green",
        "garbled", f"```html\n{PAGE}```", "```css\nheader {\n    background-color: green;\n}\n```",
    )
    assert result.startswith("Applied (")
    assert after.count("<html") == 1


def test_fix_on_a_missing_file_points_at_insert(tmp_path):
    agent = Agent(Scripted(), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)
    result, _ = agent._handle_action(parse_action("FIX nope.html add things"), None)
    assert "INSERT nope.html 0" in result


def test_none_of_the_above_is_offered_only_when_adding():
    blocks = focus.split_blocks(PAGE)
    adding = focus.candidates(blocks, "add a border to the header")
    assert f"[{len(adding) + 1}] NONE of the above" in focus.choose_prompt(
        "add a border to the header", "index.html", adding)
    assert focus.parse_choice(str(len(adding) + 1), len(adding)) == "NEW"

    changing = focus.candidates(blocks, "make the header green")
    assert "NONE of the above" not in focus.choose_prompt("make the header green", "index.html", changing)


def test_adding_something_the_file_lacks_skips_the_question():
    blocks = focus.split_blocks(PAGE)
    assert focus.adds_something_absent(blocks, "add a footer with contact info")
    assert not focus.adds_something_absent(blocks, "add another card")


def test_unrequested_new_file_is_questioned_once(tmp_path):
    # Seen live: "add a footer with contact info" became INSERT footer.html 0
    # in a project whose only page is index.html.
    (tmp_path / "index.html").write_text(PAGE)
    agent = Agent(Scripted(), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)
    agent.progress.start("add a footer with contact info")
    agent.progress.set_project_files([("index.html", 27)])
    insert = parse_action("INSERT footer.html 0\n```html\n<footer>hi</footer>\n```")

    result, _ = agent._handle_action(insert, None)
    assert "FIX index.html" in result and not (tmp_path / "footer.html").exists()

    agent._handle_action(insert, None)
    assert (tmp_path / "footer.html").exists(), "sending it again means it's wanted"


def test_named_new_file_goes_straight_through(tmp_path):
    (tmp_path / "index.html").write_text(PAGE)
    agent = Agent(Scripted(), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)
    agent.progress.start("create about.html with a short bio")
    agent.progress.set_project_files([("index.html", 27)])
    agent._handle_action(parse_action("INSERT about.html 0\n```html\n<p>bio</p>\n```"), None)
    assert (tmp_path / "about.html").exists()


LIST_PAGE = """<html>
<body>
    <div class="stalls">
        <div class="stall">
            <h2>Fruit</h2>
        </div>
        <div class="stall">
            <h2>Bread</h2>
        </div>
    </div>
    <div class="contact">
        <p>Call us</p>
    </div>
</body>
</html>
"""


def test_another_of_a_repeated_thing_goes_after_the_last_one_without_asking():
    blocks = focus.split_blocks(LIST_PAGE)
    spot = focus.obvious_place(blocks, "add a fifth stall called Fresh Flowers")
    assert (spot.start, spot.end) == (7, 9)


def test_words_later_in_the_request_dont_trigger_the_rule():
    # "links to ... Stalls" is not "add a stall": seen live, a nav bar landed
    # after the last stall card.
    blocks = focus.split_blocks(LIST_PAGE)
    assert focus.obvious_place(blocks, "add a navigation bar with links to Home, Stalls and Contact") is None


def test_css_class_rules_know_where_they_are_used():
    page = '<style>\n    .book-btn {\n        color: blue;\n    }\n</style>\n<button class="book-btn">Book Now</button>\n'
    rule = next(b for b in focus.split_blocks(page) if b.head == ".book-btn {")
    assert rule.used_by == '<button class="book-btn">Book Now</button>'
    assert focus.candidates(focus.split_blocks(page), "make the Book Now buttons red")[0] is not None


def test_joined_requests_split_but_pronouns_stay_attached():
    assert focus.split_request("make the header green and add a footer") == ["make the header green", "add a footer"]
    assert focus.split_request("a red and white header") == ["a red and white header"]
    assert focus.split_request("add a nav bar and make it sticky") == ["add a nav bar and make it sticky"]


def test_stray_closing_tags_are_trimmed_from_new_markup():
    assert focus.trim_stray_closers("</div>\n</div>\n<footer>hi</footer>") == "<footer>hi</footer>"


def test_empty_rewrite_removes_the_block(tmp_path):
    choices = focus.candidates(focus.split_blocks(PAGE), "remove the card styles")
    pick = 1 + [b.head for b in choices].index(".card {")
    result, after, _ = _run_fix(tmp_path, "FIX index.html remove the card styles", str(pick), "```css\n```")
    assert result.startswith("Applied (removed")
    assert ".card {" not in after


def test_line_edits_on_an_unread_file_are_pointed_at_fix(tmp_path):
    (tmp_path / "index.html").write_text(PAGE)
    agent = Agent(Scripted(), tmp_path, Session(tmp_path), Console(file=io.StringIO()),
                  auto_apply=True, self_review_enabled=False)
    agent.progress.start("add a fifth stall")
    edit = parse_action("INSERT index.html 10\n```html\n<li>x</li>\n```")

    result, _ = agent._handle_action(edit, None)
    assert "FIX index.html" in result and (tmp_path / "index.html").read_text() == PAGE

    agent.progress.record("READ index.html", "--- index.html ---")
    other = parse_action("INSERT index.html 21\n```html\n        <p>x</p>\n```")
    result, _ = agent._handle_action(other, None)
    assert result.startswith("Applied"), "after a READ, line numbers are fine"
