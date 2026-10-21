"""Live benchmark: real requests against the real Jan model, judged by code.

usage: python bench/bench.py <out dir> [task ids...] [--runs N] [--no-review]

Needs Jan's Local API Server running on 127.0.0.1:1338.
"""

import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
OUT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE.parent))

from rich.console import Console  # noqa: E402

from local_edit.agent import Agent  # noqa: E402
from local_edit.client import JanClient  # noqa: E402
from local_edit.state import Session  # noqa: E402
from local_edit.validate import html_unclosed_tags, validate  # noqa: E402

PAGE = FIXTURES / "stalls.html"


def css_rule(text, selector):
    m = re.search(r"(?m)^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", text)
    return m.group(1) if m else ""


def colour_is(value, name):
    value = value.lower()
    if name in value:
        return True
    for h in re.findall(r"#([0-9a-f]{6}|[0-9a-f]{3})\b", value):
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        if name == "green" and g > r + 40 and g > b + 20:
            return True
        if name == "red" and r > g + 60 and r > b + 60:
            return True
        if name == "blue" and b > r + 40 and b > g + 20:
            return True
    return False


def body_of(text):
    m = re.search(r"<body[^>]*>(.*)</body>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


def check_header_green(t):
    return colour_is(css_rule(t, "header"), "green") or bool(
        re.search(r"<header[^>]*style=\"[^\"]*background[^\"]*green", t, re.IGNORECASE))


def check_footer(t):
    body = body_of(t)
    return "<footer" in body.lower() and ("@" in body or "contact" in body.lower())


def check_title(t):
    m = re.search(r"<title>(.*?)</title>", t, re.IGNORECASE | re.DOTALL)
    return bool(m and "market stalls" in m.group(1).lower())


def check_buttons_red(t):
    return colour_is(css_rule(t, ".book-btn"), "red")


def check_stall_added(t):
    body = body_of(t)
    grid, form, new = body.find('class="stalls"'), body.find('class="booking-form"'), body.find("Fresh Flowers")
    return body.count('class="stall"') >= 5 and grid < new < form


def check_form_removed(t):
    return 'class="booking-form"' not in body_of(t) and "<form" not in body_of(t)


def check_nav(t):
    body = body_of(t)
    nav = body.lower().find("<nav")
    return nav != -1 and nav < body.find('class="stalls"') and body.lower().count("<a ", nav) >= 3


def check_about(t, root):
    about = root / "about.html"
    return about.exists() and "<body" in about.read_text().lower() and validate("about.html", about.read_text()).ok


def _calc(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location("calc_under_test", root / "calc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_py_empty(t, root):
    try:
        m = _calc(root)
        return m.average([]) == 0 and m.average([2, 4]) == 3 and m.total([1, 2]) == 3
    except Exception:
        return False


def check_py_median(t, root):
    try:
        m = _calc(root)
        return m.median([3, 1, 2]) == 2 and m.average([2, 4]) == 3
    except Exception:
        return False


def check_h2_blue(t):
    inline = re.compile(r"<h2[^>]*style=\"[^\"]*color:\s*(blue|#[0-9a-f]{3,6})", re.IGNORECASE)
    m = inline.search(t)
    # an inline style on the card's <h2> makes the titles blue just as well
    if m and colour_is(m.group(1), "blue") and not inline.search(PAGES_ORIG[CURRENT[0]]):
        return True
    return (colour_is(css_rule(t, ".stall-card h2"), "blue")
            and not colour_is(css_rule(PAGES_ORIG[CURRENT[0]], ".stall-card h2"), "blue"))


def check_phone_in_footer(t):
    m = re.search(r"<footer.*?</footer>", body_of(t), re.DOTALL | re.IGNORECASE)
    return bool(m and re.search(r"\d{3}[\s.-]?\d{3,4}", m.group(0)))


PAGES = {
    "p1": FIXTURES / "p1.html",
    "p2": FIXTURES / "p2.html",
}
PAGES_ORIG = {k: v.read_text() for k, v in PAGES.items()}
CURRENT = [None]

CALC = "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n\ndef total(numbers):\n    return sum(numbers)\n"


TASKS = {
    "green": ("make the header background green", [check_header_green]),
    "footer": ("add a footer with contact info", [check_footer]),
    "both": ("make the header background green and add a footer with contact info",
             [check_header_green, check_footer]),
    "title": ("change the page title to Market Stalls", [check_title]),
    "red": ("make the Book Now buttons red", [check_buttons_red]),
    "stall": ("add a fifth stall called Fresh Flowers", [check_stall_added]),
    "remove": ("remove the booking form", [check_form_removed], {"max_lost": 40}),
    "nav": ("add a navigation bar with links to Home, Stalls and Contact", [check_nav]),
    "about": ("create an about.html page with a short bio", [check_about]),
    "pyempty": ("make average() in calc.py return 0 for an empty list instead of crashing",
                [check_py_empty], {"file": "calc.py"}),
    "p1green": ("make the header background green", [check_header_green], {"page": "p1"}),
    "p1title": ("change the page title to Market Stalls", [check_title], {"page": "p1"}),
    "p1footer": ("add a footer with contact info", [check_footer], {"page": "p1"}),
    "p1blue": ("make the stall card titles blue", [check_h2_blue], {"page": "p1"}),
    "p2green": ("make the header background green", [check_header_green], {"page": "p2"}),
    "p2title": ("change the page title to Market Stalls", [check_title], {"page": "p2"}),
    "p2phone": ("add a phone number to the footer", [check_phone_in_footer], {"page": "p2"}),
    "p2blue": ("make the stall card titles blue", [check_h2_blue], {"page": "p2"}),
    "scratch": ("I want a html website that shows bookable stalls in the city center",
                [lambda t, root: True], {"empty": True}),
    "pymedian": ("add a median(numbers) function to calc.py", [check_py_median], {"file": "calc.py"}),
}


def run(task_id, n, review):
    request, checks, *extra = TASKS[task_id]
    opts = extra[0] if extra else {}
    target = opts.get("file", "index.html")
    root = OUT / f"{task_id}-{n}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    CURRENT[0] = opts.get("page")
    if opts.get("empty"):
        pass
    elif target == "calc.py":
        (root / "calc.py").write_text(CALC)
    elif opts.get("page"):
        shutil.copy(PAGES[opts["page"]], root / "index.html")
    else:
        shutil.copy(PAGE, root / "index.html")
    if opts.get("empty"):
        t0 = time.time()
        agent = Agent(JanClient(model="Jan-v3.5-4B-Q4_K_XL", max_tokens=512), root, Session(root),
                      Console(file=(root / "run.log").open("w"), force_terminal=False, width=140),
                      auto_apply=True, self_review_enabled=review)
        agent.chat_turn(request)
        pages = [p for p in root.rglob("*.html")]
        good = [p for p in pages if validate(p.name, p.read_text()).ok and not html_unclosed_tags(p.read_text())
                and "stall" in p.read_text().lower() and len(p.read_text().splitlines()) > 40]
        passed = bool(good)
        print(f"{'PASS' if passed else 'FAIL'} {task_id}-{n}: pages={[str(p.relative_to(root)) for p in pages]} "
              f"complete={[len(p.read_text().splitlines()) for p in good]} {time.time() - t0:.0f}s", flush=True)
        return passed
    before = (root / target).read_text()
    log = (root / "run.log").open("w")
    client = JanClient(model="Jan-v3.5-4B-Q4_K_XL", max_tokens=512)
    calls = [0]
    real = client.stream_chat

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    client.stream_chat = counted
    agent = Agent(client, root, Session(root), Console(file=log, force_terminal=False, width=140),
                  auto_apply=True, self_review_enabled=review)
    t0 = time.time()
    agent.chat_turn(request)
    after = (root / target).read_text()
    ok_checks = [c.__name__ for c in checks
                 if (c(after, root) if c.__code__.co_argcount == 2 else c(after))]
    valid = validate(target, after).ok and (target != "index.html" or not html_unclosed_tags(after))
    lost = len(before.splitlines()) - len(after.splitlines())
    intact = lost < opts.get("max_lost", 8) and after.count("<html") <= 1
    passed = len(ok_checks) == len(checks) and valid and intact
    print(f"{'PASS' if passed else 'FAIL'} {task_id}-{n}: checks {len(ok_checks)}/{len(checks)} valid={valid} "
          f"intact={intact} (lines {len(before.splitlines())}->{len(after.splitlines())}) "
          f"calls={calls[0]} {time.time() - t0:.0f}s", flush=True)
    return passed


if __name__ == "__main__":
    args = sys.argv[2:]
    runs = 1
    if "--runs" in args:
        i = args.index("--runs")
        runs = int(args[i + 1])
        del args[i : i + 2]
    review = "--no-review" not in args
    args = [a for a in args if a != "--no-review"]
    ids = args or list(TASKS)
    total = sum(run(t, n, review) for t in ids for n in range(1, runs + 1))
    print(f"=== {total}/{len(ids) * runs} passed", flush=True)
