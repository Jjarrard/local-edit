"""Live benchmark for bigger tasks: several files, big files, vague asks.

Each task runs through /work (plan -> todos -> FIX steps) against the real
Jan model, and is judged by code: files parse, tests pass, nothing lost.

usage: python bench/big.py <out dir> [task ids...] [--runs N]

Needs Jan's Local API Server running on 127.0.0.1:1338.
"""

import ast
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE.parent))

from rich.console import Console  # noqa: E402

from local_edit.agent import Agent  # noqa: E402
from local_edit.client import JanClient  # noqa: E402
from local_edit.state import Session  # noqa: E402
from local_edit.validate import html_unclosed_tags, validate  # noqa: E402

MODEL = "Jan-v3.5-4B-Q4_K_XL"


# -- fixtures --------------------------------------------------------------------

def _page(title, nav, main):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>City Stalls</h1>
        <nav>
{nav}
        </nav>
    </header>
    <main>
{main}
    </main>
    <footer>
        <p>City Stalls Ltd</p>
    </footer>
</body>
</html>
"""


NAV = """            <a href="index.html">Home</a>
            <a href="contact.html">Contact</a>"""

STYLE = """body {
    font-family: Arial, sans-serif;
    margin: 0;
    color: #222;
}

header {
    background: #336699;
    color: white;
    padding: 20px;
}

nav a {
    color: white;
    margin-right: 15px;
}

main {
    padding: 20px;
}

.btn {
    background: #336699;
    color: white;
    padding: 8px 16px;
    border: none;
}

footer {
    background: #eee;
    padding: 10px 20px;
}
"""


def site(root):
    (root / "index.html").write_text(_page("City Stalls", NAV, """        <h2>Book a stall</h2>
        <p>Stalls in the city centre, every weekend.</p>
        <button class="btn">Book Now</button>"""))
    (root / "about.html").write_text(_page("About - City Stalls", NAV, """        <h2>About us</h2>
        <p>We have run the city market since 1998.</p>"""))
    (root / "contact.html").write_text(_page("Contact - City Stalls", NAV, """        <h2>Contact</h2>
        <p>Email hello@citystalls.test</p>"""))
    (root / "style.css").write_text(STYLE)


_THINGS = ["apple", "bread", "candle", "drill", "envelope", "fabric", "glove", "hammer", "ink", "jar",
           "kettle", "lamp", "magnet", "nail", "oil", "paper", "rope", "soap", "tape", "umbrella"]


def big_py(root):
    """~900 lines of plausible inventory code with one bug in the middle."""
    parts = ['"""Inventory helpers for the stall shop."""\n\nimport math\n']
    for i, thing in enumerate(_THINGS):
        parts.append(f'''

def total_{thing}_stock(rows):
    """Sum the {thing} count over every warehouse row."""
    total = 0
    for row in rows:
        if row.get("item") != "{thing}":
            continue
        count = row.get("count", 0)
        if count < 0:
            raise ValueError("negative {thing} count")
        total += count
    return total


def {thing}_reorder_needed(stock, minimum={10 + i}):
    """True when {thing} stock has dropped below the reorder level."""
    if stock is None:
        return True
    return stock < minimum


def format_{thing}_line(name, count, price):
    """One line of the {thing} report."""
    label = name.strip().title()
    value = round(count * price, 2)
    return f"{{label:<20}} {{count:>5}} {{value:>10.2f}}"
''')
        if i == 11:
            parts.append('''

def apply_discount(price, percent):
    """Price after taking `percent` off, e.g. apply_discount(200, 25) -> 150."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be 0-100")
    discount = price * percent / 100
    return round(discount, 2)
''')
    (root / "inventory.py").write_text("".join(parts))


CALC = '''def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def total(numbers):
    return sum(numbers)
'''

TEST_CALC = '''from calc import average, total


def test_average():
    assert average([2, 4]) == 3


def test_average_empty():
    assert average([]) == 0


def test_total():
    assert total([1, 2, 3]) == 6
'''


def calc_project(root):
    (root / "calc.py").write_text(CALC)
    (root / "test_calc.py").write_text(TEST_CALC)


def stalls(root):
    shutil.copy(HERE / "fixtures" / "stalls.html", root / "index.html")


_JS_HEAD = """// Stall booking cart for the city market.

const PRICES = { small: 20, medium: 35, large: 50 };

function lineTotal(item) {
  const price = PRICES[item.size];
  if (price === undefined) {
    throw new Error("unknown stall size: " + item.size);
  }
  return price;
}

function cartTotal(cart) {
  let total = 0;
  for (const item of cart) {
    total += lineTotal(item);
  }
  return total;
}

function applyDiscount(total, code) {
  if (code === "SAVE10") {
    return total * 0.9;
  }
  return total;
}
"""

_JS_DOM = """
if (typeof document !== "undefined") {
  const cart = [];
  document.getElementById("add-btn").addEventListener("click", () => {
    const size = document.getElementById("size").value;
    cart.push({ id: Date.now(), size: size, quantity: 1 });
    document.getElementById("total").textContent = cartTotal(cart).toFixed(2);
  });
}
"""


def cart_js(root):
    """~320 lines of booking-cart script; lineTotal ignores quantity and
    applyDiscount only takes upper-case codes."""
    parts = [_JS_HEAD]
    for thing in _THINGS[:16]:
        cap = thing.capitalize()
        parts.append(f"""
function format{cap}Label(stall) {{
  const name = (stall.name || "").trim();
  if (!name) {{
    return "{cap} stall";
  }}
  return name + " ({thing})";
}}

function count{cap}Bookings(bookings) {{
  let count = 0;
  for (const b of bookings) {{
    if (b.kind === "{thing}") {{
      count += 1;
    }}
  }}
  return count;
}}
""")
    parts.append(_JS_DOM)
    (root / "app.js").write_text("".join(parts))
    (root / "index.html").write_text(_page("Book a stall", NAV, """        <select id="size"><option>small</option><option>large</option></select>
        <button id="add-btn" class="btn">Add</button>
        <p>Total: <span id="total">0.00</span></p>""").replace(
        "</body>", '    <script src="app.js"></script>\n</body>'))
    (root / "style.css").write_text(STYLE)


_JS_RUNNER = """
const fs = require("fs"), vm = require("vm");
const ctx = {};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
const out = {};
for (const [k, v] of Object.entries(ctx)) if (typeof v === "function") out["fn:" + k] = v.toString();
try { out.result = vm.runInContext(process.argv[2], ctx); } catch (e) { out.error = String(e); }
console.log(JSON.stringify(out));
"""


def _js(root, expr):
    """Run app.js in a sandbox, evaluate `expr` in it; also returns the
    source of every top-level function, to see what else was touched."""
    import json

    r = subprocess.run(["node", "-e", _JS_RUNNER, str(root / "app.js"), expr], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "node failed"}
    return json.loads(r.stdout)


def _js_others_changed(root, before_root_text, allowed):
    tmp = root / ".before.js"
    tmp.write_text(before_root_text)
    try:
        import json

        r = subprocess.run(["node", "-e", _JS_RUNNER, str(tmp), "0"], capture_output=True, text=True)
        before_fns = {k: v for k, v in json.loads(r.stdout).items() if k.startswith("fn:")}
    finally:
        tmp.unlink()
    after = _js(root, "0")
    return [k[3:] for k, v in before_fns.items() if k[3:] not in allowed and after.get(k) != v]


def _js_task(expr, expect, allowed):
    def check(root, before):
        out = _js(root, f"JSON.stringify({expr})")
        if "error" in out:
            return False, f"error: {out['error']}"
        got = out["result"]
        changed = _js_others_changed(root, before["app.js"], allowed)
        valid = validate("app.js", (root / "app.js").read_text()).ok
        return got == expect and not changed and valid, f"got {got} want {expect} others changed={changed[:4]} valid={valid}"
    return check


check_js_quantity = _js_task(
    '[cartTotal([{size:"small",quantity:3}]), cartTotal([{size:"large",quantity:2},{size:"small",quantity:1}])]',
    "[60,120]", {"lineTotal", "cartTotal"})
check_js_case = _js_task('[applyDiscount(100,"save10"), applyDiscount(100,"SAVE10"), applyDiscount(100,"nope")]',
                         "[90,90,100]", {"applyDiscount"})
check_js_remove = _js_task('removeItem([{id:1,size:"small"},{id:2,size:"large"}], 1)',
                           '[{"id":2,"size":"large"}]', {"removeItem"})


# -- checks ------------------------------------------------------------------------

def _html_ok(text):
    return validate("x.html", text).ok and not html_unclosed_tags(text) and text.lower().count("<html") == 1


def _nav(text):
    m = re.search(r"<nav[^>]*>(.*?)</nav>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


def check_about_links(root, before):
    pages = ["index.html", "about.html", "contact.html"]
    bad = [p for p in pages if 'href="about.html"' not in _nav((root / p).read_text())
           or not _html_ok((root / p).read_text())
           or "Contact</a>" not in (root / p).read_text()]
    return not bad, f"bad pages: {bad}" if bad else "all 3 pages"


def _green(value):
    value = value.lower()
    if "green" in value:
        return True
    for h in re.findall(r"#([0-9a-f]{6}|[0-9a-f]{3})\b", value):
        h = "".join(c * 2 for c in h) if len(h) == 3 else h
        r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
        if g > r + 40 and g > b + 20:
            return True
    return False


def check_green(root, before):
    css = (root / "style.css").read_text()
    m = re.search(r"(?m)^header\s*\{([^}]*)\}", css)
    ok = bool(m and _green(m.group(1))) and css.count("{") >= STYLE.count("{")
    untouched = all((root / p).read_text() == before[p] for p in ("index.html", "about.html", "contact.html"))
    return ok, f"header green={bool(m and _green(m.group(1)))} html untouched={untouched}"


def _functions(text):
    tree = ast.parse(text)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def check_discount(root, before):
    text = (root / "inventory.py").read_text()
    try:
        after_fns = _functions(text)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    before_fns = _functions(before["inventory.py"])
    changed = [n for n in before_fns if n != "apply_discount" and after_fns.get(n) != before_fns[n]]
    ns = {}
    exec(compile(text, "inventory.py", "exec"), ns)
    try:
        right = ns["apply_discount"](200, 25) == 150 and ns["apply_discount"](80, 50) == 40
    except Exception as e:
        return False, f"apply_discount raised {e!r}"
    return right and not changed, f"discount right={right} other fns changed={changed[:5]}"


def check_todo_app(root, before):
    need = ["index.html", "style.css", "app.js"]
    found = {p.name: p for p in root.rglob("*") if p.name in need and ".local-edit" not in p.parts}
    missing = [n for n in need if n not in found]
    if missing:
        return False, f"missing {missing} (have {[str(p.relative_to(root)) for p in root.rglob('*.*') if '.local-edit' not in p.parts]})"
    html, js = found["index.html"].read_text(), found["app.js"].read_text()
    js_ok = subprocess.run(["node", "--check", str(found["app.js"])], capture_output=True).returncode == 0
    linked = "style.css" in html and "app.js" in html
    behaviour = "addEventListener" in js or "onclick" in (js + html).lower()
    dom = "createElement" in js or "innerHTML" in js
    deletes = bool(re.search(r"remove|splice|filter|delete", js, re.IGNORECASE))
    ok = _html_ok(html) and js_ok and linked and behaviour and dom and deletes
    return ok, f"html ok={_html_ok(html)} js parses={js_ok} linked={linked} events={behaviour} dom={dom} delete={deletes}"


def check_median(root, before):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root,
                       capture_output=True, text=True)
    tests = (root / "test_calc.py").read_text()
    ns = {}
    try:
        exec(compile((root / "calc.py").read_text(), "calc.py", "exec"), ns)
        med = ns["median"]([3, 1, 2]) == 2 and ns["median"]([4, 1, 3, 2]) == 2.5
    except Exception as e:
        med = f"error {e!r}"
    kept = all(t in tests for t in ("test_average", "test_average_empty", "test_total"))
    ok = r.returncode == 0 and med is True and "median" in tests and kept
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    return ok, f"median right={med} test added={'median' in tests} old tests kept={kept} pytest: {last}"


def check_modern(root, before):
    after = (root / "index.html").read_text()
    lost = len(before["index.html"].splitlines()) - len(after.splitlines())
    stalls_kept = after.count('class="stall"') >= before["index.html"].count('class="stall"')
    ok = after != before["index.html"] and _html_ok(after) and lost < 8 and stalls_kept
    return ok, f"changed={after != before['index.html']} valid={_html_ok(after)} lines lost={lost} stalls kept={stalls_kept}"


TASKS = {
    "aboutlinks": (site, "add an About link (about.html) to the nav on every page", check_about_links),
    "green": (site, "make the site's header green", check_green),
    "discount": (big_py, "apply_discount in inventory.py returns the discount amount instead of the "
                         "discounted price - fix it", check_discount),
    "todoapp": (None, "build a todo list web app with index.html, style.css and app.js: add tasks, "
                      "tick them done, delete them", check_todo_app),
    "median": (calc_project, "add a median(numbers) function to calc.py and a test for it in test_calc.py",
               check_median),
    "modern": (stalls, "make the page look more modern", check_modern),
    "jsquantity": (cart_js, "the cart total ignores how many of each stall are booked (the quantity) - fix it",
                   check_js_quantity),
    "jscase": (cart_js, "discount codes should work in lowercase too, like save10", check_js_case),
    "jsremove": (cart_js, "add a removeItem(cart, id) function to app.js that returns the cart without "
                          "the item with that id", check_js_remove),
}


def run(task_id, n):
    setup, goal, check = TASKS[task_id]
    root = OUT / f"{task_id}-{n}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    if setup:
        setup(root)
    before = {p.name: p.read_text() for p in root.iterdir() if p.is_file()}
    client = JanClient(model=MODEL, max_tokens=512)
    calls, prompt_chars = [0], [0]
    real = client.stream_chat

    def counted(messages, *a, **k):
        calls[0] += 1
        prompt_chars[0] = max(prompt_chars[0], sum(len(m["content"]) for m in messages))
        return real(messages, *a, **k)

    client.stream_chat = counted
    log = (root.parent / f"{task_id}-{n}.log").open("w")
    agent = Agent(client, root, Session(root), Console(file=log, force_terminal=False, width=140),
                  auto_apply=True)
    t0 = time.time()
    agent.work(goal)
    todos = [(t.status, t.text) for t in agent.session.todos.items]
    try:
        passed, detail = check(root, before)
    except Exception as e:
        passed, detail = False, f"check crashed: {e!r}"
    print(f"{'PASS' if passed else 'FAIL'} {task_id}-{n}: {detail} | todos={len(todos)} "
          f"{sum(s == 'done' for s, _ in todos)} done | calls={calls[0]} biggest prompt={prompt_chars[0]} chars "
          f"{time.time() - t0:.0f}s", flush=True)
    for status, text in todos:
        print(f"      [{status}] {text}", flush=True)
    return passed


if __name__ == "__main__":
    args = sys.argv[2:]
    runs = 1
    if "--runs" in args:
        i = args.index("--runs")
        runs = int(args[i + 1])
        del args[i: i + 2]
    ids = args or list(TASKS)
    total = sum(run(t, n) for t in ids for n in range(1, runs + 1))
    print(f"=== {total}/{len(ids) * runs} passed", flush=True)
