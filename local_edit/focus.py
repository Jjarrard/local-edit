"""FIX: the harness finds the code, the model only rewrites it.

A 4B model can't be trusted to name lines (EDIT 15-100 for a 3-line rule)
or to quote them (it writes the file it wants instead). What it can do is
answer a small closed question and rewrite a short block it is shown. So a
FIX is broken into steps that each fit that:

  1. code splits the file into blocks - CSS rules, HTML elements,
     functions - by indentation and closing lines, no parser needed
  2. code ranks the blocks against the request by shared words
  3. model: "which of these numbered blocks needs to change? or NEW"
  4. model: "rewrite this block so that ..." - or, for NEW, "after which
     block does it go?" then "write only the new part"
  5. code splices the result in at boundaries it already knows

The model never sees the whole file and never writes a line number, and
every prompt is a few hundred tokens however big the file is - the same
shape as local-search, where code does the searching and cutting and the
model answers small questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_REWRITE_LINES = 60   # never ask the model to rewrite more than this at once
MAX_CHOICES = 8
MAX_PLACES = 16

_CLOSER_RE = re.compile(r"^\s*(\}|\)|\]|</|end\b|fi\b|done\b|esac\b)")
_WORD_RE = re.compile(r"[a-z][a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "make", "add", "change", "into", "from", "some", "to",
    "can", "you", "please", "should", "would", "want", "need", "file", "page", "website", "site",
    "html", "css", "code", "just", "also", "new", "all", "its", "use", "more", "less", "bit",
}


@dataclass
class Block:
    start: int  # 1-indexed, inclusive
    end: int
    depth: int
    lines: list[str]
    used_by: str = ""  # for a CSS class rule: a line of markup that uses it

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    @property
    def head(self) -> str:
        return self.lines[0].strip()

    @property
    def indent(self) -> int:
        return _indent(self.lines[0])

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def split_blocks(text: str) -> list[Block]:
    """Every block in the file, outermost first, in file order.

    A block is a line plus everything indented under it, plus a closing line
    (`}`, `</div>`, `end`) back at its own indentation. Blocks are
    listed and then split into their children too, so both "the whole
    <main>" and "this one card inside it" can be chosen.
    """
    lines = text.splitlines()
    out: list[Block] = []

    def walk(lo: int, hi: int, depth: int) -> None:
        i = lo
        while i < hi:
            if not lines[i].strip():
                i += 1
                continue
            ind = _indent(lines[i])
            j = i + 1
            while j < hi and (not lines[j].strip() or _indent(lines[j]) > ind):
                j += 1
            has_closer = j < hi and _indent(lines[j]) == ind and bool(_CLOSER_RE.match(lines[j]))
            if has_closer:
                j += 1
            end = j - 1
            while end > i and not lines[end].strip():
                end -= 1
            out.append(Block(i + 1, end + 1, depth, lines[i : end + 1]))
            if end > i:
                inner_hi = end if has_closer else end + 1
                walk(i + 1, inner_hi, depth + 1)
            i = j

    walk(0, len(lines), 0)
    _link_css_to_markup(out, lines)
    return out


_CLASS_RULE_RE = re.compile(r"^\.([a-zA-Z][\w-]*)")


def _link_css_to_markup(blocks: list[Block], lines: list[str]) -> None:
    """Note where each `.class {` rule is used, e.g. `.book-btn` ->
    `<button class="book-btn">Book Now</button>`. People name things by what
    the page shows ("the Book Now buttons"); CSS names them by class. The
    usage line bridges the two, for ranking and for the model's choice."""
    for b in blocks:
        m = _CLASS_RULE_RE.match(b.head)
        if not m or b.size < 2:
            continue
        pattern = re.compile(r"class=\"[^\"]*\b" + re.escape(m.group(1)) + r"\b")
        use = next((ln.strip() for ln in lines if pattern.search(ln)), "")
        b.used_by = use[:100]


# Code abbreviates what people spell out: "buttons" must find `.book-btn`.
_ABBREVIATIONS = {
    "btn": "button", "bg": "background", "nav": "navigation", "img": "image", "msg": "message",
    "txt": "text", "num": "number", "idx": "index", "cfg": "config", "auth": "login",
    "colour": "color", "grey": "gray",
}
_STATE_RE = re.compile(r":(hover|focus|active|visited|before|after)\b")


_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


def _words(text: str) -> set[str]:
    # cartTotal -> cart, total: people say "the cart total", JS says
    # cartTotal, and as one word it matched nothing (snake_case already splits)
    words = set(_WORD_RE.findall(_CAMEL_RE.sub(r"\1 \2", text).lower()))
    words |= {_ABBREVIATIONS[w] for w in words if w in _ABBREVIATIONS}
    # crude stems so "colours"/"colour", "buttons"/"button" meet
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 4}


def request_words(request: str) -> set[str]:
    return {w for w in _words(request) if w not in _STOPWORDS}


def _score(block: Block, words: set[str]) -> float:
    hits = len(words & (_words(block.text) | _words(block.used_by)))
    head_hits = len(words & _words(block.head))
    # The head line names a block ("header {", "<footer>", "def save"),
    # so a match there counts double; big blocks are mildly penalised so
    # the specific rule beats the container it sits in.
    score = (hits + head_hits) / (1 + block.size / 80)
    # `.btn:hover` is a variant of `.btn` - the base rule is what "make the
    # buttons red" means, unless the request is about hovering itself.
    state = _STATE_RE.search(block.head)
    if state and state.group(1) not in words:
        score *= 0.5
    return score


def outline(blocks: list[Block], limit: int = MAX_CHOICES) -> list[Block]:
    """The file's main multi-line blocks (functions, rules, sections), for
    when the request shares no words with the code ("handle empty lists")
    and the model has to judge from what the blocks are."""
    main = [b for b in blocks if b.depth <= 1 and 2 <= b.size <= MAX_REWRITE_LINES]
    return main[:limit]


def candidates(blocks: list[Block], request: str, limit: int = MAX_CHOICES) -> list[Block]:
    """The blocks most likely to need changing, back in file order."""
    words = request_words(request)
    # Single lines deep inside a block (one CSS property, one statement)
    # would crowd the list; near the top level they're real elements
    # (<title>, an import) and stay.
    usable = [b for b in blocks if b.size <= MAX_REWRITE_LINES and (b.size > 1 or b.depth <= 2)]
    ranked = sorted(usable, key=lambda b: (-_score(b, words), b.size))
    picked: list[Block] = []
    same_head: dict[str, int] = {}
    for b in ranked:
        if len(picked) >= limit or _score(b, words) <= 0:
            break
        # Four identical `<div class="stall">` cards would fill half the
        # list; two are enough to show the model that kind of block.
        if same_head.get(b.head, 0) >= 2:
            continue
        same_head[b.head] = same_head.get(b.head, 0) + 1
        picked.append(b)
    # A single line inside another listed block is already covered by it,
    # and three slices of one click handler crowd out the function that
    # actually needs the change.
    picked = [b for b in picked if b.size > 1 or not any(
        o is not b and o.size > 1 and o.start <= b.start and b.end <= o.end for o in picked)]
    return sorted(picked, key=lambda b: b.start)


_WRAPPER_RE = re.compile(r"^<(!doctype|html|head|body|style|script)\b", re.IGNORECASE)


def obvious_place(blocks: list[Block], request: str) -> Block | None:
    """Where "add another X" goes, when code can tell without asking.

    If the page has a run of look-alike blocks (stall cards, list items)
    whose first line shares a word with the request, the new one goes after
    the last of them. Rule first, model only when the rule has no answer.
    """
    added = _added_thing(request)
    if not added:
        return None
    best = None
    for b in places(blocks):
        twins = [x for x in blocks if x.head == b.head and x.depth == b.depth and x.size >= 2]
        if len(twins) >= 2 and twins[-1] is b and added & _words(b.head):
            best = b
    return best


_ADDED_RE = re.compile(
    r"\b(?:add|insert|create|put|append)\s+(?:a|an|another|one|one more|some|\w+th|\w+nd|\w+rd|\w+st|new)?\s*"
    r"((?:[\w-]+\s+){0,2}?[\w-]+?)(?=\s+(?:with|to|called|named|that|for|which|showing|about|in|on|at)\b|[,.]|$)",
    re.IGNORECASE,
)


def _added_thing(request: str) -> set[str]:
    """The words naming what's being added: "add a fifth stall called X" ->
    {stall}; "add a navigation bar with links to Stalls" -> {navigation, bar}."""
    m = _ADDED_RE.search(request)
    return request_words(m.group(1)) if m else set()


def places(blocks: list[Block], limit: int = MAX_PLACES) -> list[Block]:
    """Where new code could go, as "after this block".

    Document wrappers (<body>, <style>, ...) are left out: "after <body>"
    means after </body>, outside the page. Of a run of sibling blocks only
    the last is offered - that's where a new CSS rule or page section is
    appended - which keeps 17 CSS rules from crowding out the page layout.
    """
    def parent_of(b: Block) -> Block | None:
        enclosing = [p for p in blocks if p.depth == b.depth - 1 and p.start <= b.start and b.end <= p.end]
        return enclosing[-1] if enclosing else None

    spots = [
        b for b in blocks
        if b.depth <= 2 and not _WRAPPER_RE.match(b.head) and not _CLOSER_RE.match(b.lines[0])
    ]
    last_child: dict[int, Block] = {}
    for b in spots:
        parent = parent_of(b)
        key = id(parent) if parent else -1
        if b.depth >= 2:
            last_child[key] = b  # later siblings overwrite earlier ones
    keep = [b for b in spots if b.depth < 2 or last_child.get(id(parent_of(b)) if parent_of(b) else -1) is b]
    # The last of a run of look-alike siblings at any depth - the 4th stall
    # card, the last <li> - is where "add another one" goes, however deep
    # the list sits.
    groups: dict[tuple[int, str], list[Block]] = {}
    for b in blocks:
        if b.size >= 2:
            parent = parent_of(b)
            groups.setdefault((id(parent) if parent else -1, b.head), []).append(b)
    for twins in groups.values():
        if len(twins) >= 2 and twins[-1] not in keep:
            keep.append(twins[-1])
    keep.sort(key=lambda b: (b.depth, b.start))
    return sorted(keep[:limit], key=lambda b: b.start)


def _preview(block: Block, max_lines: int = 4) -> str:
    shown = block.lines if block.size <= max_lines + 1 else block.lines[:max_lines] + ["    ..."]
    base = block.indent
    return "\n".join("    " + ln.expandtabs(4)[base:] if ln.strip() else "" for ln in shown)


_ADDING_RE = re.compile(r"\b(add|adds|adding|insert|create|append|include|put|new|another|more)\b", re.IGNORECASE)


def is_adding(request: str) -> bool:
    return bool(_ADDING_RE.search(request))


def adds_something_absent(blocks: list[Block], request: str) -> bool:
    """ "add a footer" to a page with no footer anywhere: nothing to choose
    from, it's new code - decided by code, without a question."""
    added = _added_thing(request)
    if not added:
        return False
    return not any(added & _words(b.text) for b in blocks)


def choose_prompt(request: str, rel: str, choices: list[Block]) -> str:
    listing = "\n\n".join(
        f"[{n}] lines {b.start}-{b.end}:\n{_preview(b)}" + (f"\n    (used by: {b.used_by})" if b.used_by else "")
        for n, b in enumerate(choices, 1)
    )
    # "Something new" is a numbered option, not a word to type: small models
    # pick from a list far more reliably than they volunteer "NEW". It's
    # only offered when the task adds something - otherwise "handle empty
    # lists" gets a second copy of the function instead of a fixed one.
    new = len(choices) + 1
    none_option = (
        f"[{new}] NONE of the above - the task adds a new part that isn't in the file yet\n\n"
        if is_adding(request) else ""
    )
    return (
        f"Task: {request}\nFile: {rel}\n\nParts of the file:\n\n{listing}\n\n"
        f"{none_option}"
        "Which ONE number fits the task? Pick the most specific match - a rule or element\n"
        "named after the thing in the task beats a general one. For adding, removing or\n"
        "changing something ON the page pick the HTML element, not its CSS - CSS only when\n"
        "the task is about looks (colour, size, font, spacing). Reply with just the number."
    )


def place_prompt(request: str, rel: str, spots: list[Block]) -> str:
    listing = "\n".join(
        f"[{n}] after lines {b.start}-{b.end}: {'  ' * b.depth}{b.head[:70]}" for n, b in enumerate(spots, 1)
    )
    return (
        f"Task: {request}\nFile: {rel}\n\nNew code must be added. The file's structure:\n{listing}\n\n"
        "After which numbered part should the new code go? Reply with just the number."
    )


_SIGNATURE_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:def|function|class)\s+\w+|^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"
)
MAX_SIGNATURES = 8
_NAME_RE = re.compile(r"(?:def|function|class|const|let|var)\s+(\w+)")


def signatures(blocks: list[Block], skip: Block, request: str = "", limit: int = MAX_SIGNATURES) -> list[str]:
    """One line per other top-level function a rewrite can use: the ones
    the block calls, then ones the request names, then the nearest. Shown
    only the cart total loop, a small model invented `item.stallPrice`;
    one line saying `function lineTotal(item)` exists was what it lacked."""
    words = request_words(request)
    called = set(re.findall(r"\b(\w+)\s*\(", skip.text))
    found = []
    for b in blocks:
        if b.depth == 0 and b is not skip and _SIGNATURE_RE.match(b.head):
            m = _NAME_RE.search(b.head)
            name = m.group(1) if m else ""
            rank = (0 if name in called else 1, -len(words & _words(name)), abs(b.start - skip.start))
            found.append((rank, b.head.rstrip("{: ").strip()))
    return [sig for _, sig in sorted(found)[:limit]]


def rewrite_prompt(request: str, rel: str, block: Block, others: list[str] | None = None) -> str:
    context = ("Other functions in this file you can use:\n" + "\n".join(f"  {o}" for o in others) + "\n\n"
               if others else "")
    return (
        f"Task: {request}\nFile: {rel} (lines {block.start}-{block.end} shown)\n\n{context}```\n{block.text}\n```\n\n"
        "Rewrite ONLY these lines so the task is done. Keep everything else in them the same.\n"
        "Reply with the complete new version of these lines in one ``` block, nothing else."
    )


def _tail(block: Block, max_lines: int = 4) -> str:
    """The end of a block - what sits directly above code inserted after it."""
    if block.size <= max_lines + 1:
        return block.text
    return "\n".join([block.lines[0], "    ..."] + block.lines[-(max_lines - 1):])


def new_code_prompt(request: str, rel: str, after: Block) -> str:
    return (
        f"Task: {request}\nFile: {rel}\n\nThe new code goes right after this (already complete - "
        f"don't close or repeat any of it):\n```\n{_tail(after)}\n```\n\n"
        "Write ONLY the new code to add there - not the existing code, not the whole file.\n"
        "Reply with it in one ``` block, nothing else."
    )


def parse_choice(reply: str, count: int) -> int | str | None:
    """A 1-based choice, "NEW", or None if the reply says neither."""
    text = re.sub(r"<think>.*?(</think>|$)", "", reply, flags=re.DOTALL | re.IGNORECASE).strip()
    m = re.search(r"\d+", text)
    if m and 1 <= int(m.group()) <= count:
        return int(m.group())
    if m and int(m.group()) == count + 1:
        return "NEW"  # the "none of the above" option choose_prompt adds
    if re.search(r"\bNEW\b", text, re.IGNORECASE):
        return "NEW"
    return None


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|$)", re.DOTALL)


def extract_code(reply: str) -> str | None:
    """The fenced code in a reply: None if there's no block at all, "" for
    an empty block - which, for a rewrite, means "delete these lines"."""
    m = _FENCE_RE.search(reply)
    if m is None:
        return None
    code = m.group(1).rstrip()
    return code if code.strip() else ""


def looks_like_whole_file(code: str, block: Block | None) -> bool:
    """The model wrote the entire file instead of the part it was asked for."""
    markers = ("<!doctype", "<html")
    lowered = code.lower()
    if block is not None and any(m in block.text.lower() for m in markers):
        return False
    return any(m in lowered for m in markers)


_LONE_CLOSER_RE = re.compile(r"^\s*</([a-zA-Z][\w-]*)\s*>\s*$")
_LONE_OPENER_RE = re.compile(r"^\s*<([a-zA-Z][\w-]*)(?:\s[^>]*)?>\s*$")


def trim_stray_closers(code: str) -> str:
    """Drop closing tags dangling at the start of new markup.

    Shown the end of a <div> and asked for "a footer after it", a small
    model often re-closes things first: `</div>\n</div>\n<footer>`. Those
    closers belong to the existing code, not the new part; leaving them in
    would close the page's containers early.
    """
    lines = code.splitlines()
    while lines and (not lines[0].strip() or _LONE_CLOSER_RE.match(lines[0])):
        lines.pop(0)
    return "\n".join(lines)


_AND_THEN_RE = re.compile(
    r"\s*(?:,\s*)?\b(?:and|then|also|plus)\s+(?=(?:also\s+)?(?:add|make|change|remove|delete|put|set|turn|"
    r"move|create|insert|update|use|give|replace|rename|fix)\b)",
    re.IGNORECASE,
)


_PRONOUN_RE = re.compile(r"\b(it|its|them|they|this|that|these|those)\b", re.IGNORECASE)


def split_request(request: str) -> list[str]:
    """ "make the header green and add a footer" -> two requests.

    Each FIX handles one block. Joined requests otherwise get half-done: the
    header changes, the footer is forgotten, and the step reads as success.
    Only splits before a verb, so "a red and white header" stays whole.
    """
    parts: list[str] = []
    for part in (p.strip(" ,.") for p in _AND_THEN_RE.split(request)):
        if not part:
            continue
        # "add a nav bar and make it sticky": "it" needs the part before it.
        if parts and _PRONOUN_RE.search(part):
            parts[-1] = f"{parts[-1]} and {part}"
        else:
            parts.append(part)
    return parts or [request]


def match_indent(code: str, target_indent: int) -> str:
    """Shift a block so its first line sits at `target_indent`, keeping the
    relative indentation inside it."""
    lines = code.splitlines()
    first = next((ln for ln in lines if ln.strip()), None)
    if first is None:
        return code
    shift = target_indent - _indent(first)
    if shift == 0:
        return code
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        elif shift > 0:
            out.append(" " * shift + ln)
        else:
            out.append(ln[min(-shift, _indent(ln)):])
    return "\n".join(out)


_TOP_LEVEL_RE = re.compile(
    r"^(?:def|async\s+def|class|import|from|@|#|if\s+__name__|[A-Za-z_][\w, ]*\s*(?::[^=]+)?=(?!=)|\"\"\"|''')"
)


def is_body_code(code: str) -> bool:
    """Python that only makes sense inside a function: `if not numbers:`,
    `return price`. Asked to "handle empty lists in median()", a small model
    picks "new code" and writes the guard alone - placed after the function,
    at the top level, it breaks the module."""
    first = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
    return bool(first) and not _TOP_LEVEL_RE.match(first)


MAX_HOOK_LINES = 30
_HOOK_RE = re.compile(r"\b(id|class|name)=\"|<(form|input|button|ul|ol|table|select|textarea)\b", re.IGNORECASE)


def markup_hooks(html: str, limit: int = MAX_HOOK_LINES) -> list[str]:
    """The lines of a page that scripts and styles hang off - ids, classes,
    form controls. A new app.js has to use the ids index.html already has;
    this is the slice of the page that says what they are."""
    hooks = [ln.strip() for ln in html.splitlines() if _HOOK_RE.search(ln)]
    return [h[:120] for h in hooks[:limit]]


def create_prompt(rel: str, what: str, goal: str, context: str) -> str:
    goal_line = f"Overall goal: {goal}\n" if goal and goal.strip() != what.strip() else ""
    return (
        f"{goal_line}Write the new file {rel}. It should: {what}\n\n"
        f"{context}"
        f"Reply with the COMPLETE content of {rel} in one ``` block, nothing else."
    )


def _top_level_names(tree) -> set[str]:
    import ast

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def complete_imports(text: str, modules: dict[str, str]) -> tuple[str, list[str]]:
    """Add names a file uses but never imported to its existing
    `from <module> import ...` line.

    A model adding test_median() to test_calc.py writes the test and leaves
    `from calc import average, total` alone - the test then dies on a
    NameError. Which import is missing is a fact code can look up: the name
    is used, bound nowhere in the file, and defined at the top of a sibling
    module the file already imports from. `modules` maps module name to
    source. Returns the new text and the names added.
    """
    import ast
    import builtins

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, []
    bound = set(dir(builtins))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (used if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
    missing = used - bound
    if not missing:
        return text, []
    lines = text.splitlines()
    added: list[str] = []
    for node in tree.body:
        if not (isinstance(node, ast.ImportFrom) and node.module in modules and node.lineno == node.end_lineno):
            continue
        line = lines[node.lineno - 1]
        if "(" in line or "*" in line or ";" in line or "#" in line:
            continue
        try:
            offered = _top_level_names(ast.parse(modules[node.module]))
        except SyntaxError:
            continue
        new = sorted((missing & offered) - set(added))
        if new:
            lines[node.lineno - 1] = line.rstrip() + ", " + ", ".join(new)
            added += new
    if not added:
        return text, []
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), added


def _function_scopes(tree):
    import ast

    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def undefined_names(text: str) -> set[tuple[str, str]]:
    """(function, name) pairs where a function reads a name that is bound
    nowhere - not in it, not at module level, not a builtin.

    Rewriting `return round(discount, 2)` from a request that said
    "discount_amount", a small model writes `price - discount_amount`: valid
    syntax, a NameError the first time it runs. Compared before and after an
    edit, so only names the edit introduced count.
    """
    import ast
    import builtins

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    module_bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            module_bound |= set(node.names)
    for node in tree.body:
        for sub in ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else []:
            if isinstance(sub, ast.Name) and not isinstance(sub.ctx, ast.Load):
                module_bound.add(sub.id)
    out: set[tuple[str, str]] = set()
    for fn in _function_scopes(tree):
        bound = set()
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and not isinstance(sub.ctx, ast.Load):
                bound.add(sub.id)
            elif isinstance(sub, ast.arg):
                bound.add(sub.arg)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                bound.add(sub.name)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in bound | module_bound:
                out.add((fn.name, sub.id))
    return out


def enclosing_function(blocks: list[Block], block: Block) -> Block | None:
    """The top-level def/class around a line chosen inside it, if it's
    small enough to rewrite whole. One line alone hides the names it can
    use; the function shows them."""
    if block.depth == 0:
        return None
    for b in blocks:
        if (b.depth == 0 and b.start <= block.start and block.end <= b.end
                and re.match(r"(async\s+def|def|class)\s", b.head) and b.size <= MAX_REWRITE_LINES):
            return b
    return None


_LOOKS_RE = re.compile(
    r"\b(colou?rs?|background|font|fonts|size|bigger|smaller|spacing|margin|padding|border|rounded|shadow|"
    r"bold|italic|width|height|style|styling|darker|lighter|theme)\b",
    re.IGNORECASE,
)
_STYLESHEET_RE = re.compile(r"<link\b[^>]*rel=[\"']?stylesheet[^>]*>", re.IGNORECASE)
_HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']", re.IGNORECASE)


def stylesheet_for(html: str, request: str) -> str | None:
    """The local stylesheet a looks-only change to this page belongs in:
    the page has no <style> of its own and links exactly one local .css."""
    if not _LOOKS_RE.search(request) or re.search(r"<style\b", html, re.IGNORECASE):
        return None
    hrefs = [m.group(1) for tag in _STYLESHEET_RE.findall(html) for m in [_HREF_RE.search(tag)] if m]
    local = [h for h in hrefs if not re.match(r"(?i)(https?:)?//", h)]
    return local[0] if len(local) == 1 else None
