"""Cheap structural validation of a proposed edit, before it touches disk.

A small model's single most common damaging failure is an edit that is
*plausible* but syntactically broken - a dedented line, a dropped bracket, a
half-replaced block. Catching that here turns a silent corruption into a
specific, actionable error the model can retry against, and costs nothing
compared to another model round-trip.

Validation is best-effort by design: an unknown file type is not an error,
and a validator that can't run never blocks an edit.
"""

from __future__ import annotations

import ast
import configparser
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - exercised only on Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - tomli not installed
        tomllib = None  # type: ignore[assignment]


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""

    @classmethod
    def good(cls) -> "ValidationResult":
        return cls(True)

    @classmethod
    def bad(cls, message: str) -> "ValidationResult":
        return cls(False, message)


def _validate_python(text: str) -> ValidationResult:
    try:
        ast.parse(text)
    except SyntaxError as exc:
        where = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return ValidationResult.bad(f"Python syntax error at {where}: {exc.msg}")
    except ValueError as exc:  # e.g. null bytes
        return ValidationResult.bad(f"Python parse error: {exc}")
    return ValidationResult.good()


def _validate_json(text: str) -> ValidationResult:
    if not text.strip():
        return ValidationResult.good()
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult.bad(f"JSON error at line {exc.lineno}, column {exc.colno}: {exc.msg}")
    return ValidationResult.good()


def _validate_ini(text: str) -> ValidationResult:
    try:
        configparser.ConfigParser().read_string(text)
    except configparser.Error as exc:
        return ValidationResult.bad(f"INI/config error: {exc}")
    return ValidationResult.good()


def _validate_toml(text: str) -> ValidationResult:
    if tomllib is None:
        # No TOML parser available (Python 3.10 without `tomli` installed).
        # TOML and INI are similar but not compatible - inline tables, arrays
        # of tables, and multi-line strings are all valid TOML that
        # `configparser` would misreport as broken - so best-effort here
        # means skipping validation, not running an incompatible parser.
        return ValidationResult.good()
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return ValidationResult.bad(f"TOML error: {exc}")
    return ValidationResult.good()


def _balanced_delimiters(text: str) -> ValidationResult:
    """Bracket balance check for languages we have no real parser for.

    Deliberately conservative: it ignores anything inside strings or comments
    and only reports a definite mismatch, because a false positive here would
    block a perfectly good edit.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())
    stack: list[str] = []

    i = 0
    n = len(text)
    line = 1
    in_line_comment = False
    in_block_comment = False
    quote: str | None = None

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if ch == "\n":
            line += 1
            in_line_comment = False
            i += 1
            continue

        if in_line_comment:
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue

        if ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack:
                return ValidationResult.bad(f"unbalanced '{ch}' at line {line} (nothing open to close)")
            if stack[-1] != pairs[ch]:
                return ValidationResult.bad(f"mismatched '{ch}' at line {line}")
            stack.pop()
        i += 1

    if stack:
        return ValidationResult.bad(f"{len(stack)} unclosed '{stack[-1]}' by end of file")
    return ValidationResult.good()


BRACKET_LANGUAGES = {".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rs", ".css"}

# suffix -> validator, so a new language/tool can be wired in with one line
# instead of editing the dispatch logic in `validate()`.
VALIDATORS: dict[str, Callable[[str], ValidationResult]] = {".py": _validate_python, ".json": _validate_json}
VALIDATORS.update({suffix: _validate_ini for suffix in (".ini", ".cfg")})
VALIDATORS[".toml"] = _validate_toml
VALIDATORS.update({suffix: _balanced_delimiters for suffix in BRACKET_LANGUAGES})


_HTML_CONTAINERS = ("html", "head", "body", "style", "script")


def html_unclosed_tags(text: str) -> list[str]:
    """Container tags an HTML file opens but never closes.

    Not a validator: a page legitimately written in several INSERTs is
    unfinished between them, so this can't block an edit. It exists so the
    model is told the page is incomplete instead of reporting success - an
    unclosed <style> swallows the whole body and renders as a blank page.
    """
    lowered = text.lower()
    unclosed = []
    for tag in _HTML_CONTAINERS:
        opens = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", lowered))
        closes = lowered.count(f"</{tag}>")
        if opens > closes:
            unclosed.append(tag)
    return unclosed


_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_TAG_LINE_RE = re.compile(r"^\s*</?[a-zA-Z][\w-]*[\s>/]", re.MULTILINE)


def _validate_html(text: str) -> ValidationResult:
    """Catch markup pasted into a <style> block.

    A line-range EDIT that misses by a few dozen lines lands page markup in
    the CSS - the browser then ignores it as bad CSS and the page loses its
    content. CSS never has a line starting with a tag, so that's a definite
    mistake. (<script> isn't checked: JS template strings hold markup.)
    """
    end = re.search(r"</html\s*>", text, re.IGNORECASE)
    if end and text[end.end():].strip():
        line = text.count("\n", 0, end.end()) + 2
        return ValidationResult.bad(
            f"content after </html> at line {line} - page content belongs inside <body>"
        )
    for block in _STYLE_BLOCK_RE.finditer(text):
        tag = _TAG_LINE_RE.search(block.group(1))
        if tag:
            line = text.count("\n", 0, block.start(1) + tag.start()) + 1
            return ValidationResult.bad(
                f"HTML markup inside the <style> block at line {line} ({tag.group(0).strip()}...) - "
                "page content belongs after </style> and </head>, inside <body>"
            )
    return ValidationResult.good()


VALIDATORS.update({suffix: _validate_html for suffix in (".html", ".htm")})


_BALANCED_TAGS = ("div", "section", "main", "header", "footer", "nav", "article", "aside", "form",
                  "ul", "ol", "table", "tr", "button", "a", "p", "span", "label", "select")


def html_tag_balance(text: str) -> dict[str, int]:
    """Opened minus closed, per container tag, ignoring <style>/<script>
    bodies (JS template strings hold markup)."""
    body = re.sub(r"<(style|script)\b.*?</\1\s*>", "", text, flags=re.DOTALL | re.IGNORECASE).lower()
    out = {}
    for tag in _BALANCED_TAGS:
        opened = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", body))
        closed = len(re.findall(rf"</{tag}\s*>", body))
        if opened != closed:
            out[tag] = opened - closed
    return out


def html_balance_change(old_text: str, new_text: str) -> str | None:
    """How an edit changed tag balance, if it did - an extra </div> or a
    <footer> never closed. Compared to before, not to perfection, so a page
    that was already a little sloppy can still be edited."""
    before, after = html_tag_balance(old_text), html_tag_balance(new_text)
    for tag in sorted(set(before) | set(after)):
        delta = after.get(tag, 0) - before.get(tag, 0)
        if delta > 0:
            return f"it opens {delta} more <{tag}> than it closes"
        if delta < 0:
            return f"it closes {-delta} more <{tag}> than it opens (a stray </{tag}>)"
    return None


def register_validator(suffix: str, fn: Callable[[str], ValidationResult]) -> None:
    """Register (or override) the validator used for a given file extension.

    `suffix` should include the leading dot (e.g. ".yaml"). Call this before
    `validate()` is used to add support for a language/tool this module
    doesn't already know about, without touching `validate()` itself.
    """
    VALIDATORS[suffix.lower()] = fn


def validate(rel_path: str, text: str) -> ValidationResult:
    """Validate proposed file content by extension. Unknown types always pass."""
    suffix = Path(rel_path).suffix.lower()
    validator = VALIDATORS.get(suffix)
    if validator is None:
        return ValidationResult.good()
    return validator(text)
