"""Diff rendering, an optional model self-review pass, and the human gate.

Small models make mistakes often enough that edits should never land
silently. Every EDIT/INSERT/DELETE goes through: render diff -> optional
self-review by the model -> human confirmation (unless auto-apply is on).
"""

from __future__ import annotations

import re

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .client import JanClient


def render_diff(console: Console, diff_text: str, title: str) -> None:
    if not diff_text.strip():
        console.print(Panel("(no textual change)", title=title, border_style="yellow"))
        return
    body = Text()
    for line in diff_text.splitlines():
        style = "dim"
        if line.startswith("+++") or line.startswith("---"):
            style = "bold"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        body.append(line + "\n", style=style)
    console.print(Panel(body, title=title, border_style="blue"))


# Self-review only needs a short verdict, not a full reply - capping it well
# below the main generation budget keeps it from taking as long as (or
# longer than) the edit it's reviewing, especially for reasoning models that
# would otherwise spend their whole budget on a <think> block.
REVIEW_MAX_TOKENS = 400


CONTEXT_LINES = 4


def snippet(text: str, start: int, end: int, context: int = CONTEXT_LINES) -> str:
    """A numbered window around a changed region, for review context."""
    lines = text.splitlines()
    lo = max(1, start - context)
    hi = min(len(lines), end + context)
    if lo > len(lines):
        return "(empty file)"
    return "\n".join(f"{i:>4}: {lines[i - 1]}" for i in range(lo, hi + 1)) or "(empty file)"


def build_review_messages(task_text: str, rel: str, before: str, after: str) -> list[dict]:
    """Ask for a verdict on before/after text rather than a unified diff.

    Small models misread diffs badly - given "-  return a - b / +  return a + b"
    they routinely describe the removed line as the result and reject a
    correct fix. Showing the two states plainly removes that trap.
    """
    prompt = (
        f"A developer was asked to: {task_text}\n\n"
        f"File: {rel}\n\n"
        f"BEFORE (current file):\n{before}\n\n"
        f"AFTER (what the file becomes if this change is accepted):\n{after}\n\n"
        "Judge only the AFTER state. Does AFTER accomplish the task and stay valid code?\n"
        "Ignore what BEFORE did wrong - that is the bug being fixed.\n"
        "Answer with one word, YES or NO, then one short sentence of reason."
    )
    return [
        {"role": "system", "content": "You are a careful, terse code reviewer. You answer YES or NO."},
        {"role": "user", "content": prompt},
    ]


def parse_review_verdict(reply: str) -> tuple[bool, str]:
    reply = reply.strip()
    # Reasoning models front-load a <think> block, so the verdict can land
    # anywhere in the reply (and no single stop sequence is safe to use,
    # since "\n\n" reliably shows up inside their reasoning). Take the last
    # standalone YES/NO in the text as the conclusion.
    matches = re.findall(r"\bYES\b|\bNO\b", reply, re.IGNORECASE)
    if not matches:
        return True, reply or "(no verdict given)"
    ok = matches[-1].upper() != "NO"
    return ok, reply


def self_review(client: JanClient, task_text: str, rel: str, before: str, after: str) -> tuple[bool, str]:
    """Ask the model to sanity-check a change. Best-effort: on any ambiguity
    or error we default to allowing it through.

    This is the silent, non-streaming version - Agent uses the streamed
    equivalent (_run_self_review) so the review is visible and can be cut
    short by the same repetition guard as the main generation.
    """
    try:
        reply = client.chat(
            build_review_messages(task_text, rel, before, after), max_tokens=REVIEW_MAX_TOKENS
        )
    except Exception as exc:  # noqa: BLE001 - self-review is advisory, never fatal
        return True, f"(self-review unavailable: {exc})"
    return parse_review_verdict(reply)


def ask_human(console: Console, prompt: str = "Apply this edit?") -> str:
    """Returns one of: apply, reject, quit.

    Delegates to ui.confirm so buffered keystrokes typed during generation
    are discarded first - an approval must be a deliberate keypress made
    after the diff was on screen. Enter = apply, matching the diff already
    being on screen as the thing being confirmed.
    """
    from . import ui

    decision = ui.confirm(console, prompt, default="yes")
    return {"yes": "apply", "no": "reject", "stop": "quit"}[decision]
