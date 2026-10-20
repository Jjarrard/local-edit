"""A small, fixed-size note of what the model is doing, and trimming of old turns.

A 4B model on a 4k window loses the plot two ways: the request scrolls out
of the window under file dumps, or the window fills and it starts guessing.
So instead of a growing transcript the model gets:

  - a progress note on every turn - the request, the files touched, the last
    few steps and how each went - built by the harness from what actually
    happened, never from the model's own (unreliable) summary, and capped
    so it can't grow with the session;
  - older turns with their bulk (file contents, command output, code
    blocks) cut to a one-line stub, since the note already says what
    happened.
"""

from __future__ import annotations

import re
from collections import OrderedDict, deque

MAX_STEPS = 5
MAX_FILES = 5
MAX_LINE = 110
MAX_GOAL = 240
MAX_NOTE_CHARS = 900

# Messages newer than this many from the end are left exactly as they were.
RECENT_UNTRIMMED = 2
# Tool output older than that, longer than this, is cut to its first line.
TRIM_OVER_CHARS = 300

_TOOL_OUTPUT_PREFIXES = ("--- ", "[LIST", "[GREP", "$ ")
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_EXIT_CODE_RE = re.compile(r"\(exit code (-?\d+)\)")


def _short(text: str, limit: int = MAX_LINE) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class Progress:
    def __init__(self) -> None:
        self.goal: str | None = None
        self.earlier_goal: str | None = None
        self.steps: deque[str] = deque(maxlen=MAX_STEPS)
        self.files: OrderedDict[str, int] = OrderedDict()
        self.last_problem: str | None = None
        # Whether a DONE for this request has already been double-checked.
        self.done_checked = False

    def start(self, goal: str) -> None:
        """A new request. The previous one is kept (one level only), because
        follow-ups like "it's still white" mean nothing on their own."""
        goal = _short(goal, MAX_GOAL)
        if goal == self.goal:
            return
        if self.goal:
            self.earlier_goal = _short(self.goal, MAX_LINE)
        self.goal = goal
        self.steps.clear()
        self.last_problem = None
        self.done_checked = False

    def note_file(self, rel: str, line_count: int) -> None:
        self.files.pop(rel, None)
        self.files[rel] = line_count
        while len(self.files) > MAX_FILES:
            self.files.popitem(last=False)

    def record(self, label: str, result_text: str | None) -> None:
        """One step: what was tried (`label`) and how the harness answered."""
        first = (result_text or "").strip().splitlines()[0] if (result_text or "").strip() else ""
        if first.startswith(("ERROR", "Your ", "User rejected", "User declined", "That ")):
            detail = (result_text or "").strip().splitlines()
            reason = next((ln for ln in detail[1:] if ln.startswith("Reviewer said:")), "")
            self.last_problem = _short(f"{label}: {first} {reason}".strip(), 2 * MAX_LINE)
            outcome = "failed"
        elif first.startswith("No change"):
            outcome = "no change"
        elif first.startswith("$ "):
            code = _EXIT_CODE_RE.search(result_text or "")
            outcome = f"exit {code.group(1)}" if code else "ran"
        else:
            outcome = "ok"
            if first.startswith("Applied"):
                self.last_problem = None
        self.steps.append(_short(f"{label} -> {outcome}"))

    def render(self) -> str:
        if not self.goal:
            return ""
        lines = ["[progress note from local-edit - facts, not a command to copy]", f"Request: {self.goal}"]
        if self.earlier_goal:
            lines.append(f"Before that: {self.earlier_goal}")
        if self.files:
            lines.append("Files touched: " + ", ".join(f"{rel} ({n} lines)" for rel, n in self.files.items()))
        if self.steps:
            lines.append("Recent steps, oldest first:")
            lines.extend(f"  {s}" for s in self.steps)
        if self.last_problem:
            lines.append(f"Unresolved problem: {self.last_problem}")
        lines.append("Reply with ONE command that moves the request forward, or DONE if it is finished.")
        note = "\n".join(lines)
        while len(note) > MAX_NOTE_CHARS and len(lines) > 3:
            # Shed the oldest step first; the request and the problem matter most.
            idx = next((i for i, ln in enumerate(lines) if ln.startswith("  ")), None)
            if idx is None:
                break
            del lines[idx]
            note = "\n".join(lines)
        return note[:MAX_NOTE_CHARS]


def trim_old_message(msg: dict) -> dict:
    """The one-line version of an older turn.

    Only bulk is cut: tool output (file dumps, listings, command output) and
    code blocks the model wrote. The user's own words are left alone.
    """
    content = msg.get("content", "")
    if msg.get("role") == "assistant":
        def stub(m: re.Match) -> str:
            n = len(m.group(1).splitlines())
            return m.group(0) if n <= 3 else f"[{n}-line code block - handled, see progress note]"
        trimmed = _FENCED_BLOCK_RE.sub(stub, content)
    elif msg.get("role") == "user" and len(content) > TRIM_OVER_CHARS:
        parts = content.split("\n\n")
        out = []
        for part in parts:
            if part.startswith(_TOOL_OUTPUT_PREFIXES) and len(part) > TRIM_OVER_CHARS:
                head = part.splitlines()[0]
                if head.startswith("$ "):
                    code = _EXIT_CODE_RE.search(part)
                    head += f" {code.group(0)}" if code else ""
                out.append(f"{_short(head)} [older output trimmed - run it again if you need it]")
            else:
                out.append(part)
        trimmed = "\n\n".join(out)
    else:
        return msg
    return msg if trimmed == content else {**msg, "content": trimmed}
