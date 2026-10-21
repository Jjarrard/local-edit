"""The command language a tiny model speaks, and the parser for it.

Small (1-3B) local models are unreliable at JSON tool-calling and at
reproducing whole files verbatim. So instead of function-calling we use a
tiny, line-oriented command grammar with exactly one action per turn, and
edits are always expressed as line ranges against a file the model was just
shown. The parser is deliberately forgiving: it scans for the first
recognized command line rather than requiring the whole response to match,
because small models pad their output with filler prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COMMANDS = ("SAY", "LIST", "READ", "GREP", "RUN", "TODO", "FIX", "CREATE", "EDIT", "INSERT", "DELETE", "DONE")

# Models dress their commands up. DeepSeek-R1 writes "COMMAND: INSERT x 0";
# others use markdown bullets, numbered steps, block quotes, bold, or inline
# backticks. All of these carry the same instruction, so the parser accepts
# the decoration rather than treating a well-formed command as chat.
_CMD_PREFIX = (
    r"(?:[>\-\*•]+[ \t]*)?"          # > quote, - bullet, * bullet
    r"(?:\d+[.)][ \t]*)?"                 # 1. numbered step
    r"(?:[`*_]{1,3}[ \t]*)?"              # `code` / **bold** / _italic_
    r"(?:(?:COMMAND|ACTION|NEXT|STEP|TOOL)[ \t]*[:\-][ \t]*)?"  # COMMAND: label
    r"(?:[`*_]{1,3}[ \t]*)?"              # bold/backticks after the label too
)
_CMD_LINE_RE = re.compile(
    r"^[ \t]*" + _CMD_PREFIX + r"(" + "|".join(COMMANDS) + r")\b(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"```[^\n]*\n(.*)$", re.DOTALL)


@dataclass
class Action:
    kind: str  # SAY, LIST, READ, GREP, TODO_ADD, TODO_DONE, TODO_LIST, EDIT, INSERT, DELETE, DONE
    preamble: str = ""  # any chat text before the command line
    path: str | None = None
    start: int | None = None
    end: int | None = None
    pattern: str | None = None
    glob: str = "*"
    payload: str | None = None  # fenced block content for EDIT/INSERT
    payload_truncated: bool = False
    todo_id: int | None = None
    text: str | None = None  # SAY message / TODO ADD text / RUN command line
    raw: str = ""


class ParseError(ValueError):
    pass


def _extract_fenced_block(text: str) -> tuple[str | None, bool]:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).rstrip("\n"), False
    m = _FENCE_OPEN_RE.search(text)
    if m:
        return m.group(1).rstrip("\n"), True
    return None, False


_TRAILING_DECORATION_RE = re.compile(r"[`*_]+$")


def _strip_trailing_decoration(text: str) -> str:
    """Drop closing markdown emphasis so `**EDIT a.py 1-1**` parses.

    Only trailing backticks/asterisks/underscores go - the argument text
    itself is left untouched.
    """
    return _TRAILING_DECORATION_RE.sub("", text).strip()


def _parse_range(spec: str, label: str) -> tuple[int, int]:
    spec = spec.strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", spec)
    if not m:
        m = re.match(r"^(\d+)$", spec)
        if not m:
            raise ParseError(f"expected a line range like 12-14 for {label}, got '{spec}'")
        n = max(1, int(m.group(1)))
        return n, n
    # Lines are 1-indexed, but models routinely say 0 for "the top"
    # (EDIT index.html 0-53). Read that as line 1 rather than failing.
    return max(1, int(m.group(1))), max(1, int(m.group(2)))


# Markers that mean the model stopped issuing a command and started
# role-playing the rest of the conversation - inventing tool output and
# further turns. Seen in the wild: a model emitted "You: INSERT ... Result:
# 3 + 4 = 7 ... DONE", fabricated the entire success, and never edited
# anything. Everything from such a marker onward is fiction and must be cut
# before it reaches the parser or, worse, gets stored as history where it
# reinforces the same behaviour next turn.
_HALLUCINATED_TURN_RE = re.compile(
    r"^\s*(Result|Output|You|User|Assistant|Human|AI|System)\s*:",
    re.IGNORECASE,
)


def strip_hallucinated_continuation(raw_text: str) -> str:
    """Cut a reply at the point it starts fabricating the conversation.

    Fence-aware: these markers are only treated as fabrication when they
    appear outside a ``` block, so a legitimate code payload containing
    "Output:" is left intact.
    """
    lines = raw_text.splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _HALLUCINATED_TURN_RE.match(line):
            return "\n".join(lines[:i]).rstrip()
    return raw_text


# A shell command the model wrote without the RUN keyword. Only explicitly
# marked forms count - a labelled line, a "$" prompt, or a shell code fence -
# so ordinary prose is never mistaken for something to execute.
_LABELLED_SHELL_RE = re.compile(
    r"^[ \t]*(?:COMMAND|ACTION|STEP|TOOL|SHELL|BASH|TERMINAL)[ \t]*[:\-][ \t]*(?P<cmd>\S.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_DOLLAR_SHELL_RE = re.compile(r"^[ \t]*\$[ \t]+(?P<cmd>\S.*)$", re.MULTILINE)
_SHELL_FENCE_RE = re.compile(r"```(?:bash|sh|shell|console|zsh)\s*\n(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)
# First token must look like a program name, not the start of a sentence.
_PROGRAM_TOKEN_RE = re.compile(r"^[A-Za-z_./][\w./+-]*$")


def _looks_like_shell(command: str) -> bool:
    command = command.strip().strip("`").strip()
    if not command:
        return False
    # Prose ends in sentence punctuation; shell commands essentially never do.
    # Without this, "COMMAND: is a word I use." would be treated as runnable.
    if command.endswith((".", "!", "?")) and " " in command:
        return False
    first = command.split()[0]
    return bool(_PROGRAM_TOKEN_RE.match(first))


def find_bare_shell_command(text: str) -> str | None:
    """Extract a shell command written without the RUN keyword, if any."""
    for pattern in (_LABELLED_SHELL_RE, _DOLLAR_SHELL_RE):
        match = pattern.search(text)
        if match:
            candidate = _strip_trailing_decoration(match.group("cmd").strip().strip("`").strip())
            if _looks_like_shell(candidate):
                return candidate
    fence = _SHELL_FENCE_RE.search(text)
    if fence:
        for line in fence.group("body").splitlines():
            line = line.strip().lstrip("$").strip()
            if line and not line.startswith("#") and _looks_like_shell(line):
                return line
    return None


_FILE_COMMANDS = ("FIX", "CREATE", "EDIT", "INSERT", "DELETE")


def _later_file_command(text: str, from_pos: int) -> Action | None:
    """The first EDIT/INSERT/DELETE after `from_pos` that sits outside a code
    block and parses cleanly, with everything before it as the preamble."""
    for later in _CMD_LINE_RE.finditer(text, from_pos):
        if later.group(1).upper() not in _FILE_COMMANDS:
            continue
        if text.count("```", 0, later.start()) % 2:
            continue  # inside a fence - e.g. a SQL "DELETE FROM" line
        try:
            action = parse_action(text[later.start():])
        except ParseError:
            continue
        action.preamble = text[: later.start()].strip()
        action.raw = text
        return action
    return None


def parse_action(raw_text: str) -> Action:
    """Parse the first recognized command out of a model response.

    If nothing matches, the whole response is treated as a SAY (plain chat).
    Raises ParseError only when a command keyword is recognized but its
    arguments are malformed.
    """
    text = raw_text.strip()
    if not text:
        return Action(kind="SAY", text="(empty response)", raw=raw_text)

    m = _CMD_LINE_RE.search(text)
    if not m:
        # No local-edit keyword - but the model may have written a bare shell
        # command ("COMMAND: mkdir assets", "$ ls", a ```bash block), which
        # is plainly an instruction to run something rather than chat.
        shell_command = find_bare_shell_command(text)
        if shell_command:
            return Action(kind="RUN", text=shell_command, raw=raw_text)
        return Action(kind="SAY", text=text, raw=raw_text)

    keyword = m.group(1).upper()
    rest_of_line = _strip_trailing_decoration(m.group(2).strip())
    preamble = text[: m.start()].strip()
    after_cmd = text[m.end() :]

    if keyword == "SAY":
        # "SAY I'll create the page.\n\nINSERT index.html 0\n```..." is one
        # command, not chat: the SAY is narration in front of the real
        # action. Treating it as chat throws away the whole file the model
        # just spent its budget writing.
        file_action = _later_file_command(text, m.end())
        if file_action is not None:
            return file_action
        message = (rest_of_line + "\n" + after_cmd).strip()
        # "SAY the script tag is already there. TODO DONE 4" - the verdict
        # is at the end; read as chat, the todo stalls and gets blocked.
        done = re.search(r"\bTODO\s+DONE\s+(\d+)\W*$", message, re.IGNORECASE)
        if done:
            return Action(kind="TODO_DONE", preamble=message[: done.start()].strip(),
                          todo_id=int(done.group(1)), raw=raw_text)
        return Action(kind="SAY", preamble=preamble, text=message or "(empty)", raw=raw_text)

    if keyword == "DONE":
        return Action(kind="DONE", preamble=preamble, text=rest_of_line, raw=raw_text)

    if keyword == "LIST":
        path = rest_of_line.split()[0] if rest_of_line.split() else "."
        return Action(kind="LIST", preamble=preamble, path=path, raw=raw_text)

    if keyword == "READ":
        parts = rest_of_line.split()
        if not parts:
            raise ParseError("READ needs a path, e.g. READ src/app.py")
        path = parts[0]
        start: int | None = None
        end: int | None = None
        if len(parts) > 1:
            range_m = re.match(r"^(\d+)\s*-\s*(\d+)$", parts[1])
            if range_m:
                start, end = max(1, int(range_m.group(1))), max(1, int(range_m.group(2)))
            elif parts[1].isdigit():
                start = max(1, int(parts[1]))
                if len(parts) > 2 and parts[2].isdigit():
                    end = max(1, int(parts[2]))
        return Action(kind="READ", preamble=preamble, path=path, start=start, end=end, raw=raw_text)

    if keyword == "GREP":
        gm = re.match(r"^(.*?)(?:\s+in\s+(\S+))?$", rest_of_line, re.IGNORECASE)
        pattern = gm.group(1).strip().strip('"').strip("'") if gm else rest_of_line
        path = gm.group(2) if gm and gm.group(2) else "."
        if not pattern:
            raise ParseError("GREP needs a search term, e.g. GREP TODO in src")
        return Action(kind="GREP", preamble=preamble, pattern=pattern, path=path, raw=raw_text)

    if keyword == "RUN":
        # Everything after RUN is the shell command, verbatim. A fenced block
        # is accepted too, since models habitually wrap commands in ```.
        command = rest_of_line.strip()
        if not command:
            fenced, _ = _extract_fenced_block(after_cmd)
            command = (fenced or "").strip().splitlines()[0].strip() if fenced else ""
        if not command:
            raise ParseError("RUN needs a command, e.g. RUN pytest -q")
        command = command.strip("`").strip()
        return Action(kind="RUN", preamble=preamble, text=command, raw=raw_text)

    if keyword == "TODO":
        tm = re.match(r"^(ADD|DONE|LIST)\b\s*(.*)$", rest_of_line, re.IGNORECASE)
        if not tm:
            raise ParseError("TODO must be followed by ADD, DONE, or LIST")
        sub = tm.group(1).upper()
        arg = tm.group(2).strip()
        if sub == "ADD":
            if not arg:
                raise ParseError("TODO ADD needs text, e.g. TODO ADD add input validation")
            return Action(kind="TODO_ADD", preamble=preamble, text=arg, raw=raw_text)
        if sub == "DONE":
            if not arg.isdigit():
                raise ParseError("TODO DONE needs a numeric id, e.g. TODO DONE 2")
            return Action(kind="TODO_DONE", preamble=preamble, todo_id=int(arg), raw=raw_text)
        return Action(kind="TODO_LIST", preamble=preamble, raw=raw_text)

    if keyword == "FIX":
        # FIX <path> <what to change, in words> - the harness finds the lines.
        fm = re.match(r"^(\S+?)[:,]?(?:\s+(.*))?$", rest_of_line)
        if not fm:
            raise ParseError("FIX needs a path and what to change, e.g. FIX index.html make the header green")
        return Action(
            kind="FIX", preamble=preamble, path=fm.group(1),
            text=(fm.group(2) or "").strip(" :-") or None, raw=raw_text,
        )

    if keyword == "CREATE":
        # CREATE <path> <what it's for> - the harness asks for the content
        # in a prompt of its own. Small models say this unprompted when a
        # todo reads "Create app.js ...", so it's a command, not an error.
        cm = re.match(r"^(\S+?)[:,]?(?:\s+(.*))?$", rest_of_line)
        if not cm:
            raise ParseError("CREATE needs a path, e.g. CREATE app.js handle adding and deleting tasks")
        payload, truncated = _extract_fenced_block(after_cmd)
        if payload is not None:
            # it wrote the content anyway - that's an INSERT of a new file
            return Action(
                kind="INSERT", preamble=preamble, path=cm.group(1), start=0,
                payload=payload, payload_truncated=truncated, raw=raw_text,
            )
        return Action(
            kind="CREATE", preamble=preamble, path=cm.group(1),
            text=(cm.group(2) or "").strip(" :-") or None, raw=raw_text,
        )

    if keyword == "EDIT":
        parts = rest_of_line.split()
        if len(parts) < 2:
            raise ParseError("EDIT needs a path and a line range, e.g. EDIT src/app.py 12-14")
        path = parts[0]
        start, end = _parse_range(parts[1], "EDIT")
        payload, truncated = _extract_fenced_block(after_cmd)
        if payload is None:
            raise ParseError("EDIT must be followed by a ``` fenced code block with the replacement")
        return Action(
            kind="EDIT", preamble=preamble, path=path, start=start, end=end,
            payload=payload, payload_truncated=truncated, raw=raw_text,
        )

    if keyword == "INSERT":
        parts = rest_of_line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            raise ParseError("INSERT needs a path and a line number, e.g. INSERT src/app.py 0")
        path = parts[0]
        after_line = int(parts[1])
        payload, truncated = _extract_fenced_block(after_cmd)
        if payload is None:
            raise ParseError("INSERT must be followed by a ``` fenced code block with the new lines")
        return Action(
            kind="INSERT", preamble=preamble, path=path, start=after_line,
            payload=payload, payload_truncated=truncated, raw=raw_text,
        )

    if keyword == "DELETE":
        parts = rest_of_line.split()
        if len(parts) < 2:
            raise ParseError("DELETE needs a path and a line range, e.g. DELETE src/app.py 12-14")
        path = parts[0]
        start, end = _parse_range(parts[1], "DELETE")
        return Action(kind="DELETE", preamble=preamble, path=path, start=start, end=end, raw=raw_text)

    raise ParseError(f"unrecognized command '{keyword}'")


MAX_PLANNED_TODOS = 7


def parse_todo_adds(raw_text: str) -> list[str]:
    """Pull out `TODO ADD <text>` lines, used during planning.

    Small models degenerate into repeating the same line dozens of times;
    guard against that by stopping at MAX_PLANNED_TODOS and skipping
    consecutive duplicates rather than trusting the model's own "3-7 items"
    instruction to hold.
    """
    items: list[str] = []
    seen_lower: set[str] = set()
    for line in raw_text.splitlines():
        if len(items) >= MAX_PLANNED_TODOS:
            break
        m = re.match(r"^\s*TODO\s+ADD\s+(.+)$", line, re.IGNORECASE)
        if not m:
            continue
        text = m.group(1).strip()
        key = text.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        items.append(text)
    return items


FORMAT_REMINDER = (
    "ERROR: {error}\n"
    "Reply with exactly ONE command, as the first line of your response:\n"
    "  SAY <message>                 - talk to the user\n"
    "  LIST [path]                   - list files\n"
    "  READ <path> [start-end]       - view file lines (numbered)\n"
    "  GREP <text> [in <path>]       - search for text\n"
    "  RUN <command>                 - run a shell command\n"
    "  TODO ADD <text>               - add a todo\n"
    "  TODO DONE <id>                - complete a todo\n"
    "  TODO LIST                     - show todos\n"
    "  FIX <path> <what to change>   - change an existing file; say what in words\n"
    "  CREATE <path> <what it does>  - make a new file; say what in words\n"
    "  EDIT <path> <start>-<end>     - replace those lines, followed by a ``` block\n"
    "  INSERT <path> <line>          - insert after that line, followed by a ``` block\n"
    "  DELETE <path> <start>-<end>   - delete those lines\n"
    "  DONE                          - finished this task\n"
)


_RUN_DOC = """  RUN <command>                 run a shell command (output comes back to you)
"""

SYSTEM_PROMPT_TEMPLATE = """You are local-edit, a coding assistant in the project at {root}.

Reply with ONE command and STOP. You will be shown its real result, then you send the
next command. Never write the result yourself - never invent output or extra turns.

COMMANDS (your reply must start with one of these):
  SAY <message>               talk to the user (never put code here)
  LIST [path]                 list files
  READ <path> [start-end]     show numbered file lines
  GREP <text> [in <path>]     search
{run_doc}  TODO ADD <text> | TODO DONE <id> | TODO LIST
  FIX <path> <what to change> change an existing file - say WHAT in words, no code.
                              local-edit finds the right lines and asks you to rewrite them
  CREATE <path> <what it does> make a new file - say WHAT in words, no code.
                              local-edit asks you for its content next
  INSERT <path> <line>        insert after that line, then a ``` block
  EDIT <path> <start>-<end>   replace exactly those lines, then a ``` block
  DELETE <path> <start>-<end> delete those lines
  DONE                        task complete

RULES:
- New file: CREATE <path> <what it does>. Do not READ it first. Examples:
    CREATE style.css styles for the todo list
    CREATE app.js add, tick off and delete tasks
- New folder: just INSERT a file inside it (INSERT assets/style.css 0) - missing parent
  folders are created automatically. For an empty folder, RUN mkdir -p assets.
- If the request doesn't name a file, it is about a file that already exists (see
  "Project files") - change that file, don't create a new one.
- Existing file: use FIX, one change at a time. No need to READ it first. Examples:
    FIX index.html make the header background green
    FIX index.html add a footer with contact details
    FIX index.html remove the booking form
    FIX app.py handle an empty list in average()
  Two changes = two FIX commands, one after the other.
- Code goes only in an INSERT/EDIT ``` block. Code in a SAY is thrown away.
- Use relative paths (src/app.py).
- On ERROR, fix that specific thing. Never resend an identical failing command.
"""


def build_system_prompt(root: str, allow_run: bool = True) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(root=root, run_doc=_RUN_DOC if allow_run else "")
