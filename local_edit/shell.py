"""Running shell commands on the model's behalf, safely.

A 1-3B model will confidently emit nonsense, so the guarantees here are
structural rather than trust-based:

  - Every command is classified BLOCKED / NEEDS_APPROVAL / SAFE before it
    runs. Only a conservative read-only allowlist is SAFE (auto-runnable);
    anything unrecognized needs a human yes, and a small denylist of
    catastrophic or outward-facing commands can't run at all.
  - stdin is /dev/null, so a command that waits for input fails fast instead
    of hanging the agent forever.
  - Everything runs with cwd pinned to the project root, under a wall-clock
    timeout, with output truncated before it reaches the model's context.

This is deliberately not a sandbox: it's a seatbelt around a tool the user
has asked to point at their own project.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

BLOCKED = "blocked"
NEEDS_APPROVAL = "needs_approval"
SAFE = "safe"

DEFAULT_TIMEOUT = 60
MAX_OUTPUT_CHARS = 4000
MAX_OUTPUT_LINES = 120

# Per-project override file: <project>/.local-edit/command_rules.json. Lets a
# team tune the classifier for their own toolchain without editing source.
# Schema (all keys optional):
#   {"allow_programs": ["mvn"], "deny_programs": ["ssh"],
#    "allow_subcommands": {"docker": ["build"]}}
# Overrides can only ADD to the built-in allow/deny lists, never remove from
# BLOCKED_PATTERNS - a project config is not allowed to re-enable `rm -rf`.
COMMAND_RULES_FILE = ".local-edit/command_rules.json"

# Commands that are read-only enough to run without asking every time.
SAFE_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "file", "stat", "du", "df",
    "find", "grep", "rg", "tree", "which", "echo", "date", "whoami", "uname",
    "diff", "sort", "uniq", "cut", "basename", "dirname", "realpath",
}
SAFE_SUBCOMMANDS = {
    "git": {"status", "diff", "log", "show", "branch", "remote", "ls-files", "rev-parse", "blame"},
    "npm": {"test", "run", "ls"},
    "cargo": {"test", "build", "check", "fmt"},
    "go": {"test", "build", "vet", "fmt"},
    "poetry": {"run", "show"},
    "pip": {"list", "show", "freeze"},
    "docker": {"ps", "images", "logs", "inspect", "version", "info"},
    "kubectl": {"get", "describe", "logs", "version", "explain", "config"},
}
# Test/build runners: safe to invoke, they're the whole point of verification.
SAFE_PROGRAMS = {"pytest", "python", "python3", "node", "make", "ruff", "mypy", "eslint", "tsc", "jest"}

# Patterns that never run, no matter who asks.
BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]", re.I), "recursive/forced delete"),
    (re.compile(r"\bmkfs\b", re.I), "filesystem format"),
    (re.compile(r"\bdd\b.*\bof=/dev/", re.I), "raw device write"),
    (re.compile(r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:", re.S), "fork bomb"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.I), "power state change"),
    (re.compile(r"\bsudo\b|\bsu\s", re.I), "privilege escalation"),
    (re.compile(r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/", re.I), "permissions wipe on a root path"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh", re.I), "piping a download into a shell"),
    (re.compile(r">\s*/dev/(sd|nvme|disk)", re.I), "raw disk write"),
    (re.compile(r"\brm\b[^|;&]*\s(/|~|\$HOME)\s*$", re.I), "delete of a root/home path"),
    (re.compile(r"\bgit\s+push\b", re.I), "pushing to a remote (outward-facing; run it yourself)"),
    (re.compile(r"\b(shred|wipefs)\b", re.I), "secure erase"),
    (re.compile(r"\bhistory\s+-c\b|\bunset\s+HISTFILE\b", re.I), "shell history tampering"),
    (re.compile(r"\bgit\s+reset\s+(--hard|--merge|--keep)\b", re.I), "discards uncommitted work irrecoverably"),
    (re.compile(r"\bgit\s+clean\s+(-[a-zA-Z]*f[a-zA-Z]*|--force)\b", re.I), "irrecoverably deletes untracked files"),
    (re.compile(r"\bdocker\s+system\s+prune\b", re.I), "irrecoverably deletes docker images/containers/volumes"),
    (re.compile(r"\bdocker\s+(rm|rmi)\s+(-[a-zA-Z]*f[a-zA-Z]*|--force)\b", re.I), "force-removes docker containers/images"),
    (re.compile(r"\bkubectl\s+delete\b.*(--all\b|--all-namespaces\b)", re.I), "deletes an entire class of cluster resources"),
]

# Shell metacharacters that chain extra commands - force approval so a safe
# looking prefix can't smuggle something else in behind it.
CHAINING = re.compile(r"[;&|]|\$\(|`|\n")


@dataclass
class CommandCheck:
    verdict: str
    reason: str = ""
    # True when the command names a path outside the project. Such a command
    # is never auto-run, not even under --auto-apply: the project root is the
    # boundary the user agreed to, and RUN must respect it like the file
    # tools do. (Observed: a model cd'd into a sibling directory and created
    # files there while the session was pointed somewhere else entirely.)
    outside_project: bool = False


_ABS_PATH_RE = re.compile(r"(?<![\w=])(~/|/)[^\s'\"();|&]*")


def _paths_outside_root(command: str, root: Path) -> list[str]:
    """Absolute paths in `command` that fall outside the project root."""
    root_resolved = root.resolve()
    outside = []
    for match in _ABS_PATH_RE.finditer(command):
        raw = match.group(0)
        candidate = Path(raw).expanduser()
        # Ignore system paths used as programs (/bin/sh, /usr/bin/python).
        if candidate.parts[:2] in ((os.sep, "bin"), (os.sep, "usr"), (os.sep, "opt"), (os.sep, "sbin")):
            continue
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root_resolved)
        except (ValueError, OSError):
            outside.append(raw)
    return outside


@dataclass
class CommandResult:
    command: str
    exit_code: int
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class CommandRules:
    allow_programs: set[str]
    deny_programs: set[str]
    allow_subcommands: dict[str, set[str]]


def load_command_rules(root: Path | None) -> CommandRules:
    """Read `<root>/.local-edit/command_rules.json`, if present.

    Malformed or missing config is silent and inert (empty rules) - a broken
    override file must never itself block or unblock anything.
    """
    empty = CommandRules(set(), set(), {})
    if root is None:
        return empty
    path = root / COMMAND_RULES_FILE
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    allow_programs = {str(p) for p in data.get("allow_programs", []) if isinstance(p, str)}
    deny_programs = {str(p) for p in data.get("deny_programs", []) if isinstance(p, str)}
    allow_subcommands = {
        str(prog): {str(s) for s in subs if isinstance(s, str)}
        for prog, subs in data.get("allow_subcommands", {}).items()
        if isinstance(subs, list)
    }
    return CommandRules(allow_programs, deny_programs, allow_subcommands)


def classify(command: str, root: Path | None = None) -> CommandCheck:
    """Decide whether a command may run, needs a human, or is refused."""
    cmd = (command or "").strip()
    if not cmd:
        return CommandCheck(BLOCKED, "empty command")

    for pattern, why in BLOCKED_PATTERNS:
        if pattern.search(cmd):
            return CommandCheck(BLOCKED, why)

    escapes: list[str] = []
    if root is not None:
        escapes = _paths_outside_root(cmd, root)
        if escapes:
            shown = ", ".join(escapes[:2])
            return CommandCheck(
                NEEDS_APPROVAL,
                f"writes outside the project: {shown}",
                outside_project=True,
            )

    if CHAINING.search(cmd):
        return CommandCheck(NEEDS_APPROVAL, "chains multiple commands")

    try:
        parts = shlex.split(cmd)
    except ValueError:
        return CommandCheck(NEEDS_APPROVAL, "could not parse quoting")
    if not parts:
        return CommandCheck(BLOCKED, "empty command")

    program = Path(parts[0]).name
    rules = load_command_rules(root)
    if program in rules.deny_programs:
        return CommandCheck(NEEDS_APPROVAL, f"{program} is on this project's deny list")
    if program in rules.allow_programs:
        return CommandCheck(SAFE, f"{program} is on this project's allow list")
    if program in SAFE_COMMANDS or program in SAFE_PROGRAMS:
        return CommandCheck(SAFE, "read-only or test/build command")
    subcommands = SAFE_SUBCOMMANDS.get(program, set()) | rules.allow_subcommands.get(program, set())
    if subcommands:
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub in subcommands:
            return CommandCheck(SAFE, f"{program} {sub} is read-only")
        return CommandCheck(NEEDS_APPROVAL, f"{program} subcommand is not on the read-only list")
    return CommandCheck(NEEDS_APPROVAL, "not a recognized read-only command")


def truncate_output(text: str) -> str:
    lines = text.splitlines()
    clipped = False
    if len(lines) > MAX_OUTPUT_LINES:
        head = lines[: MAX_OUTPUT_LINES // 2]
        tail = lines[-(MAX_OUTPUT_LINES // 2) :]
        lines = head + [f"... ({len(lines) - len(head) - len(tail)} lines omitted) ..."] + tail
        clipped = True
    out = "\n".join(lines)
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"
        clipped = True
    if clipped and not out.endswith("truncated)"):
        out += "\n(output truncated)"
    return out


def run(command: str, cwd: Path, timeout: int = DEFAULT_TIMEOUT) -> CommandResult:
    """Execute a command with cwd pinned, stdin closed, and a hard timeout.

    Callers are responsible for classify()-ing and getting approval first.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,  # never block waiting for input
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return CommandResult(
            command,
            exit_code=124,
            output=truncate_output(partial) + f"\n(timed out after {timeout}s)",
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(command, exit_code=127, output=f"failed to start: {exc}")

    return CommandResult(command, exit_code=proc.returncode, output=truncate_output(proc.stdout or ""))
