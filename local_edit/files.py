"""Path-safe, line-range file primitives.

Tiny local models can't reliably reproduce a whole file, so every edit is
expressed as a line range against a file the model was just shown (with
line numbers). Everything here is 1-indexed and inclusive, matching what
gets printed to the model.
"""

from __future__ import annotations

import contextlib
import difflib
import fnmatch
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_READ_LINES = 300
MAX_LIST_ENTRIES = 300
MAX_GREP_RESULTS = 60

IGNORE_DIRS = {".git", ".local-edit", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}


class PathError(ValueError):
    pass


def resolve(root: Path, rel: str) -> Path:
    """Resolve `rel` against `root`, refusing anything that escapes it."""
    rel = (rel or "").strip().strip('"').strip("'")
    if not rel or rel == ".":
        rel = "."
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise PathError(f"path '{rel}' is outside the project root") from None
    return candidate


def read_lines(root: Path, rel: str, start: int | None = None, end: int | None = None) -> str:
    path = resolve(root, rel)
    if not path.is_file():
        raise PathError(f"no such file: {rel}")
    lines = path.read_text(errors="replace").splitlines()
    n = len(lines)
    if start is None:
        start = 1
    if end is None:
        end = min(n, start + MAX_READ_LINES - 1)
    start = max(1, start)
    end = min(n, end)
    if start > n:
        return f"(file '{rel}' has only {n} lines)"
    if end - start + 1 > MAX_READ_LINES:
        end = start + MAX_READ_LINES - 1
    out = [f"{i:>5}: {lines[i - 1]}" for i in range(start, end + 1)]
    header = f"--- {rel} (lines {start}-{end} of {n}) ---"
    return header + "\n" + "\n".join(out)


def list_tree(root: Path, rel: str = ".", max_depth: int = 3) -> str:
    base = resolve(root, rel)
    if not base.exists():
        raise PathError(f"no such path: {rel}")
    if base.is_file():
        return str(base.relative_to(root.resolve()))

    lines: list[str] = []

    def walk(dir_: Path, depth: int) -> None:
        if len(lines) >= MAX_LIST_ENTRIES or depth > max_depth:
            return
        try:
            entries = sorted(dir_.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if len(lines) >= MAX_LIST_ENTRIES:
                # The walk is recursive: only the first level to hit the cap
                # adds the marker, not every ancestor on the way back up.
                if lines[-1] != "... (truncated)":
                    lines.append("... (truncated)")
                return
            if entry.name in IGNORE_DIRS:
                continue
            relpath = entry.relative_to(root.resolve())
            if entry.is_dir():
                lines.append(f"{relpath}/")
                walk(entry, depth + 1)
            else:
                lines.append(str(relpath))

    walk(base, 1)
    return "\n".join(lines) if lines else "(empty)"


def grep(root: Path, pattern: str, rel: str = ".", glob: str = "*") -> str:
    base = resolve(root, rel)
    pattern_lower = pattern.lower()
    hits: list[str] = []

    def search_file(path: Path) -> None:
        if len(hits) >= MAX_GREP_RESULTS:
            return
        try:
            text = path.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            return
        relpath = path.relative_to(root.resolve())
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern_lower in line.lower():
                hits.append(f"{relpath}:{i}: {line.strip()[:200]}")
                if len(hits) >= MAX_GREP_RESULTS:
                    return

    if base.is_file():
        search_file(base)
    else:
        for path in sorted(base.rglob("*")):
            if len(hits) >= MAX_GREP_RESULTS:
                break
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            if path.is_file() and fnmatch.fnmatch(path.name, glob):
                search_file(path)

    if not hits:
        return f"(no matches for '{pattern}')"
    return "\n".join(hits)


@dataclass
class EditResult:
    rel: str
    old_text: str
    new_text: str
    diff: str
    is_new_file: bool = False


def diff_texts(rel: str, old_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{rel}", tofile=f"b/{rel}")
    return "".join(diff)


def plan_replace(root: Path, rel: str, start: int, end: int, new_text: str) -> EditResult:
    path = resolve(root, rel)
    if not path.is_file():
        raise PathError(f"no such file: {rel}")
    old_text = path.read_text(errors="replace")
    lines = old_text.splitlines()
    n = len(lines)
    if start < 1 or end < start or end > n:
        raise PathError(f"line range {start}-{end} is out of bounds for '{rel}' ({n} lines)")
    replacement = new_text.splitlines()
    new_lines = lines[: start - 1] + replacement + lines[end:]
    new_full = "\n".join(new_lines) + ("\n" if old_text.endswith("\n") or not old_text else "\n")
    return EditResult(rel, old_text, new_full, diff_texts(rel, old_text, new_full))


def plan_insert(root: Path, rel: str, after_line: int, new_text: str) -> EditResult:
    path = resolve(root, rel)
    is_new = not path.exists()
    old_text = "" if is_new else path.read_text(errors="replace")
    lines = old_text.splitlines()
    n = len(lines)
    if after_line < 0 or after_line > n:
        raise PathError(f"insert point {after_line} is out of bounds for '{rel}' ({n} lines)")
    insertion = new_text.splitlines()
    new_lines = lines[:after_line] + insertion + lines[after_line:]
    new_full = "\n".join(new_lines) + "\n"
    return EditResult(rel, old_text, new_full, diff_texts(rel, old_text, new_full), is_new_file=is_new)


def plan_delete(root: Path, rel: str, start: int, end: int) -> EditResult:
    path = resolve(root, rel)
    if not path.is_file():
        raise PathError(f"no such file: {rel}")
    old_text = path.read_text(errors="replace")
    lines = old_text.splitlines()
    n = len(lines)
    if start < 1 or end < start or end > n:
        raise PathError(f"line range {start}-{end} is out of bounds for '{rel}' ({n} lines)")
    new_lines = lines[: start - 1] + lines[end:]
    new_full = "\n".join(new_lines) + ("\n" if new_lines else "")
    return EditResult(rel, old_text, new_full, diff_texts(rel, old_text, new_full))


def write(root: Path, rel: str, new_text: str) -> Path:
    """Write `new_text` to `rel`, atomically.

    Writes to a temp file in the same directory first, then renames it over
    the target. `os.replace` is atomic on both POSIX and Windows, so a
    process killed mid-write (or a crash) can never leave the file half
    written - the target either has the old content or the new content,
    never a mix.
    """
    path = resolve(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(new_text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path
