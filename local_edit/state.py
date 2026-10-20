"""Todo queue and session persistence.

Everything lives under <project>/.local-edit/ so a session (todos + chat
history) survives across restarts of the terminal app.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_BLOCKED = "blocked"


@dataclass
class TodoItem:
    id: int
    text: str
    status: str = STATUS_PENDING
    note: str = ""


MAX_TODOS = 40  # circuit breaker: a looping model must not be able to grow the queue without bound


class TodoStore:
    def __init__(self) -> None:
        self.items: list[TodoItem] = []
        self._next_id = 1

    def add(self, text: str) -> TodoItem | None:
        """Returns None (adds nothing) once MAX_TODOS is reached."""
        if len(self.items) >= MAX_TODOS:
            return None
        item = TodoItem(id=self._next_id, text=text)
        self._next_id += 1
        self.items.append(item)
        return item

    def get(self, todo_id: int) -> TodoItem | None:
        return next((t for t in self.items if t.id == todo_id), None)

    def mark(self, todo_id: int, status: str, note: str = "") -> TodoItem | None:
        item = self.get(todo_id)
        if item:
            item.status = status
            if note:
                item.note = note
        return item

    def next_pending(self) -> TodoItem | None:
        return next((t for t in self.items if t.status == STATUS_PENDING), None)

    def pending_count(self) -> int:
        return sum(1 for t in self.items if t.status == STATUS_PENDING)

    def render(self) -> str:
        if not self.items:
            return "(no todos)"
        marks = {STATUS_PENDING: " ", STATUS_IN_PROGRESS: "~", STATUS_DONE: "x", STATUS_BLOCKED: "!"}
        lines = [f"[{marks.get(t.status, '?')}] {t.id}. {t.text}" for t in self.items]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"items": [asdict(t) for t in self.items], "next_id": self._next_id}

    @classmethod
    def from_dict(cls, data: dict) -> "TodoStore":
        store = cls()
        store.items = [TodoItem(**d) for d in data.get("items", [])]
        store._next_id = data.get("next_id", len(store.items) + 1)
        return store


class Session:
    """Owns the todo store, chat history, and a small edit journal, persisted to disk."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state_dir = project_root / ".local-edit"
        self.backup_dir = self.state_dir / "backups"
        self.history: list[dict] = []  # chat messages excluding the system prompt
        self.todos = TodoStore()
        self.applied_edits: list[dict] = []  # journal for /undo

    @property
    def state_file(self) -> Path:
        return self.state_dir / "session.json"

    def load(self, include_history: bool = True) -> int:
        """Restore saved state. Returns how many saved chat messages there were.

        With include_history=False the chat starts empty but todos and the
        /undo journal still come back: a small model reading its own failed
        attempts from last time tends to repeat them, so a fresh launch
        shouldn't resume the conversation unless asked to.
        """
        if not self.state_file.exists():
            return 0
        try:
            data = json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return 0
        saved_history = data.get("history", [])
        self.history = saved_history if include_history else []
        self.todos = TodoStore.from_dict(data.get("todos", {}))
        self.applied_edits = data.get("applied_edits", [])
        return len(saved_history)

    def save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "history": self.history,
            "todos": self.todos.to_dict(),
            "applied_edits": self.applied_edits,
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def add_message(self, role: str, content: str) -> None:
        """Append a message, merging into the previous one if it has the same
        role. Several call sites can legitimately want to add two `user`
        messages back to back (a tool result immediately followed by a new
        task prompt); some chat templates (Gemma's, notably) hard-require
        strict user/assistant alternation and error out otherwise, so this
        keeps history alternating no matter what the caller does."""
        if self.history and self.history[-1]["role"] == role:
            self.history[-1]["content"] += "\n\n" + content
        else:
            self.history.append({"role": role, "content": content})

    def backup_and_record(self, rel: str, old_text: str, new_text: str, is_new_file: bool) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_name = rel.replace("/", "__")
        backup_path = self.backup_dir / f"{stamp}-{safe_name}.bak"
        backup_path.write_text(old_text)
        self.applied_edits.append(
            {
                "rel": rel,
                "backup": str(backup_path),
                "is_new_file": is_new_file,
                "time": stamp,
            }
        )

    def last_edit(self) -> dict | None:
        return self.applied_edits[-1] if self.applied_edits else None

    def pop_last_edit(self) -> dict | None:
        return self.applied_edits.pop() if self.applied_edits else None
