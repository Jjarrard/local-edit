"""Persisted user preferences (model choices, toggles).

Lives in ~/.local-edit/config.json rather than in the project, because which
models exist is a property of the machine running Jan, not of the code being
edited. Every read is defensive: a corrupt or hand-edited config degrades to
defaults instead of crashing the app on startup.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def config_dir() -> Path:
    override = os.environ.get("LOCAL_EDIT_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local-edit"


def migrate_legacy_dir(new: Path) -> None:
    """Move a `.janedit` dir (this tool's old name) to `new`, once.

    Leaves both alone if `new` already exists, so nothing is ever overwritten.
    """
    old = new.with_name(".janedit")
    if old.is_dir() and not new.exists():
        try:
            old.rename(new)
        except OSError:
            pass


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass
class Config:
    """User preferences. Field names double as the on-disk keys."""

    code_model: str | None = None       # the capable model: writes and edits code
    fast_model: str | None = None       # the cheap model: planning, review, command work
    base_url: str = "http://127.0.0.1:1338/v1"
    auto_apply: bool = False
    self_review: bool = True
    allow_run: bool = True
    recent_models: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "Config":
        path = config_path()
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in known}
        try:
            cfg = cls(**clean)
        except TypeError:
            return cls()
        if not isinstance(cfg.recent_models, list):
            cfg.recent_models = []
        return cfg

    def save(self) -> None:
        path = config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2))
            tmp.replace(path)  # atomic: a crash mid-write can't corrupt the config
        except OSError:
            pass  # preferences are a convenience, never worth crashing over

    def remember_model(self, model: str) -> None:
        if not model:
            return
        recents = [m for m in self.recent_models if m != model]
        recents.insert(0, model)
        self.recent_models = recents[:10]

    def order_models(self, available: list[str]) -> list[str]:
        """Most-recently-used first, so the picker opens on what you actually use."""
        recent = [m for m in self.recent_models if m in available]
        rest = sorted(m for m in available if m not in recent)
        return recent + rest
