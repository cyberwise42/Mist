"""Persistent input history, shared by `mist tui` and `mist chat`.

Plain one-entry-per-line text file — the same format GNU readline's own
history file uses, so `mist chat` can point `readline` straight at it and
get real up/down recall for free, sharing history with the TUI's own
recall/autofill (see mist.tui.app.HistoryInput) without any format
translation.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_MAX_ENTRIES = 500


class HistoryStore:
    def __init__(self, path: Path | str, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.path = Path(path).expanduser()
        self.max_entries = max_entries
        self._entries: list[str] = self._load()

    def _load(self) -> list[str]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [line for line in lines if line.strip()]

    def all(self) -> list[str]:
        """Oldest first, matching the on-disk order."""
        return list(self._entries)

    def add(self, entry: str) -> None:
        entry = entry.strip()
        if not entry:
            return
        # Re-submitting an entry moves it to the most-recent position instead
        # of leaving a stale duplicate earlier in the file.
        self._entries = [e for e in self._entries if e != entry]
        self._entries.append(entry)
        self._entries = self._entries[-self.max_entries:]
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self._entries) + "\n", encoding="utf-8")
