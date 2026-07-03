"""SQLite-backed memory and session store.

Retrieval over recall: the agent never depends on the model remembering.
FTS5 gives cheap, dependency-free lexical search that works well when the
query terms come from the current user message.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    title TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,          -- user | assistant | tool
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    kind TEXT DEFAULT 'note',    -- note | fact | preference
    created_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(content, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;
"""


class MemoryStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- sessions ------------------------------------------------------
    def new_session(self, title: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (created_at, title) VALUES (?, ?)",
            (time.time(), title),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_turn(self, session_id: int, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self.conn.commit()

    def recent_turns(self, session_id: int, limit: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- memories ------------------------------------------------------
    def remember(self, content: str, kind: str = "note") -> int:
        cur = self.conn.execute(
            "INSERT INTO memories (content, kind, created_at) VALUES (?, ?, ?)",
            (content, kind, time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """FTS5 search; falls back to recency if the query has no usable terms."""
        terms = " OR ".join(
            t for t in "".join(c if c.isalnum() else " " for c in query).split()
            if len(t) > 2
        )
        if terms:
            try:
                rows = self.conn.execute(
                    "SELECT m.content FROM memories_fts f "
                    "JOIN memories m ON m.id = f.rowid "
                    "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                    (terms, top_k),
                ).fetchall()
                if rows:
                    return [r["content"] for r in rows]
            except sqlite3.OperationalError:
                pass
        rows = self.conn.execute(
            "SELECT content FROM memories ORDER BY id DESC LIMIT ?", (top_k,)
        ).fetchall()
        return [r["content"] for r in rows]
