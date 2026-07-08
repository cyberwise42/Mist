"""Pure (no-Textual) parsing/rendering for the `/memories` and `/history`
TUI commands — kept separate from `mist/tui/app.py` so the command grammar
is unit-testable without spinning up the Textual App.

Both commands share one shape: list (optionally filtered), `forget <id>`
for an immediate single-row delete, and `clear <pattern> [--yes]` for a
bulk delete — the first call is always a dry run (nothing deleted, just a
preview + count), and only a literal re-run with `--yes` appended actually
deletes. There is no dialog/modal system in this TUI (see `mist/tui/app.py`,
where `/kill` and `/new` both already act immediately once their state
check passes) — `--yes` is the only confirmation mechanism, by design.
"""
from __future__ import annotations

from mist.memory.store import MemoryStore

_PREVIEW_CHARS = 80
_CLEAR_PREVIEW_ROWS = 10


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def render_memories_command(store: MemoryStore, rest: str) -> str:
    parts = rest.split()
    if parts and parts[0] == "forget":
        if len(parts) != 2 or not parts[1].isdigit():
            return "Usage: /memories forget <id>"
        memory_id = int(parts[1])
        return (f"Deleted memory #{memory_id}." if store.forget_memory(memory_id)
                else f"No memory #{memory_id}.")
    if parts and parts[0] == "clear":
        args = parts[1:]
        yes = "--yes" in args
        pattern_parts = [a for a in args if a != "--yes"]
        if not pattern_parts:
            return "Usage: /memories clear <keyword> [--yes]"
        pattern = " ".join(pattern_parts)
        if yes:
            deleted = store.delete_memories_matching(pattern)
            return f"Deleted {deleted} memor{'y' if deleted == 1 else 'ies'} matching {pattern!r}."
        matches = store.list_memories(limit=10_000, like=pattern)
        if not matches:
            return f"No memories match {pattern!r}."
        lines = [f"{len(matches)} memories match {pattern!r}:"]
        for m in matches[:_CLEAR_PREVIEW_ROWS]:
            lines.append(f"  #{m['id']} [{m['kind']}] {_preview(m['content'])}")
        if len(matches) > _CLEAR_PREVIEW_ROWS:
            lines.append(f"  ... and {len(matches) - _CLEAR_PREVIEW_ROWS} more")
        lines.append(f'Re-run "/memories clear {pattern} --yes" to delete these {len(matches)} memories.')
        return "\n".join(lines)
    keyword = rest.strip() or None
    rows = store.list_memories(limit=20, like=keyword)
    if not rows:
        return "No memories match that keyword." if keyword else "No memories stored."
    lines = [f"#{m['id']} [{m['kind']}] {_preview(m['content'])}" for m in rows]
    return "\n".join(lines)


def render_history_command(store: MemoryStore, rest: str) -> str:
    parts = rest.split()
    if parts and parts[0] == "forget":
        if len(parts) != 2 or not parts[1].isdigit():
            return "Usage: /history forget <turn_id>"
        turn_id = int(parts[1])
        return (f"Deleted turn #{turn_id}." if store.delete_turn(turn_id)
                else f"No turn #{turn_id}.")
    if parts and parts[0] == "clear":
        args = parts[1:]
        yes = "--yes" in args
        args = [a for a in args if a != "--yes"]
        if not args or not args[0].isdigit():
            return "Usage: /history clear <session_id> [keyword] [--yes]"
        session_id = int(args[0])
        pattern = " ".join(args[1:]) or None
        if yes:
            deleted = store.delete_turns_matching(session_id, pattern)
            return f"Deleted {deleted} turn{'s' if deleted != 1 else ''} from session {session_id}."
        matches = store.list_turns(session_id, limit=10_000, like=pattern)
        if not matches:
            scope = f" matching {pattern!r}" if pattern else ""
            return f"No turns in session {session_id}{scope}."
        lines = [f"{len(matches)} turns in session {session_id}"
                 + (f" match {pattern!r}:" if pattern else ":")]
        for t in matches[:_CLEAR_PREVIEW_ROWS]:
            lines.append(f"  #{t['id']} [{t['role']}] {_preview(t['content'])}")
        if len(matches) > _CLEAR_PREVIEW_ROWS:
            lines.append(f"  ... and {len(matches) - _CLEAR_PREVIEW_ROWS} more")
        suffix = f" {pattern}" if pattern else ""
        lines.append(f'Re-run "/history clear {session_id}{suffix} --yes" '
                     f"to delete these {len(matches)} turns.")
        return "\n".join(lines)
    if parts and parts[0].isdigit():
        session_id = int(parts[0])
        keyword = " ".join(parts[1:]) or None
        rows = store.list_turns(session_id, limit=20, like=keyword)
        if not rows:
            return (f"No turns in session {session_id} match that keyword."
                     if keyword else f"No turns in session {session_id}.")
        return "\n".join(f"#{t['id']} [{t['role']}] {_preview(t['content'])}" for t in rows)
    sessions = store.list_sessions(limit=20)
    if not sessions:
        return "No sessions."
    lines = []
    for s in sessions:
        title = s["title"] or "(untitled)"
        flag = " [compacted]" if s["compacted"] else ""
        lines.append(f"#{s['id']} {title} — {s['turn_count']} turns{flag}")
    return "\n".join(lines)
