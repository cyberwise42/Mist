"""Pure formatting helpers for the TUI's status line and tool-call display —
no Textual imports, so these are directly unit-testable without a pilot."""
from __future__ import annotations

import json


def format_tool_call(tool: str, detail: str, max_len: int = 64) -> str:
    """Renders a compact `tool(key=value, ...)` summary instead of dumping
    `detail`'s raw JSON — falls back to `tool(...)` on anything unparseable.
    `detail`'s contract (a JSON args string, see agent.py's TurnEvent) is
    unchanged; this is a display-only transform.

    A `shell` tool's `command` is shown IN FULL (never elided): an operator
    monitors Mist by reading this feed, and a truncated command hides what is
    actually being run against the target. The TUI log wraps (wrap=True), so a
    long command spans lines cleanly. Every other tool/arg stays compact so the
    feed isn't flooded by e.g. a large write_file body."""
    try:
        args = json.loads(detail) if detail else {}
    except (ValueError, TypeError):
        args = None
    if not isinstance(args, dict):
        return f"{tool}(...)"
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", " ")
        show_full = tool == "shell" and k == "command"
        if not show_full and len(s) > 40:
            s = s[:37] + "…"
        parts.append(f"{k}={s!r}" if isinstance(v, str) else f"{k}={s}")
    inner = ", ".join(parts)
    if tool != "shell" and len(inner) > max_len:  # shell commands are never capped
        inner = inner[: max_len - 1] + "…"
    return f"{tool}({inner})"


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes:.0f}m{rest:02.0f}s"


def format_status(*, state: str, elapsed: float | None, model: str,
                  used_tokens: int, budget: int, session_id: int,
                  queued: int = 0) -> str:
    """One `·`-separated status line: state (+ live elapsed time while a
    turn/mission runs), model, real context-token usage (from
    MistAgent.last_context_tokens — not a synthetic estimate), session id,
    and a queued-message count when non-zero."""
    pct = min(100, round(100 * used_tokens / budget)) if budget else 0
    head = state
    if elapsed is not None:
        head = f"{state} {_format_elapsed(elapsed)}"
    parts = [head, model, f"ctx {pct}% ({used_tokens}/{budget})", f"session {session_id}"]
    if queued:
        parts.append(f"{queued} queued")
    return "  ·  ".join(parts)
