"""Diagnose a wedged mission.

Two mission runs stalled with an identical, hard-to-reproduce signature: the
process went fully idle mid-turn (event loop parked, worker threads idle, no
subprocess, no in-flight LLM call) after a tool had already run — its result
was produced but never propagated, and the mission simply stopped advancing
with no pause/error marker. A plain `py-spy dump` can't show *which* suspended
coroutine is parked, so the exact frame was never captured.

This module is the instrument for catching it red-handed. `mission_stall_
watchdog` runs as its OWN asyncio task — so it keeps firing even while every
mission coroutine is parked — and when the mission stream goes silent past a
threshold it calls back to dump every pending task's stack (via
`format_pending_task_stacks`) into the mission log. The next stall then names
the parked coroutine and its exact await, turning "reproduce and guess" into a
concrete fix. It's diagnostic-only: it never cancels or alters the mission.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Iterable


def format_pending_task_stacks(tasks: Iterable[asyncio.Task]) -> str:
    """Format the suspended stack of every not-done task — for each, the coro's
    qualified name and the file:line/qualname of every frame it's parked in.
    This is exactly what identifies where a wedged mission is stuck. Wrapped so
    a problem reading any one task can never crash the watchdog that calls it."""
    out: list[str] = []
    for t in tasks:
        try:
            if t.done():
                continue
            coro = t.get_coro()
            code = getattr(coro, "cr_code", None)
            qual = getattr(code, "co_qualname", None) or repr(coro)
            out.append(f"--- task {t.get_name()!r}  ({qual}) ---")
            frames = t.get_stack(limit=30)
            if not frames:
                out.append("    (suspended with no python frames)")
            for f in frames:
                out.append(f"    {f.f_code.co_filename}:{f.f_lineno}  {f.f_code.co_qualname}")
        except Exception as exc:  # diagnostics must never take down the loop
            out.append(f"--- task <unreadable: {exc!r}> ---")
    return "\n".join(out) if out else "(no pending tasks)"


def effective_stall_threshold(configured: float, shell_timeout: float,
                              margin: float = 120.0) -> float:
    """A tool can legitimately run right up to the shell timeout with no mission
    event in between (a slow gobuster/nuclei), so a watchdog threshold BELOW
    that timeout guarantees false-positive "stall" dumps on healthy long scans.
    Floor the configured value to `shell_timeout + margin` so the watchdog can
    only fire once a turn has been silent LONGER than any tool could legitimately
    take — i.e. a genuine wedge. 0 (disabled) passes through unchanged."""
    if not configured or configured <= 0:
        return 0.0
    return max(configured, shell_timeout + margin)


async def mission_stall_watchdog(
    get_last_progress: Callable[[], float],
    on_stall: Callable[[float], None],
    *,
    threshold: float = 90.0,
    poll_interval: float = 15.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Poll `get_last_progress()` — a monotonic timestamp the mission consumer
    updates on every event. When the gap since the last progress first crosses
    `threshold`, call `on_stall(idle_seconds)` exactly ONCE for that stall
    episode (not once per poll), then re-arm only after progress resumes. Runs
    forever until cancelled; `now`/`sleep` are injectable for deterministic
    tests."""
    armed = True
    while True:
        await sleep(poll_interval)
        idle = now() - get_last_progress()
        if idle >= threshold:
            if armed:
                on_stall(idle)
                armed = False
        else:
            armed = True
