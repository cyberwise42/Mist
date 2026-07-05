"""Mission mode: autonomous, multi-turn pursuit of a single objective.

A mission repeatedly drives ``MistAgent.astream_turn`` — each "respond" is
treated as an interim status update, not a stopping point, and the loop
synthesizes the next turn itself until the model calls the ``finish_objective``
tool, a safety limit is hit, or the operator intervenes. ``MissionControl`` is
the cooperative handle an operator-facing surface (the TUI) uses to pause,
resume, or steer a mission in flight; killing one outright is done by
cancelling the asyncio task driving it (see mist.tui.app), since asyncio has
no native way to pause a task but cancellation works everywhere.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class MissionControl:
    """Mutable control surface shared between a running mission and its
    driver. Pausing is cooperative (checked between turns, never mid-turn) so
    an in-flight tool call always finishes cleanly rather than being cut off."""
    paused: bool = False
    notes: list[str] = field(default_factory=list)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def pop_notes(self) -> list[str]:
        notes, self.notes = self.notes, []
        return notes

    async def wait_while_paused(self, poll_interval: float = 0.25) -> None:
        while self.paused:
            await asyncio.sleep(poll_interval)


@dataclass
class MissionEvent:
    """One step of a running mission, as emitted by ``MistAgent.astream_mission``.

    kind: forwards every ``TurnEvent`` kind ("delta", "tool_start",
    "tool_result", "done", "error") as-is, plus mission-level kinds:
    "turn_start" (a new autonomous turn began), "paused", "resumed",
    "stuck" (repeated the same tool call — auto-paused for review),
    "finished" (objective complete or a limit was hit).
    """
    kind: str
    text: str = ""
    tool: str = ""
    detail: str = ""
