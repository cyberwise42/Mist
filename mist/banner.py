"""Shared welcome banner and command reference for Mist's interactive
surfaces (`mist chat` and `mist tui`). Kept dependency-free (no textual, no
rich) so both the plain console loop and the Textual app can import it."""
from __future__ import annotations

BANNER = r"""
   __  _________________
  /  |/  /  _/ __/_  __/
 / /|_/ // /_\ \  / /
/_/  /_/___/___/ /_/
""".strip("\n")

TAGLINE = "context-frugal agent for penetration testing & security research"

COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help message"),
    ("/clear", "Clear the visible transcript"),
    ("/new", "Start a fresh session (new memory context)"),
    ("/models", "List models available on the backend"),
    ("/model [name]", "Show the active model, or switch to <name>"),
    ("/mission <goal>", "Work an objective autonomously until done (TUI only)"),
    ("/pause", "Pause the running mission after its current step"),
    ("/resume", "Resume a paused mission"),
    ("/kill", "Stop the running mission immediately (Ctrl+C also works)"),
    ("/compact", "Fold old sessions into long-term memory now"),
    ("/memories [kw]", "List memories, optionally filtered; forget/clear to delete"),
    ("/history [id]", "List sessions or a session's turns; forget/clear/rename/export"),
    ("/quit, /exit", "Quit Mist"),
]


def help_lines() -> list[str]:
    width = max(len(cmd) for cmd, _ in COMMANDS)
    return [f"{cmd.ljust(width)}   {desc}" for cmd, desc in COMMANDS]
