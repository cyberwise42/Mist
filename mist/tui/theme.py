"""Mist's TUI color theme: a dark navy background with an amber accent,
in the spirit of Hermes/Claude Code's own terminal look.

Hex constants are exported alongside the Textual `Theme` because Rich
markup (used for everything written into the `#transcript` RichLog, and
for the onboarding `Panel`'s border) has no notion of Textual's `$accent`-
style CSS variables — Rich only understands literal color names/hex codes
in its own markup tags (`[#f0a860]...[/]`). Both need the same literal
values so the transcript text and the CSS-styled chrome never drift apart.
"""
from __future__ import annotations

from textual.theme import Theme

ACCENT = "#f0a860"
SUCCESS = "#4ade80"
WARNING = "#f5c26b"
ERROR = "#ff6b6b"

MIST_THEME = Theme(
    name="mist-dark",
    primary="#f0a860",
    secondary="#7dd3fc",
    accent=ACCENT,
    warning=WARNING,
    error=ERROR,
    success=SUCCESS,
    foreground="#e5e9f0",
    background="#0b0f1a",
    surface="#121826",
    panel="#182234",
    boost="#1c2740",
    dark=True,
)
