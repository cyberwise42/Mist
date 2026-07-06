"""Builds the one-time onboarding panel written into the TUI's transcript
on mount — a bordered Rich Panel (not a pinned widget) so it scrolls away
naturally as the conversation grows, the same way Hermes/Claude Code's own
onboarding screens behave."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from mist.banner import BANNER, TAGLINE
from mist.tui.theme import ACCENT

if TYPE_CHECKING:
    from mist.core.agent import MistAgent

# The built-in tool set is small and fixed, so a hardcoded category map is
# simpler and more honest than inventing a data-driven taxonomy Mist's tools
# don't actually have (Tool only carries `keywords` for ranking, not a
# category).
TOOL_CATEGORIES: dict[str, str] = {
    "read_file": "file",
    "write_file": "file",
    "search_files": "file",
    "shell": "exec",
    "remember": "knowledge",
    "write_skill": "knowledge",
    "spawn_subagents": "orchestration",
    "finish_objective": "orchestration",
}


def _tool_category(name: str) -> str:
    return TOOL_CATEGORIES.get(name, "general")


def _grouped(items, key_fn, label_fn) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for it in items:
        groups.setdefault(key_fn(it), []).append(label_fn(it))
    return groups


def build_onboarding_panel(agent: "MistAgent") -> Panel:
    tools = agent.tools.all()
    skills = agent.skills.skills
    tool_groups = _grouped(tools, lambda t: _tool_category(t.name), lambda t: t.name)
    skill_groups = _grouped(skills, lambda s: s.category, lambda s: s.name)

    lines: list[Text] = [
        Text(BANNER, style=f"bold {ACCENT}", justify="center"),
        Text(TAGLINE, style="dim italic", justify="center"),
        Text(""),
        Text(f"model     {agent.cfg.backend} / {agent.cfg.model}"),
        Text(f"workspace {agent.cfg.workspace_root}"),
        Text(f"session   {agent.session_id}"),
    ]

    if tool_groups:
        lines.append(Text(""))
        lines.append(Text("Available Tools", style="bold"))
        for cat, names in sorted(tool_groups.items()):
            lines.append(Text(f"  {cat}: " + ", ".join(sorted(names))))

    if skill_groups:
        lines.append(Text(""))
        lines.append(Text("Available Skills", style="bold"))
        for cat, names in sorted(skill_groups.items()):
            lines.append(Text(f"  {cat}: " + ", ".join(sorted(names))))

    lines.append(Text(""))
    lines.append(Text(f"{len(tools)} tools · {len(skills)} skills · /help for commands",
                      style="dim"))

    return Panel(Group(*lines), border_style=ACCENT, box=box.ROUNDED, padding=(1, 2))
