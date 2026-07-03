"""Two-tier skill system.

Tier 1: name + one-line description (cheap, always available to the router).
Tier 2: full SKILL.md body (loaded only when routed to).

Routing is deliberately dumb-and-cheap: keyword overlap scoring. Small models
don't need (and can't afford) an LLM call just to pick a skill. Swap in an
embedding scorer later if lexical routing proves insufficient.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    path: Path

    def body(self) -> str:
        return self.path.read_text(encoding="utf-8")


_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = _FRONT.match(text)
    meta: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


class SkillRouter:
    def __init__(self, library_path: str | Path):
        self.skills: list[Skill] = []
        root = Path(library_path)
        if root.is_dir():
            for md in sorted(root.rglob("SKILL.md")):
                meta = _parse_frontmatter(md.read_text(encoding="utf-8"))
                self.skills.append(Skill(
                    name=meta.get("name", md.parent.name),
                    description=meta.get("description", ""),
                    path=md,
                ))

    def route(self, query: str, max_candidates: int = 4) -> list[Skill]:
        """Rank skills by keyword overlap with the query; return top candidates
        that score above zero."""
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for skill in self.skills:
            hay = set(re.findall(r"[a-z0-9]+", (skill.name + " " + skill.description).lower()))
            score = len(q_tokens & hay)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:max_candidates]]
