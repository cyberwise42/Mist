"""Two-tier skill system.

Tier 1: name + one-line description (cheap, always available to the router).
Tier 2: full SKILL.md body (loaded only when routed to).

Routing is deliberately dumb-and-cheap first: keyword overlap scoring. Small
models don't need (and can't afford) an LLM call just to pick a skill. Only
when lexical matching finds nothing do we fall back to a small embedding
model (e.g. bge-small) for a semantic pass — paraphrases and synonyms that
share no tokens with a skill's description still get routed correctly, at
the cost of one embedding call instead of zero.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mist.llm.embeddings import EmbeddingClient, cosine_similarity


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    category: str = "general"

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
    def __init__(self, library_path: str | Path,
                 embeddings: EmbeddingClient | None = None,
                 embedding_threshold: float = 0.35):
        self.library_path = Path(library_path)
        self.embeddings = embeddings
        self.embedding_threshold = embedding_threshold
        self.skills: list[Skill] = []
        self._skill_vecs: list[list[float]] | None = None
        self._load()

    def _load(self) -> None:
        self.skills = []
        if self.library_path.is_dir():
            for md in sorted(self.library_path.rglob("SKILL.md")):
                meta = _parse_frontmatter(md.read_text(encoding="utf-8"))
                self.skills.append(Skill(
                    name=meta.get("name", md.parent.name),
                    description=meta.get("description", ""),
                    path=md,
                    category=meta.get("category", "general"),
                ))

    def reload(self) -> None:
        """Re-scan the skills library and drop the stale embedding cache.
        Called after `write_skill` writes a new SKILL.md so it becomes
        routable in the same process, without a restart."""
        self._load()
        self._skill_vecs = None

    def _skill_text(self, skill: Skill) -> str:
        return f"{skill.name}: {skill.description}"

    def _ensure_skill_vecs(self) -> None:
        if self._skill_vecs is None:
            self._skill_vecs = self.embeddings.embed([self._skill_text(s) for s in self.skills])

    def _route_semantic(self, query: str, max_candidates: int) -> list[Skill]:
        """Embedding fallback used only when lexical matching finds nothing.
        Any error talking to the embedding backend just means no candidates —
        it must never take the whole turn down."""
        try:
            self._ensure_skill_vecs()
            [q_vec] = self.embeddings.embed([query])
        except Exception:
            return []
        scored = sorted(
            ((cosine_similarity(q_vec, vec), skill)
             for skill, vec in zip(self.skills, self._skill_vecs)),
            key=lambda pair: -pair[0],
        )
        return [skill for score, skill in scored[:max_candidates]
                if score >= self.embedding_threshold]

    def route(self, query: str, max_candidates: int = 4) -> list[Skill]:
        """Rank skills by keyword overlap with the query; return top candidates
        that score above zero. Falls back to semantic similarity (if an
        embedding client is configured) when no skill shares a keyword."""
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for skill in self.skills:
            hay = set(re.findall(r"[a-z0-9]+", (skill.name + " " + skill.description).lower()))
            score = len(q_tokens & hay)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        if scored:
            return [s for _, s in scored[:max_candidates]]
        if self.embeddings is not None and self.skills:
            return self._route_semantic(query, max_candidates)
        return []
