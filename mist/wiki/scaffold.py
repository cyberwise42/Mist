"""Deterministic, idempotent wiki skeleton creation.

Unlike the wiki's day-to-day ingest/query/lint mechanics (which live as
prose in skills_library/llm-wiki/SKILL.md and run over the model's own
read_file/write_file/search_files tool calls), the *skeleton* itself is
created here in Python rather than left to the model: getting directory
names, frontmatter conventions, and the starter tag taxonomy right and
consistent every time is exactly the kind of imperative, order-sensitive
bootstrap Mist already keeps out of model hands elsewhere (see
`write_skill` in mist/tools/registry.py, which writes real frontmatter
rather than asking the model to hand-write it).
"""
from __future__ import annotations

from pathlib import Path

SUBDIRS = ("raw", "entities", "concepts", "comparisons", "queries")

SCHEMA_TEMPLATE = """# Wiki schema

This file defines the conventions and tag taxonomy for this wiki. Read it
before creating or editing any page. If a new tag is needed, add it here
first, then use it — tags not listed below should not appear on a page.

## Directories

- `raw/` — untouched source dumps (recon output, tool logs, pasted text,
  fetched pages). Read but never modify these once written.
- `entities/` — a specific thing: a target host, a service, a CVE, a tool,
  a credential.
- `concepts/` — a technique or vulnerability class that generalizes across
  targets (not tied to one box).
- `comparisons/` — head-to-head notes (e.g. tool A vs tool B for a given
  job).
- `queries/` — substantial synthesized answers worth keeping, filed here
  rather than re-derived next time the same question comes up.

## Page frontmatter (required)

```
---
title: <short title>
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>
type: entity | concept | comparison | query
tags: [<from the taxonomy below>]
sources: [<raw/... paths this page draws from>]
---
```

Optional fields, add only when they apply:
- `confidence: high | medium | low` — set medium/low for opinion-heavy or
  fast-moving topics (e.g. an unconfirmed exploit chain).
- `contested: true` — set when a page has unresolved contradictions.
- `contradictions: [page-name, ...]` — pages this one conflicts with.

## Raw source frontmatter (required)

```
---
source_url: <url, or "pasted" for manually entered text>
ingested: <YYYY-MM-DD>
sha256: <sha256sum of the body, via the shell tool>
---
```

On re-ingesting the same source: recompute the sha256, compare to the
stored value — skip if identical, flag drift if different.

## Tag taxonomy (starter set — add more here as needed, don't invent tags
elsewhere)

- `target` — a specific host/box in scope
- `service` — a network service (name + version when known)
- `cve` — a specific CVE or vulnerability advisory
- `technique` — an exploitation or enumeration technique
- `tool` — a specific tool (nmap, gobuster, hashcat, ...)
- `credential` — a credential class or cracked/found credential note
- `network` — network topology, segmentation, or pivoting notes

## Page size

Split a page into sub-topics with cross-links once it exceeds ~200 lines.

## Cross-referencing

Every new or updated page should link to at least 2 other pages via
`[[wikilinks]]` where a genuine relationship exists.
"""

INDEX_TEMPLATE = """# Wiki index

Catalog of every page, grouped by type, one-line summary each. Keep
alphabetical within each section. Read this before creating any new page,
to avoid duplicates.

## Entities

## Concepts

## Comparisons

## Queries
"""

LOG_TEMPLATE = """# Wiki log

Append-only chronological record of wiki actions. Format:
`## [YYYY-MM-DD] action | subject`

Read the last 20-30 entries at the start of a session before touching the
wiki, to avoid repeating already-logged work.
"""


def init_wiki(root: Path) -> list[str]:
    """Create the wiki skeleton under `root` if it doesn't already exist.
    Never overwrites an existing file. Returns the paths actually created
    (relative to `root`), so callers can report what happened — an empty
    list means the wiki was already fully initialized."""
    root = Path(root)
    created: list[str] = []

    for sub in SUBDIRS:
        path = root / sub
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(f"{sub}/")

    for name, template in (
        ("SCHEMA.md", SCHEMA_TEMPLATE),
        ("index.md", INDEX_TEMPLATE),
        ("log.md", LOG_TEMPLATE),
    ):
        path = root / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(template, encoding="utf-8")
            created.append(name)

    return created
