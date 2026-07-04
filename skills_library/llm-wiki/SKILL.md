---
name: llm-wiki
description: Build, ingest into, query, or audit the wiki knowledge base — start a wiki, add a source, what do we know about a target or CVE or tool, check the wiki, audit the wiki
---

# The llm-wiki: a compounding knowledge base

This is a meta-skill: it describes a *process*, not domain content. The
wiki lives under the configured wiki root (run `mist wiki-init` once to
create it if `SCHEMA.md` doesn't exist yet) and is reached with the
`read_file`, `write_file`, and `search_files` tools — there is no dedicated
wiki API, these are the same generic tools used everywhere else in Mist.

This is distinct from the skill library (see the `pentest-wiki` skill): a
wiki page records a *fact* about a specific target, CVE, tool, or finding —
it is not auto-loaded into context, you consult it on demand. A skill
records a reusable *procedure* and is auto-loaded by keyword match. If a
technique is confirmed to work, write it as a skill; if it's a fact worth
remembering about this engagement, put it in the wiki.

## Mandatory startup orientation

Before creating or editing anything in the wiki, in this order:
1. `read_file` `SCHEMA.md` — the conventions and tag taxonomy.
2. `read_file` `index.md` — what pages already exist.
3. `read_file` `log.md` — at least the last 20-30 entries.

Skipping this causes duplicate pages, missed cross-references,
contradicting the schema's conventions, and repeating already-logged work.

## Ingesting a source

1. Capture the raw material into `raw/` in this exact order, to avoid
   shell-quoting the body text (never pass body text inline in a `shell`
   command — only file paths):
   a. `write_file` the raw body only (no frontmatter yet) to its `raw/`
      path — `shell` (`curl -o <path> <url>`) for a URL, `write_file`
      directly for pasted text or recon output.
   b. `shell` `sha256sum <path>` on that file to get its hash.
   c. `write_file` the same path again, this time prepending frontmatter
      (`source_url`, `ingested` date, `sha256` from step b) above the body.
2. On re-ingesting the same source: recompute the sha256 and compare to the
   stored value. Skip if identical. If different, flag the drift on the
   page(s) that cite it rather than silently overwriting.
3. `search_files` the wiki for pages already covering the entities/concepts
   this source mentions, before creating new ones.
4. Only create or update a page when an entity/concept has **2+ source
   mentions, or is central to this one source**. Skip passing mentions and
   minor details.
5. Every new or updated page must link to at least 2 related pages via
   `[[wikilinks]]` where a genuine relationship exists.
6. Add new pages to `index.md`, alphabetically within their section.
7. Append a line to `log.md`: `## [YYYY-MM-DD] ingest | <source title>`.

Split a page into sub-topics with cross-links once it passes ~200 lines.

## Answering a question from the wiki

1. `read_file` `index.md` to find candidate pages.
2. If the wiki has grown large, `search_files` across it for key terms too.
3. `read_file` the relevant pages and synthesize an answer, citing which
   pages it came from.
4. If the answer is substantial (not a trivial one-line lookup), file it
   into `queries/` or `comparisons/` so it doesn't need re-deriving next
   time.
5. Append the query and filing status to `log.md`.

## Auditing the wiki

Check for, and report grouped by severity (broken links > orphans > source
drift > contested pages > stale content > style issues):
1. Broken `[[wikilinks]]` — links pointing at pages that don't exist.
2. Orphan pages — `search_files` for `[[page-name]]`; zero hits means
   nothing links to it.
3. `index.md` entries that don't match what's actually on disk, or pages
   missing from `index.md`.
4. Frontmatter problems: missing required fields, tags not in `SCHEMA.md`'s
   taxonomy.
5. Stale pages — `updated` much older than sources that mention the same
   entity/concept.
6. Pages with `contested: true` or a `contradictions:` field.
7. Source drift — sha256 mismatch on a `raw/` file since it was last cited.
8. Pages over ~200 lines (split candidates).

## Handling contradictions

When new information conflicts with an existing page: newer sources
generally supersede older ones by date. If it's genuinely contradictory
rather than just an update, record both positions with their dates and
sources, set `contradictions: [other-page-name]` in frontmatter on both
pages, and flag it for the user rather than silently picking one.

## What NOT to do

- Don't create a page for a tag not yet in `SCHEMA.md`'s taxonomy — add the
  tag to `SCHEMA.md` first.
- Don't skip the startup orientation because the wiki "seems small."
- Don't put target-specific secrets in raw exploit content across engagements
  where that would be a scoping violation — use judgment on what's meant to
  generalize versus what's engagement-specific.
