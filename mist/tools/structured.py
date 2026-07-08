"""Tier 2 of Mist's tool-output handling: deterministic structured
extraction for the recon/scanning tools that actually blow the char budget
(nmap, nuclei, gobuster, ffuf, searchsploit).

This runs *before* the head+tail truncation in `_combine_output` — the goal
is to keep the real signal (port tables, findings, hit lines) and drop the
high-volume boilerplate (banners, progress bars, raw fingerprint blobs)
deterministically, rather than leaving it to a truncation cut that doesn't
know which half of the text matters. Falls back to `None` for anything
unrecognized, which lets the caller fall through to today's plain
truncation (backed by tier 1's full-output artifact as a safety net).

Confirmed against real incidents this was built to fix: an `nmap` scan's
`fingerprint-strings`/`SF:` submission blob crowded out the actual port
table under truncation; an `nuclei` scan's startup banner ate the whole
budget before a critical finding line ever appeared; an `ffuf` run lost
progress-bar spam that hid the real signal — the target's Next.js
catch-all page returning an identical response size for virtually every
path tried, meaning the useful information was "this target returns
200-with-identical-size for everything, exclude that size and re-scan,"
not a shorter list of the same false hits.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_TOOL_LEADING_WORD_RE = re.compile(r"^\s*(?:sudo\s+)?(\S+)")
_KNOWN_TOOLS = {"nmap", "nuclei", "gobuster", "ffuf", "searchsploit"}


def detect_tool(command: str) -> str | None:
    """Sniffs the leading word of a shell command (tolerant of a `sudo`
    prefix and a full path like `/usr/bin/nmap`). Callers on the SSH shell
    backend must pass the *pre-wrap* command — before `_shell_ssh` splices
    in its own `cd ... &&` prefix — or this would sniff `cd` instead."""
    m = _TOOL_LEADING_WORD_RE.match(command)
    if not m:
        return None
    name = Path(m.group(1)).name
    return name if name in _KNOWN_TOOLS else None


def summarize_tool_output(tool: str | None, stdout: str, stderr: str) -> str | None:
    """Dispatches to a per-tool summarizer. Returns None for an
    unrecognized (or no) tool, or for a tool that's already compact enough
    that summarizing it would be pure overhead (searchsploit)."""
    if tool == "nmap":
        return _summarize_nmap(stdout, stderr)
    if tool == "nuclei":
        return _summarize_nuclei(stdout, stderr)
    if tool in ("gobuster", "ffuf"):
        return _summarize_content_discovery(stdout, stderr)
    return None


_SF_LINE_RE = re.compile(r"^SF:")


def _summarize_nmap(stdout: str, stderr: str) -> str:
    """Keeps the port/service/version table and script-output headers;
    drops nmap's `fingerprint-strings`/`SF:` submission blob, whose raw HTTP
    dump is a nmap.org-submission artifact, not something a caller ever
    acts on — and was confirmed live to crowd out the real port table under
    truncation."""
    kept: list[str] = []
    dropped = 0
    for line in stdout.splitlines():
        if _SF_LINE_RE.match(line):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        kept.append(f"[... {dropped} lines of raw fingerprint-strings (SF:) submission "
                    "data omitted — service/version info above is already extracted ...]")
    result = "\n".join(kept)
    if stderr.strip():
        result += f"\n[stderr]\n{stderr}"
    return result


# nuclei's real finding lines look like `[template-id] [protocol] [severity]
# matched-at` — 2+ bracketed groups right at the start of the line. Its
# informational/banner lines (`[INF] ...`, template-loading chatter) only
# ever have a single bracketed tag before free text, so this distinguishes
# them without needing to match nuclei's exact severity vocabulary.
_NUCLEI_FINDING_RE = re.compile(r"^(?:\[[^\]]+\]\s*){2,}\S")
_NUCLEI_SUMMARY_RE = re.compile(r"\d+\s+(?:matches|results?)\s+found", re.IGNORECASE)


def _summarize_nuclei(stdout: str, stderr: str) -> str | None:
    lines = [l for l in stdout.splitlines() if l.strip()]

    # -json/-jsonl output: each line is a JSON object. Parse directly for a
    # complete, not-truncated list of every finding regardless of count.
    json_objs = []
    for line in lines:
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            json_objs.append(obj)
    if json_objs and len(json_objs) >= max(1, len(lines) // 2):
        parts = []
        for obj in json_objs:
            tid = obj.get("template-id") or obj.get("templateID") or "?"
            severity = (obj.get("info") or {}).get("severity", "?")
            matched = obj.get("matched-at") or obj.get("host") or "?"
            parts.append(f"[{tid}] [{severity}] {matched}")
        return "\n".join(parts)

    # Plain-text output: bracketed finding lines plus any "N matches found"
    # style summary, dropping banner/template-loading chatter.
    findings = [l for l in lines if _NUCLEI_FINDING_RE.match(l)]
    summary_lines = [l for l in lines if _NUCLEI_SUMMARY_RE.search(l)]
    if not findings and not summary_lines:
        # Nothing recognizable extracted — safer to fall back to today's
        # truncation than to claim "no findings" when this simply failed to
        # parse nuclei's actual output shape.
        return None
    result = "\n".join(findings + summary_lines)
    if stderr.strip():
        result += "\n[stderr omitted: banner/template-loading chatter]"
    return result


_HIT_STATUS_RE = re.compile(r"Status:\s*(\d+)")
_HIT_SIZE_RE = re.compile(r"Size:\s*(\d+)")


def _summarize_content_discovery(stdout: str, stderr: str) -> str | None:
    """gobuster/ffuf: keep `Status:` hit lines, drop `Progress:`/`Duration:`
    lines unconditionally. Additionally: if most hits share an identical
    response size, that's very likely a catch-all/SPA route responding 200
    to everything, not real content discovery — surface that explicitly
    instead of a wall of near-identical false positives (confirmed live: an
    ffuf run against a Next.js catch-all page returned Status 200, an
    identical Size, for virtually every path tried)."""
    hits = [l for l in stdout.splitlines() if _HIT_STATUS_RE.search(l)]
    if not hits:
        return None

    sizes = [m.group(1) for l in hits if (m := _HIT_SIZE_RE.search(l))]
    parts = hits
    if sizes and len(sizes) >= 5:
        common_size, count = Counter(sizes).most_common(1)[0]
        if count / len(sizes) > 0.8:
            statuses = [m.group(1) for l in hits if (m := _HIT_STATUS_RE.search(l))]
            common_status = Counter(statuses).most_common(1)[0][0] if statuses else "?"
            distinct = [l for l in hits
                       if not (m := _HIT_SIZE_RE.search(l)) or m.group(1) != common_size]
            parts = [
                f"{count}/{len(sizes)} paths returned Status {common_status}, Size "
                f"{common_size} (likely a catch-all/SPA route — re-scan with "
                f"-fs {common_size} to exclude it).",
            ]
            if distinct:
                parts.append(f"Remaining {len(distinct)} hit(s) not matching that size:")
                parts.extend(distinct[:30])

    result = "\n".join(parts)
    if stderr.strip():
        result += "\n[stderr omitted: progress/banner chatter]"
    return result
