---
name: htb-engagement
description: HTB HackTheBox machine box CTF engagement rules and deliverables - complete a HackTheBox target, follow HTB rules of engagement, produce an HTB skills writeup and fine-tuning dataset
---

# HTB machine engagement

Rules and deliverables for an HTB (HackTheBox) machine objective — applies whenever the
mission names an HTB/HackTheBox machine and target IP.

## Workspace

Create/use `~/Desktop/HTB/<machine-name-or-IP>/` for all scans, notes, exploit code, and
downloads for this box.

## Rules of engagement

- No live/online brute-forcing (Hydra, Medusa, ffuf against login forms, password spraying).
  Offline hash cracking (John, Hashcat) is fine once hashes are obtained.
- If a vhost/hostname is discovered, add it to `/etc/hosts` before continuing.
- Do not look up walkthroughs or writeups for this box online — solve it directly.

## Continuous documentation (regardless of chat updates)

Maintain `~/Desktop/HTB/<machine>/notes.md`, appended to as you go: every command run
(verbatim) with a one-line rationale, full raw output of key commands, each decision point
(tried/worked/failed and why), exact exploit payloads/scripts (with source if adapted from a
public PoC), the privesc vector and how you found it, and any credentials/hashes/secrets
discovered, timestamped in order. This log is the source material for both deliverables below —
be thorough.

## Chat updates (minimal — 3-5 sentences max, no raw command dumps)

After initial recon/scanning completes; after a foothold vector is identified (before
exploiting); upon user.txt obtained; after a privesc vector is identified (before exploiting);
upon root.txt obtained; final summary when both deliverables are written.

## Deliverable 1 — Skills file: `htb/skills/<machine-name>.md`

Generalized, reusable writeup abstracted from this box — teach the pattern, not just this
instance:

```
# <Machine Name> — <OS> — <Difficulty>

## Summary
One paragraph: attack path overview (initial vector -> foothold -> privesc -> root).

## Recon
- Tools/commands used and what they revealed
- Key findings (ports, services, versions)

## Enumeration
- Service-by-service breakdown
- Vhosts discovered / /etc/hosts entries added
- Interesting endpoints, files, misconfigurations

## Initial Foothold
- Vulnerability/technique used (name/CVE if applicable)
- Exact exploitation steps
- Why this worked (root cause)

## Privilege Escalation
- Vulnerability/technique used
- Exact steps
- Why this worked (root cause)

## Flags
- user.txt location & method of access
- root.txt location & method of access

## Lessons / Generalizable Patterns
- What class of vulnerability this represents
- How to recognize this pattern on other boxes
- Tools most useful for this category
```

## Deliverable 2 — Fine-tuning dataset: `htb/fine_tune/<exploit-slug>.jsonl`

Name `<exploit-slug>` after the primary exploitation technique (short, lowercase, hyphenated —
e.g. `log4shell-rce`, `smb-null-session`, `suid-python-privesc`). If foothold and privesc are
distinct techniques, pick the more novel one, or hyphenate both if roughly equal weight (e.g.
`xxe-to-suid-tar`). State the chosen slug and reasoning in the final chat summary.

One JSON object per line, instruction-tuning format:
`{"instruction": "...", "input": "...", "output": "..."}`

Include at least one example each for: initial recon -> next-step reasoning; service
enumeration -> vulnerability identification; exploitation -> foothold reasoning;
post-exploitation enumeration -> privesc vector identification; privesc exploitation -> root
reasoning. Plus one full-chain example:

`{"instruction": "Perform a full penetration test walkthrough of this HTB machine from initial
scan to root.", "input": "Target IP: <IP>", "output": "<condensed end-to-end narrative with
commands, in order>"}`

Formatting: instruction phrasing generic/reusable, machine-specific context in input, output
factual and command-grounded (no narrative fluff). Keep file naming consistent so multiple
machines' `.jsonl` files can later be concatenated into one corpus by technique.
