"""Deterministic engagement-state ledger.

A local model on a bounded token budget has no persistent picture of the
target: turn-to-turn it sees only a sliding window of recent chat plus a short
recap of the last few tool outputs. So findings from early in the mission (the
open ports, the vhosts already added to /etc/hosts, the scans already run)
scroll out of view, and — unable to see what it already knows — the model falls
back on the one thing always in context, the methodology script, and re-runs
`nmap`, re-echoes `/etc/hosts`, re-runs `gobuster`. It executes the playbook
instead of responding to the system.

This ledger fixes the dumb-redundancy half of that. It watches every tool
command+output and extracts, PURELY DETERMINISTICALLY (no LLM call, so it can't
hallucinate its own state), the durable facts — open ports/services, known
hosts, and which recon actions are already done — and renders a compact block
that gets pinned into every decision prompt. With ports and a DONE list always
in view, re-running a finished scan becomes visibly pointless, and the model
can make its next move off actual state rather than reciting the checklist.

It does NOT try to track creds, leads, or anything requiring judgement — those
are the LLM-assisted layer, deliberately left out of this deterministic pass.
"""
from __future__ import annotations

import re

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_QUOTED_RE = re.compile(r'"([^"]+)"')
# An nmap "open" port line: "22/tcp   open  ssh     OpenSSH 9.6p1 (Ubuntu ...)".
_NMAP_PORT_RE = re.compile(
    r"^(\d{1,5})/(?:tcp|udp)\s+open\s+(\S+)(?:\s+(.*\S))?\s*$", re.MULTILINE)
# An IP + hostname pair, as written into /etc/hosts (`echo '10.10.10.5 host.htb'`).
_HOST_PAIR_RE = re.compile(
    r"(?:\d{1,3}\.){3}\d{1,3}\s+([A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}[A-Za-z0-9.-]*)")

# (label, command-pattern) — a completed recon action, detected off the command.
# Ordered: the first match wins for a given command so a vhost ffuf isn't also
# counted as directory fuzzing.
_ACTION_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("vhost-fuzz", re.compile(r"gobuster\s+vhost|-H\s+['\"]?Host:\s*FUZZ|Host:\s*FUZZ", re.I)),
    ("dir-fuzz", re.compile(r"gobuster\s+dir|\bffuf\b[^|]*FUZZ|dirsearch|feroxbuster|\bdirb\b", re.I)),
    ("port-scan", re.compile(r"\bnmap\b[^|]*-p-", re.I)),
    ("service-scan", re.compile(r"\bnmap\b[^|]*-s[VC]", re.I)),
    ("vuln-scan", re.compile(r"\bnmap\b[^|]*--script[= ]\S*vuln|\bnuclei\b", re.I)),
    ("searchsploit", re.compile(r"\bsearchsploit\b", re.I)),
    ("nikto", re.compile(r"\bnikto\b", re.I)),
    ("whatweb", re.compile(r"\bwhatweb\b", re.I)),
    ("smb-enum", re.compile(r"\bsmbclient\b|\bsmbmap\b|\benum4linux\b|\bcrackmapexec\b|\bnxc\b", re.I)),
    ("nfs-enum", re.compile(r"\bshowmount\b|mount\s+-t\s+nfs", re.I)),
]
_URL_HOST_RE = re.compile(r"https?://([A-Za-z0-9][A-Za-z0-9.-]*)")

# A KEY=value / KEY: value assignment whose key names a secret. Targets the
# deterministic loot an engagement actually turns up — .env / config /
# docker-compose / git-history creds — which is the single highest-value thing
# found and exactly what the model was losing when the output scrolled out of
# context (it read a git commit's .env with DB_PASSWORD and walked right past it).
_ASSIGN_RE = re.compile(
    r"^[ \t]*(?:export[ \t]+)?([A-Za-z][\w.-]*)[ \t]*[=:][ \t]*(.+?)[ \t]*$", re.MULTILINE)
_SECRET_KEY_RE = re.compile(
    r"passw(?:or)?d|username|secret|api[_-]?key|apikey|access[_-]?key|"
    r"private[_-]?key|auth[_-]?token|_token$|^token$|credential", re.IGNORECASE)
# Placeholder / unset values that are NOT real creds. Note: common real
# usernames (admin/root/user) are deliberately NOT here — they're valid loot.
_CRED_PLACEHOLDER = {"null", "none", "nil", "undefined", "password", "passwd",
                     "changeme", "changethis", "example", "secret", "token",
                     "your_password", "your-password", "supersecret", "xxx"}


def _is_placeholder_value(v: str) -> bool:
    v = v.strip().strip("'\"").strip()
    if len(v) < 3 or len(v) > 100:
        return True
    low = v.lower()
    return low in _CRED_PLACEHOLDER or low.startswith(("<", "${", "your", "changeme", "example", "xxx"))


class EngagementState:
    """Accumulates deterministic facts about the target from tool output and
    renders them as a compact, always-injected situational block."""

    def __init__(self, objective: str = "") -> None:
        m = _IPV4_RE.search(objective)
        self.target_ip: str | None = m.group(0) if m else None
        q = _QUOTED_RE.search(objective)
        self.target_name: str | None = q.group(1) if q else None
        self.ports: dict[int, tuple[str, str]] = {}   # port -> (service, version)
        self.hosts: list[str] = []                     # known hostnames (ordered, unique)
        self.actions: list[str] = []                   # completed recon labels (ordered, unique)
        self.creds: list[str] = []                     # discovered KEY=value secrets (ordered, unique)

    def observe(self, command: str, output: str) -> None:
        """Fold one tool call's command+output into the state. Safe to call on
        every tool result; non-matching content is simply ignored."""
        command = command or ""
        output = output or ""
        # Open ports/services from any nmap output present.
        for port, service, version in _NMAP_PORT_RE.findall(output):
            self.ports[int(port)] = (service, (version or "").strip()[:40])
        # Credentials found in tool output (.env / config / git history / etc.).
        for key, value in _ASSIGN_RE.findall(output):
            if _SECRET_KEY_RE.search(key) and not _is_placeholder_value(value):
                entry = f"{key}={value.strip().strip(chr(39) + chr(34))}"
                if entry not in self.creds:
                    self.creds.append(entry)
        # Hostnames written to /etc/hosts (the model's own vhost bookkeeping).
        if "/etc/hosts" in command:
            for host in _HOST_PAIR_RE.findall(command):
                self._add_host(host)
        # Completed recon actions — first matching label only, tagged with the
        # target host when the command carries a URL host (so re-runs against a
        # NEW host still register, but a repeat against the same one doesn't).
        for label, pat in _ACTION_PATTERNS:
            if pat.search(command):
                hm = _URL_HOST_RE.search(command)
                tag = f"{label}:{hm.group(1)}" if hm and label in ("dir-fuzz", "vhost-fuzz") else label
                if tag not in self.actions:
                    self.actions.append(tag)
                break

    def _add_host(self, host: str) -> None:
        host = host.strip().lower()
        if host and host not in self.hosts:
            self.hosts.append(host)

    def render(self) -> str:
        """Compact block for injection. Empty (returns "") until there's at
        least a target, so an opening turn isn't cluttered with blank fields."""
        if not (self.target_ip or self.target_name or self.ports or self.hosts
                or self.actions or self.creds):
            return ""
        lines = ["CURRENT ENGAGEMENT STATE (facts already gathered — act on these; do NOT "
                 "re-run anything under DONE or re-add a HOST you already have):"]
        # Creds first — they're the loot, and losing them was the whole problem.
        if self.creds:
            lines.append("CREDS (USE THESE before any more enumeration — try them for ssh, the "
                         "web-app/service login, or the database they name; do not walk past "
                         "them): " + " · ".join(self.creds[:12]))
        tgt = " ".join(x for x in (self.target_name, self.target_ip) if x)
        if tgt:
            lines.append(f"TARGET {tgt}")
        if self.ports:
            lines.append("PORTS  " + " · ".join(
                f"{p}/{svc}" + (f" {ver}" if ver else "")
                for p, (svc, ver) in sorted(self.ports.items())))
        if self.hosts:
            lines.append("HOSTS  " + ", ".join(self.hosts[:15]))
        if self.actions:
            lines.append("DONE   " + " · ".join(self.actions[:20]))
        return "\n".join(lines)
