"""Pre-execution command-safety gate for the `shell` tool.

Not a sandbox, and not an allowlist of "safe" commands — a pentest agent
legitimately needs to run almost anything against its target. This is a
narrow, configurable backstop against the handful of catastrophic patterns
that would irreversibly destroy or lock the operator out of the execution
host itself, which here is real infrastructure the operator owns (see
tools.shell.backend: ssh in a real config, often the operator's own box),
not a disposable container that resets on its own. Mirrors the idea behind
Hermes's tirith/command_allowlist gate and its `--yolo` bypass flag.
"""
from __future__ import annotations

import re

DEFAULT_DENY_PATTERNS: list[str] = [
    r"rm\s+-rf\s+(/|~|\$HOME)\s*($|[;&|])",   # rm -rf / (or ~, $HOME), not e.g. rm -rf /tmp/x
    r"rm\s+-rf\s+/\*",
    r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|xvd)",  # dd ... of=/dev/sdX — raw disk overwrite
    r"\bmkfs\.",
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:",          # classic fork bomb
    r">\s*/dev/(sd|nvme|hd|xvd)[a-z0-9]",        # redirecting output onto a raw block device
    r"\bchmod\s+-R\s+777\s+/(\s|$)",
    r"\bchown\s+-R\s+\S+\s+/(\s|$)",
    r"\b(shutdown|reboot|poweroff|halt)\b",      # takes down the execution host itself
    r"\binit\s+0\b",
    r"\biptables\s+(-F|--flush)\b",              # can cut off the operator's own SSH session
    r"\bufw\s+disable\b",
]


def check_command_dangerous(command: str, patterns: list[str] | None = None) -> str | None:
    """Returns a human-readable reason if `command` matches a deny pattern,
    else None. `patterns` defaults to DEFAULT_DENY_PATTERNS; pass
    `security.deny_patterns` from config to let an operator extend or
    (by overriding the whole list) narrow it."""
    for pattern in (DEFAULT_DENY_PATTERNS if patterns is None else patterns):
        if re.search(pattern, command):
            return f"matched deny pattern {pattern!r}"
    return None
