"""Tier 1 of Mist's tool-output handling: persist the complete, untruncated
stdout/stderr of a tool call to disk *before* any truncation happens.

Truncation (head+tail split, see `_truncate_raw_output`/`_truncate_tool_output`)
is a lossy *view* onto a tool's output, sized to fit a small model's context
budget — it was never meant to mean the rest of the data is gone forever.
Before this module, it was: nothing wrote the full output anywhere, not even
the durable mission-log file (which receives the already-truncated string).
This writes it under the wiki's `raw/` convention (already scaffolded and
documented for exactly this — "untouched source dumps: recon output, tool
logs...", see `mist/wiki/scaffold.py`) so a model that needs more than the
truncated view can `read_file` the pointer this appends.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Artifact:
    rel_path: Path  # relative to the wiki root — what read_file expects
    stdout_chars: int
    stderr_chars: int

    def pointer(self) -> str:
        """Appended as a suffix *after* the (possibly truncated) tool
        result. Both truncation layers keep a tail slice of the string
        (`text[-tail_chars:]`), and this pointer is well under the ~1360
        char tail budget at default settings — so it survives every
        downstream truncation pass automatically, without any change
        needed to the truncation functions themselves.

        Explicitly says `read_file`, not `shell`/`grep`/`cat` — confirmed
        live: a model tried to `grep` this path directly and got "No such
        file or directory", since `rel_path` is relative to the wiki root
        (what `read_file` resolves against), not `shell`'s cwd (the
        workspace root, or a genuinely different machine entirely on the
        SSH backend) — the artifact lives wherever Mist itself runs, which
        a shell command has no guaranteed access to at all."""
        return (f"\n[full output: {self.rel_path.as_posix()} "
                f"({self.stdout_chars} stdout / {self.stderr_chars} stderr chars) — "
                f"use read_file, not shell/grep, to see more]")


class ArtifactStore:
    """Writes full tool output under `<wiki_root>/<dir_name>/<day>/<stamp>-
    <random>.txt`. A random suffix (not a counter) — `_run_subprocess` runs
    concurrently across subagent threads, and a counter would race."""

    def __init__(self, wiki_root: Path | str, dir_name: str = "raw/tool-output",
                 min_chars_to_persist: int = 500):
        self.wiki_root = Path(wiki_root).expanduser()
        self.dir_name = dir_name
        self.min_chars_to_persist = min_chars_to_persist

    def write(self, command: str, stdout: str, stderr: str) -> Artifact | None:
        """Returns None (writes nothing) if the combined output is smaller
        than `min_chars_to_persist` — not worth a file for a one-line
        result — or if the write itself fails for any reason (a full disk,
        a read-only mount): an artifact-write failure must never take the
        tool call down with it."""
        total = len(stdout) + len(stderr)
        if total < self.min_chars_to_persist:
            return None
        now = time.localtime()
        day = time.strftime("%Y%m%d", now)
        stamp = time.strftime("%H%M%S", now)
        rel_path = Path(self.dir_name) / day / f"{stamp}-{uuid.uuid4().hex[:8]}.txt"
        full_path = self.wiki_root / rel_path
        body = f"$ {command}\n{'=' * 60}\n--- stdout ---\n{stdout}"
        if stderr:
            body += f"\n--- stderr ---\n{stderr}"
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(body, encoding="utf-8")
        except OSError:
            return None
        return Artifact(rel_path=rel_path, stdout_chars=len(stdout), stderr_chars=len(stderr))
