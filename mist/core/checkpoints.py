"""Filesystem checkpoint/rollback: a shadow git repo per workspace
directory, snapshotting its contents before risky shell/write_file calls
so a bad mission action can be rolled back. Mirrors Hermes's `checkpoints`
subsystem (status/list/prune/clear) — see mist/cli.py's `checkpoints`
command group.

"Shadow" means the real workspace directory is never touched with a .git
of its own (which would pollute an HTB engagement folder or the wiki) —
each repo lives under `base_dir`, addressed by a hash of the workspace's
absolute path, and driven via `git --git-dir=... --work-tree=...` pointed
back at the real directory.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_GIT_TIMEOUT = 30.0


@dataclass
class CheckpointInfo:
    workspace_path: str
    repo_dir: Path
    commit_count: int
    size_bytes: int


class CheckpointStore:
    def __init__(self, base_dir: Path | str = "~/.mist/checkpoints"):
        self.base_dir = Path(base_dir).expanduser()

    def _slug(self, workspace_path: Path) -> str:
        return hashlib.sha256(str(workspace_path.resolve()).encode()).hexdigest()[:16]

    def _repo_dir(self, workspace_path: Path) -> Path:
        return self.base_dir / self._slug(Path(workspace_path))

    def _run(self, workspace_path: Path, *args: str, check: bool = True
             ) -> subprocess.CompletedProcess:
        git_dir = self._repo_dir(workspace_path) / ".git"
        cmd = ["git", f"--git-dir={git_dir}", f"--work-tree={workspace_path}",
               "-c", "user.email=mist@localhost", "-c", "user.name=mist", *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=check,
                              timeout=_GIT_TIMEOUT)

    def ensure(self, workspace_path: Path | str) -> None:
        workspace_path = Path(workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)
        repo_dir = self._repo_dir(workspace_path)
        git_dir = repo_dir / ".git"
        if git_dir.is_dir():
            return
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", f"--git-dir={git_dir}", "init", "--quiet"],
                       check=True, timeout=_GIT_TIMEOUT)
        # Records the real path alongside the hash-addressed repo dir —
        # the hash alone can't be reversed back to a path for `list_all`.
        (repo_dir / "workspace_path.txt").write_text(str(workspace_path.resolve()),
                                                      encoding="utf-8")

    def snapshot(self, workspace_path: Path | str, message: str) -> bool:
        """Stages and commits the current state of workspace_path. Returns
        False if the workspace doesn't exist yet or there's nothing to
        commit (an empty `git commit` exits non-zero — expected, not an
        error) — True only on an actual new commit."""
        workspace_path = Path(workspace_path)
        if not workspace_path.is_dir():
            return False
        self.ensure(workspace_path)
        self._run(workspace_path, "add", "-A")
        result = self._run(workspace_path, "commit", "-m", message, "--quiet", check=False)
        return result.returncode == 0

    def status(self, workspace_path: Path | str) -> CheckpointInfo | None:
        workspace_path = Path(workspace_path)
        repo_dir = self._repo_dir(workspace_path)
        if not (repo_dir / ".git").is_dir():
            return None
        return self._info(repo_dir, str(workspace_path.resolve()))

    def _info(self, repo_dir: Path, workspace_path: str) -> CheckpointInfo:
        result = subprocess.run(
            ["git", f"--git-dir={repo_dir / '.git'}", "log", "--oneline"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT, check=False,
        )
        commit_count = len(result.stdout.splitlines()) if result.returncode == 0 else 0
        size_bytes = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        return CheckpointInfo(workspace_path, repo_dir, commit_count, size_bytes)

    def list_all(self) -> list[CheckpointInfo]:
        if not self.base_dir.is_dir():
            return []
        infos = []
        for repo_dir in sorted(self.base_dir.iterdir()):
            path_file = repo_dir / "workspace_path.txt"
            if not (repo_dir / ".git").is_dir() or not path_file.is_file():
                continue
            infos.append(self._info(repo_dir, path_file.read_text(encoding="utf-8").strip()))
        return infos

    def rollback(self, workspace_path: Path | str, ref: str = "HEAD") -> bool:
        """Hard-resets workspace_path's tracked files to match `ref`
        (default: the last snapshot). Returns False if there's no
        checkpoint repo for this path at all."""
        workspace_path = Path(workspace_path)
        if not (self._repo_dir(workspace_path) / ".git").is_dir():
            return False
        result = self._run(workspace_path, "reset", "--hard", ref, check=False)
        return result.returncode == 0

    def prune(self) -> int:
        """Deletes checkpoint repos whose recorded workspace no longer
        exists on disk (an HTB engagement folder that got cleaned up,
        etc.) and runs `git gc` on the rest. Returns the count removed."""
        if not self.base_dir.is_dir():
            return 0
        removed = 0
        for repo_dir in list(self.base_dir.iterdir()):
            path_file = repo_dir / "workspace_path.txt"
            if not (repo_dir / ".git").is_dir() or not path_file.is_file():
                continue
            recorded_path = Path(path_file.read_text(encoding="utf-8").strip())
            if not recorded_path.is_dir():
                shutil.rmtree(repo_dir, ignore_errors=True)
                removed += 1
                continue
            subprocess.run(["git", f"--git-dir={repo_dir / '.git'}", "gc", "--quiet"],
                           capture_output=True, timeout=_GIT_TIMEOUT, check=False)
        return removed

    def clear(self) -> None:
        """Deletes the entire checkpoint base — all rollback history for
        every workspace, across every project."""
        shutil.rmtree(self.base_dir, ignore_errors=True)
