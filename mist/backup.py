"""One-command backup/restore for Mist's own state — config, memory DB,
history, and the wiki knowledge base. Before this, backing any of it up
meant a manual `cp` of mist.db (done by hand, repeatedly, before every
destructive DB operation this project has needed so far). Mirrors
Hermes's `backup`/`import` commands.

The zip's internal layout is two top-level prefixes, `mist_home/` (mirrors
~/.mist/) and `wiki/` (mirrors the configured wiki root — a separate,
independently-located directory, not nested under ~/.mist/), plus a
manifest.json recording where each came from so `restore_backup` can put
things back even if the current config points somewhere else.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from mist.config import MistConfig

BACKUP_SUBDIR = "backups"
MIST_HOME = Path("~/.mist").expanduser()


def default_backup_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return MIST_HOME / BACKUP_SUBDIR / f"mist-backup-{timestamp}.zip"


def create_backup(cfg: MistConfig, output_path: Path | str | None = None) -> Path:
    output_path = Path(output_path).expanduser().resolve() if output_path else default_backup_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mist_home = MIST_HOME
    wiki_root = cfg.wiki_root
    backups_dir = (mist_home / BACKUP_SUBDIR).resolve()

    manifest = {"mist_home": str(mist_home), "wiki_root": str(wiki_root),
               "created_at": datetime.now(timezone.utc).isoformat()}

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        if mist_home.is_dir():
            for path in sorted(mist_home.rglob("*")):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved == backups_dir or backups_dir in resolved.parents:
                    continue  # never recurse into prior backups
                zf.write(path, arcname=f"mist_home/{path.relative_to(mist_home)}")
        if wiki_root.is_dir() and wiki_root.resolve() != mist_home.resolve():
            for path in sorted(wiki_root.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=f"wiki/{path.relative_to(wiki_root)}")
    return output_path


def restore_backup(zip_path: Path | str, target_mist_home: Path | str | None = None,
                   target_wiki_root: Path | str | None = None) -> dict[str, int]:
    """Extracts a backup created by create_backup(). Destinations default
    to the paths recorded in the backup's own manifest.json, but can be
    overridden (e.g. restoring onto a machine with a differently-located
    wiki). Returns a count of files restored per section."""
    zip_path = Path(zip_path).expanduser()
    restored = {"mist_home": 0, "wiki": 0}
    with zipfile.ZipFile(zip_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        mist_home = Path(target_mist_home).expanduser() if target_mist_home \
            else Path(manifest["mist_home"]).expanduser()
        wiki_root = Path(target_wiki_root).expanduser() if target_wiki_root \
            else Path(manifest["wiki_root"]).expanduser()
        for name in zf.namelist():
            if name == "manifest.json":
                continue
            if name.startswith("mist_home/"):
                dest = mist_home / name[len("mist_home/"):]
                section = "mist_home"
            elif name.startswith("wiki/"):
                dest = wiki_root / name[len("wiki/"):]
                section = "wiki"
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            restored[section] += 1
    return restored
