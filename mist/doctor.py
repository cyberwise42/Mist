"""`mist doctor`: diagnose config/connectivity issues before they surface
mid-mission as a confusing hang or an empty error message.

Motivated directly by real incidents this project has hit: `mist tui`
looking hung for two minutes because the Ollama server was unreachable
(no fast, explicit check existed to catch that instantly at startup), and
config.yaml values drifting out of sync with reality with nothing to catch
it. Each check is a small, independent, pure-ish function returning a
`CheckResult` so they're unit-testable without a real backend; `run_all`
wires them together for the CLI.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mist.config import MistConfig
from mist.llm.client import LLMClient, compute_max_tokens, compute_num_ctx
from mist.wiki import init_wiki

Status = str  # "ok" | "warn" | "fail"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


def check_config_loads(cfg: MistConfig) -> CheckResult:
    # If we got a MistConfig object at all, the YAML parsed and validated
    # against the Pydantic schema — this check exists mainly so the report
    # always has at least one guaranteed-passing line confirming that.
    return CheckResult("config", "ok", f"backend={cfg.backend} model={cfg.model}")


def check_wiki_initialized(cfg: MistConfig) -> CheckResult:
    # Case-insensitive: confirmed live against a real, long-populated wiki
    # that has a lowercase schema.md from before the scaffold's SCHEMA.md
    # convention solidified — a case-sensitive check reported a false
    # "not initialized" against a wiki that very much was.
    if not cfg.wiki_root.is_dir():
        return CheckResult("wiki", "warn",
                           f"not initialized at {cfg.wiki_root} — run `mist wiki-init` "
                           "(or `mist doctor --fix`)")
    has_schema = any(p.name.lower() == "schema.md" for p in cfg.wiki_root.iterdir() if p.is_file())
    if has_schema:
        return CheckResult("wiki", "ok", f"initialized at {cfg.wiki_root}")
    return CheckResult("wiki", "warn",
                       f"not initialized at {cfg.wiki_root} — run `mist wiki-init` "
                       "(or `mist doctor --fix`)")


def check_backend_reachable(cfg: MistConfig) -> tuple[CheckResult, list[str] | None]:
    """Returns the check plus the model list on success (None on failure) —
    callers use the list for check_model_available without a second round
    trip."""
    llm = LLMClient(cfg.backend, cfg.base_url, cfg.model, cfg.api_key)
    try:
        models = llm.list_models()
    except Exception as exc:
        return (CheckResult("backend", "fail",
                            f"{cfg.base_url} unreachable ({type(exc).__name__}: {exc})"),
                None)
    return CheckResult("backend", "ok", f"{cfg.base_url} reachable, {len(models)} model(s)"), models


def _normalize_model_name(name: str) -> str:
    # Ollama's own `/api/tags` always includes a tag (":latest" if the user
    # never pulled one explicitly), but a config can reasonably omit it
    # ("bge-m3" instead of "bge-m3:latest") — confirmed live: this exact
    # mismatch made a genuinely-available embedding model report as
    # missing. Normalizing both sides the same way avoids that false
    # negative without needing every config to spell out ":latest".
    return name if ":" in name else f"{name}:latest"


def _model_is_available(configured: str, models: list[str]) -> bool:
    if configured in models:
        return True
    return _normalize_model_name(configured) in {_normalize_model_name(m) for m in models}


def check_model_available(cfg: MistConfig, models: list[str] | None) -> CheckResult:
    if models is None:
        return CheckResult("model", "warn", "skipped — backend was unreachable")
    if _model_is_available(cfg.model, models):
        return CheckResult("model", "ok", f"{cfg.model} is available")
    return CheckResult("model", "fail",
                       f"{cfg.model} not found on {cfg.base_url} — available: "
                       f"{', '.join(models[:8])}{', ...' if len(models) > 8 else ''}")


def check_num_ctx(cfg: MistConfig) -> CheckResult:
    if cfg.backend != "ollama":
        return CheckResult("num_ctx", "warn", f"skipped — {cfg.backend} backend doesn't expose this")
    llm = LLMClient(cfg.backend, cfg.base_url, cfg.model, cfg.api_key)
    discovered = llm.discover_context_length()
    if discovered is None:
        return CheckResult("num_ctx", "warn",
                           "could not discover the model's real context length "
                           "(server unreachable, or an older Ollama) — num_ctx won't be set")
    desired = cfg.context.token_budget + cfg.generation.max_tokens
    num_ctx = compute_num_ctx(cfg.context.token_budget, cfg.generation.max_tokens, discovered)
    effective_max_tokens = compute_max_tokens(cfg.generation.max_tokens, cfg.context.token_budget,
                                              discovered)
    if desired > discovered:
        return CheckResult("num_ctx", "warn",
                           f"context.token_budget + generation.max_tokens ({desired}) exceeds "
                           f"{cfg.model}'s real context window ({discovered}) — max_tokens will "
                           f"be capped to {effective_max_tokens} at runtime")
    return CheckResult("num_ctx", "ok",
                       f"{cfg.model} supports {discovered} tokens; will request num_ctx={num_ctx}")


def check_embeddings(cfg: MistConfig) -> CheckResult:
    if not cfg.embeddings.enabled:
        return CheckResult("embeddings", "ok", "disabled")
    llm = LLMClient(cfg.embeddings.backend, cfg.embeddings.base_url, cfg.embeddings.model,
                    cfg.embeddings.api_key)
    try:
        models = llm.list_models()
    except Exception as exc:
        return CheckResult("embeddings", "warn",
                           f"{cfg.embeddings.base_url} unreachable ({type(exc).__name__}) — "
                           "skill routing falls back to keyword-only matching")
    if not _model_is_available(cfg.embeddings.model, models):
        return CheckResult("embeddings", "warn",
                           f"{cfg.embeddings.model} not found on {cfg.embeddings.base_url}")
    return CheckResult("embeddings", "ok", f"{cfg.embeddings.model} available")


def check_shell_backend(cfg: MistConfig) -> CheckResult:
    if cfg.tools.shell.backend != "ssh":
        return CheckResult("shell", "ok", "local backend")
    ssh = cfg.tools.shell.ssh
    if ssh.key_path and not Path(ssh.key_path).expanduser().is_file():
        return CheckResult("shell", "fail", f"key_path {ssh.key_path} does not exist")
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
    if ssh.key_path:
        args += ["-i", str(Path(ssh.key_path).expanduser())]
    args += ["-p", str(ssh.port), f"{ssh.user}@{ssh.host}" if ssh.user else ssh.host, "echo ok"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return CheckResult("shell", "fail", f"SSH to {ssh.host}:{ssh.port} timed out")
    if proc.returncode != 0:
        return CheckResult("shell", "fail",
                           f"SSH to {ssh.host}:{ssh.port} failed: {proc.stderr.strip()[:200]}")
    return CheckResult("shell", "ok", f"SSH to {ssh.host}:{ssh.port} reachable")


def check_db_writable(cfg: MistConfig) -> CheckResult:
    try:
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        probe = cfg.db_path.parent / ".mist_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult("db", "fail", f"{cfg.db_path.parent} not writable: {exc}")
    return CheckResult("db", "ok", str(cfg.db_path))


def run_all(cfg: MistConfig) -> list[CheckResult]:
    results = [check_config_loads(cfg), check_wiki_initialized(cfg), check_db_writable(cfg)]
    backend_result, models = check_backend_reachable(cfg)
    results.append(backend_result)
    results.append(check_model_available(cfg, models))
    results.append(check_num_ctx(cfg))
    results.append(check_embeddings(cfg))
    results.append(check_shell_backend(cfg))
    return results


def fix(cfg: MistConfig) -> list[str]:
    """Applies safe, idempotent fixes for checks that support one. Returns
    a list of human-readable descriptions of what was actually changed."""
    fixed = []
    if check_wiki_initialized(cfg).status != "ok":
        created = init_wiki(cfg.wiki_root)
        if created:
            fixed.append(f"initialized wiki at {cfg.wiki_root} ({len(created)} file(s)/dir(s))")
    return fixed
