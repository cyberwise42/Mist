"""Configuration loading and validation."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ContextConfig(BaseModel):
    token_budget: int = 6000
    history_turns: int = 6
    max_tool_output_chars: int = 2000
    max_tool_steps: int = 8  # bound on tool calls per turn; raise for longer chains
                             # (multi-step recon, full llm-wiki ingest) on stronger models
    compress_on_overflow: bool = False
    # When token_budget forces history out of the assembled prompt,
    # _fit_budget (mist/core/agent.py) normally just drops the oldest
    # messages with no trace at all. When this is on, the dropped messages
    # are summarized first (via the same aux model as tier 3 tool-output
    # compression, config.summarizer) and folded into one compact recap
    # message in their place — off by default: an extra LLM call on every
    # turn that overflows budget, not free.


class MemoryConfig(BaseModel):
    db_path: str = "~/.mist/mist.db"
    top_k: int = 3
    keep_recent_sessions: int = 1   # sessions `mist compact` always leaves alone
    compact_min_turns: int = 4      # sessions shorter than this are skipped


class SkillsConfig(BaseModel):
    library_path: str = "skills_library"
    max_candidates: int = 4


class WikiConfig(BaseModel):
    root_path: str = "~/.mist/wiki"


class WorkspaceConfig(BaseModel):
    root_path: str = "~/.mist/workspace"  # shell's cwd, and (with extra_roots below) an
                                          # absolute-path escape hatch read_file/write_file/
                                          # search_files allow outside the wiki root
    extra_roots: list[str] = Field(default_factory=list)
    # Additional absolute directories read_file/write_file/search_files may reach, beyond
    # wiki_root and workspace.root_path. `shell` itself is never sandboxed by path — it can
    # already write anywhere, e.g. an operator-documented convention like
    # `~/Desktop/HTB/<machine>/` (see the htb-engagement skill). Without an entry here for
    # that same directory, write_file/read_file/search_files can't reach it: a real mission
    # tried `write_file` there (per that skill's own documented convention) and was refused,
    # landing its notes under wiki_root instead while shell's raw scan output stayed under
    # the HTB directory — the same engagement split across two unrelated trees.
    mission_root: str = ""
    # When set, each mission automatically gets its own subdirectory here
    # (named after the target IP extracted from the objective, or a
    # slugified hostname if no IP is present) as the shell's cwd for that
    # mission — instead of every mission ever run sharing one flat
    # workspace.root_path folder regardless of target. Confirmed live: a
    # single shared workspace ended up with scan output, downloaded
    # exploits, and payloads from multiple unrelated HTB machines all mixed
    # together in one directory. Empty (default) disables this — missions
    # keep using workspace.root_path unconditionally, today's behavior.


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    backend: str = "ollama"          # ollama | vllm
    model: str = "bge-small"         # `ollama pull bge-small`
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    similarity_threshold: float = 0.35  # min cosine similarity to accept a semantic match


class ArtifactConfig(BaseModel):
    """Tier 1 of tool-output handling: persists the complete, untruncated
    stdout/stderr of every tool call under `<wiki_root>/<dir>/` before any
    truncation runs (see mist/tools/artifacts.py) — turns "truncated" into
    a lossy *view* with the full data still available, not permanent data
    loss."""
    enabled: bool = True
    dir: str = "raw/tool-output"      # relative to the wiki root
    min_chars_to_persist: int = 500   # skip persisting trivially small output


class SummarizerLLMConfig(BaseModel):
    """Tier 3 of tool-output handling: an optional, independently-
    configured second model (same pattern as EmbeddingConfig) that
    compresses tool output tiers 1-2 don't already fit — the long tail of
    ad hoc commands (see mist/core/tool_compressor.py). Off by default:
    tiers 1-2 already cover the tools that actually blow the budget, so
    this rarely needs to fire."""
    enabled: bool = False
    backend: str = "ollama"          # ollama | vllm
    model: str = ""
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    trigger_chars: int = 6000     # only invoked above this size
    max_input_chars: int = 20000  # hard cap on what's sent to the aux model at all


class ShellSSHConfig(BaseModel):
    host: str = ""
    user: str = ""
    port: int = 22
    key_path: str = ""
    timeout: int = 120  # higher than the local backend's 60s: SSH round-trips
                        # plus real recon commands need more headroom


class ShellConfig(BaseModel):
    backend: str = "local"   # local | ssh
    ssh: ShellSSHConfig = Field(default_factory=ShellSSHConfig)


class StructuredToolsConfig(BaseModel):
    """Tier 2 of tool-output handling: deterministic structured extraction
    for nmap/nuclei/gobuster/ffuf (see mist/tools/structured.py) — keeps
    the real signal (port tables, findings, hit lines) and drops
    high-volume boilerplate before the char-budget truncation ever has to
    choose what to cut."""
    enabled: bool = True
    tools: list[str] = Field(default_factory=list)  # empty = every tool structured.py knows


class BrowserConfig(BaseModel):
    """Headless-browser render tool (`browse`). Renders a page in chromium on
    the shell host (`--dump-dom`, JS executed) and returns its text/forms/links
    — what curl can't get, because curl never runs the page's JavaScript. The
    value is the rendered *text* (a text-only model can't use a screenshot);
    visual browsing is a separate, multimodal concern."""
    enabled: bool = True
    binary: str = "chromium"     # remote binary name (chromium-browser / google-chrome-stable / ...)
    timeout_seconds: int = 45    # hard per-render cap via coreutils `timeout`, independent of ssh.timeout
    virtual_time_ms: int = 8000  # chromium --virtual-time-budget: how long to let the page's JS/timers run


class ToolsConfig(BaseModel):
    max_exposed: int = 6  # ranked tools shown per decision. Sized to fit the always-useful
                          # core set so none is silently crowded out of the ranked pool:
                          # {read_file, write_file, search_files, shell, browse, remember}.
                          # (Was 5 before `browse` joined the core set; raising it kept
                          # `remember` reachable — see the tool-select regression tests.)
    enabled: list[str] | None = None  # opt-in allow-list; None = every tool Mist supports
    shell: ShellConfig = Field(default_factory=ShellConfig)
    structured: StructuredToolsConfig = Field(default_factory=StructuredToolsConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)


class CheckpointConfig(BaseModel):
    """Filesystem checkpoint/rollback (mist/core/checkpoints.py) — a shadow
    git repo snapshotting a workspace directory before every shell/
    write_file call, so a bad mission action can be rolled back. Off by
    default: it shells out to git on every mutating tool call, real but
    modest overhead not every casual chat session needs."""
    enabled: bool = False
    base_dir: str = "~/.mist/checkpoints"


class SecurityConfig(BaseModel):
    """Pre-execution command-safety gate for the `shell` tool — see
    mist/tools/safety.py. Not a sandbox or an allowlist; a narrow backstop
    against catastrophic self-destructive commands (rm -rf /, a raw disk
    dd, a fork bomb, shutdown/reboot of the execution host, flushing
    iptables and cutting off the operator's own SSH session, etc.)."""
    command_safety_enabled: bool = True
    deny_patterns: list[str] | None = None  # None = use safety.DEFAULT_DENY_PATTERNS
    target_scope_enforced: bool = True  # block a `shell` command that aims a recon/attack
                                        # tool (nmap/curl/ffuf/mount/...) at an IPv4 that is
                                        # NOT the mission's authorized target — the guard
                                        # against an autonomous run drifting to a hallucinated
                                        # or unrelated host. Reverse-shell/callback IPs (the
                                        # attacker's own listener) are exempt. Off = no scope
                                        # gate (prior behavior).


class SubagentConfig(BaseModel):
    enabled: bool = True
    max_workers: int = 4    # concurrent subagent calls; raise this against vLLM
    tools_enabled: bool = True  # give subagents their own read/write/shell/remember loop
    max_steps: int = 6      # bounded tool-loop length per subagent task


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 1024
    think: bool = True  # reasoning-model <think> traces on the free-text answer step;
                        # forced off regardless on schema-constrained calls (see LLMClient)
    keep_alive: str | None = None  # Ollama-only: how long to keep the model resident in
                                    # VRAM between calls (sent as the request's top-level
                                    # `keep_alive`). Left None, Ollama uses its own default
                                    # (~5 min), which is SHORTER than a single long tool step
                                    # (the ssh shell timeout is 480s; a full nuclei scan ~3
                                    # min) — so a slow scan lets the model unload and the next
                                    # LLM call pays a full cold reload before it can generate.
                                    # Accepts Ollama's duration syntax ("30m", "60m") or -1 to
                                    # pin forever. Ignored by the vLLM backend.
    decision_max_tokens: int | None = 2048  # num_predict cap for the constrained tool-DECISION
                                    # call specifically, separate from max_tokens (which sizes
                                    # the free-text answer step). A decision emits a tiny JSON
                                    # action, so it never needs a big budget — but a forced-think
                                    # decision (the mission opening-turn reasoning) runs think-ON
                                    # and, inheriting a large max_tokens, could reason for the
                                    # full budget (~10 min at max_tokens: 40000) UNGUARDED: the
                                    # decision call is non-streaming, so stream_no_content_timeout
                                    # (which only covers the streamed answer step) can't bound it.
                                    # Effective cap is min(this, max_tokens); a non-thinking
                                    # decision is unaffected (it stops at the JSON either way).
                                    # None = no separate cap (use max_tokens).
    stream_max_content_tokens: int | None = None  # hard cap on VISIBLE answer content in a
                                    # streamed reply (approx, ~4 chars/token). Distinct from
                                    # stream_no_content_timeout (which catches a reply producing
                                    # NOTHING): this catches one producing too MUCH — e.g. a
                                    # mission turn that, with no real tool output to report,
                                    # streamed a long fabricated findings report. num_predict/
                                    # max_tokens caps TOTAL generation, but that budget also
                                    # covers reasoning; this bounds the visible answer itself.
                                    # Aborts with a truncation notice once exceeded. None = off.
    stream_no_content_timeout: float | None = None  # seconds; guards the streamed answer step
                                    # against a runaway <think>. A reasoning model can spend its
                                    # whole (large) max_tokens budget "thinking" and never emit a
                                    # visible answer — with max_tokens sized for minutes of
                                    # reasoning that's a multi-minute stall producing nothing. When
                                    # set, the stream aborts if no visible content has appeared
                                    # within this many seconds and returns a short explanatory
                                    # message instead. None (default) = no guard, wait for the full
                                    # token budget to run out (prior behavior).


class MissionConfig(BaseModel):
    max_turns: int = 40          # autonomous turns before a mission force-stops
    max_seconds: float = 3600.0  # wall-clock budget before a mission force-stops
    stuck_repeat_threshold: int = 3  # identical tool+args calls in a row -> auto-pause
    near_duplicate_threshold: int = 5  # coarsely-similar tool+args calls in a row (same
                                        # tool/target, only a quoted literal or number
                                        # differs, e.g. varying a search query or a grep
                                        # flag) -> auto-pause. Looser signal than the exact
                                        # match above, so it needs more repeats to fire.
    manual_probe_threshold: int = 8    # consecutive `curl`/`wget` shell calls in a row
                                        # (each to a genuinely different path, so neither
                                        # check above fires) -> auto-pause. A real mission
                                        # hand-guessed at a dozen+ invented API paths one at
                                        # a time instead of running a content-discovery
                                        # scanner (gobuster/ffuf) once told to.
    respond_streak_threshold: int = 2  # consecutive mission turns that end in "respond"
                                        # with no tool call at all -> auto-pause. Invisible
                                        # to every check above (they're all keyed off
                                        # tool_start events) — a real mission "responded"
                                        # with a prose plan instead of acting, burning its
                                        # whole token budget still "thinking" and never
                                        # calling a tool, which could otherwise run
                                        # undetected all the way to max_turns/max_seconds.
    unmount_shares_on_end: bool = True  # SSH backend only: when a mission ends, best-effort
                                        # force-lazy-unmount the NFS/SMB shares it mounted on
                                        # the shell host (scoped to /mnt and the engagement
                                        # workspace tree). Left mounted, they accumulate
                                        # run-to-run — and a `hard` mount to a target IP that
                                        # later changes (HTB revert) wedges FOREVER, hanging
                                        # every subsequent filesystem op that touches it. A
                                        # real incident: four stale mounts (two to a dead IP)
                                        # silently poisoned later sessions. Never touches the
                                        # local machine Mist itself runs on.


class MistConfig(BaseModel):
    backend: str = "ollama"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    base_url: str = "http://10.48.48.10:11434"
    api_key: str = ""
    history_path: str = "~/.mist/history"  # shared recall/autofill, mist chat + mist tui
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    subagents: SubagentConfig = Field(default_factory=SubagentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    summarizer: SummarizerLLMConfig = Field(default_factory=SummarizerLLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @property
    def history_file(self) -> Path:
        return Path(os.path.expanduser(self.history_path))

    @property
    def db_path(self) -> Path:
        return Path(os.path.expanduser(self.memory.db_path))

    @property
    def wiki_root(self) -> Path:
        return Path(os.path.expanduser(self.wiki.root_path))

    @property
    def workspace_root(self) -> Path:
        return Path(os.path.expanduser(self.workspace.root_path))

    @property
    def extra_workspace_roots(self) -> tuple[Path, ...]:
        return tuple(Path(os.path.expanduser(p)) for p in self.workspace.extra_roots)


def resolve_config_path(path: str | None = None) -> Path | None:
    """The file load_config() would actually read, or None if none of the
    candidates exist (meaning built-in defaults are in effect) — shared
    with `mist config path`/`mist config edit` so they don't duplicate or
    drift from load_config()'s own candidate order."""
    candidates = [path] if path else [
        os.environ.get("MIST_CONFIG"),
        os.path.expanduser("~/.mist/config.yaml"),
        "config.yaml",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def load_config(path: str | None = None) -> MistConfig:
    resolved = resolve_config_path(path)
    if resolved is not None:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return MistConfig(**data)
    return MistConfig()
