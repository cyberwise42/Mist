# Mist

A lightweight, context-frugal agent harness for **small local LLMs** served over
**vLLM** or **Ollama**.

Mist is inspired by harnesses like Hermes Agent, but designed around one hard
constraint: small models (3B–14B) degrade quickly with long, noisy contexts and
unreliable free-form tool calling. Every subsystem is built to keep the working
context tiny and the model's output constrained.

## Design principles

1. **Context is rebuilt, not accumulated.** Each turn assembles a fresh prompt:
   system + top-k memory snippets + active skill + last N turns. Dead tool
   output never rides along.
2. **Retrieval over recall.** Memory lives in SQLite (FTS5). The model never
   "remembers" — Mist retrieves.
3. **Constrained tool calls.** Tool invocations are forced into strict JSON via
   JSON-mode (Ollama) or guided decoding (vLLM). No free-text parsing.
4. **Tiny tool surface per call.** A router selects the 3–5 relevant tools and
   at most a few skill candidates per turn. The model never sees the full
   catalog.
5. **Two-tier skills.** Skill descriptions are always cheap to list; full
   SKILL.md bodies load only when routed to. Keyword overlap picks the
   candidate first; a small embedding model (`bge-small`) only gets called
   when lexical matching finds nothing.
6. **Compaction over accumulation.** Old sessions don't just sit in SQLite —
   `mist compact` folds each one into a handful of durable memories via a
   single LLM call, then never touches it again.

## Running on Apple Silicon (M4 Pro / M4 Max)

Ollama is the practical default on a Mac — it accelerates via Metal and needs
no extra setup. **vLLM does not run on Apple GPUs** (CUDA/ROCm only); point
`base_url` at a remote vLLM server if you want its continuous batching (see
"Subagents" below).

Model sizing by unified memory (Ollama's tags are Q4_K_M-quantized by default,
leaving headroom for the OS and the `bge-small` embedding model):

| Unified memory | Suggested model      |
|-----------------|----------------------|
| 36–48 GB        | `qwen2.5:14b` (default) |
| 64 GB           | `qwen2.5:32b`         |
| 128 GB          | `qwen2.5:32b` (fp16) or `llama3.1:70b` |

```bash
ollama pull qwen2.5:14b
ollama pull bge-small   # embedding fallback for skill routing
```

## Quick start

```bash
git clone https://github.com/cyberwise42/Mist.git
cd Mist
pip install -e ".[dev]"

# Ollama backend
mist chat --backend ollama --model qwen2.5:14b

# vLLM backend (OpenAI-compatible server, e.g. a remote GPU box)
mist chat --backend vllm --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-14B-Instruct

# Fold old sessions into long-term memory
mist compact
```

## Layout

```
mist/
  core/       # agent loop, context builder, turn budget, batch summarizer, subagent spawning
  llm/        # backend clients (ollama / vllm via OpenAI-compatible API) + embeddings
  memory/     # SQLite store: sessions, memories, FTS5 search, compaction tracking
  skills/     # skill loader, router (two-tier, keyword-first with embedding fallback)
  tools/      # built-in tools + registry + JSON-schema constraints
skills_library/  # SKILL.md files (portable, agentskills.io-style)
tests/
```

## Configuration

Copy `config.example.yaml` to `~/.mist/config.yaml` (or pass `--config`).
Key knobs:

- `context.token_budget` — hard cap on assembled prompt tokens (default 6000)
- `context.history_turns` — recent turns kept verbatim (default 6)
- `memory.top_k` — memory snippets retrieved per turn (default 3)
- `memory.keep_recent_sessions` / `memory.compact_min_turns` — what `mist compact` leaves alone
- `skills.max_candidates` — skill descriptions surfaced per turn (default 4)
- `embeddings.enabled` / `embeddings.model` / `embeddings.similarity_threshold` — semantic
  fallback for skill routing when keyword overlap is empty
- `tools.max_exposed` — tools exposed per call (default 5)
- `subagents.enabled` / `subagents.max_workers` — concurrent subtask fan-out

## Subagents

The `spawn_subagents` tool lets the model fan a task out into independent,
context-isolated subtasks that run **concurrently** as plain HTTP calls. Under
Ollama this mostly just adds overhead — one model serves one request at a
time in practice. Under **vLLM**, continuous batching schedules those
concurrent requests together on the GPU, so wall-clock time for N independent
subtasks can approach the time for one. Raise `subagents.max_workers` when
pointed at a vLLM server to exploit this.

## Batch summarization

`mist compact` finds sessions that aren't the active one, have at least
`memory.compact_min_turns` turns, and haven't been compacted yet; each gets a
single LLM call that extracts durable facts/preferences/notes, written to the
`memories` table (searchable via FTS5, same as `remember`). The session is
then marked compacted and skipped by future runs. Options:
`mist compact --keep-recent 2 --min-turns 6`.

## Status

Early scaffold. Core loop, memory store, skill router, batch summarizer,
subagent spawning, and both backends are functional; expect rough edges.

## License

MIT
