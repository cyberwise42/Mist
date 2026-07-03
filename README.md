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
   SKILL.md bodies load only when routed to.

## Quick start

```bash
git clone https://github.com/cyberwise42/Mist.git
cd Mist
pip install -e ".[dev]"

# Ollama backend
mist chat --backend ollama --model qwen2.5:7b

# vLLM backend (OpenAI-compatible server)
mist chat --backend vllm --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

## Layout

```
mist/
  core/       # agent loop, context builder, turn budget
  llm/        # backend clients (ollama / vllm via OpenAI-compatible API)
  memory/     # SQLite store: sessions, memories, FTS5 search
  skills/     # skill loader, router (two-tier)
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
- `skills.max_candidates` — skill descriptions surfaced per turn (default 4)
- `tools.max_exposed` — tools exposed per call (default 5)

## Status

Early scaffold. Core loop, memory store, skill router, and both backends are
functional; expect rough edges.

## License

MIT
