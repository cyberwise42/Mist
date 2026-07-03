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


class MemoryConfig(BaseModel):
    db_path: str = "~/.mist/mist.db"
    top_k: int = 3
    keep_recent_sessions: int = 1   # sessions `mist compact` always leaves alone
    compact_min_turns: int = 4      # sessions shorter than this are skipped


class SkillsConfig(BaseModel):
    library_path: str = "skills_library"
    max_candidates: int = 4


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    backend: str = "ollama"          # ollama | vllm
    model: str = "bge-small"         # `ollama pull bge-small`
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    similarity_threshold: float = 0.35  # min cosine similarity to accept a semantic match


class ToolsConfig(BaseModel):
    max_exposed: int = 5


class SubagentConfig(BaseModel):
    enabled: bool = True
    max_workers: int = 4    # concurrent subagent calls; raise this against vLLM


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 1024


class MistConfig(BaseModel):
    backend: str = "ollama"
    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    subagents: SubagentConfig = Field(default_factory=SubagentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @property
    def db_path(self) -> Path:
        return Path(os.path.expanduser(self.memory.db_path))


def load_config(path: str | None = None) -> MistConfig:
    candidates = [path] if path else [
        os.environ.get("MIST_CONFIG"),
        os.path.expanduser("~/.mist/config.yaml"),
        "config.yaml",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            with open(candidate, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return MistConfig(**data)
    return MistConfig()
