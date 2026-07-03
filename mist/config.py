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


class SkillsConfig(BaseModel):
    library_path: str = "skills_library"
    max_candidates: int = 4


class ToolsConfig(BaseModel):
    max_exposed: int = 5


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 1024


class MistConfig(BaseModel):
    backend: str = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
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
