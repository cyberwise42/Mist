"""Embedding client for semantic fallback search (e.g. skill routing).

Small local embedding models (bge-small, nomic-embed-text) are cheap enough to
run alongside a chat model and only get invoked when cheap lexical matching
fails — the same "cheap first, smarter only when needed" trade-off the rest
of Mist makes for tools and skills.
"""
from __future__ import annotations

import math

import httpx


class EmbeddingClient:
    def __init__(self, backend: str, base_url: str, model: str = "bge-small",
                 api_key: str = "", timeout: float = 30.0):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend == "ollama":
            return self._ollama(texts)
        if self.backend == "vllm":
            return self._vllm(texts)
        raise ValueError(f"Unknown embedding backend: {self.backend}")

    # ------------------------------------------------------------------
    def _ollama(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _vllm(self, texts: list[str]) -> list[list[float]]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=headers,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
