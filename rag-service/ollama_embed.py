"""
ollama_embed.py — Ollama Embedding Client
==========================================
Handles all communication with the Ollama embedding model (nomic-embed-text).

This module always talks to OLLAMA_EMBED_HOST (port 11435) — never to the
chat model on port 11434. Keeping them separate prevents the two models
from conflicting on the GPU queue.

Why a separate port?
--------------------
Ollama processes requests serially. When a developer uses @rag in Continue,
two things fire almost simultaneously:
    1. Continue sends the chat request -> port 11434 (llama3.1:8b)
    2. Continue calls /context which needs to embed the query -> needs Ollama

If both hit the same port, the embedding request queues behind the chat
request and may return empty results. A second Ollama instance on port 11435
dedicated to embeddings solves this completely.

Start the second instance (Windows):
    $env:OLLAMA_HOST = "127.0.0.1:11435"
    $env:OLLAMA_MODELS = "C:\\Users\\<user>\\.ollama\\models"
    ollama serve

Start the second instance (Linux):
    OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/path/to/models ollama serve &

Ollama API compatibility
------------------------
Different Ollama versions expose different embedding endpoints:
    - /api/embeddings (legacy): {"prompt": "..."} -> {"embedding": [...]}
    - /api/embed      (current): {"input": "..."}  -> {"embeddings": [[...]]}

This module tries both, so it works across Ollama versions.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# OLLAMA_EMBED_HOST is the dedicated embedding instance (port 11435).
# Falls back to OLLAMA_HOST if not set, but you should always set both.
_CHAT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_EMBED_HOST = os.getenv("OLLAMA_EMBED_HOST", "").strip()
_BASE = (_EMBED_HOST or _CHAT_HOST).rstrip("/")

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str, timeout_s: float = 30.0, retries: int = 2) -> list[float]:
    """
    Embed a text string using Ollama's embedding model.

    Returns a list of floats representing the embedding vector.
    The vector length depends on the model (nomic-embed-text produces 768-dim vectors).

    These vectors are stored in ChromaDB and used at query time to find
    semantically similar chunks — code that means something similar to
    the user's query, not just keyword matches.

    Args:
        text:      The text to embed (a code chunk or a user query)
        timeout_s: Request timeout in seconds
        retries:   Number of retry attempts per endpoint on failure

    Raises:
        RuntimeError: If all endpoints fail after retries
    """
    # Try both endpoints in order — legacy first, then current
    candidates: list[tuple[str, dict[str, Any], str]] = [
        (
            f"{_BASE}/api/embeddings",
            {"model": OLLAMA_EMBED_MODEL, "prompt": text},
            "embedding",   # Response key for this endpoint
        ),
        (
            f"{_BASE}/api/embed",
            {"model": OLLAMA_EMBED_MODEL, "input": text},
            "embeddings",  # Response key for this endpoint (returns a list of lists)
        ),
    ]

    last_err: Exception | None = None
    for url, payload, mode in candidates:
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=timeout_s)

                # 404 means this endpoint doesn't exist in this Ollama version
                # Move on to the next candidate
                if r.status_code == 404:
                    break

                r.raise_for_status()
                data = r.json()

                # Parse response based on which endpoint we hit
                if mode == "embedding":
                    emb = data.get("embedding")
                    if isinstance(emb, list) and emb:
                        return [float(x) for x in emb]

                if mode == "embeddings":
                    embs = data.get("embeddings")
                    if isinstance(embs, list) and embs and isinstance(embs[0], list) and embs[0]:
                        return [float(x) for x in embs[0]]

                raise RuntimeError(f"Unexpected embedding response: {data}")

            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.4 * (attempt + 1))  # Brief backoff before retry

    raise RuntimeError(
        f"Failed to embed via Ollama at {_BASE!r}. "
        f"Tried /api/embeddings and /api/embed. "
        f"Is the embedding instance running on {_BASE}?"
    ) from last_err