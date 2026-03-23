"""
api.py — FastAPI RAG Service
============================
The main entry point for the RAG service. Run this with uvicorn to start the API server.

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Endpoints:
    POST /context   — Called by Continue.dev @rag. Returns relevant code chunks.
    GET  /search    — Manual semantic search against an indexed repo.
    GET  /repomap   — Returns a structural map of a repo (files, classes, functions).
    GET  /          — Health check.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request

from indexer import CHROMA_PATH, repo_slug
from ollama_embed import embed_text
from repomap import build_repomap

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Local RAG Service", version="0.1.0")


# ─── Helper Functions ────────────────────────────────────────────────────────

def _resolve_repo(repo: str) -> Path:
    """
    Resolve a repo path string to an absolute Path.

    Accepts either:
    - An absolute or relative filesystem path (e.g. /home/user/myrepo)
    - A ChromaDB collection slug (e.g. home-user-myrepo) — looks up stored metadata

    Raises HTTP 404 if the path or collection cannot be found.
    """
    p = Path(repo)
    if p.exists() and p.is_dir():
        return p.resolve()

    # Fall back to treating it as a collection slug and look up its stored path
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(name=repo)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Unknown repo or slug: {repo!r}") from e

    meta = col.metadata or {}
    repo_path = meta.get("repo")
    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Collection {repo!r} has no repo metadata")

    rp = Path(str(repo_path))
    if not (rp.exists() and rp.is_dir()):
        raise HTTPException(
            status_code=404,
            detail=f"Repo path from collection metadata not found: {repo_path!r}",
        )
    return rp.resolve()


def _collection_for_repo(repo_root: Path):
    """
    Get or create a ChromaDB collection for the given repo root.
    Collection name is derived from the repo path (slugified).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=repo_slug(repo_root),
        metadata={"repo": str(repo_root)},
    )


def _cap_tokens(text: str, max_tokens: int = 400) -> str:
    """
    Roughly cap a text chunk to max_tokens words.
    This is a word-count approximation, not a true token count.
    Prevents any single chunk from consuming too much of the model's context window.
    """
    parts = text.split()
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens])


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check. Returns the ChromaDB storage path so you can confirm it's running."""
    return {"ok": True, "chroma_path": os.path.abspath(CHROMA_PATH)}


@app.get("/repomap")
def repomap(repo: str = Query(..., description="Repo path or collection slug")):
    """
    Returns a plain-text structural map of the repo.

    The map lists every file with its classes and functions + line numbers.
    This gives the model a high-level understanding of the whole codebase
    without injecting full file contents.

    Example:
        src/auth.py
          function_definition login (12-34)
          function_definition logout (36-45)
        src/models.py
          class_definition User (10-60)

    Usage:
        curl "http://localhost:8001/repomap?repo=/path/to/repo"
    """
    repo_root = _resolve_repo(repo)
    return {"content": build_repomap(repo_root)}


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="Natural language search query"),
    repo: str = Query(..., description="Repo path or collection slug"),
):
    """
    Semantic search against an indexed repo's ChromaDB collection.

    Embeds the query using nomic-embed-text (port 11435), then finds the
    top 4 most similar code chunks in ChromaDB.

    Returns chunks with file path and line number headers so the model
    knows exactly where each piece of code came from.

    Usage:
        curl "http://localhost:8001/search?query=how+does+auth+work&repo=/path/to/repo"

    Note: Repo must be indexed first with:
        python indexer.py --repo /path/to/repo
    """
    repo_root = _resolve_repo(repo)
    col = _collection_for_repo(repo_root)

    # Embed the query and search ChromaDB for similar chunks
    q_emb = embed_text(query)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    if not docs:
        return {"content": ""}

    # Format each chunk with a header showing file path and line range
    chunks: list[str] = []
    for d, m in zip(docs, metas, strict=False):
        if not isinstance(d, str):
            continue
        header = ""
        if isinstance(m, dict):
            header = (
                f"{m.get('path', '?')}:"
                f"{m.get('start_line', '?')}-{m.get('end_line', '?')} "
                f"[{m.get('kind', '?')}]"
            )
        chunks.append(header + "\n" + _cap_tokens(d, 400))

    return {"content": "\n\n---\n\n".join(chunks).strip() + "\n"}


@app.post("/context")
async def context(body: dict[str, Any], request: Request):
    """
    The main endpoint called by Continue.dev's HTTP context provider.

    Continue sends a POST with this JSON body:
        {
            "query":         "",                          # Always empty — do not use
            "fullInput":     "where is add defined?",    # The actual user query
            "workspacePath": "file:///z%3A/myproject",   # URL-encoded workspace root
            "options":       {}
        }

    Quirks handled here:
    1. query is always empty — we use fullInput instead
    2. workspacePath is URL-encoded (file:///z%3A/...) — we decode it
    3. workspacePath is the workspace ROOT, not the repo subfolder
       — we fall back to DEFAULT_REPO from .env

    Returns a list of context items that Continue injects into the model prompt:
        [
            {
                "name":        "Codebase RAG",
                "description": "src/auth.py:12-34",
                "content":     "def login(user, pw): ..."
            },
            ...
        ]

    Test this endpoint directly before using @rag in Continue:
        python -c "
        import requests
        r = requests.post('http://localhost:8001/context', json={
            'query': '',
            'fullInput': 'where is the add function?',
            'workspacePath': 'file:///z%3A/your/repo'
        })
        print(r.status_code, r.text)
        "

    Expect: a non-empty JSON array. Empty array [] means DEFAULT_REPO is wrong
    or the repo hasn't been indexed yet.
    """
    # Use fullInput — query is always empty from Continue
    query = str(body.get("query") or body.get("fullInput") or "").strip()
    if not query:
        return []

    # Decode the URL-encoded workspace path Continue sends
    # e.g. "file:///z%3A/New-Dev-AI-Project" -> "Z:/New-Dev-AI-Project"
    workspace_path = body.get("workspacePath", "")
    workspace_path = unquote(workspace_path).replace("file:///", "")

    # Fix Windows drive letter capitalisation (z: -> Z:)
    if len(workspace_path) >= 2 and workspace_path[1] == ":":
        workspace_path = workspace_path[0].upper() + workspace_path[1:]

    # DEFAULT_REPO overrides workspacePath — needed because Continue sends the
    # workspace root, not the specific repo subfolder that was indexed
    repo = os.getenv("DEFAULT_REPO", workspace_path)
    repo_root = Path(repo).resolve()

    if not repo_root.exists():
        return []

    # Search ChromaDB for relevant chunks
    col = _collection_for_repo(repo_root)
    q_emb = embed_text(query)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]
    print("Distances:", distances)


    # Build the context items list Continue expects
    items: list[dict[str, str]] = []
    # Always include repo structure as first context item
    items.append({
        "name": "Repo Structure",
        "description": "File and function map",
        "content": build_repomap(repo_root),
    })

    for d, m, dist in zip(docs, metas, distances, strict=False):
        if not isinstance(d, str):
            continue
        if dist > 400: # filter out chunks that don't match high
            continue
        header = ""
        if isinstance(m, dict):
            header = f"{m.get('path', '?')}:{m.get('start_line', '?')}-{m.get('end_line', '?')}"
        items.append(
            {
                "name": "Codebase RAG",
                "description": header or "RAG chunk",
                "content": _cap_tokens(d, 400),
            }
        )
        print("Distances:", dist)

    return items