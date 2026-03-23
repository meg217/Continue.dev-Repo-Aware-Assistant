"""
test_indexer.py — End-to-End Pipeline Test
===========================================
Verifies the full indexing and retrieval pipeline works correctly.

What this tests:
    1. Indexes the test repo in tests/test_repo/
    2. Queries ChromaDB for "How do we add two integers?"
    3. Asserts that the add() function was retrieved

Run this after first-time setup to confirm everything is wired up:
    python test_indexer.py

Prerequisites:
    - Both Ollama instances must be running (ports 11434 and 11435)
    - .env must be configured correctly
    - tests/test_repo/ must exist (a small Python file with an add() function)

If this passes, the full pipeline works:
    nomic-embed-text -> ChromaDB indexing -> semantic search -> correct retrieval
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from indexer import CHROMA_PATH, reindex_repo, repo_slug
from ollama_embed import embed_text


def main() -> int:
    # ── Step 1: Locate the test repo ─────────────────────────────────────────
    repo_root = Path(__file__).parent / "tests" / "test_repo"
    if not repo_root.exists():
        raise SystemExit(
            f"Missing test repo at {repo_root}\n"
            "Create tests/test_repo/ with a small Python file containing an add() function."
        )

    # ── Step 2: Index the test repo ──────────────────────────────────────────
    print(f"Indexing test repo: {repo_root}")
    n = reindex_repo(repo_root)
    assert n > 0, f"Expected at least one chunk indexed, got {n}"
    print(f"Indexed {n} chunks")

    # ── Step 3: Query ChromaDB ────────────────────────────────────────────────
    slug = repo_slug(repo_root)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(slug)

    query = "How do we add two integers?"
    print(f"\nQuerying: {query!r}")

    emb = embed_text(query)
    res = col.query(query_embeddings=[emb], n_results=4, include=["documents", "metadatas"])

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    # ── Step 4: Assert correct retrieval ─────────────────────────────────────
    content = "\n".join([d for d in docs if isinstance(d, str)])
    print(f"\nTop result:\n{docs[0] if docs else '(none)'}")

    assert "def add" in content, (
        "Expected to find the add() function in search results.\n"
        "This means either indexing failed or semantic search isn't working."
    )

    # ── Step 5: Show metadata ─────────────────────────────────────────────────
    print("\nAll retrieved chunks:")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        path = meta.get("path", "?") if isinstance(meta, dict) else "?"
        start = meta.get("start_line", "?") if isinstance(meta, dict) else "?"
        end = meta.get("end_line", "?") if isinstance(meta, dict) else "?"
        kind = meta.get("kind", "?") if isinstance(meta, dict) else "?"
        print(f"  [{i+1}] {path}:{start}-{end} [{kind}]")

    print("\nOK: Full pipeline test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())