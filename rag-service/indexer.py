"""
indexer.py — Repo Chunker & ChromaDB Indexer
=============================================
Walks a code repository, splits it into chunks, embeds each chunk using
nomic-embed-text, and stores the results in ChromaDB.

Run this once per repo before using the RAG service:
    python indexer.py --repo /path/to/your/repo

Re-run whenever your code changes significantly. Each run deletes and
recreates the collection for that repo (full re-index).

How chunking works
------------------
For each file in the repo, we try two strategies in order:

1. SEMANTIC chunking (preferred)
   Uses tree-sitter to parse the file as code and extract meaningful
   symbols — functions, classes, methods, etc. Each symbol becomes its
   own chunk, keeping related code together.

   Supported languages: .py .js .jsx .ts .tsx .go .java .rs
                        .c .h .cpp .cc .hpp .cs .rb .php .lua

2. SLIDING WINDOW fallback
   If tree-sitter can't parse the file (e.g. markdown, yaml, unknown
   extension), we fall back to a simple sliding window: 500-character
   windows with 100-character overlap, so context isn't lost at edges.

Each chunk gets a stable SHA1 ID based on its content and position,
so re-indexing the same unchanged file produces the same IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from dotenv import load_dotenv
from tree_sitter import Node
from tree_sitter_languages import get_parser

from ollama_embed import embed_text

load_dotenv()

# Where ChromaDB stores its data on disk (set in .env)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

# ─── Language Config ─────────────────────────────────────────────────────────

# Maps file extensions to tree-sitter language names.
# Files with these extensions get semantic (tree-sitter) chunking.
# All other files fall back to sliding window chunking.
LANG_BY_SUFFIX: dict[str, str] = {
    ".py":  "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".ts":  "typescript",
    ".tsx": "typescript",
    ".go":  "go",
    ".java":"java",
    ".rs":  "rust",
    ".c":   "c",
    ".h":   "c",
    ".cpp": "cpp",
    ".cc":  "cpp",
    ".hpp": "cpp",
    ".cs":  "c_sharp",
    ".rb":  "ruby",
    ".php": "php",
    ".lua": "lua",
}

# These tree-sitter node types are considered meaningful symbols worth
# extracting as their own chunks. Anything else (imports, literals, etc.)
# is skipped during semantic chunking.
SYMBOL_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "enum_declaration",
    "trait_item",       # Rust trait
    "impl_item",        # Rust impl block
    "namespace_definition",
}

# Directories to skip when walking the repo.
# Add any other generated or irrelevant directories here.
IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules",
    ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "Lib", "Scripts", "Include",   # Windows venv internals
    ".idea", ".vscode"
    # "docs",                        # Sphinx / generated docs
}


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """A single indexable unit of code from a file."""
    file_path: str   # Repo-relative path with forward slashes (e.g. src/auth.py)
    start_line: int  # 1-based line number where this chunk starts
    end_line: int    # 1-based line number where this chunk ends
    content: str     # The actual text content of this chunk
    kind: str        # "semantic" (tree-sitter) or "window" (sliding window fallback)


# ─── Utility Functions ────────────────────────────────────────────────────────

def repo_slug(repo_root: Path) -> str:
    """
    Convert a repo path to a safe ChromaDB collection name.

    ChromaDB collection names must be lowercase alphanumeric + hyphens.
    Example: /home/user/my-project -> home-user-my-project
    """
    raw = str(repo_root.resolve()).replace("\\", "/").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    if not slug:
        slug = "repo"
    if len(slug) > 80:
        slug = slug[:80].rstrip("-")
    return slug


def _iter_code_files(repo_root: Path) -> Iterable[Path]:
    """Walk the repo and yield all code files, skipping ignored directories."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Modify dirnames in-place to prevent os.walk from descending into ignored dirs
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in LANG_BY_SUFFIX:
                yield p


def _node_text(src: bytes, node: Node) -> str:
    """Extract the raw text of a tree-sitter node from the source bytes."""
    return src[node.start_byte: node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: Node, src: bytes) -> str:
    """
    Try to find the name of a symbol node (function name, class name, etc.).
    Falls back to '<anonymous>' if no name can be found.
    """
    # First try named child fields
    for field in ("name", "identifier"):
        child = node.child_by_field_name(field)
        if child is not None:
            return _node_text(src, child).strip()
    # Then try any identifier-type child node
    for child in node.children:
        if child.type in {"identifier", "type_identifier", "property_identifier"}:
            return _node_text(src, child).strip()
    return "<anonymous>"


def _walk(node: Node) -> Iterable[Node]:
    """Depth-first walk of all nodes in a tree-sitter parse tree."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


# ─── Chunking Strategies ─────────────────────────────────────────────────────

def chunk_file_semantic(repo_root: Path, file_path: Path) -> list[Chunk]:
    """
    STRATEGY 1: Semantic chunking using tree-sitter.

    Parses the file as code and extracts meaningful symbols (functions,
    classes, etc.) as individual chunks. This keeps related code together
    and provides clean boundaries for the model to reason about.

    Returns an empty list if the file type is not supported or parsing fails,
    which triggers the sliding window fallback in chunk_repo().
    """
    lang = LANG_BY_SUFFIX.get(file_path.suffix.lower())
    if not lang:
        return []

    try:
        parser = get_parser(lang)
    except Exception:
        return []

    src = file_path.read_bytes()
    tree = parser.parse(src)

    # Get repo-relative path for use in chunk headers
    rel = file_path.resolve().relative_to(repo_root.resolve())
    rel_s = str(rel).replace("\\", "/")

    chunks: list[Chunk] = []
    for n in _walk(tree.root_node):
        if n.type not in SYMBOL_NODE_TYPES:
            continue
        text = _node_text(src, n).strip()
        if not text:
            continue
        name = _find_name(n, src)
        # Prepend a header so the model knows what file/line/type this chunk is
        header = f"{rel_s}:{n.start_point[0] + 1}-{n.end_point[0] + 1} {n.type} {name}\n"
        chunks.append(
            Chunk(
                file_path=rel_s,
                start_line=n.start_point[0] + 1,
                end_line=n.end_point[0] + 1,
                content=header + text,
                kind="semantic",
            )
        )

    chunks.sort(key=lambda c: (c.file_path, c.start_line, c.end_line))
    return chunks


def chunk_file_sliding_window(
    repo_root: Path,
    file_path: Path,
    window: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    STRATEGY 2: Sliding window chunking (fallback).

    Used when tree-sitter can't parse a file (e.g. markdown, yaml, config).
    Splits the file into overlapping character windows.

    window=500  : each chunk is up to 500 characters
    overlap=100 : consecutive chunks share 100 characters to avoid
                  cutting context at chunk boundaries

    Example with window=10, overlap=3:
        "abcdefghij" -> ["abcdefghij", "hijklmno", ...]
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    rel = file_path.resolve().relative_to(repo_root.resolve())
    rel_s = str(rel).replace("\\", "/")

    step = max(1, window - overlap)
    chunks: list[Chunk] = []
    for start in range(0, max(1, len(text)), step):
        end = min(len(text), start + window)
        piece = text[start:end].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                file_path=rel_s,
                start_line=1,   # No line info available for window chunks
                end_line=1,
                content=f"{rel_s} window[{start}:{end}]\n{piece}",
                kind="window",
            )
        )
        if end >= len(text):
            break
    return chunks


def chunk_repo(repo_root: Path) -> list[Chunk]:
    """
    Chunk all code files in the repo.
    Uses semantic chunking where possible, sliding window as fallback.
    """
    repo_root = repo_root.resolve()
    out: list[Chunk] = []
    for fp in _iter_code_files(repo_root):
        semantic_chunks = chunk_file_semantic(repo_root, fp)
        if semantic_chunks:
            out.extend(semantic_chunks)
        else:
            # tree-sitter couldn't parse this file — use sliding window
            out.extend(chunk_file_sliding_window(repo_root, fp))
    return out


# ─── Stable ID ───────────────────────────────────────────────────────────────

def _stable_id(repo_slug_s: str, chunk: Chunk) -> str:
    """
    Generate a stable, unique ID for a chunk based on its content and location.

    Using SHA1 of the chunk's key attributes means:
    - The same chunk always gets the same ID (safe to re-index)
    - Different chunks always get different IDs
    - Re-indexing an unchanged file produces identical IDs (no duplicates)
    """
    h = hashlib.sha1()
    h.update(repo_slug_s.encode("utf-8"))
    h.update(b"\0")
    h.update(chunk.file_path.encode("utf-8"))
    h.update(b"\0")
    h.update(str(chunk.start_line).encode("utf-8"))
    h.update(b":")
    h.update(str(chunk.end_line).encode("utf-8"))
    h.update(b"\0")
    h.update(chunk.kind.encode("utf-8"))
    h.update(b"\0")
    h.update(chunk.content[:2000].encode("utf-8", errors="replace"))
    return h.hexdigest()


# ─── Main Indexing Function ───────────────────────────────────────────────────

def reindex_repo(repo_root: Path) -> int:
    """
    Index (or re-index) a repo into ChromaDB.

    Steps:
    1. Delete any existing collection for this repo
    2. Create a fresh collection
    3. Chunk all code files
    4. Embed each chunk via nomic-embed-text (port 11435)
    5. Store chunks + embeddings in ChromaDB

    Returns the number of chunks indexed.

    Note: Full re-index on every run — no incremental updates yet.
    For large repos this can be slow (1-2 sec per chunk for embedding).
    """
    repo_root = repo_root.resolve()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    name = repo_slug(repo_root)

    # Delete existing collection to start fresh
    try:
        client.delete_collection(name)
        print(f"Deleted existing collection: {name!r}")
    except Exception:
        pass  # Collection didn't exist, that's fine

    col = client.get_or_create_collection(
        name=name,
        metadata={"repo": str(repo_root)},
    )

    chunks = chunk_repo(repo_root)
    if not chunks:
        print("No chunks found — check that the repo path is correct and contains code files.")
        return 0

    print(f"Found {len(chunks)} chunks. Embedding...")

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    embs: list[list[float]] = []

    for i, ch in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Embedding chunk {i + 1}/{len(chunks)}...")
        ids.append(_stable_id(name, ch))
        docs.append(ch.content)
        metas.append({
            "path": ch.file_path,
            "start_line": ch.start_line,
            "end_line": ch.end_line,
            "kind": ch.kind,
        })
        embs.append(embed_text(ch.content))

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(chunks)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Index a repo into ChromaDB using Ollama embeddings.",
        epilog="Example: python indexer.py --repo /path/to/my/project",
    )
    ap.add_argument("--repo", required=True, help="Path to the repo root to index")
    args = ap.parse_args()

    repo_root = Path(args.repo)
    if not repo_root.exists():
        print(f"Error: repo path does not exist: {repo_root}")
        return 1

    print(f"Indexing: {repo_root.resolve()}")
    n = reindex_repo(repo_root)
    print(f"\nDone. Indexed {n} chunks into collection {repo_slug(repo_root)!r}")
    print(f"ChromaDB stored at: {os.path.abspath(CHROMA_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())