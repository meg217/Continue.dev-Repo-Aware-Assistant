"""
repomap.py — Repository Structure Map Generator
================================================
Builds a compact plain-text map of a codebase.

The repo map lists every code file with its classes and functions,
including line numbers. It gives the model structural awareness of
the whole project without injecting full file contents.

Example output:
    src/auth.py
      class_definition AuthManager (10-60)
      function_definition login (15-30)
      function_definition logout (32-45)
    src/models.py
      class_definition User (5-50)
      class_definition Session (52-80)
    src/utils.py
      (no symbols)

This is included in every @rag response alongside the retrieved code
chunks, so the model always knows the overall shape of the codebase.

Run standalone to generate a map for any repo:
    python repomap.py --repo /path/to/repo
    python repomap.py --repo /path/to/repo --out repomap.txt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tree_sitter import Node
from tree_sitter_languages import get_parser


# ─── Language Config ─────────────────────────────────────────────────────────

# Same language map as indexer.py — files with these extensions
# get tree-sitter parsing for symbol extraction.
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

# Symbol types to include in the repo map.
# We only want named, meaningful constructs — not imports, literals, etc.
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
    "trait_item",
    "impl_item",
    "namespace_definition",
}

# Directories to exclude from the map.
IGNORE_DIRS = {
    "docs",                        # Sphinx / generated docs
    ".git", ".hg", ".svn",
    "node_modules",
    ".venv", "__pycache__", ".mypy_cache",
    "Lib", "Scripts", "Include",   # Windows venv internals
}


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Symbol:
    """A named symbol (function, class, etc.) found in a source file."""
    kind: str        # tree-sitter node type (e.g. "function_definition")
    name: str        # Symbol name (e.g. "login")
    start_line: int  # 1-based line number
    end_line: int


# ─── File Walking ─────────────────────────────────────────────────────────────

def _iter_files(repo_root: Path) -> Iterable[Path]:
    """Walk the repo and yield all code files, skipping ignored directories."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in LANG_BY_SUFFIX:
                yield p


# ─── tree-sitter Helpers ─────────────────────────────────────────────────────

def _node_text(src: bytes, node: Node) -> str:
    """Extract text of a tree-sitter node from the raw source bytes."""
    return src[node.start_byte: node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: Node, src: bytes) -> str:
    """Extract the name of a symbol node, or '<anonymous>' if not found."""
    for field in ("name", "identifier"):
        child = node.child_by_field_name(field)
        if child is not None:
            return _node_text(src, child).strip()
    for child in node.children:
        if child.type in {"identifier", "type_identifier", "property_identifier"}:
            return _node_text(src, child).strip()
    return "<anonymous>"


def _walk(node: Node) -> Iterable[Node]:
    """Depth-first traversal of all nodes in a parse tree."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


# ─── Symbol Extraction ────────────────────────────────────────────────────────

def extract_symbols(file_path: Path) -> list[Symbol]:
    """
    Parse a source file with tree-sitter and extract all named symbols.

    Returns an empty list if:
    - The file extension isn't in LANG_BY_SUFFIX
    - tree-sitter fails to load the parser for that language
    - The file has no recognizable symbols

    Files with no symbols are still included in the repo map with
    "(no symbols)" to show the model the file exists.
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

    symbols: list[Symbol] = []
    for n in _walk(tree.root_node):
        if n.type not in SYMBOL_NODE_TYPES:
            continue
        symbols.append(Symbol(
            kind=n.type,
            name=_find_name(n, src),
            start_line=n.start_point[0] + 1,
            end_line=n.end_point[0] + 1,
        ))

    symbols.sort(key=lambda s: (s.start_line, s.end_line, s.name))
    return symbols


# ─── Repo Map Builder ─────────────────────────────────────────────────────────

def build_repomap(repo_root: Path) -> str:
    """
    Build a plain-text repo map for the given repo root.

    Walks all code files, extracts symbols, and formats them as:
        relative/path/to/file.py
          symbol_kind symbol_name (start_line-end_line)

    Files with no parseable symbols show "(no symbols)" instead.
    The map is sorted alphabetically by file path.
    """
    repo_root = repo_root.resolve()
    lines: list[str] = []

    for fp in sorted(_iter_files(repo_root)):
        rel = fp.resolve().relative_to(repo_root)
        rel_s = str(rel).replace("\\", "/")
        syms = extract_symbols(fp)

        lines.append(rel_s)
        if not syms:
            lines.append("  (no symbols)")
        else:
            for s in syms:
                lines.append(f"  {s.kind} {s.name} ({s.start_line}-{s.end_line})")

    return "\n".join(lines).strip() + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a structural repo map using tree-sitter.",
        epilog="Example: python repomap.py --repo /path/to/repo --out map.txt",
    )
    ap.add_argument("--repo", required=True, help="Path to the repo root")
    ap.add_argument("--out", default="", help="Output file path (prints to stdout if omitted)")
    args = ap.parse_args()

    content = build_repomap(Path(args.repo))

    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Repo map written to: {args.out}")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())