# Local AI Dev Assistant — RAG Service

A fully offline, codebase-aware AI assistant for your IDE.  
Ask questions about your code in VS Code and get answers grounded in your actual project — no internet, no cloud.

---

## What this does

When you type `@rag where is the add function?` in Continue (VS Code), this service:

1. Receives the query from Continue
2. Embeds it using `nomic-embed-text` (a local embedding model)
3. Searches ChromaDB for the most similar code chunks from your indexed repo
4. Returns those chunks to Continue as context items
5. Continue injects them into the prompt before sending to Ollama
6. The model answers using your actual code

---

## Project Structure

```
rag-service/
├── api.py            ← FastAPI server — start here, this is the entry point
├── indexer.py        ← Indexes a repo into ChromaDB (run once per repo)
├── ollama_embed.py   ← Talks to the embedding Ollama instance (port 11435)
├── repomap.py        ← Builds a plain-text structural map of a repo
├── test_indexer.py   ← End-to-end pipeline test
├── .env              ← Your configuration (copy from .env.example)
├── chroma/           ← ChromaDB storage (created automatically on first index)
└── tests/
    └── test_repo/    ← Small test repo for test_indexer.py
```

---

## Prerequisites

Before starting, you need:

- **Python 3.11+**
- **Ollama** installed and running ([ollama.com](https://ollama.com/download))
- **Two models pulled:**
  ```bash
  ollama pull llama3.1:8b        # or your preferred chat model
  ollama pull nomic-embed-text   # embedding model
  ```
- **Continue.dev** installed in VS Code

---

## Step 1 — Start Two Ollama Instances

> **Why two?** Ollama processes requests one at a time. When you use `@rag`,
> Continue fires a chat request (for the answer) and an embedding request (for the search)
> almost simultaneously. Two instances give each its own queue and prevent conflicts.
> Both models fit in 8GB VRAM — llama3.1:8b uses ~5GB, nomic-embed-text uses ~300MB.

**Terminal 1** — This is probably already running as a service. Leave it alone.

```
# Ollama chat instance on port 11434 (default)
# Serves: llama3.1:8b
```

**Terminal 2** — Start a second instance for embeddings only.

Windows (PowerShell):

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_MODELS = "C:\Users\<your-user>\.ollama\models"
ollama serve
```

Linux / Mac:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=~/.ollama/models ollama serve
```

Verify both are running:

```bash
curl http://localhost:11434/api/tags   # chat instance
curl http://localhost:11435/api/tags   # embed instance
```

---

## Step 2 — Set Up the Python Environment

```bash
cd rag-service

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # Linux / Mac

# Install dependencies with requirements.txt
pip install -r requirements.txt
# Or install manually
pip install fastapi uvicorn chromadb tree-sitter tree-sitter-languages requests python-dotenv
```

---

## Step 3 — Configure .env

Copy `.env.example` to `.env` and edit it:

```bash
cp .env.example .env
```

`.env` contents:

```env
# Port 11434 = chat model
OLLAMA_HOST=http://localhost:11434

# Port 11435 = embedding model (dedicated instance)
OLLAMA_EMBED_HOST=http://localhost:11435
OLLAMA_EMBED_MODEL=nomic-embed-text

# Where ChromaDB stores data (created automatically)
CHROMA_PATH=./chroma

# The repo to search — must match the path you used when indexing
# Continue sends your workspace root, not the subfolder, so we use this override
DEFAULT_REPO=/path/to/your/repo
```

> **Important:** `DEFAULT_REPO` must be the exact path you pass to `indexer.py`.
> If you index `/home/user/myrepo` but set `DEFAULT_REPO=/home/user/myrepo/`,
> the path won't match the stored collection and searches will return empty.

---

## Step 4 — Index Your Repo

This walks your codebase, splits it into chunks, embeds each chunk,
and stores everything in ChromaDB. Run this once, then re-run whenever
your code changes significantly.

```bash
python indexer.py --repo /path/to/your/repo
```

You'll see output like:

```
Indexing: /path/to/your/repo
Deleted existing collection: 'path-to-your-repo'
Found 142 chunks. Embedding...
  Embedding chunk 1/142...
  Embedding chunk 11/142...
  ...
Done. Indexed 142 chunks into collection 'path-to-your-repo'
ChromaDB stored at: ./chroma
```

Run `python repomap.py --repo <drive>:/<path>/minimal` to view the repo map after indexing.

### How chunking works

The indexer uses two strategies:

**1. Semantic chunking (preferred)**  
Uses [tree-sitter](https://tree-sitter.github.io/tree-sitter/) to parse files as code
and extract meaningful symbols — functions, classes, methods. Each becomes its own chunk
with clean, logical boundaries. The model gets complete function definitions, not fragments.

Supported file types: `.py .js .jsx .ts .tsx .go .java .rs .c .h .cpp .cs .rb .php .lua`

**2. Sliding window fallback**  
For files tree-sitter can't parse (markdown, yaml, config files, etc.), the indexer
falls back to overlapping 500-character windows with 100-character overlap.
The overlap ensures context isn't lost at chunk boundaries.

> You can tell which strategy was used by looking at the `kind` field in search results:  
> `[semantic]` = tree-sitter parsed, `[window]` = sliding window fallback.

---

## Step 5 — Start the API Server

**Terminal 3:**

```bash
cd rag-service
source .venv/bin/activate   # or Activate.ps1 on Windows
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
```

---

## Step 6 — Test Before Using Continue

Always test the `/context` endpoint directly first. This tells you if the
pipeline is working before debugging Continue integration.

```python
python -c "
import requests
r = requests.post('http://localhost:8001/context', json={
    'query': '',
    'fullInput': 'where is the add function defined?',
    'workspacePath': 'file:///z%3A/your/workspace'
})
print(r.status_code)
print(r.text)
"
```

**Good response** — a non-empty array with real code chunks:

```json
[
  {
    "name": "Codebase RAG",
    "description": "src/math.py:10-12",
    "content": "src/math.py:10-12 function_definition add\ndef add(x, y):\n    return x + y"
  }
]
```

**Empty array `[]`** — means `DEFAULT_REPO` in `.env` doesn't match the indexed path.
Double-check both and re-index if needed.

**500 error** — means the embedding model on port 11435 isn't responding.
Check that the second Ollama instance is running.

---

## Step 7 — Configure Continue

Edit `~/.continue/config.yaml` (located at `C:\Users\<user>\.continue\config.yaml` on Windows):

```yaml
models:
  - name: Llama 3.1 8B
    provider: ollama
    model: llama3.1:8b
    apiBase: http://localhost:11434   # or server IP for production
    contextLength: 8192

context:
  - provider: file
  - provider: http
    params:
      url: http://localhost:8001/context   # or server IP for production
      title: rag
      displayTitle: Codebase RAG
```

> **Note:** Use `context:` not `contextProviders:` — Continue changed this in their YAML format.
> Use a single HTTP provider pointing to `/context` — there is a confirmed Continue bug
> where multiple HTTP providers in YAML config only shows the first one.

Reload Continue (Ctrl+Shift+P → "Continue: Reload config"), then try:

```
@rag where is the add function defined?
```

You should see `4 context items` appear above the response, and the model
should cite the correct file and line numbers.

---

## API Endpoints Reference

### `POST /context`

The main endpoint called by Continue. Accepts Continue's POST body and returns context items.


| Field in body   | Notes                                              |
| --------------- | -------------------------------------------------- |
| `query`         | Always empty from Continue — ignored               |
| `fullInput`     | The actual user query — this is what gets embedded |
| `workspacePath` | URL-encoded workspace root — decoded internally    |


Returns: `[{name, description, content}, ...]`

---

### `GET /search`

Manual semantic search. Useful for debugging and testing retrieval quality.

```bash
curl "http://localhost:8001/search?query=how+does+add+function+work&repo=/path/to/repo"
```

Returns: `{content: "file.py:10-20 [semantic]\ndef login..."}` 

---

### `GET /repomap`

Returns the structural map of a repo (files, classes, functions, line numbers).

```bash
curl "http://localhost:8001/repomap?repo=/path/to/repo"
```

Useful to see what tree-sitter found in your repo, and to verify
the correct symbols are being extracted.

---

### `GET /`

Health check.

```bash
curl http://localhost:8001/
# {"ok": true, "chroma_path": "./chroma"}
```

---

## Running the End-to-End Test

```bash
python test_indexer.py
```

This indexes a small test repo (`tests/test_repo/`) and asserts that
a semantic query finds the `add()` function. If this passes, the full
pipeline is working: embedding → indexing → ChromaDB search → retrieval.

---

## Troubleshooting

`**@rag` shows nothing / empty context items**  
→ Test `/context` directly (Step 6). If empty, check `DEFAULT_REPO` in `.env`.

**500 error in uvicorn logs**  
→ The embedding model isn't responding. Check port 11435:

```bash
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "input": "test"}'
```

**Sphinx docs / config files showing up in search results**  
→ Add `docs` to `IGNORE_DIRS` in `indexer.py` and re-index.

**ChromaDB collection path mismatch**  
→ List collections and check stored paths:

```python
python -c "
import chromadb
c = chromadb.PersistentClient(path='./chroma')
for col in c.list_collections():
    print(col.name, col.metadata)
"
```

**Indexed from wrong path / collection mess**  
→ Delete everything and re-index fresh:

```python
python -c "
import chromadb
c = chromadb.PersistentClient(path='./chroma')
for col in c.list_collections():
    c.delete_collection(col.name)
    print('Deleted:', col.name)
"
python indexer.py --repo /path/to/your/repo
```

---

## Key Concepts

**RAG (Retrieval-Augmented Generation)**  
Instead of sending the model your whole codebase (impossible — too many tokens),
we pre-index it into a vector database. At query time, we find the most relevant
chunks and inject only those into the prompt.

**Embeddings**  
A way to represent text as a list of numbers (a vector) where similar meanings
produce similar vectors. `nomic-embed-text` converts both your code chunks (at
index time) and your queries (at search time) into these vectors. ChromaDB finds
the closest ones.

**tree-sitter**  
A fast, language-aware parser that understands the structure of code. Instead of
blindly splitting at every 500 characters, tree-sitter lets us split at meaningful
boundaries — function definitions, class definitions, etc. This keeps related code
together and dramatically improves retrieval quality.

**ChromaDB**  
A local vector database. No server required — it stores everything in a folder on
disk (`./chroma`). One collection per repo.
