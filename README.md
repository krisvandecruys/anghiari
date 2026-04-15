# Anghiari

> *The Battle of Anghiari was lost to time, yet studied and interpreted for centuries. So are the techniques of adversaries.*

Anghiari searches free-text descriptions of attack behaviors against [MITRE ATT&CK](https://attack.mitre.org/) Enterprise techniques. Give it a sentence about what an attacker did; it returns the technique ID, a confidence score, and the reasoning behind the match.

Everything runs 100% locally. No API keys. No external calls at query time.

## How it works

```
your description
      │
      ▼
 Harrier embedder          microsoft/harrier-oss-v1-0.6b
 (semantic search)    ──▶  ChromaDB (local)  ──▶  top-5 candidates
      │
      ▼
 Nemotron 4B LLM           nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
 (phase 1: pick best technique from candidates)
      │
      ▼
 Nemotron 4B LLM
 (phase 2: resolve to subtechnique if applicable)
      │
      ▼
 { technique_id, name, confidence, rationale }
```

Both models are downloaded automatically from HuggingFace Hub on first use.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- ~6 GB free disk space (models + index)
- Apple Silicon Mac recommended (Metal GPU acceleration)

## Installation

```bash
git clone https://github.com/your-username/anghiari
cd anghiari
uv sync
```

## Usage

### Build the index (once)

Downloads the MITRE ATT&CK STIX bundle, embeds all techniques with Harrier, and stores them in a local ChromaDB index.

```bash
anghiari index
```

Add `--force` to re-download fresh STIX data.

### Search

```bash
anghiari search "adversary dumped credentials from LSASS memory"
```

```json
{
  "best_match": {
    "technique_id": "T1003.001",
    "name": "OS Credential Dumping: LSASS Memory",
    "confidence": 0.94,
    "rationale": "The description directly references LSASS memory, the primary target of OS credential dumping via tools like Mimikatz or ProcDump."
  },
  "candidates": [...]
}
```

Pass `--top-k` to control how many candidates the LLM reasons over:

```bash
anghiari search --top-k 10 "used living-off-the-land binaries to blend in with normal admin activity"
```

### REST API

```bash
anghiari api
anghiari api --port 9000        # custom port
anghiari api --reload           # dev mode
# → http://localhost:8000
# → http://localhost:8000/schema  (OpenAPI docs)
```

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "PowerShell used to download and execute a remote payload"}' | jq .
```

### MCP server

For use with Claude Desktop or any MCP client:

```bash
anghiari mcp
```

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "anghiari": {
      "command": "anghiari",
      "args": ["mcp"]
    }
  }
}
```

## Data

The index and model weights are excluded from version control (see `.gitignore`). After cloning, run `anghiari index` once to populate `data/`. Both HuggingFace models are cached in `~/.cache/huggingface/hub/`.

## Project structure

```
src/anghiari/
├── __init__.py      public API: search_technique, SearchResult, TechniqueMatch
├── cli.py           Typer CLI  ← anghiari search / index / api / mcp
├── mapper.py        core pipeline: embed → search → LLM phase 1 → LLM phase 2
├── indexer.py       STIX fetch, parse, embed, store in ChromaDB
├── embedder.py      Harrier wrapper: embed_query / embed_documents
├── prompt.py        LLM prompt builders
├── models.py        Pydantic types
├── api.py           Litestar REST API
└── mcp.py           FastMCP server
```

## License

MIT
