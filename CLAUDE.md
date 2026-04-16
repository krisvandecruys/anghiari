# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
uv sync                        # install / sync deps
uv run anghiari index          # build the vector index (first-time setup)
uv run anghiari search "..."   # run a search
uv run anghiari api            # start the REST API
uv run anghiari mcp            # start the MCP server
```

After `uv sync` the `anghiari` binary is also available directly if the venv is activated.

No test suite or linter is configured yet.

## Architecture

The pipeline runs in three phases for every query:

1. **Embed** (`embedder.py`) — encodes the query with the Harrier sentence-transformer (`microsoft/harrier-oss-v1-0.6b`), applying an instruction prefix for query vectors only.
2. **Vector search** (`mapper.py → _get_collection()`) — queries a ChromaDB cosine-similarity index of all MITRE ATT&CK techniques.
3. **LLM rerank** (`mapper.py → _llm_call()`) — a local Nemotron 4B GGUF model (via llama-cpp-python) picks the best match from the top-k candidates, then optionally resolves it to a subtechnique in a second call.

The index is built once by `indexer.py`: it fetches the MITRE ATT&CK STIX bundle, extracts `attack-pattern` objects, embeds them, and persists them to ChromaDB. A subtechnique parent→children map is stored separately as JSON for the phase-3 resolution step.

Both models are lazy singletons (loaded on first use) cached in `~/.cache/huggingface/hub/` by HuggingFace. The STIX bundle, ChromaDB index, and subtechnique map live in `~/.cache/anghiari/` by default.

## Configuration

`src/anghiari/config.py` is the single source of truth for all tuneable values. On first run it writes `~/.config/anghiari/config.toml` with commented defaults.

- `get_config()` returns the active singleton; `set_config()` replaces it (called once in the CLI callback).
- Every module that needs config calls `get_config()` lazily inside functions, never at module import time, so the CLI callback has time to set a custom path first.
- The `--config/-c FILE` global CLI option (defined in the `@app.callback()` in `cli.py`) overrides the default config path for every subcommand.
- `Config` exposes computed path properties (`stix_cache`, `chroma_dir`, `subtech_map`) derived from `cache_dir`.

## Interfaces

- **CLI** (`cli.py`) — Typer app; four commands: `search`, `index`, `api`, `mcp`.
- **REST API** (`api.py`) — Litestar; single endpoint `POST /search` with `{query, top_k}`. OpenAPI docs at `/schema`.
- **MCP server** (`mcp.py`) — FastMCP over stdio; one tool `search_attack_technique`. Intended for Claude Desktop.
- **Public Python API** (`__init__.py`) — `search_technique(query, top_k) → SearchResult`.

## Key data shapes

`SearchResult` (returned by everything): `{best_match: TechniqueMatch, candidates: list[dict]}`.  
`TechniqueMatch`: `{technique_id, name, confidence: float 0–1, rationale: str}`.
