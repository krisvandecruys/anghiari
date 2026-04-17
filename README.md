# Anghiari

> *The Battle of Anghiari was lost to time, yet studied and interpreted for centuries. So are the techniques of adversaries.*

[![Battle of Anghiari](https://upload.wikimedia.org/wikipedia/commons/c/c4/Peter_Paul_Ruben%27s_copy_of_the_lost_Battle_of_Anghiari.jpg)](https://commons.wikimedia.org/wiki/File:Peter_Paul_Ruben%27s_copy_of_the_lost_Battle_of_Anghiari.jpg)

Anghiari searches free-text descriptions of attack behaviors against [MITRE ATT&CK](https://attack.mitre.org/) Enterprise techniques. Give it a sentence about what an attacker did; it returns the technique ID, a verbal confidence level, and the reasoning behind the match.

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

Pass any threat description or report as an argument, via stdin, or with `--file`. Anghiari will chunk the text, scan it against ATT&CK techniques, and present an annotated text map showing exactly which passage triggered which technique, alongside LLM-generated rationale.

```bash
anghiari search -f examples/iab_vishing_campaign.txt
```

```text
TECHNIQUE SCAN  (4 matches)
──────────────────────────────────────────────────────────────
[1] 0.648  T1598.004    Phishing for Information: Spearphishing Voice reconnaissance  [HIGH]
    ↳ "The adversary then executes a vishing campaign, and calls its victims leveraging the local language. The threat actor im…"
    ↳ The text describes a vishing (voice phishing) campaign to establish trust, directly matching Spearphishing Voice.
...

ANNOTATED TEXT
──────────────────────────────────────────────────────────────
An IAB associated with other attacks in Belgium initiates a targeted attack against a remote office. The attack begins with a series of abnormal or undeliverable emails sent days before the attack...
```

For automation, pass `--json` to receive structured output of all matches, complete with their character offsets and LLM rationales.

```bash
anghiari search --json "adversary dumped credentials from LSASS memory"
```

Pass `--top` to control how many techniques are evaluated:

```bash
anghiari search --top 10 "used living-off-the-land binaries to blend in with normal admin activity"
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

## Configuration

On first run, `~/.config/anghiari/config.toml` is created with commented defaults. Edit it to change models, LLM parameters, prompts, cache location, and more.

Pass `--config <file>` before any subcommand to use a different config file:

```bash
anghiari --config ~/my-config.toml search "..."
```

## Data

The index and model weights are excluded from version control. After cloning, run `anghiari index` once to populate `~/.cache/anghiari/`. HuggingFace model weights are cached separately in `~/.cache/huggingface/hub/`. Both paths are configurable in `config.toml`.

## Project structure

```
src/anghiari/
├── __init__.py      public API: search_technique, SearchResult, TechniqueMatch
├── cli.py           Typer CLI  ← anghiari search / index / api / mcp
├── config.py        config loader — reads ~/.config/anghiari/config.toml
├── mapper.py        core pipeline: embed → search → LLM phase 1 → LLM phase 2
├── indexer.py       STIX fetch, parse, embed, store in ChromaDB
├── embedder.py      Harrier wrapper: embed_query / embed_documents
├── prompt.py        LLM prompt builders
├── models.py        Pydantic types (incl. Confidence literal)
├── api.py           Litestar REST API
└── mcp.py           FastMCP server
```

## License

MIT
