"""
Configuration loader for Anghiari.

The CLI can create ~/.config/anghiari/config.toml with commented defaults.
Pass --config <path> on the CLI to use a different file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_DIR = Path("~/.config/anghiari").expanduser()
_CONFIG_FILE = _CONFIG_DIR / "config.toml"
_DEFAULT_CACHE_DIR = Path("~/.cache/anghiari").expanduser()

_DEFAULT_RERANKER_INSTRUCTION = (
    "Given an attack description, determine whether the MITRE ATT&CK technique candidate "
    "is a true match."
)

_DEFAULT_TOML = """\
# Anghiari configuration
# Generated on first run — edit freely.
# Reference: https://github.com/your-username/anghiari

# Directory for downloaded data (STIX bundle, ChromaDB index, subtechnique map).
# HuggingFace model weights are always cached in ~/.cache/huggingface/hub/.
cache_dir = "~/.cache/anghiari"

[embedder]
model_id = "microsoft/harrier-oss-v1-0.6b"
query_prefix = "Instruct: Retrieve the relevant MITRE ATT&CK technique.\\nQuery: "
batch_size = 32

[reranker]
model_id = "Qwen/Qwen3-Reranker-4B"
high_threshold = 0.50
medium_threshold = 0.20
instruction = "Given an attack description, determine whether the MITRE ATT&CK technique candidate is a true match."

[search]
top_k = 5

[api]
host = "0.0.0.0"
port = 8000

[stix]
url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
fetch_timeout = 60

"""


@dataclass
class EmbedderConfig:
    model_id: str = "microsoft/harrier-oss-v1-0.6b"
    query_prefix: str = (
        "Instruct: Retrieve the relevant MITRE ATT&CK technique.\nQuery: "
    )
    batch_size: int = 32


@dataclass
class RerankerConfig:
    model_id: str = "Qwen/Qwen3-Reranker-4B"
    high_threshold: float = 0.50
    medium_threshold: float = 0.20
    instruction: str = _DEFAULT_RERANKER_INSTRUCTION


@dataclass
class SearchConfig:
    top_k: int = 5


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class StixConfig:
    url: str = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    fetch_timeout: int = 60


@dataclass
class Config:
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    api: APIConfig = field(default_factory=APIConfig)
    stix: StixConfig = field(default_factory=StixConfig)

    @property
    def stix_cache(self) -> Path:
        return self.cache_dir / "enterprise-attack.json"

    @property
    def chroma_dir(self) -> Path:
        return self.cache_dir / "chroma_db"

    @property
    def subtech_map(self) -> Path:
        return self.cache_dir / "subtechnique_map.json"


def _build_config(data: dict) -> Config:
    defaults = Config()
    d_emb = data.get("embedder", {})
    d_reranker = data.get("reranker", {})
    d_search = data.get("search", {})
    d_api = data.get("api", {})
    d_stix = data.get("stix", {})
    return Config(
        cache_dir=Path(data.get("cache_dir", str(defaults.cache_dir))).expanduser(),
        embedder=EmbedderConfig(
            model_id=d_emb.get("model_id", defaults.embedder.model_id),
            query_prefix=d_emb.get("query_prefix", defaults.embedder.query_prefix),
            batch_size=d_emb.get("batch_size", defaults.embedder.batch_size),
        ),
        reranker=RerankerConfig(
            model_id=d_reranker.get("model_id", defaults.reranker.model_id),
            high_threshold=d_reranker.get(
                "high_threshold", defaults.reranker.high_threshold
            ),
            medium_threshold=d_reranker.get(
                "medium_threshold", defaults.reranker.medium_threshold
            ),
            instruction=d_reranker.get("instruction", defaults.reranker.instruction),
        ),
        search=SearchConfig(
            top_k=d_search.get("top_k", defaults.search.top_k),
        ),
        api=APIConfig(
            host=d_api.get("host", defaults.api.host),
            port=d_api.get("port", defaults.api.port),
        ),
        stix=StixConfig(
            url=d_stix.get("url", defaults.stix.url),
            fetch_timeout=d_stix.get("fetch_timeout", defaults.stix.fetch_timeout),
        ),
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_config: Config | None = None


def get_config() -> Config:
    """Return the active config, loading from the default path if not yet set."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(cfg: Config) -> None:
    """Set the active config (called once at CLI startup)."""
    global _config
    _config = cfg


def load_config(path: Path | None = None, *, create_default: bool = False) -> Config:
    """Load config from *path*, or from the default location.

    If *path* is None and the default config file doesn't exist, pure-Python
    defaults are returned unless *create_default* is true, in which case the
    default config file is written first.

    Raises FileNotFoundError if an explicit *path* is given but doesn't exist.
    """
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as f:
            return _build_config(tomllib.load(f))

    if not _CONFIG_FILE.exists():
        if create_default:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _CONFIG_FILE.write_text(_DEFAULT_TOML)
        return Config()

    with _CONFIG_FILE.open("rb") as f:
        return _build_config(tomllib.load(f))
