"""
Configuration loader for Anghiari.

On first run, creates ~/.config/anghiari/config.toml with commented defaults.
Pass --config <path> on the CLI to use a different file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_DIR = Path("~/.config/anghiari").expanduser()
_CONFIG_FILE = _CONFIG_DIR / "config.toml"
_DEFAULT_CACHE_DIR = Path("~/.cache/anghiari").expanduser()

_DEFAULT_SYSTEM_PROMPT = (
    "You are a MITRE ATT&CK expert. Given an attack description and candidate techniques, "
    "identify the single best match. Respond ONLY with valid JSON — no markdown, no extra text. "
    "Express confidence as one of (ordered lowest to highest): GUESS < LOW < MEDIUM < HIGH < CERTAIN. "
    "GUESS = weak signal, speculative match; CERTAIN = definitively matches, near-unmistakable."
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
query_prefix = "Retrieve semantically similar text: "
batch_size = 32

[llm]
repo_id = "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
filename = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
n_ctx = 8192
n_gpu_layers = -1    # -1 = offload all layers to GPU (Metal on Apple Silicon)
chat_format = "chatml"
max_tokens = 512
temperature = 0.1

[search]
top_k = 5

[api]
host = "0.0.0.0"
port = 8000

[stix]
url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
fetch_timeout = 60

[prompts]
system = "You are a MITRE ATT&CK expert. Given an attack description and candidate techniques, identify the single best match. Respond ONLY with valid JSON — no markdown, no extra text. Express confidence as one of (ordered lowest to highest): GUESS < LOW < MEDIUM < HIGH < CERTAIN. GUESS = weak signal, speculative match; CERTAIN = definitively matches, near-unmistakable."
description_truncate_phase1 = 300
description_truncate_phase2 = 400
"""


@dataclass
class EmbedderConfig:
    model_id: str = "microsoft/harrier-oss-v1-0.6b"
    query_prefix: str = "Retrieve semantically similar text: "
    batch_size: int = 32


@dataclass
class LLMConfig:
    repo_id: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
    filename: str = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    chat_format: str = "chatml"
    max_tokens: int = 512
    temperature: float = 0.1


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
class PromptsConfig:
    system: str = _DEFAULT_SYSTEM_PROMPT
    description_truncate_phase1: int = 300
    description_truncate_phase2: int = 400


@dataclass
class Config:
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    api: APIConfig = field(default_factory=APIConfig)
    stix: StixConfig = field(default_factory=StixConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)

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
    d_llm = data.get("llm", {})
    d_search = data.get("search", {})
    d_api = data.get("api", {})
    d_stix = data.get("stix", {})
    d_prompts = data.get("prompts", {})
    return Config(
        cache_dir=Path(data.get("cache_dir", str(defaults.cache_dir))).expanduser(),
        embedder=EmbedderConfig(
            model_id=d_emb.get("model_id", defaults.embedder.model_id),
            query_prefix=d_emb.get("query_prefix", defaults.embedder.query_prefix),
            batch_size=d_emb.get("batch_size", defaults.embedder.batch_size),
        ),
        llm=LLMConfig(
            repo_id=d_llm.get("repo_id", defaults.llm.repo_id),
            filename=d_llm.get("filename", defaults.llm.filename),
            n_ctx=d_llm.get("n_ctx", defaults.llm.n_ctx),
            n_gpu_layers=d_llm.get("n_gpu_layers", defaults.llm.n_gpu_layers),
            chat_format=d_llm.get("chat_format", defaults.llm.chat_format),
            max_tokens=d_llm.get("max_tokens", defaults.llm.max_tokens),
            temperature=d_llm.get("temperature", defaults.llm.temperature),
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
        prompts=PromptsConfig(
            system=d_prompts.get("system", defaults.prompts.system),
            description_truncate_phase1=d_prompts.get(
                "description_truncate_phase1", defaults.prompts.description_truncate_phase1
            ),
            description_truncate_phase2=d_prompts.get(
                "description_truncate_phase2", defaults.prompts.description_truncate_phase2
            ),
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


def load_config(path: Path | None = None) -> Config:
    """Load config from *path*, or from the default location.

    If *path* is None and the default config file doesn't exist, it is created
    with commented defaults and pure-Python defaults are returned.

    Raises FileNotFoundError if an explicit *path* is given but doesn't exist.
    """
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as f:
            return _build_config(tomllib.load(f))

    if not _CONFIG_FILE.exists():
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(_DEFAULT_TOML)
        return Config()

    with _CONFIG_FILE.open("rb") as f:
        return _build_config(tomllib.load(f))
