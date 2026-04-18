"""
Anghiari — maps the inscrutable to the known.

Maps free-text attack descriptions to MITRE ATT&CK Enterprise techniques
using local semantic embeddings (Harrier) and local reranking.
"""

from importlib.metadata import PackageNotFoundError, version

from .mapper import search_technique
from .models import SearchResult, TechniqueMatch

try:
    __version__ = version("anghiari")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "search_technique", "SearchResult", "TechniqueMatch"]
