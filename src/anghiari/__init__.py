"""
Anghiari — maps the inscrutable to the known.

Maps free-text attack descriptions to MITRE ATT&CK Enterprise techniques
using local semantic embeddings (Harrier) and local LLM reasoning (Nemotron).
"""

from .__about__ import __version__
from .mapper import search_technique
from .models import SearchResult, TechniqueMatch

__all__ = ["__version__", "search_technique", "SearchResult", "TechniqueMatch"]
