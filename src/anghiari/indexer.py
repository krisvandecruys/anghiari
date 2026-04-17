"""
Build the ChromaDB vector index and subtechnique map from MITRE ATT&CK STIX data.
"""

import json
import re
import sys
from collections import defaultdict

import chromadb
import requests

from .embedder import embed_documents

COLLECTION_NAME = "mitre_techniques"


def fetch_stix() -> dict:
    from .config import get_config

    cfg = get_config()
    if cfg.stix_cache.exists():
        print(f"Using cached STIX data from {cfg.stix_cache}")
        return json.loads(cfg.stix_cache.read_text())
    print(f"Fetching STIX bundle from {cfg.stix.url} ...")
    resp = requests.get(cfg.stix.url, timeout=cfg.stix.fetch_timeout)
    resp.raise_for_status()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.stix_cache.write_text(resp.text)
    print(f"Saved to {cfg.stix_cache}")
    return resp.json()


def extract_techniques(bundle: dict) -> list[dict]:
    techniques = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        mitre_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                mitre_id = ref.get("external_id", "")
                break
        if not mitre_id or not re.match(r"^T\d{4}", mitre_id):
            continue

        tactic = ""
        phases = obj.get("kill_chain_phases", [])
        if phases:
            tactic = phases[0].get("phase_name", "")

        description = obj.get("description", "").strip()
        if len(description) > 1000:
            description = description[:1000].rsplit(" ", 1)[0] + "..."

        techniques.append(
            {
                "mitre_id": mitre_id,
                "name": obj.get("name", ""),
                "description": description,
                "tactic": tactic,
                "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
            }
        )

    # Prefix subtechnique names with their parent name (e.g. "Phishing: Spearphishing Voice")
    parent_names = {
        t["mitre_id"]: t["name"] for t in techniques if not t["is_subtechnique"]
    }
    for t in techniques:
        if t["is_subtechnique"]:
            parent_id = t["mitre_id"].split(".")[0]
            if parent_name := parent_names.get(parent_id):
                t["name"] = f"{parent_name}: {t['name']}"

    print(f"Parsed {len(techniques)} techniques (including subtechniques)")
    return techniques


def build_subtechnique_map(techniques: list[dict]) -> dict[str, list[dict]]:
    subtech_map: dict[str, list[dict]] = defaultdict(list)
    for t in techniques:
        if t["is_subtechnique"]:
            parent_id = t["mitre_id"].split(".")[0]
            subtech_map[parent_id].append(t)
    return dict(subtech_map)


def embed_and_index(techniques: list[dict]) -> None:
    from .config import get_config

    chroma_dir = get_config().chroma_dir
    texts = [f"{t['name']}. {t['description']}" for t in techniques]

    print(f"Embedding {len(texts)} techniques ...")
    embeddings = embed_documents(texts, show_progress=True)

    print(f"Storing in ChromaDB at {chroma_dir} ...")
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[t["mitre_id"] for t in techniques],
        embeddings=embeddings.tolist(),
        metadatas=techniques,
        documents=texts,
    )
    print(f"Indexed {collection.count()} techniques in ChromaDB")


def main() -> None:
    import logging

    # Quick configure for standalone indexer
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from .config import get_config

    cfg = get_config()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    bundle = fetch_stix()
    techniques = extract_techniques(bundle)

    if not techniques:
        print("No techniques found — aborting.", file=sys.stderr)
        sys.exit(1)

    subtech_map = build_subtechnique_map(techniques)
    cfg.subtech_map.write_text(json.dumps(subtech_map, indent=2))
    parent_count = len(subtech_map)
    subtech_count = sum(len(v) for v in subtech_map.values())
    print(
        f"Subtechnique map: {parent_count} parents, "
        f"{subtech_count} subtechniques → {cfg.subtech_map}"
    )

    embed_and_index(techniques)
    print("Index build complete.")
