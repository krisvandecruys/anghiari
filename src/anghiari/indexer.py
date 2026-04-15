"""
Build the ChromaDB vector index and subtechnique map from MITRE ATT&CK STIX data.

Run with:  gioconda-index
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import chromadb
import requests

from .embedder import embed_documents

DATA_DIR = Path("data")
STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
STIX_CACHE = DATA_DIR / "enterprise-attack.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
SUBTECH_MAP_FILE = DATA_DIR / "subtechnique_map.json"
COLLECTION_NAME = "mitre_techniques"


def fetch_stix() -> dict:
    if STIX_CACHE.exists():
        print(f"Using cached STIX data from {STIX_CACHE}")
        return json.loads(STIX_CACHE.read_text())
    print(f"Fetching STIX bundle from {STIX_URL} ...")
    resp = requests.get(STIX_URL, timeout=60)
    resp.raise_for_status()
    DATA_DIR.mkdir(exist_ok=True)
    STIX_CACHE.write_text(resp.text)
    print(f"Saved to {STIX_CACHE}")
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

        techniques.append({
            "mitre_id": mitre_id,
            "name": obj.get("name", ""),
            "description": description,
            "tactic": tactic,
            "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
        })

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
    texts = [f"{t['name']}. {t['description']}" for t in techniques]

    print(f"Embedding {len(texts)} techniques ...")
    embeddings = embed_documents(texts, show_progress=True)

    print(f"Storing in ChromaDB at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

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
    DATA_DIR.mkdir(exist_ok=True)

    bundle = fetch_stix()
    techniques = extract_techniques(bundle)

    if not techniques:
        print("No techniques found — aborting.", file=sys.stderr)
        sys.exit(1)

    subtech_map = build_subtechnique_map(techniques)
    SUBTECH_MAP_FILE.write_text(json.dumps(subtech_map, indent=2))
    parent_count = len(subtech_map)
    subtech_count = sum(len(v) for v in subtech_map.values())
    print(
        f"Subtechnique map: {parent_count} parents, "
        f"{subtech_count} subtechniques → {SUBTECH_MAP_FILE}"
    )

    embed_and_index(techniques)
    print("Index build complete.")
