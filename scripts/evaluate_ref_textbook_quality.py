#!/usr/bin/env python3
"""Evaluate retrieval quality for ref-* Qdrant collections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import ingest_reference_textbooks_qdrant as ref_ingest


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class QueryCase:
    collection: str
    query: str
    keywords: tuple[str, ...]


CASES: list[QueryCase] = [
    QueryCase("ref-finance", "How do you calculate cost of goods sold?", ("cost of goods sold", "cogs", "inventory", "ending inventory", "beginning inventory")),
    QueryCase("ref-finance", "What is contribution margin?", ("contribution margin", "variable cost", "sales revenue", "break-even")),
    QueryCase("ref-negotiation", "What is the bracketing gambit?", ("bracketing", "gambit", "offer", "counteroffer", "range")),
    QueryCase("ref-negotiation", "When should you use higher authority?", ("higher authority", "authority", "approval", "decision maker", "negotiation tactic")),
    QueryCase("ref-engineering", "What is geometric dimensioning and tolerancing?", ("geometric dimensioning", "tolerancing", "gd&t", "tolerance", "symbol")),
    QueryCase("ref-engineering", "What are datum references?", ("datum", "datum reference", "reference frame", "feature control frame")),
    QueryCase("ref-networking", "What is the OSI model?", ("osi", "layer", "physical", "data link", "network", "transport", "session", "presentation", "application")),
    QueryCase("ref-networking", "How does TCP handshake work?", ("tcp", "handshake", "syn", "ack", "three-way")),
    QueryCase("ref-software", "What is the microservices pattern?", ("microservices", "service", "distributed", "bounded context", "decomposition")),
    QueryCase("ref-software", "What is Big O notation?", ("big o", "asymptotic", "time complexity", "space complexity", "complexity class")),
    QueryCase("ref-ai", "What is transfer learning?", ("transfer learning", "pretrained", "fine-tune", "fine tuning", "domain adaptation")),
    QueryCase("ref-ai", "What are embeddings?", ("embedding", "vector", "semantic", "representation", "high-dimensional")),
]


def _embed_query(query: str, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434") -> list[float]:
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={"model": model, "prompt": query},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    vector = payload.get("embedding", [])
    if not vector:
        raise RuntimeError("Ollama returned empty embedding for query.")
    return vector


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_relevant(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    hits = sum(1 for kw in keywords if kw in normalized)
    return hits >= 1


def _load_chunk_lookup(source_dir: Path) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for file_name, collection in ref_ingest.FILE_COLLECTION_MAP.items():
        path = source_dir / file_name
        markdown = ref_ingest._read_text(path)
        chunks = ref_ingest._build_chunks(file_name=file_name, collection=collection, markdown=markdown)
        for chunk in chunks:
            lookup[(file_name, chunk.chunk_index)] = chunk.text
    return lookup


def main() -> int:
    source_dir = Path("/mnt/d/markdown")
    output_path = REPO_ROOT / "docs" / "REF-TEXTBOOK-QUALITY-REPORT.md"
    qdrant_url = "http://localhost:6333"
    chunk_lookup = _load_chunk_lookup(source_dir)

    results_by_collection: dict[str, list[dict]] = {}

    for case in CASES:
        query_vector = _embed_query(case.query)
        response = requests.post(
            f"{qdrant_url}/collections/{case.collection}/points/search",
            json={"vector": query_vector, "limit": 3, "with_payload": True, "with_vector": False},
            timeout=60,
        )
        response.raise_for_status()
        hits = response.json().get("result", [])
        rows: list[dict] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            source_file = str(payload.get("source_file", ""))
            chunk_index = int(payload.get("chunk_index", 0) or 0)
            chunk_text = chunk_lookup.get((source_file, chunk_index), "")
            relevant = _is_relevant(chunk_text, case.keywords)
            rows.append(
                {
                    "score": float(hit.get("score") or 0.0),
                    "source_file": source_file,
                    "heading": str(payload.get("heading", "")),
                    "chunk_index": chunk_index,
                    "relevant": relevant,
                    "chunk_text": chunk_text,
                }
            )

        relevant_count = sum(1 for row in rows if row["relevant"])
        query_relevant = relevant_count >= 2
        bucket = results_by_collection.setdefault(case.collection, [])
        bucket.append(
            {
                "query": case.query,
                "keywords": case.keywords,
                "query_relevant": query_relevant,
                "relevant_count": relevant_count,
                "results": rows,
            }
        )

    lines: list[str] = []
    lines.append("# Reference Textbook Retrieval Quality Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}")
    lines.append("- Method: Qdrant vector search (top 3) + chunk text reconstruction from source markdown.")
    lines.append("- Relevance rule: query judged `relevant` when at least 2 of top 3 hits are relevant.")
    lines.append("")

    overall_queries = 0
    overall_relevant = 0

    for collection in sorted(results_by_collection):
        entries = results_by_collection[collection]
        coll_relevant_queries = sum(1 for e in entries if e["query_relevant"])
        coll_flag = coll_relevant_queries < len(entries)
        lines.append(f"## {collection}")
        lines.append("")
        lines.append(f"- Queries relevant: {coll_relevant_queries}/{len(entries)}")
        lines.append(f"- Collection status: {'FLAGGED (needs chunking/embedding review)' if coll_flag else 'PASS'}")
        lines.append("")
        for entry in entries:
            overall_queries += 1
            overall_relevant += 1 if entry["query_relevant"] else 0
            lines.append(f"### Query: {entry['query']}")
            lines.append("")
            lines.append(
                f"- Verdict: **{'relevant' if entry['query_relevant'] else 'not relevant'}** "
                f"({entry['relevant_count']}/3 relevant results)"
            )
            lines.append("")
            for idx, row in enumerate(entry["results"], start=1):
                lines.append(
                    f"{idx}. score={row['score']:.4f} | source={row['source_file']} | "
                    f"heading={row['heading']} | chunk={row['chunk_index']} | "
                    f"result={'relevant' if row['relevant'] else 'not relevant'}"
                )
                lines.append("")
                lines.append("```text")
                lines.append((row["chunk_text"] or "[missing chunk text]").strip()[:1800])
                lines.append("```")
                lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Relevant queries: {overall_relevant}/{overall_queries}")
    lines.append(f"- Flagged collections: {', '.join([c for c, e in sorted(results_by_collection.items()) if sum(1 for x in e if x['query_relevant']) < len(e)]) or 'none'}")
    lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "relevant_queries": overall_relevant, "total_queries": overall_queries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
