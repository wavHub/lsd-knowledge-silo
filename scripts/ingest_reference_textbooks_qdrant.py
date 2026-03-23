#!/usr/bin/env python3
"""Ingest reference markdown textbooks into Qdrant using Ollama embeddings.

Workflow:
1. Read markdown files from /mnt/d/markdown
2. Split by markdown headings (with fallback section)
3. Sub-chunk long sections with overlap
4. Embed via Ollama nomic-embed-text (768d)
5. Upsert into target ref-* Qdrant collections
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


FILE_COLLECTION_MAP: dict[str, str] = {
    "managerial accounting, 16th edition.md": "ref-finance",
    "_MConverter.eu_Cost Estimation Methods and Tools.md": "ref-finance",
    "Power Negotiations.md": "ref-negotiation",
    "ASME-Y14.5-2018-R2024-Dimensioning-and-Tolerancing.md": "ref-engineering",
    "Making Embedded Systems.md": "ref-engineering",
    "Data Communications and Networking 5th Edition.md": "ref-networking",
    "Fundentals-of-Software-Architecture-an-Engineering-Approach-1.md": "ref-software",
    "Introduction_to_Algorithms_Third_Edition_(2009).md": "ref-software",
    "AI_Training_Curriculum_Resource_Table.md": "ref-ai",
}


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class ChunkItem:
    file_name: str
    collection: str
    heading: str
    text: str
    chunk_index: int
    total_chunks: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        match = HEADING_RE.match(line.strip())
        if match:
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_heading, section_text))
            current_heading = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_heading, section_text))
    return sections


def _subchunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 250) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            carry = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{carry}\n\n{para}".strip() if carry else para
        else:
            cursor = 0
            while cursor < len(para):
                nxt = min(len(para), cursor + max_chars)
                chunks.append(para[cursor:nxt])
                if nxt >= len(para):
                    break
                cursor = max(0, nxt - overlap_chars)
            current = ""

    if current:
        chunks.append(current)
    return chunks


def _build_chunks(file_name: str, collection: str, markdown: str) -> list[ChunkItem]:
    sections = _split_sections(markdown)
    raw_chunks: list[tuple[str, str]] = []
    for heading, section_text in sections:
        for sub in _subchunk_text(section_text):
            raw_chunks.append((heading, sub))

    items: list[ChunkItem] = []
    total = len(raw_chunks)
    for idx, (heading, text) in enumerate(raw_chunks, start=1):
        items.append(
            ChunkItem(
                file_name=file_name,
                collection=collection,
                heading=heading,
                text=text,
                chunk_index=idx,
                total_chunks=total,
            )
        )
    return items


def _ensure_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool = False) -> None:
    existing = {item.name for item in client.get_collections().collections}
    if collection in existing:
        if recreate:
            client.delete_collection(collection_name=collection)
            existing.remove(collection)
        else:
            info = client.get_collection(collection)
            current_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if current_size != vector_size:
                raise RuntimeError(
                    f"Collection {collection} exists with vector size {current_size}, expected {vector_size}."
                )
            return
    if collection in existing:
        info = client.get_collection(collection)
        current_size = info.config.params.vectors.size  # type: ignore[union-attr]
        if current_size != vector_size:
            raise RuntimeError(
                f"Collection {collection} exists with vector size {current_size}, expected {vector_size}."
            )
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE, on_disk=True),
    )


def _embed_text(ollama_url: str, model: str, text: str) -> list[float]:
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    vector = payload.get("embedding", [])
    if not vector:
        raise RuntimeError("Ollama returned empty embedding.")
    return vector


def _upsert_points(client: QdrantClient, collection: str, points: Iterable[qmodels.PointStruct], batch_size: int = 64) -> int:
    batch: list[qmodels.PointStruct] = []
    total = 0
    for point in points:
        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection, points=batch)
            total += len(batch)
            batch = []
    if batch:
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)
    return total


def _stable_id(item: ChunkItem) -> str:
    raw = f"{item.collection}|{item.file_name}|{item.chunk_index}|{item.total_chunks}|{item.heading}|{item.text[:128]}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"niibo-ref-{digest}"))


def run(source_dir: Path, qdrant_url: str, ollama_url: str, ollama_model: str, recreate_collections: bool = False) -> dict:
    files = []
    for file_name, collection in FILE_COLLECTION_MAP.items():
        path = source_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing expected source file: {path}")
        files.append((path, collection))

    client = QdrantClient(url=qdrant_url, timeout=120)

    started = time.time()
    report: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": qdrant_url,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "source_dir": str(source_dir),
        "files_total": len(files),
        "collections": {},
        "chunks_total": 0,
        "points_upserted_total": 0,
        "recreate_collections": recreate_collections,
    }

    collection_points: dict[str, list[qmodels.PointStruct]] = {}
    vector_size = 0

    for path, collection in files:
        markdown = _read_text(path)
        chunks = _build_chunks(path.name, collection, markdown)
        report["chunks_total"] += len(chunks)
        col_file = report["collections"].setdefault(collection, {"files": 0, "chunks": 0, "points_upserted": 0})
        col_file["files"] += 1
        col_file["chunks"] += len(chunks)

        for chunk in chunks:
            vector = _embed_text(ollama_url=ollama_url, model=ollama_model, text=chunk.text)
            if not vector_size:
                vector_size = len(vector)
            if len(vector) != vector_size:
                raise RuntimeError(f"Inconsistent vector size for {path.name}: {len(vector)} != {vector_size}")
            payload = {
                "source_file": chunk.file_name,
                "source_path": str(path),
                "heading": chunk.heading,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "chunk_text": chunk.text,
                "chunk_chars": len(chunk.text),
                "collection": chunk.collection,
                "ingested_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }
            point = qmodels.PointStruct(id=_stable_id(chunk), vector=vector, payload=payload)
            collection_points.setdefault(collection, []).append(point)

    if not vector_size:
        raise RuntimeError("No chunks built from source files.")

    for collection, points in collection_points.items():
        _ensure_collection(client, collection, vector_size=vector_size, recreate=recreate_collections)
        written = _upsert_points(client, collection, points, batch_size=64)
        report["collections"][collection]["points_upserted"] = written
        report["points_upserted_total"] += written

    report["elapsed_seconds"] = round(time.time() - started, 2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest reference markdown textbooks into Qdrant ref-* collections.")
    parser.add_argument("--source-dir", default="/mnt/d/markdown")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--report-path", default=str(Path(__file__).resolve().parents[1] / "docs" / "REF-TEXTBOOK-INGEST-REPORT.json"))
    parser.add_argument(
        "--recreate-collections",
        action="store_true",
        help="Delete and recreate target ref-* collections before upsert.",
    )
    args = parser.parse_args()

    report = run(
        source_dir=Path(args.source_dir),
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        recreate_collections=args.recreate_collections,
    )
    Path(args.report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report_path={args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
