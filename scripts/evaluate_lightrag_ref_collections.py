#!/usr/bin/env python3
"""Run LightRAG KG extraction for ref-* textbook collections with local Ollama."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import networkx as nx
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

import ingest_reference_textbooks_qdrant as ref_ingest


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class QueryCase:
    collection: str
    query: str


QUERY_CASES: tuple[QueryCase, ...] = (
    QueryCase("ref-finance", "How do you calculate cost of goods sold?"),
    QueryCase("ref-finance", "What is contribution margin?"),
    QueryCase("ref-negotiation", "What is the bracketing gambit?"),
    QueryCase("ref-negotiation", "When should you use higher authority?"),
    QueryCase("ref-engineering", "What is geometric dimensioning and tolerancing?"),
    QueryCase("ref-engineering", "What are datum references?"),
    QueryCase("ref-networking", "What is the OSI model?"),
    QueryCase("ref-networking", "How does TCP handshake work?"),
    QueryCase("ref-software", "What is the microservices pattern?"),
    QueryCase("ref-software", "What is Big O notation?"),
    QueryCase("ref-ai", "What is transfer learning?"),
    QueryCase("ref-ai", "What are embeddings?"),
)

COLLECTION_ORDER: tuple[str, ...] = (
    "ref-negotiation",
    "ref-finance",
    "ref-engineering",
    "ref-networking",
    "ref-software",
    "ref-ai",
)

KEY_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ref-finance": (
        "cost of goods sold",
        "cogs",
        "contribution margin",
        "break-even",
        "budget",
        "variance",
        "standard cost",
        "activity-based",
        "cost behavior",
        "throughput",
        "overhead",
        "inventory",
    ),
    "ref-negotiation": (
        "gambit",
        "higher authority",
        "bracketing",
        "offer",
        "counteroffer",
        "concession",
        "close",
        "leverage",
        "tactic",
        "win-win",
    ),
    "ref-engineering": (
        "gdt",
        "gd&t",
        "datum",
        "tolerance",
        "feature control frame",
        "dimensioning",
        "embedded",
        "firmware",
        "hardware",
        "interface",
    ),
    "ref-networking": (
        "osi",
        "tcp",
        "udp",
        "handshake",
        "ip",
        "routing",
        "switching",
        "subnet",
        "protocol",
        "ethernet",
    ),
    "ref-software": (
        "architecture",
        "microservices",
        "modularity",
        "algorithm",
        "big o",
        "complexity",
        "pattern",
        "trade-off",
        "scalability",
        "availability",
    ),
    "ref-ai": (
        "embedding",
        "transfer learning",
        "training",
        "inference",
        "model",
        "dataset",
        "evaluation",
        "fine-tuning",
    ),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _collection_files() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for file_name, collection in ref_ingest.FILE_COLLECTION_MAP.items():
        grouped.setdefault(collection, []).append(file_name)
    for files in grouped.values():
        files.sort()
    return grouped


def _selective_sections(file_name: str, text: str, collection: str, max_sections_per_file: int) -> list[tuple[str, str]]:
    sections = ref_ingest._split_sections(text)
    if not sections:
        return []

    keywords = KEY_SECTION_KEYWORDS.get(collection, ())
    if not keywords:
        return sections[:max_sections_per_file]

    selected: list[tuple[str, str]] = []
    lowered_keywords = tuple(k.lower() for k in keywords)
    for heading, section_text in sections:
        probe = f"{heading}\n{section_text[:1200]}".lower()
        if any(keyword in probe for keyword in lowered_keywords):
            selected.append((heading, section_text))

    if not selected:
        # Fallback: keep opening sections to guarantee coverage.
        return sections[:max_sections_per_file]
    return selected[:max_sections_per_file]


def _collection_corpus(
    source_dir: Path,
    files: list[str],
    sample_chars: int,
    kg_mode: str,
    max_sections_per_file: int,
) -> str:
    sections: list[str] = []
    for file_name in files:
        text = _read_text(source_dir / file_name).strip()
        if not text:
            continue
        if kg_mode == "full":
            selected_sections = [("Full File", text)]
        else:
            selected_sections = _selective_sections(
                file_name=file_name,
                text=text,
                collection=ref_ingest.FILE_COLLECTION_MAP.get(file_name, ""),
                max_sections_per_file=max_sections_per_file,
            )
        for heading, section_text in selected_sections:
            sections.append(f"# Source File: {file_name}\n## Section: {heading}\n\n{section_text}")
    corpus = "\n\n".join(sections).strip()
    if sample_chars > 0 and len(corpus) > sample_chars:
        return corpus[:sample_chars]
    return corpus


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _record_count(path: Path) -> int:
    payload = _load_json(path)
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return len(payload)
    return 0


def _truncate(text: str, limit: int = 700) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + " ..."


async def _run_one_collection(
    *,
    collection: str,
    files: list[str],
    source_dir: Path,
    working_root: Path,
    llm_model: str,
    sample_chars: int,
    kg_mode: str,
    max_sections_per_file: int,
) -> dict[str, Any]:
    work_dir = working_root / collection
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    corpus = _collection_corpus(
        source_dir=source_dir,
        files=files,
        sample_chars=sample_chars,
        kg_mode=kg_mode,
        max_sections_per_file=max_sections_per_file,
    )
    if not corpus:
        raise RuntimeError(f"No corpus text built for {collection}")

    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 24576, "num_predict": 512},
            "timeout": 300,
        },
        llm_model_max_async=1,
        default_llm_timeout=300,
        chunk_token_size=1000,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=0,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            model_name="nomic-embed-text",
            func=partial(
                ollama_embed.func,
                embed_model="nomic-embed-text",
                host="http://localhost:11434",
            ),
        ),
    )

    await rag.initialize_storages()

    run_result: dict[str, Any] = {
        "collection": collection,
        "status": "success",
        "files": files,
        "working_dir": str(work_dir),
        "corpus_chars": len(corpus),
        "sample_chars_limit": sample_chars,
        "kg_mode": kg_mode,
        "queries": [],
    }

    try:
        await rag.ainsert(corpus)

        graph_path = work_dir / "graph_chunk_entity_relation.graphml"
        nodes = 0
        edges = 0
        if graph_path.exists():
            graph = nx.read_graphml(graph_path)
            nodes = int(graph.number_of_nodes())
            edges = int(graph.number_of_edges())

        entities_count = _record_count(work_dir / "vdb_entities.json")
        relationships_count = _record_count(work_dir / "vdb_relationships.json")
        chunks_count = _record_count(work_dir / "vdb_chunks.json")

        run_result["graph"] = {
            "graphml_path": str(graph_path),
            "nodes": nodes,
            "edges": edges,
            "entities_count": entities_count,
            "relationships_count": relationships_count,
            "chunks_count": chunks_count,
        }

        for case in QUERY_CASES:
            if case.collection != collection:
                continue
            response = await rag.aquery(case.query, param=QueryParam(mode="hybrid"))
            run_result["queries"].append(
                {
                    "query": case.query,
                    "mode": "hybrid",
                    "response_preview": _truncate(str(response), 900),
                }
            )
    finally:
        await rag.finalize_storages()

    return run_result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).resolve()
    working_root = Path(args.working_root).resolve()
    working_root.mkdir(parents=True, exist_ok=True)

    grouped = _collection_files()
    targets: list[str] = [c for c in COLLECTION_ORDER if c in grouped]
    if args.only:
        allowed = {x.strip() for x in args.only.split(",") if x.strip()}
        targets = [c for c in targets if c in allowed]
    if not targets:
        raise RuntimeError("No target collections selected.")

    results: list[dict[str, Any]] = []
    for collection in targets:
        print(f"[hybrid-kg] starting collection={collection} mode={args.kg_mode}")
        try:
            result = await _run_one_collection(
                collection=collection,
                files=grouped[collection],
                source_dir=source_dir,
                working_root=working_root,
                llm_model=args.llm_model,
                sample_chars=args.sample_chars,
                kg_mode=args.kg_mode,
                max_sections_per_file=args.max_sections_per_file,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "collection": collection,
                "status": "failed",
                "error": str(exc),
                "files": grouped[collection],
                "working_dir": str((working_root / collection).resolve()),
                "queries": [],
            }
        if result.get("status") == "success":
            graph = result.get("graph", {})
            print(
                "[hybrid-kg] completed "
                f"collection={collection} entities={graph.get('entities_count', 0)} "
                f"relationships={graph.get('relationships_count', 0)} "
                f"chunks={graph.get('chunks_count', 0)}"
            )
        else:
            print(f"[hybrid-kg] failed collection={collection} error={result.get('error','unknown')}")
        results.append(result)

        if args.pilot_only:
            break

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "llm_model": args.llm_model,
        "embedding_model": "nomic-embed-text",
        "kg_mode": args.kg_mode,
        "max_sections_per_file": args.max_sections_per_file,
        "results": results,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="/mnt/d/markdown",
        help="Directory with source markdown files.",
    )
    parser.add_argument(
        "--working-root",
        default=str(REPO_ROOT / "data" / "lightrag-ref"),
        help="Root directory for per-collection LightRAG working dirs.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "docs" / "LIGHTRAG-EVALUATION.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2:3b",
        help="Ollama LLM model for extraction/query.",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run only the first collection in COLLECTION_ORDER (ref-negotiation).",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=60000,
        help="Limit corpus characters per collection for bounded evaluation runtime (0 disables sampling).",
    )
    parser.add_argument(
        "--kg-mode",
        choices=("selective", "full"),
        default="selective",
        help="selective=key sections only (hybrid mode), full=entire files.",
    )
    parser.add_argument(
        "--max-sections-per-file",
        type=int,
        default=28,
        help="When --kg-mode selective, cap extracted sections per file.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Optional comma-separated collection list to run.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    payload = asyncio.run(_run(args))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "collections_ran": len(payload["results"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
