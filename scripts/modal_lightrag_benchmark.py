#!/usr/bin/env python3
"""RN-046 benchmark runner with checkpointing and Modal model cache.

Scope:
- 5 collections (exclude ref-ai)
- 1 representative segment per collection
- 3 models: llama3.2:3b (local), llama3.1:8b (Modal), qwen2.5:32b (Modal)
- 15 runs total with resumability
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import modal

try:
    import ingest_reference_textbooks_qdrant as ref_ingest
except Exception:  # pragma: no cover
    ref_ingest = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR_DEFAULT = "/mnt/d/markdown"
APP_NAME = "lsd-lightrag-benchmark-h100"
REMOTE_WORK_ROOT = "/tmp/lightrag-benchmark"
EMBED_MODEL = "nomic-embed-text"
BENCH_GPU = os.getenv("MODAL_BENCH_GPU", "H100").strip() or "H100"
MODEL_3B = "llama3.2:3b"
MODEL_8B = "llama3.1:8b"
MODEL_32B = "qwen2.5:32b"

COLLECTION_ORDER = (
    "ref-finance",
    "ref-negotiation",
    "ref-engineering",
    "ref-networking",
    "ref-software",
    "ref-ai",
)

COLLECTION_TO_DIVISION = {
    "ref-finance": "finance",
    "ref-negotiation": "sales",
    "ref-engineering": "engineering",
    "ref-networking": "networking",
    "ref-software": "software",
}

DEFAULT_FILE_COLLECTION_MAP: dict[str, str] = {
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

QUERY_BY_COLLECTION: dict[str, str] = {
    "ref-finance": "How do you calculate cost of goods sold?",
    "ref-negotiation": "What is the bracketing gambit?",
    "ref-engineering": "What is geometric dimensioning and tolerancing?",
    "ref-networking": "How does TCP handshake work?",
    "ref-software": "What is the microservices pattern?",
    "ref-ai": "What are embeddings?",
}

KEYWORDS_BY_COLLECTION: dict[str, tuple[str, ...]] = {
    "ref-finance": (
        "cost of goods sold",
        "contribution margin",
        "break-even",
        "overhead",
        "inventory",
        "cost-volume-profit",
        "variance",
    ),
    "ref-negotiation": (
        "bracketing",
        "higher authority",
        "gambit",
        "counteroffer",
        "concession",
        "close",
    ),
    "ref-engineering": (
        "geometric",
        "dimensioning",
        "tolerance",
        "datum",
        "profile tolerance",
        "position tolerance",
        "feature control frame",
    ),
    "ref-networking": (
        "tcp",
        "ip",
        "network layer",
        "osi",
        "routing",
        "packet",
        "handshake",
    ),
    "ref-software": (
        "architecture",
        "microservices",
        "complexity",
        "big o",
        "pattern",
        "trade-off",
    ),
    "ref-ai": (
        "embedding",
        "transfer learning",
        "training",
        "evaluation",
        "fine-tuning",
    ),
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(?im)^\s*page\s+\d+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _split_sections(text: str) -> list[tuple[str, str]]:
    if ref_ingest is not None and hasattr(ref_ingest, "_split_sections"):
        return ref_ingest._split_sections(text)  # type: ignore[attr-defined]
    lines = _normalize_text(text).split("\n")
    sections: list[tuple[str, str]] = []
    heading = "Preamble"
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
            heading = stripped
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))
    return [item for item in sections if item[1].strip()]


def _score_section(collection: str, heading: str, body: str) -> float:
    probe = f"{heading}\n{body[:2500]}".lower()
    score = 0.0
    for keyword in KEYWORDS_BY_COLLECTION.get(collection, ()):  # pragma: no branch
        if keyword in probe:
            score += 8.0
    heading_l = heading.lower()
    if heading_l.startswith("#"):
        score += 2.0
    if "chapter" in heading_l or "section" in heading_l:
        score += 3.0
    tokens = _token_estimate(body)
    if 1200 <= tokens <= 2200:
        score += 5.0
    elif 800 <= tokens < 1200:
        score += 2.5
    elif tokens > 2200:
        score += 2.0
    return score


def _split_segment(text: str, min_tokens: int = 1200, max_tokens: int = 2200) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    target_chars_min = min_tokens * 4
    target_chars_max = max_tokens * 4
    if len(normalized) <= target_chars_max:
        return normalized

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunk_parts: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= target_chars_max:
            current = candidate
            continue
        if current:
            chunk_parts.append(current)
        current = para
    if current:
        chunk_parts.append(current)

    if not chunk_parts:
        return normalized[:target_chars_max]

    for part in chunk_parts:
        if len(part) >= target_chars_min:
            return part[:target_chars_max]
    return chunk_parts[0][:target_chars_max]


def build_segments(
    source_dir: Path,
    *,
    per_collection: int = 1,
    exclude_collections: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude = {item.strip().lower() for item in (exclude_collections or set()) if item.strip()}
    file_collection_map = (
        ref_ingest.FILE_COLLECTION_MAP
        if ref_ingest is not None and hasattr(ref_ingest, "FILE_COLLECTION_MAP")
        else DEFAULT_FILE_COLLECTION_MAP
    )
    grouped: dict[str, list[str]] = {}
    for file_name, collection in file_collection_map.items():
        if collection.strip().lower() in exclude:
            continue
        grouped.setdefault(collection, []).append(file_name)
    for file_names in grouped.values():
        file_names.sort()

    segments: list[dict[str, Any]] = []
    for collection in COLLECTION_ORDER:
        if collection.strip().lower() in exclude:
            continue
        candidates: list[dict[str, Any]] = []
        for file_name in grouped.get(collection, []):
            source_path = source_dir / file_name
            if not source_path.exists():
                continue
            raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
            sections = _split_sections(raw_text)
            for heading, body in sections:
                body_text = _split_segment(body)
                if _token_estimate(body_text) < 600:
                    continue
                score = _score_section(collection, heading, body_text)
                candidates.append(
                    {
                        "collection": collection,
                        "division": COLLECTION_TO_DIVISION.get(collection, ""),
                        "file_name": file_name,
                        "heading": heading,
                        "score": round(score, 3),
                        "token_est": _token_estimate(body_text),
                        "text": body_text,
                    }
                )
        candidates.sort(key=lambda item: (float(item["score"]), int(item["token_est"])), reverse=True)
        selected = candidates[:per_collection]
        if len(selected) < per_collection:
            fallback = [c for c in candidates if c not in selected][: per_collection - len(selected)]
            selected.extend(fallback)

        for idx, item in enumerate(selected, start=1):
            fingerprint = hashlib.sha1(
                f"{collection}|{item['file_name']}|{item['heading']}|{item['text'][:240]}".encode("utf-8")
            ).hexdigest()[:12]
            segments.append(
                {
                    "segment_id": f"{collection}-{idx:02d}-{fingerprint}",
                    "collection": collection,
                    "division": item["division"],
                    "file_name": item["file_name"],
                    "heading": item["heading"],
                    "score": item["score"],
                    "token_est": item["token_est"],
                    "text": item["text"],
                }
            )
    return segments


def _query_for_collection(collection: str) -> str:
    return QUERY_BY_COLLECTION.get(collection, "Summarize key relationships in this segment.")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"generated_at": _utcnow_iso(), "runs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("runs"), dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"generated_at": _utcnow_iso(), "runs": {}}


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_from_graph(graph) -> tuple[list[str], list[dict[str, str]]]:
    entities: list[str] = []
    for node in list(graph.nodes())[:8]:
        entities.append(str(node))

    relations: list[dict[str, str]] = []
    for from_node, to_node, data in list(graph.edges(data=True))[:8]:
        relation = "related_to"
        if isinstance(data, dict):
            relation = str(
                data.get("relation")
                or data.get("relationship")
                or data.get("description")
                or data.get("label")
                or "related_to"
            )
        relations.append({"from": str(from_node), "relation": relation, "to": str(to_node)})
    return entities, relations


async def _run_lightrag_segment(
    *,
    model_name: str,
    segment: dict[str, Any],
    work_dir: Path,
    ollama_host: str,
) -> dict[str, Any]:
    import networkx as nx
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    status = "success"
    error = ""
    nodes = 0
    edges = 0
    query_preview = ""
    sample_entities: list[str] = []
    sample_relationships: list[dict[str, str]] = []

    rag = LightRAG(
        working_dir=str(work_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        llm_model_kwargs={
            "host": ollama_host,
            "options": {"num_ctx": 16384, "num_predict": 768},
            "timeout": 1200,
        },
        llm_model_max_async=1,
        default_llm_timeout=1200,
        chunk_token_size=1200,
        chunk_overlap_token_size=120,
        entity_extract_max_gleaning=1,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            model_name=EMBED_MODEL,
            func=partial(ollama_embed.func, embed_model=EMBED_MODEL, host=ollama_host),
        ),
    )

    try:
        await rag.initialize_storages()
        await rag.ainsert(
            f"# Collection: {segment['collection']}\n"
            f"# Segment: {segment['segment_id']}\n"
            f"# Heading: {segment.get('heading','')}\n\n"
            f"{segment.get('text','')}"
        )

        graph_path = work_dir / "graph_chunk_entity_relation.graphml"
        if graph_path.exists():
            graph = nx.read_graphml(graph_path)
            nodes = int(graph.number_of_nodes())
            edges = int(graph.number_of_edges())
            sample_entities, sample_relationships = _sample_from_graph(graph)

        query_output = await rag.aquery(_query_for_collection(str(segment["collection"])), param=QueryParam(mode="hybrid"))
        query_preview = " ".join(str(query_output).split())[:700]
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass

    elapsed = round(time.time() - started, 2)
    return {
        "model": model_name,
        "segment_id": str(segment["segment_id"]),
        "collection": str(segment["collection"]),
        "division": str(segment.get("division", "")),
        "file_name": str(segment.get("file_name", "")),
        "heading": str(segment.get("heading", "")),
        "segment_tokens_est": int(segment.get("token_est", 0) or 0),
        "time_seconds": elapsed,
        "nodes_extracted": nodes,
        "edges_extracted": edges,
        "status": status,
        "error_message": error,
        "sample_entities": sample_entities,
        "sample_relationships": sample_relationships,
        "query_preview": query_preview,
    }


ollama_volume = modal.Volume.from_name("lsd-lightrag-ollama-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("lightrag-hku==1.4.11", "networkx", "numpy", "requests")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=BENCH_GPU,
    timeout=60 * 60,
    retries=0,
    volumes={"/root/.ollama": ollama_volume},
    single_use_containers=True,
)
def run_remote_segment(model_name: str, segment: dict[str, Any]) -> dict[str, Any]:
    import requests

    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    os.environ["OLLAMA_NUM_PARALLEL"] = "2"
    os.environ["OLLAMA_KEEP_ALIVE"] = "120m"
    os.environ["OLLAMA_CONTEXT_LENGTH"] = "16384"

    ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        for _ in range(90):
            try:
                requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Ollama did not become ready in time.")

        pull_embed = subprocess.run(["ollama", "pull", EMBED_MODEL], capture_output=True, text=True)
        if pull_embed.returncode != 0:
            raise RuntimeError(f"Failed pull {EMBED_MODEL}: {(pull_embed.stdout or pull_embed.stderr)[-300:]}")

        pull_model = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
        if pull_model.returncode != 0:
            raise RuntimeError(f"Failed pull {model_name}: {(pull_model.stdout or pull_model.stderr)[-300:]}")

        work_dir = Path(REMOTE_WORK_ROOT) / model_name.replace(":", "-") / str(segment["segment_id"])
        return asyncio.run(
            _run_lightrag_segment(
                model_name=model_name,
                segment=segment,
                work_dir=work_dir,
                ollama_host="http://127.0.0.1:11434",
            )
        )
    finally:
        try:
            ollama_proc.terminate()
        except Exception:
            pass


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in results:
        model = str(row.get("model", ""))
        entry = by_model.setdefault(
            model,
            {
                "runs": 0,
                "success": 0,
                "time_seconds_total": 0.0,
                "nodes_total": 0,
                "edges_total": 0,
                "by_collection": {},
            },
        )
        entry["runs"] += 1
        if row.get("status") == "success":
            entry["success"] += 1
        entry["time_seconds_total"] += float(row.get("time_seconds", 0) or 0)
        entry["nodes_total"] += int(row.get("nodes_extracted", 0) or 0)
        entry["edges_total"] += int(row.get("edges_extracted", 0) or 0)

        collection = str(row.get("collection", ""))
        coll = entry["by_collection"].setdefault(collection, {"nodes": 0, "edges": 0, "time_seconds": 0.0})
        coll["nodes"] += int(row.get("nodes_extracted", 0) or 0)
        coll["edges"] += int(row.get("edges_extracted", 0) or 0)
        coll["time_seconds"] += float(row.get("time_seconds", 0) or 0)

    for model, payload in by_model.items():
        runs = int(payload["runs"])
        payload["time_seconds_avg"] = round(float(payload["time_seconds_total"]) / max(1, runs), 2)
        payload["time_seconds_total"] = round(float(payload["time_seconds_total"]), 2)
        for coll in payload["by_collection"].values():
            coll["time_seconds"] = round(float(coll["time_seconds"]), 2)
    return by_model


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LIGHTRAG Model Benchmark (RN-046)",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- GPU profile: `{payload.get('gpu_profile', '')}`",
        "- Scope: 5 collections, 1 representative segment each, 3 models (3b/8b/32b)",
        "",
        "## Model Summary",
        "",
        "| Model | Runs | Success | Total Time (sec) | Avg Time (sec) | Nodes | Edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    by_model = payload.get("summary_by_model", {})
    for model in (MODEL_3B, MODEL_8B, MODEL_32B):
        stats = by_model.get(model, {})
        lines.append(
            "| {model} | {runs} | {success} | {total:.2f} | {avg:.2f} | {nodes} | {edges} |".format(
                model=model,
                runs=stats.get("runs", 0),
                success=stats.get("success", 0),
                total=float(stats.get("time_seconds_total", 0.0) or 0.0),
                avg=float(stats.get("time_seconds_avg", 0.0) or 0.0),
                nodes=stats.get("nodes_total", 0),
                edges=stats.get("edges_total", 0),
            )
        )

    lines.extend(
        [
            "",
            "## Per Collection (Actual Counts)",
            "",
            "| Collection | 3b Nodes | 3b Edges | 8b Nodes | 8b Edges | 32b Nodes | 32b Edges |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    collections = sorted({str(r.get("collection", "")) for r in payload.get("results", [])})
    for collection in collections:
        def pair(model: str) -> tuple[int, int]:
            for row in payload.get("results", []):
                if row.get("collection") == collection and row.get("model") == model:
                    return int(row.get("nodes_extracted", 0) or 0), int(row.get("edges_extracted", 0) or 0)
            return 0, 0

        n3, e3 = pair(MODEL_3B)
        n8, e8 = pair(MODEL_8B)
        n32, e32 = pair(MODEL_32B)
        lines.append(f"| {collection} | {n3} | {e3} | {n8} | {e8} | {n32} | {e32} |")

    lines.append("")
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(
    source_dir: str = SOURCE_DIR_DEFAULT,
    output_json: str = str(REPO_ROOT / "docs" / "LIGHTRAG-MODEL-BENCHMARK.json"),
    output_md: str = str(REPO_ROOT / "docs" / "LIGHTRAG-MODEL-BENCHMARK.md"),
    checkpoint_path: str = str(REPO_ROOT / "docs" / "LIGHTRAG-MODEL-BENCHMARK.checkpoint.json"),
    segments_output: str = str(REPO_ROOT / "docs" / "REF-TEXTBOOK-SEGMENTS.json"),
    per_collection: int = 1,
    exclude_collections: str = "ref-ai",
    rerun_failed: bool = True,
) -> None:
    source_root = Path(source_dir).resolve()
    exclude = {item.strip().lower() for item in str(exclude_collections).split(",") if item.strip()}
    segments = build_segments(source_root, per_collection=per_collection, exclude_collections=exclude)
    if not segments:
        raise RuntimeError("No benchmark segments selected.")

    Path(segments_output).parent.mkdir(parents=True, exist_ok=True)
    Path(segments_output).write_text(
        json.dumps(
            {
                "generated_at": _utcnow_iso(),
                "per_collection": per_collection,
                "segment_count": len(segments),
                "exclude_collections": sorted(exclude),
                "segments": segments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checkpoint_file = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_file)
    checkpoint["segments"] = segments
    checkpoint["models"] = [MODEL_3B, MODEL_8B, MODEL_32B]

    results: list[dict[str, Any]] = []

    for model in (MODEL_3B, MODEL_8B, MODEL_32B):
        for segment in segments:
            key = f"{model}|{segment['segment_id']}"
            existing = checkpoint["runs"].get(key)
            if existing and (existing.get("status") == "success" or not rerun_failed):
                results.append(existing)
                continue

            print(f"[benchmark] model={model} segment={segment['segment_id']} collection={segment['collection']}")
            result = run_remote_segment.remote(model, segment)

            checkpoint["runs"][key] = result
            _save_checkpoint(checkpoint_file, checkpoint)
            results.append(result)
            print(
                f"[benchmark] complete model={model} collection={result['collection']} "
                f"status={result['status']} nodes={result['nodes_extracted']} edges={result['edges_extracted']} "
                f"time={result['time_seconds']}"
            )

    summary_by_model = _summarize(results)
    payload = {
        "generated_at": _utcnow_iso(),
        "app_name": APP_NAME,
        "gpu_profile": BENCH_GPU,
        "collections": sorted({str(seg["collection"]) for seg in segments}),
        "segment_count": len(segments),
        "models": [MODEL_3B, MODEL_8B, MODEL_32B],
        "results": results,
        "summary_by_model": summary_by_model,
    }

    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(output_md)
    out_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(json.dumps({"output_json": str(out_json), "output_md": str(out_md), "checkpoint": str(checkpoint_file)}, indent=2))


if __name__ == "__main__":
    main()
