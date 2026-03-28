#!/usr/bin/env python3
"""Run resumable LightRAG extraction for project notebook domains.

- Reads ingested chunk content from Supabase knowledge_chunks.
- Aggregates chunks by document path into extraction segments.
- Uses Modal + Ollama cache for remote extraction.
- Persists graph_nodes/graph_edges/extraction_runs to Supabase.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import modal
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "lsd-lightrag-project-pass"
REMOTE_WORK_ROOT = "/tmp/lightrag-project"
EMBED_MODEL = "nomic-embed-text"
FULL_GPU = os.getenv("MODAL_FULL_GPU", "A10G").strip() or "A10G"

MODEL_3B = "llama3.2:3b"
MODEL_8B = "llama3.1:8b"
MODEL_32B = "qwen2.5:32b"

PROJECT_COLLECTIONS = ["project-binder", "project-workbook", "ops-binder"]
COLLECTION_TO_DIVISION = {name: "software" for name in PROJECT_COLLECTIONS}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _subchunk_text(text: str, max_chars: int = 7000, overlap_chars: int = 700) -> list[str]:
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


def _fetch_collection_rows(base_url: str, headers: dict[str, str], collection: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while True:
        response = requests.get(
            f"{base_url}/rest/v1/knowledge_chunks",
            headers=headers,
            params={
                "select": "id,division,collection,book,chapter,chunk_index,char_count,token_est,content,file_path,heading,document_id,document_title",
                "collection": f"eq.{collection}",
                "content": "not.is.null",
                "status": "eq.extracted",
                "order": "file_path.asc,chunk_index.asc",
                "limit": str(page_size),
                "offset": str(offset),
            },
            timeout=60,
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Failed to fetch chunks collection={collection} status={response.status_code} body={response.text[:300]}"
            )
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def build_project_segments(
    base_url: str,
    headers: dict[str, str],
    max_docs_per_collection: int = 0,
) -> list[dict[str, Any]]:
    """Build extraction segments by aggregating ingested chunks per document."""
    segments: list[dict[str, Any]] = []
    for collection in PROJECT_COLLECTIONS:
        rows = _fetch_collection_rows(base_url, headers, collection)
        doc_map: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            file_path = str(row.get("file_path") or row.get("book") or "unknown")
            doc_map.setdefault(file_path, []).append(row)

        doc_items = sorted(doc_map.items(), key=lambda x: x[0])
        if max_docs_per_collection > 0:
            doc_items = doc_items[:max_docs_per_collection]

        for file_path, doc_rows in doc_items:
            doc_rows_sorted = sorted(doc_rows, key=lambda x: int(x.get("chunk_index", 0) or 0))
            division = str(doc_rows_sorted[0].get("division") or COLLECTION_TO_DIVISION[collection])
            doc_title = str(doc_rows_sorted[0].get("document_title") or Path(file_path).stem)
            blocks: list[str] = []
            for row in doc_rows_sorted:
                heading = str(row.get("heading") or row.get("chapter") or "")
                content = str(row.get("content") or "").strip()
                if not content:
                    continue
                if heading:
                    blocks.append(f"## {heading}\n\n{content}")
                else:
                    blocks.append(content)
            if not blocks:
                continue
            merged = "\n\n".join(blocks)
            for sub_idx, sub_text in enumerate(_subchunk_text(merged), start=1):
                segment_id = hashlib.sha1(
                    f"{collection}|{file_path}|{sub_idx}|{sub_text[:200]}".encode("utf-8")
                ).hexdigest()[:16]
                segments.append(
                    {
                        "segment_id": f"{collection}-{segment_id}",
                        "collection": collection,
                        "division": division,
                        "file_name": file_path,
                        "heading": doc_title,
                        "section_index": sub_idx,
                        "sub_index": sub_idx,
                        "char_count": len(sub_text),
                        "token_est": max(1, len(sub_text) // 4),
                        "text": sub_text,
                    }
                )
    segments.sort(key=lambda x: (str(x["collection"]), str(x["file_name"]), int(x["section_index"]), int(x["sub_index"])))
    return segments


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


async def _run_lightrag_segment(
    *,
    model_name: str,
    segment: dict[str, Any],
    work_dir: Path,
    ollama_host: str,
) -> dict[str, Any]:
    import networkx as nx
    from lightrag import LightRAG
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
    entities: list[str] = []
    relations: list[dict[str, str]] = []

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
        chunk_token_size=1400,
        chunk_overlap_token_size=140,
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
            entities = [str(node) for node in list(graph.nodes())]
            for from_node, to_node, data in list(graph.edges(data=True)):
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
        "division": str(segment["division"]),
        "file_name": str(segment.get("file_name", "")),
        "heading": str(segment.get("heading", "")),
        "char_count": int(segment.get("char_count", 0) or 0),
        "token_est": int(segment.get("token_est", 0) or 0),
        "time_seconds": elapsed,
        "nodes_extracted": nodes,
        "edges_extracted": edges,
        "entities": entities,
        "relationships": relations,
        "status": status,
        "error_message": error,
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
    gpu=FULL_GPU,
    timeout=60 * 60,
    retries=0,
    volumes={"/root/.ollama": ollama_volume},
    single_use_containers=True,
)
def run_remote_segment(model_name: str, segment: dict[str, Any]) -> dict[str, Any]:
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


def choose_model_from_benchmark(benchmark_path: Path) -> str:
    if not benchmark_path.exists():
        return MODEL_8B

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    summary = payload.get("summary_by_model")
    if not isinstance(summary, dict):
        return MODEL_8B

    e3 = float((summary.get(MODEL_3B) or {}).get("edges_total", 0) or 0)
    e8 = float((summary.get(MODEL_8B) or {}).get("edges_total", 0) or 0)
    e32 = float((summary.get(MODEL_32B) or {}).get("edges_total", 0) or 0)

    if e8 >= (2.0 * max(1.0, e3)):
        if e32 >= (2.0 * max(1.0, e8)):
            return MODEL_32B
        return MODEL_8B
    return MODEL_3B


def _group_results_by_collection(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row.get("collection", "")), []).append(row)
    return grouped


def _build_graph_payload(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_seen: dict[str, dict[str, Any]] = {}
    edges_seen: dict[str, dict[str, Any]] = {}

    for row in results:
        if row.get("status") != "success":
            continue
        division = str(row.get("division", ""))
        collection = str(row.get("collection", ""))
        source_book = str(row.get("file_name", ""))
        source_chunk_id = str(row.get("segment_id", ""))
        extraction_model = str(row.get("model", ""))

        for entity in row.get("entities", []):
            label = str(entity).strip()
            if not label:
                continue
            node_id = f"{division}:{hashlib.sha1(label.lower().encode('utf-8')).hexdigest()[:20]}"
            nodes_seen.setdefault(
                node_id,
                {
                    "id": node_id,
                    "division": division,
                    "label": label,
                    "entity_type": None,
                    "properties": {"collection": collection},
                    "source_book": source_book,
                    "source_chapter": str(row.get("heading", "")),
                    "source_chunk_id": source_chunk_id,
                    "extraction_model": extraction_model,
                    "confidence": None,
                },
            )

        for rel in row.get("relationships", []):
            src_label = str(rel.get("from", "")).strip()
            dst_label = str(rel.get("to", "")).strip()
            relation = str(rel.get("relation", "related_to")).strip() or "related_to"
            if not src_label or not dst_label:
                continue
            src_id = f"{division}:{hashlib.sha1(src_label.lower().encode('utf-8')).hexdigest()[:20]}"
            dst_id = f"{division}:{hashlib.sha1(dst_label.lower().encode('utf-8')).hexdigest()[:20]}"
            edge_id = hashlib.sha1(f"{division}|{src_id}|{relation}|{dst_id}".encode("utf-8")).hexdigest()[:32]
            edges_seen.setdefault(
                edge_id,
                {
                    "id": edge_id,
                    "division": division,
                    "from_node_id": src_id,
                    "to_node_id": dst_id,
                    "relationship": relation,
                    "properties": {"collection": collection},
                    "source_book": source_book,
                    "source_chunk_id": source_chunk_id,
                    "extraction_model": extraction_model,
                    "confidence": None,
                },
            )

    return list(nodes_seen.values()), list(edges_seen.values())


def _get_supabase_creds() -> tuple[str, str]:
    keymaster = Path("/home/niiboAdmin/dev/keymaster/keymaster.py")
    supabase_url = subprocess.check_output(["python3", str(keymaster), "get", "supabase-url"], text=True).strip().rstrip("/")
    service_key = subprocess.check_output(
        ["python3", str(keymaster), "get", "supabase-service-role-key"],
        text=True,
    ).strip()
    return supabase_url, service_key


def _table_available(base_url: str, headers: dict[str, str], table: str) -> bool:
    response = requests.get(f"{base_url}/rest/v1/{table}?select=id&limit=1", headers=headers, timeout=20)
    return response.status_code != 404


def _upsert_rows(base_url: str, headers: dict[str, str], table: str, rows: list[dict[str, Any]], batch_size: int = 200) -> None:
    if not rows:
        return
    endpoint = f"{base_url}/rest/v1/{table}?on_conflict=id"
    upsert_headers = {
        **headers,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        resp = requests.post(endpoint, headers=upsert_headers, data=json.dumps(batch), timeout=120)
        if resp.status_code >= 300:
            raise RuntimeError(f"Upsert failed table={table} status={resp.status_code} body={resp.text[:400]}")


def persist_to_supabase(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    run_id: str,
    model_name: str,
    results: list[dict[str, Any]],
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {"status": "skipped", "reason": "--skip-supabase flag"}

    base_url, service_key = _get_supabase_creds()
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}

    required_tables = ["graph_nodes", "graph_edges", "extraction_runs"]
    missing = [table for table in required_tables if not _table_available(base_url, headers, table)]
    if missing:
        return {"status": "skipped", "reason": f"missing tables: {', '.join(missing)}"}

    extraction_runs_rows: list[dict[str, Any]] = []
    for collection, division in COLLECTION_TO_DIVISION.items():
        coll_rows = [r for r in results if r.get("collection") == collection]
        extraction_runs_rows.append(
            {
                "id": f"{run_id}-{collection}",
                "division": division,
                "collection": collection,
                "model": model_name,
                "gpu": FULL_GPU,
                "provider": "modal",
                "total_chunks": len(coll_rows),
                "processed_chunks": sum(1 for x in coll_rows if x.get("status") == "success"),
                "total_nodes": sum(int(x.get("nodes_extracted", 0) or 0) for x in coll_rows),
                "total_edges": sum(int(x.get("edges_extracted", 0) or 0) for x in coll_rows),
                "status": "completed",
                "started_at": _utcnow_iso(),
                "completed_at": _utcnow_iso(),
                "cost_estimate": None,
                "error_log": None,
            }
        )

    _upsert_rows(base_url, headers, "extraction_runs", extraction_runs_rows)
    _upsert_rows(base_url, headers, "graph_nodes", nodes)
    _upsert_rows(base_url, headers, "graph_edges", edges)

    return {
        "status": "ok",
        "inserted_nodes": len(nodes),
        "inserted_edges": len(edges),
        "inserted_runs": len(extraction_runs_rows),
    }


def write_graphml_outputs(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    import networkx as nx

    output_root = REPO_ROOT / "data" / "lightrag-project"
    output_root.mkdir(parents=True, exist_ok=True)

    per_collection: dict[str, nx.MultiDiGraph] = {}
    summary: dict[str, dict[str, int]] = {}

    for row in results:
        if row.get("status") != "success":
            continue
        collection = str(row.get("collection", ""))
        graph = per_collection.setdefault(collection, nx.MultiDiGraph())

        for node in row.get("entities", []):
            node_label = str(node)
            graph.add_node(node_label)

        for rel in row.get("relationships", []):
            src = str(rel.get("from", ""))
            dst = str(rel.get("to", ""))
            relation = str(rel.get("relation", "related_to"))
            if src and dst:
                graph.add_edge(src, dst, relationship=relation)

    for collection, graph in per_collection.items():
        coll_dir = output_root / collection
        coll_dir.mkdir(parents=True, exist_ok=True)
        graph_path = coll_dir / "graph_chunk_entity_relation.graphml"
        nx.write_graphml(graph, graph_path)
        summary[collection] = {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        }

    return summary


@app.local_entrypoint()
def main(
    benchmark_json: str = str(REPO_ROOT / "docs" / "LIGHTRAG-MODEL-BENCHMARK.json"),
    selected_model: str = "",
    checkpoint_path: str = str(REPO_ROOT / "docs" / "PROJECT-LIGHTRAG-PASS.checkpoint.json"),
    output_json: str = str(REPO_ROOT / "docs" / "PROJECT-LIGHTRAG-RUN-RESULT.json"),
    max_docs_per_collection: int = 0,
    max_segments_per_collection: int = 0,
    rerun_failed: bool = True,
    skip_supabase: bool = False,
) -> None:
    base_url, service_key = _get_supabase_creds()
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}
    segments = build_project_segments(base_url, headers, max_docs_per_collection=max_docs_per_collection)
    if not segments:
        raise RuntimeError("No project segments built from Supabase knowledge_chunks.")

    if max_segments_per_collection > 0:
        clipped: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for seg in segments:
            collection = str(seg["collection"])
            count = seen.get(collection, 0)
            if count >= max_segments_per_collection:
                continue
            clipped.append(seg)
            seen[collection] = count + 1
        segments = clipped

    model_name = selected_model.strip() or choose_model_from_benchmark(Path(benchmark_json))
    print(f"[project-pass] selected model: {model_name}")

    checkpoint_file = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_file)
    checkpoint["model"] = model_name
    checkpoint["segments_total"] = len(segments)

    results: list[dict[str, Any]] = []
    run_id = f"project-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

    for seg in segments:
        key = f"{model_name}|{seg['segment_id']}"
        existing = checkpoint["runs"].get(key)
        if existing and (existing.get("status") == "success" or not rerun_failed):
            results.append(existing)
            continue

        print(
            f"[project-pass] model={model_name} collection={seg['collection']} "
            f"segment={seg['segment_id']} chars={seg['char_count']}"
        )

        result = run_remote_segment.remote(model_name, seg)

        checkpoint["runs"][key] = result
        _save_checkpoint(checkpoint_file, checkpoint)
        results.append(result)
        print(
            f"[project-pass] complete collection={result['collection']} status={result['status']} "
            f"nodes={result['nodes_extracted']} edges={result['edges_extracted']} time={result['time_seconds']}"
        )

    graph_summary = write_graphml_outputs(results)
    nodes_payload, edges_payload = _build_graph_payload(results)
    supabase_status = persist_to_supabase(
        nodes=nodes_payload,
        edges=edges_payload,
        run_id=run_id,
        model_name=model_name,
        results=results,
        skip=skip_supabase,
    )

    grouped = _group_results_by_collection(results)
    per_collection = {}
    for collection, rows in grouped.items():
        per_collection[collection] = {
            "segments_total": len(rows),
            "segments_success": sum(1 for x in rows if x.get("status") == "success"),
            "time_seconds_total": round(sum(float(x.get("time_seconds", 0) or 0) for x in rows), 2),
            "nodes_total": sum(int(x.get("nodes_extracted", 0) or 0) for x in rows),
            "edges_total": sum(int(x.get("edges_extracted", 0) or 0) for x in rows),
        }

    payload = {
        "generated_at": _utcnow_iso(),
        "run_id": run_id,
        "app_name": APP_NAME,
        "gpu_profile": FULL_GPU,
        "model": model_name,
        "segments_total": len(segments),
        "results": results,
        "per_collection": per_collection,
        "graphml_summary": graph_summary,
        "supabase": supabase_status,
        "graph_nodes_payload": len(nodes_payload),
        "graph_edges_payload": len(edges_payload),
    }

    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "run_id": run_id, "supabase": supabase_status}, indent=2))


if __name__ == "__main__":
    main()
