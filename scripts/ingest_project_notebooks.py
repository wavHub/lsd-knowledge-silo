#!/usr/bin/env python3
"""Embed project notebooks into Supabase pgvector via Modal (DEC-013 schema).

- Scans project binder/workbook/ops binder markdown.
- Splits by headings and subchunks long sections.
- Embeds with Ollama nomic-embed-text running in Modal.
- Upserts into Supabase knowledge_chunks with deterministic IDs.
- Supports hash-based file dedup via checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "lsd-project-notebooks-ingest"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_DIVISION = "software"
DEFAULT_CHECKPOINT = REPO_ROOT / "docs" / "PROJECT-NOTEBOOK-INGEST-CHECKPOINT.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "PROJECT-NOTEBOOK-INGEST-REPORT.json"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
DOC_ID_RE = re.compile(r"\b([A-Z]{2,5}-\d{3})\b")

COLLECTIONS: dict[str, dict[str, Any]] = {
    "project-binder": {
        "source": "/mnt/c/Users/Niibo-Admin/Developer/joplin-notebook-mirror/02 - Project Binder Niibo/",
        "recursive": True,
        "exclude": [".git", "__pycache__"],
    },
    "project-workbook": {
        "source": "/mnt/c/Users/Niibo-Admin/Developer/joplin-notebook-mirror/01 - Project Workbook/",
        "recursive": True,
        "exclude": [".git"],
    },
    "ops-binder": {
        "source": "/mnt/c/Users/Niibo-Admin/Developer/niibo-ops-binder/",
        "recursive": True,
        "exclude": [".git"],
    },
}


@dataclass
class ChunkItem:
    chunk_id: str
    collection: str
    division: str
    book: str
    chapter: str
    section: str
    chunk_index: int
    char_count: int
    token_est: int
    chunk_hash: str
    storage_path: str
    extraction_model: str
    document_id: str
    document_title: str
    file_path: str
    heading: str
    section_name: str
    text: str


ollama_volume = modal.Volume.from_name("lsd-lightrag-ollama-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("psycopg2-binary", "pgvector", "requests")
)

app = modal.App(APP_NAME, image=image)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(?im)^\s*page\s+\d+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


def _parse_document_id(file_name: str) -> str:
    match = DOC_ID_RE.search(file_name)
    return match.group(1) if match else ""


def _derive_document_title(file_name: str, document_id: str) -> str:
    stem = Path(file_name).stem
    if document_id and stem.startswith(document_id):
        stem = stem[len(document_id) :]
    stem = stem.strip("- _")
    title = re.sub(r"[_-]+", " ", stem).strip()
    return title or stem


def _stable_chunk_id(collection: str, rel_path: str, chunk_index: int) -> str:
    digest = hashlib.sha1(f"{collection}|{rel_path}|{chunk_index}".encode("utf-8")).hexdigest()
    return f"proj-{digest}"


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"generated_at": _utcnow_iso(), "files": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("files"), dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"generated_at": _utcnow_iso(), "files": {}}


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iter_markdown_files(source: Path, recursive: bool, exclude: list[str]) -> Iterable[Path]:
    if recursive:
        iterator = source.rglob("*.md")
    else:
        iterator = source.glob("*.md")
    for path in iterator:
        if any(part in exclude for part in path.parts):
            continue
        if path.is_file():
            yield path


def _build_chunks_for_file(
    *,
    collection: str,
    division: str,
    source_root: Path,
    path: Path,
) -> list[ChunkItem]:
    raw_text = _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    sections = _split_sections(raw_text)

    rel_path = str(path.relative_to(source_root))
    rel_parts = Path(rel_path).parts
    section_name = rel_parts[0] if len(rel_parts) > 1 else ""

    document_id = _parse_document_id(path.name)
    document_title = _derive_document_title(path.name, document_id)
    book = " ".join(part for part in [document_id, document_title] if part).strip() or path.stem

    items: list[ChunkItem] = []
    chunk_index = 0
    for heading, section_text in sections:
        for sub_text in _subchunk_text(section_text):
            chunk_index += 1
            text = sub_text.strip()
            if not text:
                continue
            items.append(
                ChunkItem(
                    chunk_id=_stable_chunk_id(collection, rel_path, chunk_index),
                    collection=collection,
                    division=division,
                    book=book,
                    chapter=heading.strip(),
                    section=section_name,
                    chunk_index=chunk_index,
                    char_count=len(text),
                    token_est=max(1, len(text) // 4),
                    chunk_hash=_chunk_hash(text),
                    storage_path=rel_path,
                    extraction_model=EMBED_MODEL,
                    document_id=document_id,
                    document_title=document_title,
                    file_path=rel_path,
                    heading=heading.strip(),
                    section_name=section_name,
                    text=text,
                )
            )
    return items


def _collect_chunks(
    *,
    division: str,
    checkpoint: dict[str, Any],
    force: bool,
    max_files: int,
) -> tuple[list[ChunkItem], list[dict[str, Any]], dict[str, Any]]:
    chunks: list[ChunkItem] = []
    files_report: list[dict[str, Any]] = []
    files_state = checkpoint.get("files", {})

    for collection, meta in COLLECTIONS.items():
        source_root = Path(meta["source"]).expanduser().resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Missing source directory: {source_root}")

        exclude = list(meta.get("exclude", []))
        recursive = bool(meta.get("recursive", True))
        count = 0

        for path in sorted(_iter_markdown_files(source_root, recursive, exclude)):
            count += 1
            if max_files and count > max_files:
                break

            rel_path = str(path.relative_to(source_root))
            file_key = f"{collection}:{rel_path}"
            file_hash = _file_hash(path)
            existing = files_state.get(file_key)

            if not force and existing and existing.get("sha256") == file_hash:
                files_report.append(
                    {
                        "collection": collection,
                        "file": rel_path,
                        "status": "skipped",
                        "reason": "hash-match",
                        "sha256": file_hash,
                    }
                )
                continue

            file_chunks = _build_chunks_for_file(
                collection=collection,
                division=division,
                source_root=source_root,
                path=path,
            )
            chunks.extend(file_chunks)
            files_state[file_key] = {
                "sha256": file_hash,
                "chunk_count": len(file_chunks),
                "updated_at": _utcnow_iso(),
            }
            files_report.append(
                {
                    "collection": collection,
                    "file": rel_path,
                    "status": "queued",
                    "sha256": file_hash,
                    "chunk_count": len(file_chunks),
                }
            )

    checkpoint["files"] = files_state
    checkpoint["generated_at"] = _utcnow_iso()
    return chunks, files_report, checkpoint


def _get_supabase_db_url() -> str:
    override = os.getenv("SUPABASE_DB_URL", "").strip()
    if override:
        return override

    db_url = subprocess.check_output(
        ["python3", "/home/niiboAdmin/dev/keymaster/keymaster.py", "get", "supabase-db-url"],
        text=True,
    ).strip()
    pooler_host = os.getenv("SUPABASE_DB_POOLER_HOST", "").strip()
    pooler_port = os.getenv("SUPABASE_DB_POOLER_PORT", "").strip()
    if pooler_host:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(db_url)
        project_ref = ""
        if parsed.hostname:
            match = re.match(r"^db\\.([^.]+)\\.supabase\\.co$", parsed.hostname)
            if match:
                project_ref = match.group(1)

        username = parsed.username or ""
        if project_ref and username and not username.endswith(f".{project_ref}"):
            username = f"{username}.{project_ref}"

        userinfo = username if username else ""
        if userinfo and parsed.password:
            userinfo = f"{userinfo}:{parsed.password}"

        host = pooler_host
        port = int(pooler_port) if pooler_port else (parsed.port or 5432)
        netloc = f"{userinfo}@{host}:{port}" if userinfo else f"{host}:{port}"
        db_url = urlunparse(parsed._replace(netloc=netloc))

    return db_url


def _get_supabase_rest_creds() -> tuple[str, str]:
    base_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if base_url and service_key:
        return base_url, service_key

    base_url = subprocess.check_output(
        ["python3", "/home/niiboAdmin/dev/keymaster/keymaster.py", "get", "supabase-url"],
        text=True,
    ).strip().rstrip("/")
    service_key = subprocess.check_output(
        ["python3", "/home/niiboAdmin/dev/keymaster/keymaster.py", "get", "supabase-service-role-key"],
        text=True,
    ).strip()
    return base_url, service_key


def _batch_iter(items: list[ChunkItem], batch_size: int) -> Iterable[list[ChunkItem]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@app.function(
    gpu=os.getenv("MODAL_EMBED_GPU", "A10G"),
    timeout=60 * 60,
    retries=0,
    volumes={"/root/.ollama": ollama_volume},
    single_use_containers=True,
)
def embed_and_upsert(
    db_url: str,
    batch,
    require_embedding: bool = True,
    rest_base_url: str = "",
    rest_service_key: str = "",
) -> dict[str, Any]:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    import requests
    import socket
    from urllib.parse import urlparse

    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    os.environ["OLLAMA_NUM_PARALLEL"] = "2"
    os.environ["OLLAMA_KEEP_ALIVE"] = "120m"

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

        parsed = urlparse(db_url)
        host = parsed.hostname or ""
        port = parsed.port or 5432
        hostaddr = None
        if host:
            try:
                for res in socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM):
                    hostaddr = res[4][0]
                    break
            except socket.gaierror:
                hostaddr = None

        using_rest = bool(rest_base_url and rest_service_key)
        columns: set[str] = set()
        conn = None
        try:
            if not using_rest:
                conn = psycopg2.connect(db_url, hostaddr=hostaddr) if hostaddr else psycopg2.connect(db_url)
                conn.autocommit = False
                register_vector(conn)

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'knowledge_chunks'
                        """
                    )
                    columns = {row[0] for row in cur.fetchall()}

                if require_embedding and "embedding" not in columns:
                    raise RuntimeError("knowledge_chunks missing embedding column; cannot store pgvector embeddings.")

            insert_rows: list[dict[str, Any]] = []
            for item in batch:
                embed_text = f"{item.get('heading','').strip()}\n\n{item.get('text','').strip()}".strip()
                resp = requests.post(
                    "http://127.0.0.1:11434/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": embed_text},
                    timeout=120,
                )
                resp.raise_for_status()
                vector = resp.json().get("embedding")
                if not vector:
                    raise RuntimeError("Ollama returned empty embedding.")

                row: dict[str, Any] = {
                    "id": item["id"],
                    "division": item["division"],
                    "collection": item["collection"],
                    "book": item["book"],
                    "chapter": item["chapter"],
                    "section": item["section"],
                    "chunk_index": item["chunk_index"],
                    "char_count": item["char_count"],
                    "token_est": item["token_est"],
                    "chunk_hash": item["chunk_hash"],
                    "storage_path": item["storage_path"],
                    "status": "extracted",
                    "extraction_model": item["extraction_model"],
                    "nodes_extracted": 0,
                    "edges_extracted": 0,
                    "processing_started_at": item["processing_started_at"],
                    "processing_completed_at": item["processing_completed_at"],
                    "error_message": None,
                }

                optional_fields = {
                    "embedding": vector,
                    "content": item.get("text"),
                    "document_id": item.get("document_id"),
                    "document_title": item.get("document_title"),
                    "file_path": item.get("file_path"),
                    "heading": item.get("heading"),
                    "section_name": item.get("section_name"),
                    "metadata": item.get("metadata"),
                }
                if using_rest:
                    row.update(optional_fields)
                else:
                    for key, value in optional_fields.items():
                        if key in columns:
                            row[key] = value

                insert_rows.append(row)

            if insert_rows:
                if using_rest:
                    endpoint = f"{rest_base_url.rstrip('/')}/rest/v1/knowledge_chunks?on_conflict=id"
                    headers = {
                        "apikey": rest_service_key,
                        "Authorization": f"Bearer {rest_service_key}",
                        "Content-Type": "application/json",
                        "Prefer": "resolution=merge-duplicates,return=minimal",
                    }
                    seen_hashes: dict[str, int] = {}
                    filtered_rows: list[dict[str, Any]] = []
                    for row in insert_rows:
                        chunk_hash = row.get("chunk_hash") or ""
                        if not chunk_hash:
                            filtered_rows.append(row)
                            continue
                        if chunk_hash in seen_hashes:
                            continue
                        seen_hashes[chunk_hash] = 1
                        filtered_rows.append(row)

                    for i in range(0, len(filtered_rows), 200):
                        batch_rows = filtered_rows[i : i + 200]
                        resp = requests.post(endpoint, headers=headers, data=json.dumps(batch_rows), timeout=120)
                        if resp.status_code >= 300:
                            raise RuntimeError(
                                f"Supabase upsert failed status={resp.status_code} body={resp.text[:400]}"
                            )
                else:
                    _upsert_rows(conn, "knowledge_chunks", insert_rows)
                    conn.commit()
            return {
                "status": "ok",
                "rows": len(insert_rows),
                "embedding_model": EMBED_MODEL,
            }
        finally:
            if conn is not None:
                conn.close()
    finally:
        try:
            ollama_proc.terminate()
        except Exception:
            pass


def _upsert_rows(conn, table: str, rows: list[dict[str, Any]], batch_size: int = 200) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    cols_sql = ", ".join(columns)
    update_cols = [col for col in columns if col != "id" and col != "created_at"]
    update_sql = ", ".join([f"{col}=EXCLUDED.{col}" for col in update_cols])

    with conn.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            values = [[row[col] for col in columns] for row in batch]
            cur.executemany(
                f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT (id) DO UPDATE SET {update_sql}",
                values,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed project notebooks into Supabase pgvector via Modal.")
    parser.add_argument("--division", default=DEFAULT_DIVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--allow-missing-embedding", action="store_true")
    args, _unknown = parser.parse_known_args()

    env_batch = os.getenv("INGEST_BATCH_SIZE", "").strip()
    if env_batch:
        try:
            args.batch_size = int(env_batch)
        except ValueError:
            pass
    env_force = os.getenv("INGEST_FORCE", "").strip().lower()
    if env_force in {"1", "true", "yes", "y"}:
        args.force = True

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)

    chunks, files_report, checkpoint = _collect_chunks(
        division=args.division,
        checkpoint=checkpoint,
        force=args.force,
        max_files=args.max_files,
    )

    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    started = _utcnow_iso()
    results: list[dict[str, Any]] = []

    if not args.dry_run and chunks:
        db_url = _get_supabase_db_url()
        rest_base_url, rest_service_key = _get_supabase_rest_creds()
        for batch in _batch_iter(chunks, args.batch_size):
            payload = []
            for chunk in batch:
                payload.append(
                    {
                        "id": chunk.chunk_id,
                        "division": chunk.division,
                        "collection": chunk.collection,
                        "book": chunk.book,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "char_count": chunk.char_count,
                        "token_est": chunk.token_est,
                        "chunk_hash": chunk.chunk_hash,
                        "storage_path": chunk.storage_path,
                        "extraction_model": chunk.extraction_model,
                        "document_id": chunk.document_id,
                        "document_title": chunk.document_title,
                        "file_path": chunk.file_path,
                        "heading": chunk.heading,
                        "section_name": chunk.section_name,
                        "text": chunk.text,
                        "processing_started_at": started,
                        "processing_completed_at": _utcnow_iso(),
                    }
                )
            result = embed_and_upsert.remote(
                db_url,
                payload,
                not args.allow_missing_embedding,
                rest_base_url,
                rest_service_key,
            )
            results.append(result)

    if chunks and not args.dry_run:
        _save_checkpoint(checkpoint_path, checkpoint)

    report = {
        "generated_at": _utcnow_iso(),
        "started_at": started,
        "completed_at": _utcnow_iso(),
        "division": args.division,
        "collections": list(COLLECTIONS.keys()),
        "files": files_report,
        "chunks_total": len(chunks),
        "chunks_inserted": sum(int(r.get("rows", 0) or 0) for r in results),
        "batches": len(results),
        "dry_run": args.dry_run,
        "supabase_results": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "chunks": len(chunks)}, indent=2))


@app.local_entrypoint()
def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
