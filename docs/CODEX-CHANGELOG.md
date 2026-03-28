# Codex Change Log (lsd-knowledge-silo)

---
## 2026-03-24 19:23 ICT - Project notebook ingest script for Supabase pgvector

### Files changed and why
- `scripts/ingest_project_notebooks.py`
  - New Modal-based ingestion script for project binder/workbook/ops binder into Supabase pgvector (DEC-013).
- `docs/CODEX-CHANGELOG.md`
  - Initialized repo-local changelog and recorded this session.
- `docs/CODEX-ERROR-LOG.md`
  - Logged Qdrant connectivity failure during required pre-change check.

### Commands run and results
- `curl -s http://localhost:6333/collections`
  - Failed with exit code 7 (Qdrant unreachable).

### Tests run and pass/fail
- Not run (no ingestion executed in this session).

### Issues found and resolution
- Issue: Required Qdrant pre-check failed (localhost:6333 unreachable).
  - Resolution: Logged in `docs/CODEX-ERROR-LOG.md`; ingestion not executed yet.

### Remaining work
- Run `scripts/ingest_project_notebooks.py` against project docs when Supabase + Modal are available.
- Verify 5 test queries after ingest.

## 2026-03-24 19:41 ICT Project notebook ingest fixes
- Updated scripts/ingest_project_notebooks.py to support Modal local entrypoint, ignore unknown args, and REST upsert fallback.
- Added IPv4 resolution attempt and pooler host override support for Supabase DB URL.
- Ingest run blocked by Supabase schema missing `content`/`embedding` columns; no data inserted.

## 2026-03-24 22:13 ICT - Project notebook graph extraction runner (step 6 canary)

### Files changed and why
- `scripts/modal_lightrag_project_notebooks.py`
  - Added a dedicated Modal LightRAG runner for project domains using Supabase `knowledge_chunks` content as source.
  - Added grouped-doc segment builder, checkpointed reruns, and Supabase persistence to `graph_nodes`, `graph_edges`, and `extraction_runs`.
  - Output path set to `docs/PROJECT-LIGHTRAG-RUN-RESULT.json` and GraphML under `data/lightrag-project/`.

### Commands run and results
- `modal run scripts/modal_lightrag_project_notebooks.py --max-docs-per-collection 1 --max-segments-per-collection 1 --selected-model llama3.2:3b`
  - Completed.
  - Run ID: `project-20260324150742-6e11d0`
  - Supabase write: `inserted_nodes=76`, `inserted_edges=30`, `inserted_runs=3`.
- Post-run verification queries against Supabase REST:
  - `extraction_runs` shows 3 completed rows for `project-binder`, `project-workbook`, `ops-binder`.
  - Current totals: `graph_nodes=216`, `graph_edges=87`.

### Tests run and pass/fail
- `python3 -m py_compile scripts/modal_lightrag_project_notebooks.py`
  - Pass.
- Modal canary run with Supabase persistence
  - Pass.

### Issues found and resolution
- Long-running canary attempt with larger limits was terminated and replaced by a small deterministic canary for faster verification.

### Remaining work
- Run scaled extraction pass (larger `max_docs_per_collection` / `max_segments_per_collection`, then full) using the same checkpoint file.
