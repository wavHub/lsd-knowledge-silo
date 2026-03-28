# Codex Error Log (lsd-knowledge-silo)

## 2026-03-24 19:23 ICT Qdrant pre-check failed
- File: N/A
- Error: curl -s http://localhost:6333/collections (exit code 7)
- Cause: Qdrant not reachable on localhost:6333
- Fix: Not resolved in this session
- Prevention: Verify Qdrant service availability before pre-change check

## 2026-03-24 19:41 ICT Project notebook ingest blocked by Supabase schema
- File: scripts/ingest_project_notebooks.py
- Error: Supabase upsert failed (PGRST204) missing `content` column in `knowledge_chunks`
- Cause: Supabase schema does not include `content`/`embedding` fields expected by ingest.
- Fix: none yet (requires schema update or adjusted field mapping)
- Prevention: verify Supabase schema columns before ingest; apply DEC-013+embedding schema if missing.

## 2026-03-24 19:36 ICT Modal DB connection failed via IPv6-only host
- File: scripts/ingest_project_notebooks.py (Modal execution)
- Error: psycopg2 OperationalError, network unreachable to IPv6 Supabase host
- Cause: Supabase DB host resolves only to IPv6; Modal environment lacks IPv6 connectivity.
- Fix: added REST fallback; DB access still blocked without schema changes.
- Prevention: use REST or a pooler host with IPv4; document correct pooler endpoint.
