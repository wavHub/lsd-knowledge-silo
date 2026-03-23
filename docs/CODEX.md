# LSD Knowledge Silo — Codex Instructions

## Project
Little Spark Digital Knowledge Silo — textbook-trained domain intelligence.
Separate from Niibo/Qwinn (Azure). Connected later via FRAG.

## Location
/home/niiboAdmin/dev/lsd-knowledge-silo/

## Repo
wavHub/lsd-knowledge-silo (to be created on GitHub)

## Infrastructure
- Qdrant: localhost:6333 (35 collections, 5M+ vectors, ref-* collections are ours)
- Ollama: localhost:11434 (nomic-embed-text, llama3.2:3b)
- Supabase: lsd-knowledge-silo (project ekvsjqmtpxamqmdljckq), keys in vault
- Modal: workspace littlesparkdev, $30/mo free tier
- Python venv: /home/niiboAdmin/niibo-project/cortex-venv/

## Vault Access
```bash
source /home/niiboAdmin/niibo-project/cortex-venv/bin/activate
export AZURE_CONFIG_DIR=/home/niiboAdmin/.azure
python3 /home/niiboAdmin/dev/keymaster/keymaster.py get supabase-url
python3 /home/niiboAdmin/dev/keymaster/keymaster.py get supabase-service-role-key
python3 /home/niiboAdmin/dev/keymaster/keymaster.py get supabase-db-url
```

## What Exists Already (in qwinn repo — needs migration)
- scripts/ingest_reference_textbooks_qdrant.py
- scripts/evaluate_lightrag_ref_collections.py
- scripts/modal_lightrag_full.py
- scripts/modal_lightrag_benchmark.py
- scripts/evaluate_ref_textbook_quality.py
- scripts/render_lightrag_evaluation_report.py
- data/lightrag-ref/ and data/lightrag-ref-hybrid/
- docs/REF-TEXTBOOK-QUALITY-REPORT.md
- docs/LIGHTRAG-EVALUATION-hybrid.md
- docs/REF-TEXTBOOK-INGEST-REPORT.json

## Current State
- Stage 1 DONE: 8,178 textbook vectors in 6 Qdrant ref-* collections
- Stage 2 DONE: vectors searchable
- Stage 3 DONE: LightRAG hybrid (selective KG, 3b model, thin graphs)
- Stage 4 NEXT: Reasoning — connect textbook knowledge to produce structured output
- H100 32b benchmark: was running, Modal apps died, needs rerun

## 5 Division Silos
| Silo | Qdrant Collection | Books |
|------|------------------|-------|
| Engineering | ref-engineering | ASME Y14.5, Cost Estimation, Making Embedded Systems |
| Finance | ref-finance | Managerial Accounting 16th Ed |
| Sales | ref-negotiation | Power Negotiations (Dawson) |
| Software | ref-software | Software Architecture, Algorithms (CLRS) |
| Networking | ref-networking | Data Communications 5th Ed |

## Textbook Source
/mnt/d/markdown/ (9 files, 11.4 MB)

## Supabase Schema (from DEC-013)
One DB, 5 logical silos. Tables: graph_nodes, graph_edges, knowledge_chunks, extraction_runs, benchmark_results.

## Key Documents
- RN-043: Training Silos & Reference Library
- RN-044: Books to Actions — 7-stage workflow
- RN-046: LightRAG Model Benchmark SOP
- DEC-013: Knowledge Silo Infrastructure decision
- FL-028: Knowledge Silo Implementation follow-up

## Rules
- DEC-012: Grounding Verification — read docs before acting
- DEC-011: Commit per document
- DEC-010: D: drive data is local only — never send to external APIs
- Textbooks are published works — API embedding is acceptable
