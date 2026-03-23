-- DEC-013 / FL-028 schema for lsd-knowledge-silo.
-- Idempotent migration: safe to run multiple times.

CREATE OR REPLACE FUNCTION is_valid_division(value TEXT)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
AS $$
  SELECT value IN ('engineering', 'finance', 'sales', 'software', 'networking')
$$;

CREATE TABLE IF NOT EXISTS graph_nodes (
  id TEXT PRIMARY KEY,
  division TEXT NOT NULL CHECK (is_valid_division(division)),
  label TEXT NOT NULL,
  entity_type TEXT,
  properties JSONB NOT NULL DEFAULT '{}'::jsonb,
  source_book TEXT,
  source_chapter TEXT,
  source_chunk_id TEXT,
  extraction_model TEXT,
  confidence NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nodes_division ON graph_nodes (division);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON graph_nodes (label);

CREATE TABLE IF NOT EXISTS graph_edges (
  id TEXT PRIMARY KEY,
  division TEXT NOT NULL CHECK (is_valid_division(division)),
  from_node_id TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  to_node_id TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  relationship TEXT NOT NULL,
  properties JSONB NOT NULL DEFAULT '{}'::jsonb,
  source_book TEXT,
  source_chunk_id TEXT,
  extraction_model TEXT,
  confidence NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edges_division ON graph_edges (division);
CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges (from_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges (to_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_relationship ON graph_edges (relationship);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id TEXT PRIMARY KEY,
  division TEXT NOT NULL CHECK (is_valid_division(division)),
  collection TEXT NOT NULL,
  book TEXT NOT NULL,
  chapter TEXT,
  section TEXT,
  chunk_index INTEGER,
  char_count INTEGER,
  token_est INTEGER,
  chunk_hash TEXT UNIQUE,
  storage_path TEXT,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'processing', 'extracted', 'failed')),
  extraction_model TEXT,
  nodes_extracted INTEGER NOT NULL DEFAULT 0,
  edges_extracted INTEGER NOT NULL DEFAULT 0,
  processing_started_at TIMESTAMPTZ,
  processing_completed_at TIMESTAMPTZ,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_division ON knowledge_chunks (division);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_collection ON knowledge_chunks (collection);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_status ON knowledge_chunks (status);

CREATE TABLE IF NOT EXISTS extraction_runs (
  id TEXT PRIMARY KEY,
  division TEXT NOT NULL CHECK (is_valid_division(division)),
  collection TEXT NOT NULL,
  model TEXT NOT NULL,
  gpu TEXT,
  provider TEXT,
  total_chunks INTEGER NOT NULL DEFAULT 0,
  processed_chunks INTEGER NOT NULL DEFAULT 0,
  total_nodes INTEGER NOT NULL DEFAULT 0,
  total_edges INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'completed', 'failed')),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  cost_estimate NUMERIC,
  error_log TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extraction_runs_division ON extraction_runs (division);
CREATE INDEX IF NOT EXISTS idx_extraction_runs_collection ON extraction_runs (collection);
CREATE INDEX IF NOT EXISTS idx_extraction_runs_status ON extraction_runs (status);

CREATE TABLE IF NOT EXISTS benchmark_results (
  id TEXT PRIMARY KEY,
  division TEXT NOT NULL CHECK (is_valid_division(division)),
  collection TEXT NOT NULL,
  model TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  source_file TEXT,
  heading TEXT,
  segment_tokens_est INTEGER,
  time_seconds NUMERIC NOT NULL DEFAULT 0,
  nodes_extracted INTEGER NOT NULL DEFAULT 0,
  edges_extracted INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'success'
    CHECK (status IN ('success', 'failed')),
  error_message TEXT,
  sample_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
  sample_relationships JSONB NOT NULL DEFAULT '[]'::jsonb,
  extraction_run_id TEXT REFERENCES extraction_runs(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_division ON benchmark_results (division);
CREATE INDEX IF NOT EXISTS idx_benchmark_collection ON benchmark_results (collection);
CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_results (model);
CREATE INDEX IF NOT EXISTS idx_benchmark_segment ON benchmark_results (segment_id);

CREATE OR REPLACE FUNCTION ensure_edge_division_consistency()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  from_division TEXT;
  to_division TEXT;
BEGIN
  SELECT division INTO from_division FROM graph_nodes WHERE id = NEW.from_node_id;
  SELECT division INTO to_division FROM graph_nodes WHERE id = NEW.to_node_id;

  IF from_division IS NULL OR to_division IS NULL THEN
    RAISE EXCEPTION 'Edge references missing node(s): from=% to=%', NEW.from_node_id, NEW.to_node_id;
  END IF;

  IF from_division <> to_division THEN
    RAISE EXCEPTION 'Cross-division edge blocked: % -> % (% vs %)',
      NEW.from_node_id, NEW.to_node_id, from_division, to_division;
  END IF;

  IF NEW.division <> from_division THEN
    RAISE EXCEPTION 'Edge division mismatch: edge=% node=%', NEW.division, from_division;
  END IF;

  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_trigger
    WHERE tgname = 'trg_graph_edges_division_consistency'
  ) THEN
    CREATE TRIGGER trg_graph_edges_division_consistency
      BEFORE INSERT OR UPDATE ON graph_edges
      FOR EACH ROW
      EXECUTE FUNCTION ensure_edge_division_consistency();
  END IF;
END
$$;

ALTER TABLE graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE extraction_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE benchmark_results ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'graph_nodes'
      AND policyname = 'graph_nodes_division_isolation'
  ) THEN
    CREATE POLICY graph_nodes_division_isolation
      ON graph_nodes
      FOR ALL
      USING (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      )
      WITH CHECK (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'graph_edges'
      AND policyname = 'graph_edges_division_isolation'
  ) THEN
    CREATE POLICY graph_edges_division_isolation
      ON graph_edges
      FOR ALL
      USING (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      )
      WITH CHECK (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'knowledge_chunks'
      AND policyname = 'knowledge_chunks_division_isolation'
  ) THEN
    CREATE POLICY knowledge_chunks_division_isolation
      ON knowledge_chunks
      FOR ALL
      USING (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      )
      WITH CHECK (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'extraction_runs'
      AND policyname = 'extraction_runs_division_isolation'
  ) THEN
    CREATE POLICY extraction_runs_division_isolation
      ON extraction_runs
      FOR ALL
      USING (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      )
      WITH CHECK (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'benchmark_results'
      AND policyname = 'benchmark_results_division_isolation'
  ) THEN
    CREATE POLICY benchmark_results_division_isolation
      ON benchmark_results
      FOR ALL
      USING (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      )
      WITH CHECK (
        COALESCE(
          NULLIF((current_setting('request.jwt.claims', true)::jsonb ->> 'division'), ''),
          division
        ) = division
      );
  END IF;
END
$$;

CREATE OR REPLACE VIEW engineering_graph AS
  SELECT * FROM graph_nodes WHERE division = 'engineering';
CREATE OR REPLACE VIEW finance_graph AS
  SELECT * FROM graph_nodes WHERE division = 'finance';
CREATE OR REPLACE VIEW sales_graph AS
  SELECT * FROM graph_nodes WHERE division = 'sales';
CREATE OR REPLACE VIEW software_graph AS
  SELECT * FROM graph_nodes WHERE division = 'software';
CREATE OR REPLACE VIEW networking_graph AS
  SELECT * FROM graph_nodes WHERE division = 'networking';

CREATE OR REPLACE VIEW engineering_graph_edges AS
  SELECT * FROM graph_edges WHERE division = 'engineering';
CREATE OR REPLACE VIEW finance_graph_edges AS
  SELECT * FROM graph_edges WHERE division = 'finance';
CREATE OR REPLACE VIEW sales_graph_edges AS
  SELECT * FROM graph_edges WHERE division = 'sales';
CREATE OR REPLACE VIEW software_graph_edges AS
  SELECT * FROM graph_edges WHERE division = 'software';
CREATE OR REPLACE VIEW networking_graph_edges AS
  SELECT * FROM graph_edges WHERE division = 'networking';
