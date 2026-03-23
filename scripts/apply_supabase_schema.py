#!/usr/bin/env python3
"""Apply DEC-013 schema to Supabase and run smoke tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "supabase_schema_dec013.sql"
KEYMASTER_PATH = Path("/home/niiboAdmin/dev/keymaster/keymaster.py")


def get_secret(name: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(KEYMASTER_PATH), "get", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def get_db_url() -> str:
    env_value = os.getenv("SUPABASE_DB_URL", "").strip()
    if env_value:
        return env_value
    return get_secret("supabase-db-url")


def apply_schema(conn) -> None:
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def _insert_node(cur, node_id: str, division: str, label: str) -> None:
    cur.execute(
        """
        INSERT INTO graph_nodes (id, division, label, entity_type, source_book)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (node_id, division, label, "concept", "smoke-test"),
    )


def run_smoke_tests(conn) -> None:
    test_id = uuid.uuid4().hex[:10]
    valid_a = f"smoke-node-a-{test_id}"
    valid_b = f"smoke-node-b-{test_id}"
    invalid = f"smoke-node-invalid-{test_id}"
    edge_ok = f"smoke-edge-ok-{test_id}"
    edge_bad = f"smoke-edge-bad-{test_id}"
    edge_bad_div = f"smoke-edge-bad-div-{test_id}"

    with conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            _insert_node(cur, valid_a, "finance", "Smoke Finance A")
            _insert_node(cur, valid_b, "finance", "Smoke Finance B")

            cur.execute(
                """
                INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (edge_ok, "finance", valid_a, valid_b, "depends_on", "smoke-test"),
            )

            # Division CHECK must fail.
            division_failed = False
            try:
                _insert_node(cur, invalid, "marketing", "Bad Division")
            except Exception:
                division_failed = True
                conn.rollback()
                cur.execute("BEGIN")
                _insert_node(cur, valid_a, "finance", "Smoke Finance A")
                _insert_node(cur, valid_b, "finance", "Smoke Finance B")
                cur.execute(
                    """
                    INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (edge_ok, "finance", valid_a, valid_b, "depends_on", "smoke-test"),
                )

            if not division_failed:
                raise RuntimeError("Division CHECK smoke test failed: invalid division insert unexpectedly succeeded.")

            # Cross-division edge must fail.
            _insert_node(cur, f"smoke-node-eng-{test_id}", "engineering", "Smoke Engineering A")
            cross_failed = False
            try:
                cur.execute(
                    """
                    INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (edge_bad, "finance", valid_a, f"smoke-node-eng-{test_id}", "depends_on", "smoke-test"),
                )
            except Exception:
                cross_failed = True
                conn.rollback()
                cur.execute("BEGIN")
                _insert_node(cur, valid_a, "finance", "Smoke Finance A")
                _insert_node(cur, valid_b, "finance", "Smoke Finance B")
                cur.execute(
                    """
                    INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (edge_ok, "finance", valid_a, valid_b, "depends_on", "smoke-test"),
                )
            if not cross_failed:
                raise RuntimeError("Cross-division edge smoke test failed: invalid edge insert unexpectedly succeeded.")

            # Mismatched edge.division must fail.
            mismatch_failed = False
            try:
                cur.execute(
                    """
                    INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (edge_bad_div, "engineering", valid_a, valid_b, "depends_on", "smoke-test"),
                )
            except Exception:
                mismatch_failed = True
                conn.rollback()
                cur.execute("BEGIN")
                _insert_node(cur, valid_a, "finance", "Smoke Finance A")
                _insert_node(cur, valid_b, "finance", "Smoke Finance B")
                cur.execute(
                    """
                    INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (edge_ok, "finance", valid_a, valid_b, "depends_on", "smoke-test"),
                )
            if not mismatch_failed:
                raise RuntimeError("Edge division smoke test failed: mismatched edge division insert unexpectedly succeeded.")

            # Verify division view exposes the finance node.
            cur.execute("SELECT count(*) FROM finance_graph WHERE id = %s", (valid_a,))
            finance_count = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM engineering_graph WHERE id = %s", (valid_a,))
            engineering_count = int(cur.fetchone()[0])
            if finance_count != 1 or engineering_count != 0:
                raise RuntimeError(
                    f"View smoke test failed: finance_count={finance_count}, engineering_count={engineering_count}"
                )

            cur.execute("DELETE FROM graph_edges WHERE id IN (%s, %s, %s)", (edge_ok, edge_bad, edge_bad_div))
            cur.execute(
                "DELETE FROM graph_nodes WHERE id IN (%s, %s, %s)",
                (valid_a, valid_b, f"smoke-node-eng-{test_id}"),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Supabase schema migration and run smoke tests.")
    parser.add_argument("--skip-smoke-tests", action="store_true", help="Apply schema only.")
    args = parser.parse_args()

    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    try:
        apply_schema(conn)
        print("Schema applied: OK")
        if not args.skip_smoke_tests:
            run_smoke_tests(conn)
            print("Smoke tests: OK")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
