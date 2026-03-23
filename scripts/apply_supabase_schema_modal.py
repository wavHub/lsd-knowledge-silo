#!/usr/bin/env python3
"""Apply Supabase schema via Modal to bypass local IPv6 limits."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "supabase_schema_dec013.sql"
APP_NAME = "lsd-supabase-schema-apply"

image = modal.Image.debian_slim(python_version="3.11").pip_install("psycopg2-binary")
app = modal.App(APP_NAME, image=image)


@app.function(timeout=60 * 15)
def apply_schema_remote(db_url: str, schema_sql: str, run_smoke_tests: bool = True) -> dict:
    import psycopg2

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    report = {"schema_applied": False, "smoke_tests": None}

    def insert_node(cur, node_id: str, division: str, label: str) -> None:
        cur.execute(
            """
            INSERT INTO graph_nodes (id, division, label, entity_type, source_book)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (node_id, division, label, "concept", "smoke-test"),
        )

    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
        report["schema_applied"] = True

        if run_smoke_tests:
            test_id = uuid.uuid4().hex[:10]
            a = f"smoke-node-a-{test_id}"
            b = f"smoke-node-b-{test_id}"
            eng = f"smoke-node-eng-{test_id}"
            edge_ok = f"smoke-edge-ok-{test_id}"

            smoke = {
                "division_check": False,
                "cross_division_block": False,
                "edge_division_block": False,
                "views_ok": False,
            }

            with conn.cursor() as cur:
                cur.execute("BEGIN")
                insert_node(cur, a, "finance", "Smoke Finance A")
                insert_node(cur, b, "finance", "Smoke Finance B")
                cur.execute(
                    """
                    INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (edge_ok, "finance", a, b, "depends_on", "smoke-test"),
                )

                try:
                    insert_node(cur, f"smoke-invalid-{test_id}", "marketing", "Bad")
                except Exception:
                    smoke["division_check"] = True
                    conn.rollback()
                    cur.execute("BEGIN")
                    insert_node(cur, a, "finance", "Smoke Finance A")
                    insert_node(cur, b, "finance", "Smoke Finance B")
                    cur.execute(
                        """
                        INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (edge_ok, "finance", a, b, "depends_on", "smoke-test"),
                    )

                insert_node(cur, eng, "engineering", "Smoke Engineering")
                try:
                    cur.execute(
                        """
                        INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (f"smoke-cross-{test_id}", "finance", a, eng, "depends_on", "smoke-test"),
                    )
                except Exception:
                    smoke["cross_division_block"] = True
                    conn.rollback()
                    cur.execute("BEGIN")
                    insert_node(cur, a, "finance", "Smoke Finance A")
                    insert_node(cur, b, "finance", "Smoke Finance B")
                    cur.execute(
                        """
                        INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (edge_ok, "finance", a, b, "depends_on", "smoke-test"),
                    )

                try:
                    cur.execute(
                        """
                        INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (f"smoke-bad-div-{test_id}", "engineering", a, b, "depends_on", "smoke-test"),
                    )
                except Exception:
                    smoke["edge_division_block"] = True
                    conn.rollback()
                    cur.execute("BEGIN")
                    insert_node(cur, a, "finance", "Smoke Finance A")
                    insert_node(cur, b, "finance", "Smoke Finance B")
                    cur.execute(
                        """
                        INSERT INTO graph_edges (id, division, from_node_id, to_node_id, relationship, source_book)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (edge_ok, "finance", a, b, "depends_on", "smoke-test"),
                    )

                cur.execute("SELECT count(*) FROM finance_graph WHERE id = %s", (a,))
                fin = int(cur.fetchone()[0])
                cur.execute("SELECT count(*) FROM engineering_graph WHERE id = %s", (a,))
                eng_view = int(cur.fetchone()[0])
                smoke["views_ok"] = fin == 1 and eng_view == 0

                cur.execute("DELETE FROM graph_edges WHERE id = %s", (edge_ok,))
                cur.execute("DELETE FROM graph_nodes WHERE id IN (%s, %s, %s)", (a, b, eng))
                cur.execute("COMMIT")

            report["smoke_tests"] = smoke
    finally:
        conn.close()

    return report


@app.local_entrypoint()
def main() -> None:
    db_url = subprocess.check_output(
        ["python3", "/home/niiboAdmin/dev/keymaster/keymaster.py", "get", "supabase-db-url"],
        text=True,
    ).strip()
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    report = apply_schema_remote.remote(db_url, schema_sql, True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
