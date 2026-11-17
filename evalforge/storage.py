"""SQLite storage for runs and results, so any two runs can be compared later."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .runner import Result

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    suite TEXT NOT NULL,
    suite_path TEXT,
    created_at TEXT NOT NULL,
    git_sha TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL REFERENCES runs(id),
    case_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    tags TEXT,
    output TEXT,
    score REAL,
    passed INTEGER,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    judge_cost_usd REAL,
    checks TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
"""

DEFAULT_DB = ".evalforge/results.db"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Store:
    def __init__(self, path: str = DEFAULT_DB):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save_run(self, suite_name: str, suite_path: str, results: list[Result], note: str = "") -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?)",
                (run_id, suite_name, suite_path, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 _git_sha(), note),
            )
            self.conn.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, r.case_id, r.provider_id, r.prompt_id, json.dumps(r.tags), r.output, r.score,
                  int(r.passed), r.latency_ms, r.input_tokens, r.output_tokens, r.cost_usd,
                  r.judge_cost_usd, json.dumps(r.checks), r.error) for r in results],
            )
        return run_id

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT r.*, COUNT(x.case_id) AS n, AVG(x.score) AS mean_score, AVG(x.passed) AS pass_rate
               FROM runs r LEFT JOIN results x ON x.run_id = r.id
               GROUP BY r.id ORDER BY r.created_at DESC, r.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict:
        if run_id == "latest":
            row = self.conn.execute("SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
        else:
            row = self.conn.execute("SELECT * FROM runs WHERE id = ? OR id LIKE ?", (run_id, f"{run_id}%")).fetchone()
        if row is None:
            raise KeyError(f"No run found matching '{run_id}'")
        return dict(row)

    def load_results(self, run_id: str) -> list[Result]:
        rows = self.conn.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchall()
        return [
            Result(
                case_id=r["case_id"], provider_id=r["provider_id"], prompt_id=r["prompt_id"],
                tags=json.loads(r["tags"] or "[]"), output=r["output"] or "", score=r["score"],
                passed=bool(r["passed"]), latency_ms=r["latency_ms"], input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"], cost_usd=r["cost_usd"], judge_cost_usd=r["judge_cost_usd"],
                checks=json.loads(r["checks"] or "[]"), error=r["error"],
            )
            for r in rows
        ]
