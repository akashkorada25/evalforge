"""Baselines and regression detection, used for comparing runs and for CI gating."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .runner import Result, summarize


def snapshot(results: list[Result], suite_name: str = "") -> dict:
    """A compact, committable summary of a run: per-combo scores and per-case scores."""
    combos = {}
    for row in summarize(results):
        combos[row["combo"]] = {
            "mean_score": round(row["mean_score"], 4),
            "pass_rate": round(row["pass_rate"], 4),
            "cost_usd": round(row["cost_usd"], 6),
            "cases": {},
        }
    for r in results:
        combos[r.combo]["cases"][r.case_id] = {"score": r.score, "passed": r.passed}
    return {
        "suite": suite_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "combos": combos,
    }


def save_baseline(results: list[Result], path: str, suite_name: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(snapshot(results, suite_name), indent=2, sort_keys=True))


def load_baseline(path: str) -> dict:
    return json.loads(Path(path).read_text())


def compare(baseline: dict, current: dict, max_drop: float = 0.05) -> dict:
    """Compare two snapshots.

    A combo regresses when its mean score falls by more than `max_drop` (absolute, on a 0-1
    scale). Individual cases that passed before and fail now are listed too.
    """
    rows, regressions, newly_failing, newly_passing = [], [], [], []
    for combo, cur in current["combos"].items():
        base = baseline["combos"].get(combo)
        if base is None:
            rows.append({"combo": combo, "before": None, "after": cur["mean_score"], "delta": None, "status": "new"})
            continue
        delta = cur["mean_score"] - base["mean_score"]
        status = "regressed" if delta < -max_drop else "improved" if delta > max_drop else "unchanged"
        row = {"combo": combo, "before": base["mean_score"], "after": cur["mean_score"],
               "delta": round(delta, 4), "status": status,
               "pass_before": base["pass_rate"], "pass_after": cur["pass_rate"]}
        rows.append(row)
        if status == "regressed":
            regressions.append(row)
        for case_id, c in cur["cases"].items():
            b = base["cases"].get(case_id)
            if b and b["passed"] and not c["passed"]:
                newly_failing.append({"combo": combo, "case_id": case_id})
            elif b and not b["passed"] and c["passed"]:
                newly_passing.append({"combo": combo, "case_id": case_id})
    return {
        "max_drop": max_drop,
        "rows": rows,
        "regressions": regressions,
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "passed_gate": not regressions,
    }
