"""Judge calibration: how well does the LLM judge agree with human labels?

Input is a CSV with columns: id, input, response, human_score (1-5), and optionally reference.
EvalForge asks the judge to grade each row and reports agreement statistics.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .scorers import build_judge_prompt, parse_judge


def cohens_kappa(a: list[int], b: list[int], labels=range(1, 6), weights: str | None = "quadratic") -> float:
    """Cohen's kappa. Quadratic weights suit ordinal 1-5 ratings (a 4 vs 5 is a small miss)."""
    if len(a) != len(b) or not a:
        raise ValueError("Need two equal-length, non-empty rating lists")
    labels = list(labels)
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1
    row = [sum(r) for r in observed]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights == "quadratic":
            return ((i - j) ** 2) / ((k - 1) ** 2)
        if weights == "linear":
            return abs(i - j) / (k - 1)
        return 0.0 if i == j else 1.0

    obs = sum(w(i, j) * observed[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 if exp == 0 else 1 - obs / exp


@dataclass
class CalibrationReport:
    n: int
    kappa_quadratic: float
    kappa_unweighted: float
    exact_agreement: float
    within_one: float
    mean_human: float
    mean_judge: float
    disagreements: list[dict]


def calibrate(judge, csv_path: str, rubric: str, concurrency: int = 8) -> CalibrationReport:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Calibration CSV has no rows")

    def grade(row):
        prompt = build_judge_prompt(rubric, row.get("input", ""), row.get("reference"), row["response"])
        score, reason = parse_judge(judge.complete(prompt).text)
        return score, reason

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        graded = list(pool.map(grade, rows))

    human = [int(r["human_score"]) for r in rows]
    model = [s for s, _ in graded]
    disagreements = [
        {"id": r.get("id", str(i)), "human": h, "judge": m, "reason": reason}
        for i, (r, h, m, (_, reason)) in enumerate(zip(rows, human, model, graded))
        if abs(h - m) >= 2
    ]
    n = len(rows)
    return CalibrationReport(
        n=n,
        kappa_quadratic=cohens_kappa(human, model, weights="quadratic"),
        kappa_unweighted=cohens_kappa(human, model, weights=None),
        exact_agreement=sum(h == m for h, m in zip(human, model)) / n,
        within_one=sum(abs(h - m) <= 1 for h, m in zip(human, model)) / n,
        mean_human=sum(human) / n,
        mean_judge=sum(model) / n,
        disagreements=disagreements,
    )
