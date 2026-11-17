import json
from pathlib import Path

import pytest

from evalforge.calibration import cohens_kappa
from evalforge.cli import main
from evalforge.compare import compare, snapshot
from evalforge.config import SuiteError, load_suite, render
from evalforge.providers import ProviderError
from evalforge.runner import call_with_retries, percentile, run_suite, summarize
from evalforge.storage import Store

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "suites" / "support_summaries.yaml"


def write_suite(tmp_path, model="mock-strong", provider="mock"):
    p = tmp_path / "suite.yaml"
    p.write_text(f"""
name: tiny
providers:
  - {{id: m, provider: {provider}, model: {model}}}
prompts:
  - {{id: p1, template: "Q: {{question}}"}}
default_checks:
  - {{type: similarity, threshold: 0.9}}
cases:
""" + "".join(f"""  - id: c{i}
    tags: [t{i % 2}]
    vars: {{question: "question number {i}"}}
    expected: "answer number {i} with some words"
""" for i in range(20)))
    return p


def test_render_leaves_unknown_braces():
    assert render('Reply {"a": 1} to {name}', {"name": "Bo"}) == 'Reply {"a": 1} to Bo'


def test_suite_validation_missing_var(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("""
providers: [{id: m, provider: mock, model: mock-fast}]
prompts: [{id: p, template: "{missing}"}]
cases: [{id: c, vars: {}}]
""")
    with pytest.raises(SuiteError, match="missing"):
        load_suite(p)


def test_suite_validation_duplicate_ids(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("""
providers: [{id: m, provider: mock, model: mock-fast}]
prompts: [{id: p, template: "hi"}]
cases: [{id: c}, {id: c}]
""")
    with pytest.raises(SuiteError, match="Duplicate"):
        load_suite(p)


def test_example_suite_runs_and_is_deterministic():
    suite = load_suite(SUITE)
    a = run_suite(suite, progress=False)
    b = run_suite(suite, progress=False)
    assert len(a) == len(suite.providers) * len(suite.prompts) * len(suite.cases)
    assert [r.score for r in a] == [r.score for r in b]
    rows = summarize(a)
    assert rows == sorted(rows, key=lambda r: (-r["mean_score"], r["cost_usd"]))
    assert all(r["errors"] == 0 for r in rows)


def test_stronger_mock_model_scores_higher(tmp_path):
    strong = summarize(run_suite(load_suite(write_suite(tmp_path, "mock-strong")), progress=False))[0]
    fast = summarize(run_suite(load_suite(write_suite(tmp_path, "mock-fast")), progress=False))[0]
    assert strong["mean_score"] > fast["mean_score"]


def test_missing_api_key_is_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = run_suite(load_suite(write_suite(tmp_path, "gpt-x", "openai")), progress=False, retries=0)
    assert all(r.error and "OPENAI_API_KEY" in r.error for r in results)
    assert summarize(results)[0]["errors"] == len(results)


def test_retries_then_succeeds():
    calls = {"n": 0}

    class Flaky:
        def complete(self, prompt, system=None, reference=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderError("rate limited", retryable=True)
            return "ok"

    assert call_with_retries(Flaky(), "p", None, None, retries=3, backoff=0) == "ok"
    assert calls["n"] == 3


def test_non_retryable_error_raises_immediately():
    class Broken:
        def complete(self, *a, **k):
            raise ProviderError("bad request", retryable=False)

    with pytest.raises(ProviderError):
        call_with_retries(Broken(), "p", None, None, retries=5, backoff=0)


def test_storage_roundtrip(tmp_path):
    results = run_suite(load_suite(write_suite(tmp_path)), progress=False)
    store = Store(str(tmp_path / "r.db"))
    run_id = store.save_run("tiny", "suite.yaml", results)
    loaded = store.load_results(run_id)
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in results]
    assert store.get_run("latest")["id"] == run_id


def test_compare_detects_regression(tmp_path):
    good = run_suite(load_suite(write_suite(tmp_path, "mock-strong")), progress=False)
    bad = run_suite(load_suite(write_suite(tmp_path, "mock-fast")), progress=False)
    result = compare(snapshot(good), snapshot(bad), max_drop=0.05)
    assert not result["passed_gate"]
    assert result["regressions"][0]["delta"] < -0.05
    assert result["newly_failing"]
    assert compare(snapshot(good), snapshot(good))["passed_gate"]


def test_kappa():
    assert cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert cohens_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # Off by one everywhere should still be fairly high with quadratic weights
    assert cohens_kappa([1, 2, 3, 4, 5, 5], [2, 3, 4, 5, 4, 5]) > 0.6


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) == 0.0


def test_cli_end_to_end(tmp_path):
    db, base, html = str(tmp_path / "r.db"), str(tmp_path / "base.json"), str(tmp_path / "r.html")
    assert main(["--db", db, "run", str(SUITE), "--save-baseline", base, "--report", html]) == 0
    assert json.loads(Path(base).read_text())["combos"]
    page = Path(html).read_text()
    assert "<!DOCTYPE html>" in page and "Leaderboard" in page
    # Same suite against its own baseline passes the gate
    assert main(["--db", db, "run", str(SUITE), "--baseline", base]) == 0
    assert main(["--db", db, "compare", "latest", "latest"]) == 0
    assert main(["--db", db, "list"]) == 0
    assert main(["--db", db, "failures", "latest", "--limit", "3"]) == 0


def test_cli_gate_fails_on_regression(tmp_path):
    db, base = str(tmp_path / "r.db"), str(tmp_path / "base.json")
    assert main(["--db", db, "run", str(write_suite(tmp_path, "mock-strong")), "--save-baseline", base]) == 0
    assert main(["--db", db, "run", str(write_suite(tmp_path, "mock-fast")), "--baseline", base]) == 1


def test_calibrate_cli():
    assert main(["calibrate", str(SUITE), str(ROOT / "data" / "judge_calibration.csv")]) == 0
