"""Tests for the v0.2 features used to evaluate RAG systems."""
import sqlite3

from evalforge.cli import main
from evalforge.config import Case, load_suite
from evalforge.providers import PythonProvider, ProviderError
from evalforge.runner import run_suite
from evalforge.scorers import build_judge_prompt, run_check
from evalforge.storage import Store

FAKE_RAG = '''
def answer(prompt, system=None, retriever="good"):
    if retriever == "boom":
        raise RuntimeError("index missing")
    sources = ["tutorial/controlflow#Default Argument Values"] if retriever == "good" else ["faq/design#Why"]
    return {
        "text": "The default value is evaluated only once [1].",
        "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.0005,
        "metadata": {"sources": sources, "context": ["[1] The default value is evaluated only once."]},
    }

def plain(prompt, system=None):
    return "just text"
'''


def install_fake(tmp_path, monkeypatch):
    (tmp_path / "fake_rag.py").write_text(FAKE_RAG)
    monkeypatch.syspath_prepend(str(tmp_path))
    PythonProvider._cache.clear()


def test_python_provider_returns_metadata_and_cost(tmp_path, monkeypatch):
    install_fake(tmp_path, monkeypatch)
    comp = PythonProvider(id="rag", model="fake_rag:answer").complete("q")
    assert comp.text.startswith("The default")
    assert comp.cost_usd == 0.0005 and comp.input_tokens == 100
    assert comp.metadata["sources"][0].startswith("tutorial/controlflow")


def test_python_provider_plain_string_and_errors(tmp_path, monkeypatch):
    install_fake(tmp_path, monkeypatch)
    assert PythonProvider(id="p", model="fake_rag:plain").complete("q").text == "just text"
    for model in ("fake_rag", "fake_rag:missing", "no_such_module:x"):
        try:
            PythonProvider(id="p", model=model).complete("q")
            raise AssertionError("expected ProviderError")
        except ProviderError:
            pass


def test_sources_check_any_and_all():
    case = Case(id="c", vars={})
    ctx = {"metadata": {"sources": ["tutorial/controlflow#Defaults", "faq/programming#Copy"]}}
    assert run_check({"type": "sources", "value": ["controlflow"]}, "", case, ctx).passed
    r = run_check({"type": "sources", "value": ["controlflow", "modules"], "mode": "all"}, "", case, ctx)
    assert not r.passed and r.score == 0.5
    assert not run_check({"type": "sources", "value": ["x"]}, "", case, {"metadata": {}}).passed


def test_judge_prompt_with_context_and_without_reference():
    p = build_judge_prompt("Is it faithful?", "question: q", None, "answer", context="[1] passage")
    assert "<<CONTEXT>>[1] passage<</CONTEXT>>" in p
    assert "<<REFERENCE>><</REFERENCE>>" in p
    assert "CONTEXT" not in build_judge_prompt("r", "i", "ref", "a")


def test_rag_suite_end_to_end(tmp_path, monkeypatch):
    install_fake(tmp_path, monkeypatch)
    suite_path = tmp_path / "rag.yaml"
    suite_path.write_text("""
name: rag
providers:
  - {id: good, provider: python, model: "fake_rag:answer", params: {retriever: good}}
  - {id: bad, provider: python, model: "fake_rag:answer", params: {retriever: bad}}
  - {id: boom, provider: python, model: "fake_rag:answer", params: {retriever: boom}}
judge: {id: j, provider: mock, model: mock-strong}
prompts: [{id: q, template: "{question}"}]
default_checks:
  - {type: llm_judge, include_context: true, use_reference: false, rubric: "Faithful to context?", threshold: 1}
cases:
  - id: defaults
    vars: {question: "When are default values evaluated?"}
    checks: [{type: sources, value: [controlflow]}]
""")
    results = {r.provider_id: r for r in run_suite(load_suite(suite_path), progress=False, retries=0)}
    assert results["good"].passed and results["good"].metadata["sources"]
    assert not results["bad"].passed
    assert results["boom"].error and "index missing" in results["boom"].error

    db = str(tmp_path / "r.db")
    assert main(["--db", db, "run", str(suite_path), "--retries", "0"]) == 0
    loaded = Store(db).load_results(Store(db).get_run("latest")["id"])
    assert any(r.metadata.get("sources") for r in loaded)


def test_store_migrates_v01_database(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE runs (id TEXT PRIMARY KEY, suite TEXT NOT NULL, suite_path TEXT, created_at TEXT NOT NULL,
                       git_sha TEXT, note TEXT);
    CREATE TABLE results (run_id TEXT NOT NULL, case_id TEXT NOT NULL, provider_id TEXT NOT NULL,
        prompt_id TEXT NOT NULL, tags TEXT, output TEXT, score REAL, passed INTEGER, latency_ms REAL,
        input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, judge_cost_usd REAL, checks TEXT, error TEXT);
    """)
    conn.close()
    store = Store(str(db))
    cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(results)")}
    assert "metadata" in cols
