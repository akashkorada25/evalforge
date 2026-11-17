"""Command-line interface.

  evalforge run SUITE [--report report.html] [--baseline base.json --max-drop 0.05] [--save-baseline base.json]
  evalforge list
  evalforge compare RUN_A RUN_B [--max-drop 0.05] [--report diff.html]
  evalforge report RUN_ID [--baseline-run RUN_ID] -o report.html
  evalforge calibrate SUITE LABELS.csv [--rubric "..."]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .calibration import calibrate
from .compare import compare, load_baseline, save_baseline, snapshot
from .config import SuiteError, load_suite
from .providers import build_provider
from .report import build_report
from .runner import run_suite, summarize
from .storage import DEFAULT_DB, Store


def _table(rows: list[dict]) -> str:
    head = f"{'Model | prompt':<40} {'Score':>7} {'Pass':>7} {'p50 ms':>8} {'Cost':>10} {'Err':>4}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['combo'][:40]:<40} {r['mean_score']:>7.3f} {r['pass_rate'] * 100:>6.1f}% "
            f"{r['p50_latency_ms']:>8.0f} {('$%.4f' % r['cost_usd']):>10} {r['errors']:>4}"
        )
    return "\n".join(lines)


def _print_comparison(cmp: dict) -> None:
    print(f"\nCompared with baseline (max allowed drop {cmp['max_drop']}):")
    for r in cmp["rows"]:
        if r["delta"] is None:
            print(f"  {r['combo']:<40} new")
        else:
            print(f"  {r['combo']:<40} {r['before']:.3f} -> {r['after']:.3f} ({r['delta']:+.3f}) {r['status']}")
    for x in cmp["newly_failing"]:
        print(f"  newly failing: {x['case_id']} [{x['combo']}]")
    print("Regression gate:", "PASS" if cmp["passed_gate"] else "FAIL")


def cmd_run(args) -> int:
    suite = load_suite(args.suite)
    n = len(suite.providers) * len(suite.prompts) * len(suite.cases)
    print(f"Suite '{suite.name}': {len(suite.providers)} models x {len(suite.prompts)} prompts x "
          f"{len(suite.cases)} cases = {n} calls", file=sys.stderr)
    results = run_suite(suite, concurrency=args.concurrency, retries=args.retries)
    store = Store(args.db)
    run_id = store.save_run(suite.name, suite.path, results, note=args.note or "")
    run = store.get_run(run_id)
    print(f"\nRun {run_id}\n")
    print(_table(summarize(results)))
    errors = sorted({f"[{r.provider_id}] {r.error}" for r in results if r.error})
    if errors:
        print(f"\n{sum(1 for r in results if r.error)} calls failed. Distinct errors:")
        for e in errors[:5]:
            print(f"  {e[:300]}")
    judge_errors = sorted({c["reason"][:200] for r in results for c in r.checks
                           if c["type"] == "llm_judge" and c["reason"].startswith("Judge error")})
    if judge_errors:
        print("\nJudge errors:")
        for e in judge_errors[:3]:
            print(f"  {e}")
    if args.json_summary:
        Path(args.json_summary).write_text(json.dumps(
            {"run_id": run_id, "suite": suite.name, "summary": summarize(results)}, indent=2))

    comparison = None
    if args.baseline:
        if not Path(args.baseline).exists():
            print(f"\nBaseline {args.baseline} not found; skipping the regression check.", file=sys.stderr)
        else:
            comparison = compare(load_baseline(args.baseline), snapshot(results, suite.name), args.max_drop)
            _print_comparison(comparison)
    if args.save_baseline:
        save_baseline(results, args.save_baseline, suite.name)
        print(f"\nSaved baseline to {args.save_baseline}")
    if args.report:
        Path(args.report).write_text(build_report(results, run, comparison), encoding="utf-8")
        print(f"Report written to {args.report}")
    if comparison and not comparison["passed_gate"]:
        return 1
    if args.min_pass_rate is not None:
        worst = min(r["pass_rate"] for r in summarize(results))
        if worst < args.min_pass_rate:
            print(f"Lowest pass rate {worst:.1%} is below --min-pass-rate {args.min_pass_rate:.0%}")
            return 1
    return 0


def cmd_list(args) -> int:
    runs = Store(args.db).list_runs(args.limit)
    if not runs:
        print("No runs yet. Try: evalforge run suites/support_summaries.yaml")
        return 0
    print(f"{'Run id':<24} {'Suite':<28} {'Created (UTC)':<22} {'Results':>7} {'Score':>7} {'Pass':>7}")
    for r in runs:
        print(f"{r['id']:<24} {r['suite'][:28]:<28} {r['created_at']:<22} {r['n']:>7} "
              f"{(r['mean_score'] or 0):>7.3f} {(r['pass_rate'] or 0) * 100:>6.1f}%")
    return 0


def cmd_compare(args) -> int:
    store = Store(args.db)
    run_a, run_b = store.get_run(args.run_a), store.get_run(args.run_b)
    res_a, res_b = store.load_results(run_a["id"]), store.load_results(run_b["id"])
    cmp = compare(snapshot(res_a), snapshot(res_b), args.max_drop)
    print(f"Baseline {run_a['id']}  vs  candidate {run_b['id']}")
    _print_comparison(cmp)
    if args.report:
        Path(args.report).write_text(build_report(res_b, run_b, cmp), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0 if cmp["passed_gate"] else 1


def cmd_failures(args) -> int:
    store = Store(args.db)
    run = store.get_run(args.run_id)
    failed = [r for r in store.load_results(run["id"]) if not r.passed]
    if args.combo:
        failed = [r for r in failed if args.combo in r.combo]
    print(f"Run {run['id']}: {len(failed)} failing results\n")
    reasons: dict[str, int] = {}
    for r in failed:
        for c in r.checks:
            if not c["passed"]:
                key = f"{c['type']}: {c['reason'][:80]}" if c["type"] != "llm_judge" else "llm_judge: score below threshold"
                reasons[key] = reasons.get(key, 0) + 1
    print("Most common failure reasons:")
    for reason, n in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:>3} x {reason}")
    print()
    for r in failed[: args.limit]:
        print(f"- {r.case_id} [{r.combo}] score {r.score:.2f}")
        if r.error:
            print(f"    error: {r.error[:200]}")
        for c in r.checks:
            if not c["passed"]:
                print(f"    {c['type']}: {c['reason'][:200]}")
        print(f"    output: {r.output[:args.chars]!r}")
    return 0


def cmd_report(args) -> int:
    store = Store(args.db)
    run = store.get_run(args.run_id)
    results = store.load_results(run["id"])
    cmp = None
    if args.baseline_run:
        base = store.get_run(args.baseline_run)
        cmp = compare(snapshot(store.load_results(base["id"])), snapshot(results), args.max_drop)
    Path(args.output).write_text(build_report(results, run, cmp), encoding="utf-8")
    print(f"Report written to {args.output}")
    return 0


def cmd_calibrate(args) -> int:
    suite = load_suite(args.suite)
    if not suite.judge:
        print("The suite has no 'judge' section to calibrate.", file=sys.stderr)
        return 2
    rubric = args.rubric or next(
        (c.get("rubric") for c in suite.default_checks if c.get("type") == "llm_judge" and c.get("rubric")),
        "Is the response correct, complete and helpful?",
    )
    rep = calibrate(build_provider(suite.judge), args.labels, rubric, args.concurrency)
    print(f"Judge: {suite.judge.id} ({suite.judge.model}) on {rep.n} human-labelled examples\n")
    print(f"  Cohen's kappa (quadratic weighted): {rep.kappa_quadratic:.3f}")
    print(f"  Cohen's kappa (unweighted):         {rep.kappa_unweighted:.3f}")
    print(f"  Exact agreement:                    {rep.exact_agreement:.1%}")
    print(f"  Within 1 point:                     {rep.within_one:.1%}")
    print(f"  Mean score  human {rep.mean_human:.2f} / judge {rep.mean_judge:.2f}")
    print("\n  Rough guide: kappa > 0.8 strong, 0.6-0.8 substantial, 0.4-0.6 moderate agreement.")
    if rep.disagreements:
        print(f"\n  {len(rep.disagreements)} large disagreements (2+ points). Review these to refine the rubric:")
        for d in rep.disagreements[:10]:
            print(f"    {d['id']}: human {d['human']} vs judge {d['judge']} - {d['reason'][:90]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="evalforge", description="Evaluate and compare LLM prompts and models.")
    p.add_argument("--version", action="version", version=f"evalforge {__version__}")
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite results database (default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a test suite")
    r.add_argument("suite")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--retries", type=int, default=3)
    r.add_argument("--report", help="Write an HTML dashboard to this path")
    r.add_argument("--baseline", help="Baseline JSON to compare against (fails on regression)")
    r.add_argument("--save-baseline", help="Save this run as a baseline JSON")
    r.add_argument("--max-drop", type=float, default=0.05, help="Allowed drop in mean score (0-1)")
    r.add_argument("--min-pass-rate", type=float, help="Fail if any combination's pass rate is below this")
    r.add_argument("--note", help="Free-text note stored with the run")
    r.add_argument("--json-summary", help="Write the leaderboard as JSON to this path")
    r.set_defaults(fn=cmd_run)

    ls = sub.add_parser("list", help="List recent runs")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(fn=cmd_list)

    c = sub.add_parser("compare", help="Compare two stored runs (use 'latest' or an id prefix)")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--max-drop", type=float, default=0.05)
    c.add_argument("--report")
    c.set_defaults(fn=cmd_compare)

    fl = sub.add_parser("failures", help="Show why cases failed in a stored run")
    fl.add_argument("run_id", nargs="?", default="latest")
    fl.add_argument("--combo", help="Only show combinations containing this text")
    fl.add_argument("--limit", type=int, default=20)
    fl.add_argument("--chars", type=int, default=240, help="Characters of output to show")
    fl.set_defaults(fn=cmd_failures)

    rp = sub.add_parser("report", help="Build an HTML dashboard for a stored run")
    rp.add_argument("run_id")
    rp.add_argument("-o", "--output", default="report.html")
    rp.add_argument("--baseline-run")
    rp.add_argument("--max-drop", type=float, default=0.05)
    rp.set_defaults(fn=cmd_report)

    cal = sub.add_parser("calibrate", help="Measure judge agreement with human labels")
    cal.add_argument("suite")
    cal.add_argument("labels", help="CSV with id,input,response,human_score[,reference]")
    cal.add_argument("--rubric")
    cal.add_argument("--concurrency", type=int, default=8)
    cal.set_defaults(fn=cmd_calibrate)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (SuiteError, KeyError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
