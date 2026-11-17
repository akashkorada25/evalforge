"""Runs every provider x prompt x case combination in parallel and scores the outputs."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from .config import Suite, render
from .providers import ProviderError, build_provider
from .scorers import run_check


@dataclass
class Result:
    case_id: str
    provider_id: str
    prompt_id: str
    tags: list[str]
    output: str
    score: float
    passed: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    judge_cost_usd: float = 0.0
    checks: list[dict] = field(default_factory=list)
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def combo(self) -> str:
        return f"{self.provider_id} | {self.prompt_id}"

    def to_dict(self):
        return asdict(self)


def call_with_retries(provider, prompt, system, reference, retries: int, backoff: float):
    attempt = 0
    while True:
        try:
            return provider.complete(prompt, system=system, reference=reference)
        except ProviderError as e:
            attempt += 1
            if not e.retryable or attempt > retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


def _evaluate(provider, judge, prompt_spec, case, suite: Suite, retries: int, backoff: float) -> Result:
    prompt = render(prompt_spec.template, case.vars)
    system = render(prompt_spec.system, case.vars) if prompt_spec.system else None
    try:
        comp = call_with_retries(provider, prompt, system, case.expected, retries, backoff)
    except Exception as e:  # noqa: BLE001 - record the failure and keep going
        return Result(case.id, provider.id, prompt_spec.id, case.tags, "", 0.0, False, 0.0, 0, 0, 0.0,
                      error=str(e))

    ctx = {"judge": judge, "metadata": comp.metadata or {}}
    checks = [run_check(c, comp.text, case, ctx) for c in (suite.default_checks + case.checks)]
    score = sum(c.score for c in checks) / len(checks) if checks else 0.0
    passed = bool(checks) and all(c.passed for c in checks)
    return Result(
        case_id=case.id,
        provider_id=provider.id,
        prompt_id=prompt_spec.id,
        tags=case.tags,
        output=comp.text,
        score=round(score, 4),
        passed=passed,
        latency_ms=round(comp.latency_ms, 1),
        input_tokens=comp.input_tokens,
        output_tokens=comp.output_tokens,
        cost_usd=comp.cost_usd,
        judge_cost_usd=ctx.get("judge_cost", 0.0),
        checks=[c.to_dict() for c in checks],
        metadata=comp.metadata or {},
    )


def run_suite(suite: Suite, concurrency: int = 8, retries: int = 3, backoff: float = 1.0,
              progress: bool = True) -> list[Result]:
    providers = [build_provider(p) for p in suite.providers]
    judge = build_provider(suite.judge) if suite.judge else None
    jobs = [(p, pr, c) for p in providers for pr in suite.prompts for c in suite.cases]
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_evaluate, p, judge, pr, c, suite, retries, backoff) for p, pr, c in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if progress:
                sys.stderr.write(f"\r  Evaluating {i}/{len(jobs)}")
                sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
    results.sort(key=lambda r: (r.provider_id, r.prompt_id, r.case_id))
    return results


# ---- aggregation -----------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[Result]) -> list[dict]:
    """One row per provider x prompt combination, best mean score first."""
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(r.combo, []).append(r)
    rows = []
    for combo, rs in groups.items():
        ok = [r for r in rs if not r.error]
        lat = [r.latency_ms for r in ok]
        rows.append({
            "combo": combo,
            "provider_id": rs[0].provider_id,
            "prompt_id": rs[0].prompt_id,
            "cases": len(rs),
            "pass_rate": sum(r.passed for r in rs) / len(rs),
            "mean_score": sum(r.score for r in rs) / len(rs),
            "p50_latency_ms": percentile(lat, 50),
            "p95_latency_ms": percentile(lat, 95),
            "cost_usd": sum(r.cost_usd for r in rs),
            "judge_cost_usd": sum(r.judge_cost_usd for r in rs),
            "errors": len(rs) - len(ok),
        })
    rows.sort(key=lambda x: (-x["mean_score"], x["cost_usd"]))
    return rows


def tag_breakdown(results: list[Result]) -> dict[str, dict[str, float]]:
    """{tag: {combo: mean score}}"""
    acc: dict[str, dict[str, list[float]]] = {}
    for r in results:
        for tag in r.tags or ["untagged"]:
            acc.setdefault(tag, {}).setdefault(r.combo, []).append(r.score)
    return {t: {c: sum(v) / len(v) for c, v in combos.items()} for t, combos in sorted(acc.items())}
