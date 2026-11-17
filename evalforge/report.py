"""Builds a single self-contained HTML dashboard for a run (no server or internet needed)."""
from __future__ import annotations

import html
import json

from .runner import Result, summarize, tag_breakdown

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#3b5bdb;
--good:#2b8a3e;--bad:#c92a2a;--warn:#e67700;--bar:#dbe4ff;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1f1f1d;--ink:#ecebe6;--muted:#9a9993;
--line:#33332f;--accent:#91a7ff;--good:#69db7c;--bad:#ff8787;--warn:#ffc078;--bar:#2b3566}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;
background:var(--bg);color:var(--ink)}main{max-width:1180px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:0 0 12px}.sub{color:var(--muted)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-top:20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.kpi .value{font-size:22px;font-weight:600;margin-top:2px}.kpi .note{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.barcell{position:relative;min-width:120px}.barcell .bar{position:absolute;left:0;top:6px;bottom:6px;
background:var(--bar);border-radius:3px}.barcell span{position:relative;padding-left:6px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.pill.pass{color:var(--good);border-color:var(--good)}.pill.fail{color:var(--bad);border-color:var(--bad)}
.heat td.cell{text-align:center;font-variant-numeric:tabular-nums}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
select,input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
details{border-bottom:1px solid var(--line);padding:8px 0}summary{cursor:pointer;display:flex;gap:10px;align-items:center}
summary::-webkit-details-marker{display:none}.case-id{font-family:var(--mono);font-size:13px}
pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;
font:12px/1.5 var(--mono);margin:8px 0}.checks li{margin:2px 0}.muted{color:var(--muted)}
svg text{fill:var(--muted);font-size:11px}
"""

JS = """
const DATA = JSON.parse(document.getElementById('data').textContent);
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fill(sel, values){ const el=$(sel); values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o);}); }
fill('#f-combo', [...new Set(DATA.map(r=>r.combo))]);
fill('#f-tag', [...new Set(DATA.flatMap(r=>r.tags))].sort());
function render(){
  const combo=$('#f-combo').value, tag=$('#f-tag').value, status=$('#f-status').value, q=$('#f-q').value.toLowerCase();
  const rows = DATA.filter(r => (!combo||r.combo===combo) && (!tag||r.tags.includes(tag)) &&
    (!status || (status==='fail') !== r.passed) && (!q || (r.case_id+' '+r.output).toLowerCase().includes(q)));
  $('#count').textContent = rows.length + ' results';
  $('#cases').innerHTML = rows.map(r => `<details><summary>
    <span class="pill ${r.passed?'pass':'fail'}">${r.passed?'pass':'fail'}</span>
    <span class="case-id">${esc(r.case_id)}</span><span class="muted">${esc(r.combo)}</span>
    <span class="muted" style="margin-left:auto">score ${r.score.toFixed(2)} · ${Math.round(r.latency_ms)} ms</span></summary>
    ${r.error?`<p class="bad">Error: ${esc(r.error)}</p>`:''}
    <div class="muted">Tags: ${esc(r.tags.join(', ')||'none')}</div>
    ${(r.metadata&&r.metadata.sources&&r.metadata.sources.length)?`<div class="muted">Retrieved: ${esc(r.metadata.sources.join(' · '))}</div>`:''}
    <pre>${esc(r.output)}</pre>
    <ul class="checks">${r.checks.map(c=>`<li><span class="${c.passed?'good':'bad'}">${c.passed?'✓':'✗'}</span>
      <b>${esc(c.name||c.type)}</b> (${c.score.toFixed(2)}): ${esc(c.reason)}</li>`).join('')}</ul></details>`).join('');
}
['#f-combo','#f-tag','#f-status'].forEach(s=>$(s).addEventListener('change',render));
$('#f-q').addEventListener('input',render); render();
"""


def _e(x) -> str:
    return html.escape(str(x))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _money(x: float) -> str:
    return f"${x:.4f}" if x < 1 else f"${x:,.2f}"


def _bar(value: float, text: str) -> str:
    return f'<td class="barcell"><div class="bar" style="width:{max(0, min(1, value)) * 100:.1f}%"></div><span>{text}</span></td>'


def _heat_color(v: float) -> str:
    # red (0) -> amber (0.5) -> green (1), with alpha so it works in light and dark mode
    hue = int(120 * max(0.0, min(1.0, v)))
    return f"hsla({hue},65%,45%,0.28)"


def _scatter(rows: list[dict]) -> str:
    """Quality vs cost scatter plot as inline SVG."""
    if not rows:
        return ""
    W, H, P = 720, 300, 48
    costs = [r["cost_usd"] for r in rows]
    cmax = max(costs) or 1
    pts, legend = [], []
    for i, r in enumerate(rows, 1):
        x = P + (r["cost_usd"] / cmax) * (W - 2 * P)
        y = H - P - r["mean_score"] * (H - 2 * P)
        pts.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="var(--accent)" opacity=".85"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-weight:600">{i}</text>'
            f'<title>{_e(r["combo"])}: score {r["mean_score"]:.3f}, cost {_money(r["cost_usd"])}</title></g>'
        )
        legend.append(f'<span class="pill"><b>{i}</b> {_e(r["combo"])} · {r["mean_score"]:.3f} · {_money(r["cost_usd"])}</span>')
    grid = "".join(
        f'<line x1="{P}" x2="{W - P}" y1="{H - P - t * (H - 2 * P):.1f}" y2="{H - P - t * (H - 2 * P):.1f}" '
        f'stroke="var(--line)"/><text x="{P - 8}" y="{H - P - t * (H - 2 * P) + 4:.1f}" text-anchor="end">{t:.2f}</text>'
        for t in (0, 0.25, 0.5, 0.75, 1)
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block" role="img" '
        f'aria-label="Mean score versus cost">{grid}'
        f'<text x="{W / 2}" y="{H - 10}" text-anchor="middle">Total cost for this suite (max {_money(cmax)}) →</text>'
        f'<text x="14" y="{H / 2}" transform="rotate(-90 14 {H / 2})" text-anchor="middle">Mean score</text>'
        f'{"".join(pts)}</svg>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">{"".join(legend)}</div>'
    )


def best_value(rows: list[dict], tolerance: float = 0.02) -> dict | None:
    """Cheapest combo whose mean score is within `tolerance` of the best."""
    if not rows:
        return None
    top = max(r["mean_score"] for r in rows)
    close = [r for r in rows if r["mean_score"] >= top - tolerance]
    return min(close, key=lambda r: r["cost_usd"])


def build_report(results: list[Result], run: dict, comparison: dict | None = None) -> str:
    rows = summarize(results)
    tags = tag_breakdown(results)
    best = rows[0] if rows else None
    value = best_value(rows)
    total_cost = sum(r.cost_usd + r.judge_cost_usd for r in results)
    combos = [r["combo"] for r in rows]

    kpis = []
    if best:
        kpis.append(("Top score", f'{best["mean_score"]:.3f}', best["combo"]))
        kpis.append(("Top pass rate", _pct(max(r["pass_rate"] for r in rows)),
                     max(rows, key=lambda r: r["pass_rate"])["combo"]))
    if value and best:
        saving = 1 - value["cost_usd"] / best["cost_usd"] if best["cost_usd"] else 0
        kpis.append(("Best value", value["combo"],
                     f"Within 0.02 of the top score at {_pct(saving)} lower cost" if saving > 0 else "Also the top scorer"))
    kpis.append(("Total cost", _money(total_cost), f"{len(results)} calls, judge included"))
    if comparison:
        n = len(comparison["regressions"])
        kpis.append(("Regression gate", "PASS" if comparison["passed_gate"] else "FAIL",
                     f"{n} regressed, {len(comparison['newly_failing'])} cases newly failing"))
    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{_e(l)}</div><div class="value">{_e(v)}</div><div class="note">{_e(n)}</div></div>'
        for l, v, n in kpis
    )

    def board_row(r):
        score_text = "%.3f" % r["mean_score"]
        err_cls = "bad" if r["errors"] else ""
        return (
            f'<tr><td><b>{_e(r["provider_id"])}</b><div class="muted">{_e(r["prompt_id"])}</div></td>'
            f'{_bar(r["mean_score"], score_text)}{_bar(r["pass_rate"], _pct(r["pass_rate"]))}'
            f'<td class="num">{r["p50_latency_ms"]:.0f} / {r["p95_latency_ms"]:.0f}</td>'
            f'<td class="num">{_money(r["cost_usd"])}</td>'
            f'<td class="num {err_cls}">{r["errors"]}</td></tr>'
        )

    board = "".join(board_row(r) for r in rows)

    heat_head = "".join(f"<th>{_e(c)}</th>" for c in combos)
    heat_rows = "".join(
        f"<tr><td><b>{_e(tag)}</b></td>"
        + "".join(
            f'<td class="cell" style="background:{_heat_color(by.get(c, 0))}">{by[c]:.2f}</td>' if c in by
            else '<td class="cell muted">–</td>'
            for c in combos
        )
        + "</tr>"
        for tag, by in tags.items()
    )

    cmp_html = ""
    if comparison:
        def fmt(v):
            return "–" if v is None else f"{v:.3f}"

        def cmp_row(r):
            cls = {"regressed": "bad", "improved": "good"}.get(r["status"], "")
            delta = "–" if r["delta"] is None else "%+.3f" % r["delta"]
            return (
                f'<tr><td>{_e(r["combo"])}</td><td class="num">{fmt(r["before"])}</td>'
                f'<td class="num">{fmt(r["after"])}</td><td class="num {cls}">{delta}</td>'
                f'<td>{_e(r["status"])}</td></tr>'
            )

        cmp_rows = "".join(cmp_row(r) for r in comparison["rows"])
        failing = "".join(
            f'<li><span class="case-id">{_e(x["case_id"])}</span> <span class="muted">{_e(x["combo"])}</span></li>'
            for x in comparison["newly_failing"]
        ) or '<li class="muted">None</li>'
        cmp_html = f"""<section><h2>Compared with baseline <span class="muted">(max allowed drop {comparison['max_drop']})</span></h2>
<table><tr><th>Combination</th><th class="num">Before</th><th class="num">After</th><th class="num">Change</th><th>Status</th></tr>{cmp_rows}</table>
<h2 style="margin-top:16px">Cases that passed before and fail now</h2><ul>{failing}</ul></section>"""

    data = json.dumps([{**r.to_dict(), "combo": r.combo} for r in results]).replace("</", "<\\/")

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EvalForge · {_e(run.get('suite', ''))}</title>
<style>{CSS}</style></head><body><main>
<h1>{_e(run.get('suite', 'Evaluation'))}</h1>
<div class="sub">Run <span class="case-id">{_e(run.get('id', ''))}</span> · {_e(run.get('created_at', ''))}
{(' · commit ' + _e(run['git_sha'])) if run.get('git_sha') else ''} · {len(rows)} combinations × {len({r.case_id for r in results})} cases</div>
<div class="kpis">{kpi_html}</div>
<section><h2>Leaderboard</h2><table><tr><th>Model / prompt</th><th>Mean score</th><th>Pass rate</th>
<th class="num">Latency p50 / p95 (ms)</th><th class="num">Cost</th><th class="num">Errors</th></tr>{board}</table></section>
<section><h2>Quality vs cost</h2>{_scatter(rows)}<p class="muted">Up and to the left is better: higher score for less money.</p></section>
<section><h2>Score by tag</h2><div style="overflow-x:auto"><table class="heat"><tr><th>Tag</th>{heat_head}</tr>{heat_rows}</table></div></section>
{cmp_html}
<section><h2>Case explorer <span class="muted" id="count"></span></h2>
<div class="filters"><select id="f-combo"><option value="">All combinations</option></select>
<select id="f-tag"><option value="">All tags</option></select>
<select id="f-status"><option value="">Pass and fail</option><option value="fail">Failing only</option><option value="pass">Passing only</option></select>
<input id="f-q" placeholder="Search case id or output"></div><div id="cases"></div></section>
<p class="muted" style="margin-top:24px">Generated by EvalForge</p>
</main><script id="data" type="application/json">{data}</script><script>{JS}</script></body></html>"""
