"""Checks that score a model output. Each returns a score in [0, 1], pass/fail and a reason.

Check types (used in suite files under `checks` or `default_checks`):
  exact          output equals `value` (whitespace and case-insensitive by default)
  contains       output contains every string in `value` (list or string)
  not_contains   output contains none of the strings in `value`
  regex          output matches `pattern`
  json_schema    output is valid JSON matching a small schema (`schema`)
  max_words      output has at most `value` words
  similarity     cosine similarity to the case's `expected` is at least `threshold`
  llm_judge      a judge model grades the output 1-5 against `rubric`; passes at `threshold`.
                 `include_context: true` shows the judge the passages a RAG system retrieved
                 (metadata["context"]), for faithfulness checks; `use_reference: false` hides
                 the expected answer from the judge.
  sources        the provider's metadata["sources"] includes the strings in `value`
                 (`mode: any` by default, or `all`); for checking what a RAG system retrieved
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

STOPWORDS = set(
    "a an the and or but of to in on at for with by from is are was were be been it its this that "
    "as i you he she they we our their his her them my your me us do does did not no so if then than "
    "has have had will would can could should".split()
)


@dataclass
class CheckResult:
    type: str
    score: float
    passed: bool
    reason: str
    name: str = ""  # optional label from the suite, e.g. "faithful", to tell checks of one type apart

    def to_dict(self):
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOPWORDS]


def cosine_similarity(a: str, b: str) -> float:
    """Bag-of-words cosine similarity. Dependency-free stand-in for embeddings."""
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    norm = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return dot / norm if norm else 0.0


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


# ---- individual checks ---------------------------------------------------------------


def check_exact(check, output, case, ctx):
    expected = check.get("value", case.expected)
    if expected is None:
        return CheckResult("exact", 0.0, False, "No value or expected output to compare against")
    a, b = str(output).strip(), str(expected).strip()
    if not check.get("case_sensitive", False):
        a, b = a.lower(), b.lower()
    ok = a == b
    return CheckResult("exact", float(ok), ok, "Exact match" if ok else "Output differs from expected")


def check_contains(check, output, case, ctx):
    needles = _as_list(check.get("value"))
    hay = output.lower()
    found = [n for n in needles if n.lower() in hay]
    missing = [n for n in needles if n.lower() not in hay]
    score = len(found) / len(needles) if needles else 1.0
    ok = not missing
    return CheckResult("contains", score, ok, "All present" if ok else f"Missing: {missing}")


def check_not_contains(check, output, case, ctx):
    needles = _as_list(check.get("value"))
    hay = output.lower()
    present = [n for n in needles if n.lower() in hay]
    ok = not present
    return CheckResult("not_contains", float(ok), ok, "None present" if ok else f"Found forbidden: {present}")


def check_regex(check, output, case, ctx):
    pattern = check["pattern"]
    flags = re.I if check.get("ignore_case", True) else 0
    ok = re.search(pattern, output, flags) is not None
    return CheckResult("regex", float(ok), ok, f"Pattern {'matched' if ok else 'not found'}: {pattern}")


_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_schema(value, schema: dict, path: str = "$") -> list[str]:
    """Validate a small, commonly used subset of JSON Schema."""
    errors = []
    expected_type = schema.get("type")
    if expected_type:
        py = _TYPES[expected_type]
        is_bool = isinstance(value, bool)
        if not isinstance(value, py) or (is_bool and expected_type in ("number", "integer")):
            return [f"{path}: expected {expected_type}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate_schema(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


def check_json_schema(check, output, case, ctx):
    try:
        data = json.loads(_strip_fences(output))
    except json.JSONDecodeError as e:
        return CheckResult("json_schema", 0.0, False, f"Invalid JSON: {e.msg}")
    errors = validate_schema(data, check.get("schema", {}))
    ok = not errors
    return CheckResult("json_schema", 1.0 if ok else 0.5, ok, "Valid" if ok else "; ".join(errors[:3]))


def check_max_words(check, output, case, ctx):
    limit = int(check["value"])
    n = len(output.split())
    ok = n <= limit
    score = 1.0 if ok else max(0.0, 1 - (n - limit) / limit)
    return CheckResult("max_words", score, ok, f"{n} words (limit {limit})")


def check_similarity(check, output, case, ctx):
    reference = check.get("value", case.expected)
    if not reference:
        return CheckResult("similarity", 0.0, False, "No expected output to compare against")
    threshold = float(check.get("threshold", 0.6))
    sim = cosine_similarity(output, reference)
    ok = sim >= threshold
    return CheckResult("similarity", round(sim, 4), ok, f"Similarity {sim:.2f} (threshold {threshold})")


JUDGE_TEMPLATE = """You are a strict, fair evaluator of AI responses.
Grade the RESPONSE against the RUBRIC on a scale of 1 (very poor) to 5 (excellent).
Use the REFERENCE as an example of a good answer when one is given.

RUBRIC:
[[RUBRIC]]

INPUT:
<<INPUT>>[[INPUT]]<</INPUT>>

REFERENCE:
<<REFERENCE>>[[REFERENCE]]<</REFERENCE>>
[[CONTEXT_BLOCK]]
RESPONSE:
<<RESPONSE>>[[RESPONSE]]<</RESPONSE>>

Reply with JSON only, in this form: {"score": <integer 1-5>, "reason": "<one sentence>"}"""


CONTEXT_BLOCK = """
CONTEXT (the passages the system retrieved and was told to answer from):
<<CONTEXT>>[[CONTEXT]]<</CONTEXT>>
"""


def build_judge_prompt(rubric: str, input_text: str, reference: str | None, response: str,
                       context: str | None = None) -> str:
    block = CONTEXT_BLOCK.replace("[[CONTEXT]]", context) if context is not None else ""
    # The context block goes in before the response is substituted, so text inside the
    # response can never be mistaken for a template placeholder.
    return (
        JUDGE_TEMPLATE.replace("[[CONTEXT_BLOCK]]", block)
        .replace("[[RUBRIC]]", rubric)
        .replace("[[INPUT]]", input_text)
        .replace("[[REFERENCE]]", reference or "")
        .replace("[[RESPONSE]]", response)
    )


def _context_text(metadata: dict) -> str:
    ctx = (metadata or {}).get("context", "")
    if isinstance(ctx, list):
        return "\n\n".join(str(c) for c in ctx)
    return str(ctx)


def parse_judge(text: str) -> tuple[int, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return int(data["score"]), str(data.get("reason", ""))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1)), text.strip()[:200]
    raise ValueError(f"Could not parse judge output: {text[:200]!r}")


def check_llm_judge(check, output, case, ctx):
    judge = ctx.get("judge")
    if judge is None:
        return CheckResult("llm_judge", 0.0, False, "No judge configured in the suite")
    threshold = int(check.get("threshold", 4))
    input_text = "\n".join(f"{k}: {v}" for k, v in case.vars.items())
    reference = case.expected if check.get("use_reference", True) else None
    context = _context_text(ctx.get("metadata", {})) if check.get("include_context") else None
    prompt = build_judge_prompt(check.get("rubric", "Is the response correct and helpful?"),
                                input_text, reference, output, context)
    try:
        completion = judge.complete(prompt)
        score, reason = parse_judge(completion.text)
    except Exception as e:  # noqa: BLE001 - a broken judge should not crash the run
        return CheckResult("llm_judge", 0.0, False, f"Judge error: {e}")
    ctx.setdefault("judge_cost", 0.0)
    ctx["judge_cost"] += completion.cost_usd
    ok = score >= threshold
    return CheckResult("llm_judge", (score - 1) / 4, ok, f"Judge {score}/5: {reason}")


def check_sources(check, output, case, ctx):
    sources = [str(s).lower() for s in (ctx.get("metadata", {}) or {}).get("sources", [])]
    wanted = _as_list(check.get("value"))
    if not wanted:
        return CheckResult("sources", 0.0, False, "No expected sources given")
    found = [w for w in wanted if any(w.lower() in s for s in sources)]
    if check.get("mode", "any") == "all":
        ok = len(found) == len(wanted)
        score = len(found) / len(wanted)
    else:
        ok = bool(found)
        score = float(ok)
    reason = f"Found {found}" if ok else f"Expected {wanted} in retrieved sources {sources[:5]}"
    return CheckResult("sources", score, ok, reason)


CHECKS = {
    "exact": check_exact,
    "contains": check_contains,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "json_schema": check_json_schema,
    "max_words": check_max_words,
    "similarity": check_similarity,
    "llm_judge": check_llm_judge,
    "sources": check_sources,
}


def run_check(check: dict, output: str, case, ctx: dict) -> CheckResult:
    kind = check.get("type")
    name = str(check.get("name") or kind)
    fn = CHECKS.get(kind)
    if fn is None:
        return CheckResult(str(kind), 0.0, False, f"Unknown check type '{kind}'", name)
    try:
        result = fn(check, output or "", case, ctx)
    except Exception as e:  # noqa: BLE001
        result = CheckResult(kind, 0.0, False, f"Check error: {e}")
    result.name = name
    return result
