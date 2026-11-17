"""Loading and validating test suite files (YAML or JSON)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class SuiteError(ValueError):
    pass


@dataclass
class ProviderSpec:
    id: str
    provider: str
    model: str
    params: dict = field(default_factory=dict)
    cost_per_1m: dict | None = None


@dataclass
class PromptSpec:
    id: str
    template: str
    system: str | None = None


@dataclass
class Case:
    id: str
    vars: dict
    expected: str | None = None
    tags: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    providers: list[ProviderSpec]
    prompts: list[PromptSpec]
    cases: list[Case]
    default_checks: list[dict] = field(default_factory=list)
    judge: ProviderSpec | None = None
    description: str = ""
    path: str = ""


_VAR_RE = re.compile(r"\{(\w+)\}")


def render(template: str, variables: dict) -> str:
    """Fill {name} placeholders. Other braces (for example JSON examples) are left alone."""

    def sub(m):
        key = m.group(1)
        return str(variables[key]) if key in variables else m.group(0)

    return _VAR_RE.sub(sub, template)


def missing_vars(template: str, variables: dict) -> list[str]:
    return [v for v in _VAR_RE.findall(template) if v not in variables]


def _unique(items, kind):
    seen = set()
    for it in items:
        if it.id in seen:
            raise SuiteError(f"Duplicate {kind} id: '{it.id}'")
        seen.add(it.id)


def _provider(d: dict, where: str) -> ProviderSpec:
    for key in ("id", "provider", "model"):
        if key not in d:
            raise SuiteError(f"{where} is missing '{key}'")
    return ProviderSpec(
        id=str(d["id"]),
        provider=d["provider"],
        model=d["model"],
        params=d.get("params", {}) or {},
        cost_per_1m=d.get("cost_per_1m"),
    )


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    if not path.exists():
        raise SuiteError(f"Suite file not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise SuiteError("Suite file must contain a mapping at the top level")

    providers = [_provider(p, f"providers[{i}]") for i, p in enumerate(raw.get("providers", []))]
    prompts = []
    for i, p in enumerate(raw.get("prompts", [])):
        if "id" not in p or "template" not in p:
            raise SuiteError(f"prompts[{i}] needs 'id' and 'template'")
        prompts.append(PromptSpec(id=str(p["id"]), template=p["template"], system=p.get("system")))

    cases = []
    for i, c in enumerate(raw.get("cases", [])):
        if "id" not in c:
            raise SuiteError(f"cases[{i}] is missing 'id'")
        cases.append(
            Case(
                id=str(c["id"]),
                vars=c.get("vars", {}) or {},
                expected=c.get("expected"),
                tags=list(c.get("tags", []) or []),
                checks=list(c.get("checks", []) or []),
            )
        )

    if not providers:
        raise SuiteError("Suite needs at least one provider")
    if not prompts:
        raise SuiteError("Suite needs at least one prompt")
    if not cases:
        raise SuiteError("Suite needs at least one case")
    _unique(providers, "provider")
    _unique(prompts, "prompt")
    _unique(cases, "case")

    for prompt in prompts:
        for case in cases:
            missing = missing_vars(prompt.template, case.vars)
            if missing:
                raise SuiteError(
                    f"Prompt '{prompt.id}' uses {missing} but case '{case.id}' does not define them"
                )

    judge = _provider(raw["judge"], "judge") if raw.get("judge") else None
    return Suite(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        providers=providers,
        prompts=prompts,
        cases=cases,
        default_checks=list(raw.get("default_checks", []) or []),
        judge=judge,
        path=str(path),
    )
