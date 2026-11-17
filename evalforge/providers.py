"""Model providers. Uses only the standard library for HTTP so there are no SDK dependencies.

Every provider returns a Completion with text, token usage, latency and cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    # Extra data a provider wants to record, e.g. the sources and context a RAG system retrieved.
    metadata: dict = field(default_factory=dict)


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _post_json(url: str, payload: dict, headers: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {e.code}: {body[:300]}", retryable=e.code in RETRYABLE_STATUS) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ProviderError(f"Network error: {e}", retryable=True) from e


@dataclass
class Provider:
    id: str
    model: str
    params: dict = field(default_factory=dict)
    # USD per 1M tokens, e.g. {"input": 0.15, "output": 0.60}. Set it in the suite file;
    # prices change often, so EvalForge does not hard-code them for real providers.
    cost_per_1m: dict | None = None

    def complete(self, prompt: str, system: str | None = None, reference: str | None = None) -> Completion:
        start = time.perf_counter()
        text, in_tok, out_tok = self._call(prompt, system, reference)
        latency_ms = (time.perf_counter() - start) * 1000
        rates = self.cost_per_1m or self.default_cost()
        cost = (in_tok * rates.get("input", 0) + out_tok * rates.get("output", 0)) / 1_000_000
        return Completion(text, in_tok, out_tok, latency_ms, cost)

    def default_cost(self) -> dict:
        return {"input": 0.0, "output": 0.0}

    def _call(self, prompt: str, system: str | None, reference: str | None) -> tuple[str, int, int]:
        raise NotImplementedError


class OpenAIProvider(Provider):
    """OpenAI Chat Completions. Works with any OpenAI-compatible API via OPENAI_BASE_URL."""

    def _call(self, prompt, system, reference):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post_json(
            f"{base}/chat/completions",
            {"model": self.model, "messages": messages, **self.params},
            {"Authorization": f"Bearer {key}"},
        )
        usage = data.get("usage", {})
        text = data["choices"][0]["message"].get("content") or ""
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class AnthropicProvider(Provider):
    """Anthropic Messages API."""

    def _call(self, prompt, system, reference):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        payload = {
            "model": self.model,
            "max_tokens": self.params.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in self.params.items() if k != "max_tokens"},
        }
        if system:
            payload["system"] = system
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class OllamaProvider(Provider):
    """Local models through Ollama (free, no API key)."""

    def _call(self, prompt, system, reference):
        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post_json(
            f"{base}/api/chat",
            {"model": self.model, "messages": messages, "stream": False, "options": self.params},
            {},
        )
        return (
            data.get("message", {}).get("content", ""),
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
        )


# --------------------------------------------------------------------------------------
# Mock provider: deterministic fake models so the whole pipeline runs with no API keys.
# --------------------------------------------------------------------------------------

MOCK_MODELS = {
    # accuracy = chance of producing the reference answer; latency in ms; cost per 1M tokens
    "mock-strong": {"accuracy": 0.92, "latency": (40, 90), "cost": {"input": 2.50, "output": 10.0}},
    "mock-balanced": {"accuracy": 0.80, "latency": (20, 50), "cost": {"input": 0.40, "output": 1.60}},
    "mock-fast": {"accuracy": 0.62, "latency": (5, 20), "cost": {"input": 0.05, "output": 0.20}},
}

def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class MockProvider(Provider):
    """Simulates models of different quality, speed and cost.

    It answers with the case's reference output with probability `accuracy`, otherwise it
    returns a degraded answer. Results are deterministic for the same model + prompt,
    so reruns are reproducible. A system prompt slightly improves accuracy, so prompt
    variants produce visibly different scores in demos.
    """

    def default_cost(self):
        return MOCK_MODELS.get(self.model, MOCK_MODELS["mock-balanced"])["cost"]

    def _call(self, prompt, system, reference):
        cfg = MOCK_MODELS.get(self.model, MOCK_MODELS["mock-balanced"])
        seed = int(hashlib.sha256(f"{self.model}|{system}|{prompt}".encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        time.sleep(rng.uniform(*cfg["latency"]) / 1000)

        if "<<RESPONSE>>" in prompt:  # acting as an LLM judge
            text = self._judge(prompt, rng)
        else:
            accuracy = cfg["accuracy"] + (0.05 if system else 0.0)
            if reference and rng.random() < accuracy:
                text = reference
            elif reference:
                text = self._degrade(reference, rng)
            else:
                text = " ".join(prompt.split()[:40])
        return text, _approx_tokens((system or "") + prompt), _approx_tokens(text)

    @staticmethod
    def _degrade(reference: str, rng: random.Random) -> str:
        stripped = reference.strip()
        mode = rng.choice(["truncate", "generic", "drop_words"])
        if stripped.startswith("{"):
            # Broken or incomplete JSON
            return stripped[: max(1, len(stripped) // 2)] if mode != "generic" else "Sure! Here is the data you asked for."
        if mode == "truncate":
            words = stripped.split()
            return " ".join(words[: max(1, len(words) // 3)])
        if mode == "generic":
            return "The customer contacted support about an issue and wants it resolved."
        words = stripped.split()
        return " ".join(w for i, w in enumerate(words) if i % 2 == 0)

    @staticmethod
    def _judge(prompt: str, rng: random.Random) -> str:
        from .scorers import cosine_similarity  # local import avoids a cycle

        def between(tag):
            m = re.search(rf"<<{tag}>>(.*?)<</{tag}>>", prompt, re.S)
            return m.group(1).strip() if m else ""

        response, reference = between("RESPONSE"), between("REFERENCE") or between("CONTEXT")
        sim = cosine_similarity(response, reference) if reference else 0.6
        noise = rng.choice([-1, 0, 0, 0, 1])  # judges are noisy
        score = min(5, max(1, round(1 + sim * 4) + noise))
        return json.dumps({"score": score, "reason": f"Mock judge: similarity to reference {sim:.2f}."})


class PythonProvider(Provider):
    """Calls your own Python function, so you can evaluate a whole system (for example a RAG
    pipeline) rather than a single model call.

    `model` is "package.module:function". The function is called as
    fn(prompt, system=None, **params) and returns either a string, or a dict with
    "text" and optionally "input_tokens", "output_tokens", "cost_usd" and "metadata".
    Put whatever later checks need in metadata, e.g. {"sources": [...], "context": "..."}.
    """

    _cache: dict = {}

    def _function(self):
        if self.model not in self._cache:
            if ":" not in self.model:
                raise ProviderError(f"Python provider model must look like 'module:function', got {self.model!r}")
            module_name, fn_name = self.model.split(":", 1)
            import importlib

            try:
                module = importlib.import_module(module_name)
                self._cache[self.model] = getattr(module, fn_name)
            except (ImportError, AttributeError) as e:
                raise ProviderError(f"Cannot import {self.model}: {e}") from e
        return self._cache[self.model]

    def complete(self, prompt, system=None, reference=None):
        fn = self._function()
        start = time.perf_counter()
        try:
            out = fn(prompt, system=system, **self.params)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=bool(getattr(e, "retryable", False))) from e
        latency_ms = (time.perf_counter() - start) * 1000
        if isinstance(out, str):
            out = {"text": out}
        in_tok, out_tok = int(out.get("input_tokens", 0)), int(out.get("output_tokens", 0))
        if "cost_usd" in out:
            cost = float(out["cost_usd"])
        else:
            rates = self.cost_per_1m or {}
            cost = (in_tok * rates.get("input", 0) + out_tok * rates.get("output", 0)) / 1_000_000
        return Completion(str(out.get("text", "")), in_tok, out_tok, latency_ms, cost, dict(out.get("metadata") or {}))


PROVIDERS = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "python": PythonProvider,
}


def build_provider(spec) -> Provider:
    cls = PROVIDERS.get(spec.provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{spec.provider}'. Choose from: {', '.join(PROVIDERS)}")
    return cls(id=spec.id, model=spec.model, params=dict(spec.params), cost_per_1m=spec.cost_per_1m)
