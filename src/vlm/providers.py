"""
HTTP adapters for the VLM APIs (no vendor SDKs, just `requests`).

  anthropic         Messages API (image content block, output_config.effort)
  openai_responses  Responses API (input_image data URL, reasoning.effort)
  chat_compat       OpenAI-style /chat/completions: Gemini's OpenAI-compatible endpoint, OpenRouter, etc.
  openrouter        chat_compat against OpenRouter (usage accounting requested so real cost comes back)
  mock              offline stand-in for tests; never contacts a network

Payload builders and response parsers are pure functions so they can be unit-tested without a network.
API keys are read from the environment by the caller and are never logged or stored.
"""

import base64
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests

from src.vlm.models import ModelSpec

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/responses"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ProviderError(RuntimeError):
    pass


@dataclass
class ProviderResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reported_cost: Optional[float] = None
    raw: Dict = field(default_factory=dict)


def b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def build_anthropic_payload(spec: ModelSpec, prompt: str, png: bytes, max_tokens: int) -> Dict:
    payload = {
        "model": spec.api_model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png)}},
            {"type": "text", "text": prompt}]}],
    }
    if "effort" in spec.extra_body:
        payload["output_config"] = {"effort": spec.extra_body["effort"]}
    return payload


def parse_anthropic(resp: Dict) -> ProviderResult:
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    u = resp.get("usage", {})
    return ProviderResult(text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), 0, None, resp)


def build_openai_payload(spec: ModelSpec, prompt: str, png: bytes, max_tokens: int) -> Dict:
    payload = {
        "model": spec.api_model, "max_output_tokens": max_tokens,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64(png)}", "detail": "high"}]}],
    }
    if "effort" in spec.extra_body:
        payload["reasoning"] = {"effort": spec.extra_body["effort"]}
    return payload


def parse_openai(resp: Dict) -> ProviderResult:
    parts = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            parts += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
    u = resp.get("usage", {})
    reasoning = int((u.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
    return ProviderResult("".join(parts), int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)),
                          reasoning, None, resp)


def build_chat_payload(spec: ModelSpec, prompt: str, png: bytes, max_tokens: int) -> Dict:
    payload = {
        "model": spec.api_model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png)}"}},
            {"type": "text", "text": prompt}]}],
    }
    payload.update(spec.extra_body)
    if spec.route == "openrouter":
        payload["usage"] = {"include": True}
    return payload


def parse_chat(resp: Dict) -> ProviderResult:
    choice = (resp.get("choices") or [{}])[0].get("message", {})
    content = choice.get("content") or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    u = resp.get("usage") or {}
    reasoning = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    cost = u.get("cost")
    return ProviderResult(content, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), reasoning,
                          float(cost) if isinstance(cost, (int, float)) else None, resp)


def _post(url: str, headers: Dict, payload: Dict, timeout: float, retries: int) -> Dict:
    last = "no attempt"
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last = f"{type(e).__name__}"
        else:
            if r.status_code < 300:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in RETRY_STATUS:
                raise ProviderError(last)
        time.sleep(min(30.0, 2.0 ** attempt + random.random()))
    raise ProviderError(f"gave up after {retries + 1} attempts ({last})")


def call_model(spec: ModelSpec, prompt: str, png: bytes, api_key: str, max_tokens: int = 8000,
               timeout: float = 240.0, retries: int = 3, item: Optional[Dict] = None) -> ProviderResult:
    if spec.route == "mock":
        return call_mock(spec, item or {})
    if spec.route == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return parse_anthropic(_post(ANTHROPIC_URL, headers,
                                     build_anthropic_payload(spec, prompt, png, max_tokens), timeout, retries))
    if spec.route == "openai_responses":
        headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
        return parse_openai(_post(OPENAI_URL, headers,
                                  build_openai_payload(spec, prompt, png, max_tokens), timeout, retries))
    if spec.route in ("chat_compat", "openrouter"):
        headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
        return parse_chat(_post(spec.base_url.rstrip("/") + "/chat/completions", headers,
                                build_chat_payload(spec, prompt, png, max_tokens), timeout, retries))
    raise ProviderError(f"unknown route {spec.route!r}")


MOCK_BEHAVIOURS = ("oracle", "noisy", "empty", "garbage")


def mock_spec(kind: str) -> ModelSpec:
    if kind not in MOCK_BEHAVIOURS:
        raise ValueError(f"mock kind must be one of {MOCK_BEHAVIOURS}")
    return ModelSpec(key=f"mock-{kind}", label=f"Mock ({kind})", vendor="mock", tier="mock", route="mock",
                     api_model=kind, key_env="", openrouter_id="", price_in=1.0, price_out=1.0,
                     est_reasoning_tokens=0, context="n/a")


def call_mock(spec: ModelSpec, item: Dict) -> ProviderResult:
    """Deterministic fake answers for pipeline tests. 'oracle' peeks at the gold levels on purpose."""
    gold, price, atr = item.get("gold_levels", []), item.get("price", 1.0), item.get("atr14", 0.1)
    rng = random.Random(f"{spec.key}|{item.get('id')}")
    kind = spec.api_model
    if kind == "oracle":
        levels = [round(g, 2) for g in gold]
    elif kind == "noisy":
        levels = [round(g + rng.uniform(-0.3, 0.3) * atr, 2) for g in gold[:2]] + \
                 [round(price + rng.uniform(1, 8) * atr, 2) for _ in range(3)]
    elif kind == "empty":
        levels = []
    else:
        return ProviderResult("I cannot comment on this chart.", 5000, 20, 0)
    text = json.dumps({"resistance_levels": [{"price": p, "confidence": 0.5, "reason": "mock"} for p in levels]})
    return ProviderResult(text, 5000, 120, 0)
