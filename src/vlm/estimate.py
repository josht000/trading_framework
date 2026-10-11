"""
Token and cost estimates for a packet (image + prompt + OHLCV text).

These are ESTIMATES: image tokens follow each vendor's documented rule, text tokens use a
chars-per-token ratio typical for numeric CSV, and hidden reasoning tokens are an assumed
per-model figure. Real runs record the vendor-reported usage and cost.
"""

import math
from typing import Dict, List

import pandas as pd

from src.vlm.models import ModelSpec

SCENARIOS = {"low": 0.3, "base": 1.0, "high": 3.0}
VISIBLE_OUTPUT_TOKENS = 250
CHARS_PER_TOKEN = 2.2


def image_tokens(spec: ModelSpec, width: int, height: int) -> int:
    m = spec.image_method
    if m == "anthropic28":
        scale = min(1.0, 2576 / max(width, height), math.sqrt(4784 * 28 * 28 / (width * height)))
        w, h = int(width * scale), int(height * scale)
        return min(math.ceil(w / 28) * math.ceil(h / 28), 4784)
    if m == "openai32":
        patches = math.ceil(width / 32) * math.ceil(height / 32)
        if patches > 2500:
            f = math.sqrt(32 * 32 * 2500 / (width * height))
            patches = math.ceil(int(width * f) / 32) * math.ceil(int(height * f) / 32)
        return math.ceil(patches * spec.image_multiplier)
    if m == "gemini_fixed":
        return spec.image_tokens_fixed
    return min(math.ceil(width / 28) * math.ceil(height / 28), 4096)


def text_tokens(spec: ModelSpec, n_chars: int) -> int:
    return math.ceil(n_chars / CHARS_PER_TOKEN * spec.text_token_factor)


def estimate_call(spec: ModelSpec, width: int, height: int, text_chars: int, scenario: str = "base") -> Dict:
    t_in = image_tokens(spec, width, height) + text_tokens(spec, text_chars)
    t_out = VISIBLE_OUTPUT_TOKENS + int(spec.est_reasoning_tokens * SCENARIOS[scenario])
    cost = t_in * spec.price_in / 1e6 + t_out * spec.price_out / 1e6
    return {"image_tokens": image_tokens(spec, width, height), "input_tokens": t_in,
            "output_tokens": t_out, "cost": cost}


def model_table(models: List[ModelSpec], width: int, height: int, text_chars: int,
                n_charts: int, repeats: int, scenario: str = "base") -> pd.DataFrame:
    rows = []
    for m in models:
        e = estimate_call(m, width, height, text_chars, scenario)
        v = m.vision
        rows.append({
            "model": m.label, "key": m.key, "vendor": m.vendor, "tier": m.tier,
            "in_$/M": m.price_in, "out_$/M": m.price_out,
            "est_in_tok": e["input_tokens"], "est_out_tok": e["output_tokens"],
            "est_$/chart": e["cost"], f"est_$/run({n_charts}x{repeats})": e["cost"] * n_charts * repeats,
            "context": m.context, "open": "yes" if m.open_weights else "no",
            "arena_vision": v.get("arena_vision"), "charxiv_r": v.get("charxiv_r"),
            "chartography": v.get("chartography"), "mmmu_pro": v.get("mmmu_pro"),
            "route": m.route, "first_pass": "yes" if m.first_pass else "",
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    def cell(x):
        if isinstance(x, float):
            return "" if pd.isna(x) else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")
        return "" if x is None else str(x)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
