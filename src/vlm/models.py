"""Model registry (data/vlm/models.json): prices, API routing, image/token assumptions, vision signals."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REGISTRY_PATH = os.path.join("data", "vlm", "models.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    vendor: str
    tier: str
    route: str
    api_model: str
    key_env: str
    openrouter_id: str
    price_in: float
    price_out: float
    price_note: str = ""
    image_method: str = "generic28"
    image_tokens_fixed: int = 1120
    image_multiplier: float = 1.0
    text_token_factor: float = 1.0
    base_url: str = ""
    extra_body: Dict = field(default_factory=dict)
    est_reasoning_tokens: int = 2000
    context: str = ""
    open_weights: bool = False
    first_pass: bool = False
    vision: Dict = field(default_factory=dict)
    vision_note: str = ""

    def via(self, route: str = "native") -> "ModelSpec":
        """Return the spec re-pointed at OpenRouter when route == 'openrouter' (one key for every model)."""
        if route == "native" or self.route == "mock":
            return self
        if self.route == "local":
            raise ValueError(f"{self.key} is a local model and cannot be routed through OpenRouter")
        if route != "openrouter":
            raise ValueError(f"unknown route {route!r}; use native or openrouter")
        from dataclasses import replace
        extra = self.extra_body
        if "effort" in extra or "reasoning_effort" in extra:
            level = extra.get("effort") or extra.get("reasoning_effort")
            extra = {"reasoning": {"effort": level}}
        return replace(self, route="openrouter", api_model=self.openrouter_id, key_env=OPENROUTER_KEY_ENV,
                       base_url=OPENROUTER_URL, extra_body=extra)


def load_registry(path: str = REGISTRY_PATH) -> Dict:
    with open(path, encoding="utf8") as f:
        raw = json.load(f)
    known = set(ModelSpec.__dataclass_fields__)
    models = []
    for m in raw["models"]:
        unknown = set(m) - known
        if unknown:
            raise ValueError(f"{m.get('key')}: unknown registry fields {sorted(unknown)}")
        models.append(ModelSpec(**m))
    keys = [m.key for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model keys in registry")
    return {"as_of": raw.get("as_of"), "notes": raw.get("notes", []), "sources": raw.get("sources", {}),
            "models": models}


def select_models(registry: Dict, names: Optional[str] = None, first_pass: bool = False) -> List[ModelSpec]:
    models = registry["models"]
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        by_key = {m.key: m for m in models}
        missing = [n for n in wanted if n not in by_key]
        if missing:
            raise ValueError(f"unknown models {missing}; available: {sorted(by_key)}")
        return [by_key[n] for n in wanted]
    return [m for m in models if m.first_pass] if first_pass else list(models)
