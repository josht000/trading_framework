"""
Benchmark runner: results store, caching, spend cap, dry run.

Every call is appended to data/vlm/results/runs.jsonl and keyed by a hash of
(model, route, chart, prompt, image bytes, settings, repeat). Re-running skips keys that already
succeeded, so interrupted or repeated invocations never pay twice. `budget_usd` caps spend for one
invocation; a call is skipped when its estimated cost would push the total past the cap.
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.vlm.estimate import estimate_call
from src.vlm.models import ModelSpec
from src.vlm.packets import DEFAULT_ROOT, load_manifest
from src.vlm.prompt import build_prompt, parse_levels_strict
from src.vlm.providers import ProviderError, call_model

load_dotenv()


def results_dir(root: str) -> str:
    return os.path.join(root, "results")


def _read(path: str, mode: str = "r"):
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf8"})) as f:
        return f.read()


def run_key(spec: ModelSpec, item_id: str, variant: str, repeat: int, prompt: str, png: bytes, max_tokens: int) -> str:
    h = hashlib.sha1()
    for part in (spec.key, spec.route, spec.api_model, json.dumps(spec.extra_body, sort_keys=True), item_id, variant,
                 str(repeat), str(max_tokens), hashlib.sha1(prompt.encode()).hexdigest(),
                 hashlib.sha1(png).hexdigest()):
        h.update(part.encode())
        h.update(b"|")
    return h.hexdigest()[:20]


def load_runs(root: str = DEFAULT_ROOT) -> List[Dict]:
    path = os.path.join(results_dir(root), "runs.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def missing_keys(models: List[ModelSpec]) -> Dict[str, List[str]]:
    miss: Dict[str, List[str]] = {}
    for m in models:
        if m.route != "mock" and not os.getenv(m.key_env):
            miss.setdefault(m.key_env, []).append(m.key)
    return miss


@dataclass
class RunSummary:
    attempted: int = 0
    cached: int = 0
    ok: int = 0
    failed: int = 0
    skipped_budget: int = 0
    spent: float = 0.0


def run_benchmark(models: List[ModelSpec], root: str = DEFAULT_ROOT, variant: str = "rubric", repeats: int = 1,
                  budget_usd: float = 5.0, workers: int = 2, limit: Optional[int] = None, max_tokens: int = 8000,
                  dry_run: bool = False, only_items: Optional[List[str]] = None, log=print) -> RunSummary:
    manifest = load_manifest(root)
    items = manifest["items"]
    if only_items:
        items = [i for i in items if any(s in i["id"] for s in only_items)]
    w, h = manifest["image_px"]

    done = {r["key"] for r in load_runs(root) if r.get("status") == "ok"}
    os.makedirs(results_dir(root), exist_ok=True)
    os.makedirs(os.path.join(results_dir(root), "raw"), exist_ok=True)
    store = os.path.join(results_dir(root), "runs.jsonl")

    jobs = []
    for spec in models:
        for item in items:
            png = _read(os.path.join(root, item["chart_blind"]), "rb")
            prompt = build_prompt(variant, item, _read(os.path.join(root, item["ohlcv"])))
            for rep in range(repeats):
                key = run_key(spec, item["id"], variant, rep, prompt, png, max_tokens)
                jobs.append((spec, item, rep, key, prompt, png, len(prompt)))
    if limit:
        jobs = jobs[:limit]

    summary = RunSummary()
    lock = threading.Lock()

    def work(job):
        spec, item, rep, key, prompt, png, n_chars = job
        est = estimate_call(spec, w, h, n_chars, "base")["cost"]
        with lock:
            if key in done:
                summary.cached += 1
                return
            if summary.spent + est > budget_usd:
                summary.skipped_budget += 1
                return
            if dry_run:
                summary.spent += est
                summary.attempted += 1
                return
            summary.spent += est
            summary.attempted += 1
        t0 = time.time()
        rec = {"key": key, "model": spec.key, "route": spec.route, "item": item["id"], "repeat": rep,
               "prompt": variant, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            res = call_model(spec, prompt, png, os.getenv(spec.key_env, ""), max_tokens, item=item)
            levels = parse_levels_strict(res.text)
            cost = res.reported_cost if res.reported_cost is not None else (
                res.input_tokens * spec.price_in + res.output_tokens * spec.price_out) / 1e6
            rec.update({"status": "ok", "levels": levels or [], "parse_failed": levels is None,
                        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                        "reasoning_tokens": res.reasoning_tokens, "cost_usd": cost, "cost_reported": res.reported_cost is not None,
                        "latency_s": round(time.time() - t0, 2)})
            with open(os.path.join(results_dir(root), "raw", key + ".txt"), "w", encoding="utf8") as f:
                f.write(res.text)
            with lock:
                summary.ok += 1
                summary.spent += cost - est
        except ProviderError as e:
            rec.update({"status": "error", "error": str(e)[:400], "latency_s": round(time.time() - t0, 2), "cost_usd": 0.0})
            with lock:
                summary.failed += 1
                summary.spent -= est
        with lock:
            with open(store, "a", encoding="utf8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"  {spec.key:22s} {item['id']:24s} rep{rep} {rec['status']:5s} "
                f"${rec.get('cost_usd', 0):.4f}  {rec.get('latency_s', 0):.1f}s")

    if dry_run:
        for j in jobs:
            work(j)
        return summary
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(work, jobs))
    return summary
