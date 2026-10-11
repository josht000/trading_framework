"""
Benchmark runner: results store, caching, spend cap, dry run.

Every call is appended to data/vlm/results/runs.jsonl and keyed by a hash of the model route, chart,
prompt context, shot prefix, image bytes, settings, and repeat. Re-running skips keys that already
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
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv

from src.vlm.estimate import estimate_call
from src.vlm.fewshot import load_shots, load_split_items, shots_fingerprint
from src.vlm.models import ModelSpec
from src.vlm.packets import DEFAULT_EXPLANATIONS_ROOT, DEFAULT_ROOT, load_explanation, load_manifest
from src.vlm.prompt import build_system_prompt, packet_text, parse_levels_strict
from src.vlm.providers import ProviderError, call_model

load_dotenv()


def results_dir(root: str) -> str:
    return os.path.join(root, "results")


def _read(path: str, mode: str = "r"):
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf8"})) as f:
        return f.read()


def run_key(spec: ModelSpec, item_id: str, variant: str, repeat: int, prompt: str, png: bytes,
            max_tokens: int, shots_fp: str = "") -> str:
    h = hashlib.sha1()
    for part in (spec.key, spec.route, spec.api_model, json.dumps(spec.extra_body, sort_keys=True), item_id, variant,
                 str(repeat), str(max_tokens), hashlib.sha1(prompt.encode()).hexdigest(),
                 hashlib.sha1(png).hexdigest(), shots_fp):
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
        if m.route != "mock" and m.key_env and not os.getenv(m.key_env):
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


def _truncated(finish_reason: str) -> bool:
    reason = (finish_reason or "").lower()
    return any(x in reason for x in ("length", "max_tokens", "max_output_tokens", "incomplete"))


def run_benchmark(models: List[ModelSpec], root: str = DEFAULT_ROOT, variant: str = "rubric", repeats: int = 1,
                  budget_usd: float = 5.0, workers: int = 2, limit: Optional[int] = None, max_tokens: int = 16000,
                  dry_run: bool = False, only_items: Optional[List[str]] = None, log=print,
                  *, user_text: str = "", explanations_root: str = DEFAULT_EXPLANATIONS_ROOT,
                  split: Optional[str] = None, shot_splits: Sequence[str] = ()) -> RunSummary:
    shot_splits = tuple(shot_splits)
    if "test" in shot_splits:
        raise ValueError("the test split must never be used as few-shot examples")
    if split and split in shot_splits:
        raise ValueError(f"the {split} split cannot be used as its own few-shot examples")

    if split:
        items = load_split_items(split, explanations_root)
        w, h = 1600, 1000
        chart_key, ohlcv_key = "chart_path", "ohlcv_path"
        default_split = split
    else:
        manifest = load_manifest(root)
        items = manifest["items"]
        w, h = manifest["image_px"]
        chart_key, ohlcv_key = "chart_blind", "ohlcv"
        default_split = "manifest"
        dataset_items = {}
        for packet_split in ("train", "val"):
            try:
                dataset_items.update({i["id"]: i for i in load_split_items(packet_split, explanations_root)})
            except FileNotFoundError:
                pass

    all_ids = [i["id"] for i in items]
    if only_items:
        wanted = [s.lower() for s in only_items if s.strip()]
        items = [i for i in items if any(s in i["id"].lower() for s in wanted)]
        if not items:
            log("No charts match --items " + ",".join(only_items) +
                ". Available ids: " + ", ".join(all_ids))
            return RunSummary()

    system_prompt = build_system_prompt(variant, user_text)
    shots = load_shots(shot_splits, explanations_root) if shot_splits else []
    shots_fp = shots_fingerprint(shots) if shots else ""
    shot_chars = sum(len(s["text"]) + len(s["answer"]) for s in shots)
    context_chars = len(system_prompt) + shot_chars
    n_images = 1 + len(shots)
    shots_label = ",".join(shot_splits)

    done = {r["key"] for r in load_runs(root) if r.get("status") == "ok"}
    os.makedirs(results_dir(root), exist_ok=True)
    os.makedirs(os.path.join(results_dir(root), "raw"), exist_ok=True)
    store = os.path.join(results_dir(root), "runs.jsonl")

    jobs = []
    for spec in models:
        for item in items:
            if split:
                chart_path, ohlcv_path = item[chart_key], item[ohlcv_key]
            else:
                chart_path, ohlcv_path = os.path.join(root, item[chart_key]), os.path.join(root, item[ohlcv_key])
                packet = dataset_items.get(item["id"], {})
                if not os.path.exists(chart_path):
                    chart_path = packet.get("chart_path", chart_path)
                if not os.path.exists(ohlcv_path):
                    ohlcv_path = packet.get("ohlcv_path", ohlcv_path)
            png = _read(chart_path, "rb")
            explanation = "" if split else load_explanation(item, explanations_root)
            query_text = packet_text(item, _read(ohlcv_path), explanation)
            key_prompt = system_prompt + "\n\n" + query_text
            for rep in range(repeats):
                key = run_key(spec, item["id"], variant, rep, key_prompt, png, max_tokens, shots_fp)
                jobs.append((spec, item, rep, key, query_text, png, len(query_text)))
    if limit:
        jobs = jobs[:limit]

    summary = RunSummary()
    lock = threading.Lock()

    def work(job):
        spec, item, rep, key, query_text, png, n_chars = job
        est = estimate_call(spec, w, h, n_chars, "base", context_chars, n_images)["cost"]
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
               "prompt": variant, "split": default_split, "shots": shots_label, "n_shots": len(shots),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            res = call_model(spec, query_text, png, os.getenv(spec.key_env, ""), max_tokens,
                             item=item, system=system_prompt, shots=shots)
            levels = parse_levels_strict(res.text)
            cost = res.reported_cost if res.reported_cost is not None else (
                res.input_tokens * spec.price_in + res.output_tokens * spec.price_out) / 1e6
            with open(os.path.join(results_dir(root), "raw", key + ".txt"), "w", encoding="utf8") as f:
                f.write(res.text)
            base = {"levels": levels or [], "parse_failed": levels is None,
                    "finish_reason": res.finish_reason, "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens, "reasoning_tokens": res.reasoning_tokens,
                    "cost_usd": cost, "cost_reported": res.reported_cost is not None,
                    "latency_s": round(time.time() - t0, 2)}
            if not res.text.strip():
                rec.update(base, status="error", error="empty response from provider "
                           f"(finish_reason={res.finish_reason or 'unknown'}; check max_tokens/reasoning effort)")
            elif _truncated(res.finish_reason):
                rec.update(base, status="error", error=f"truncated response (finish_reason={res.finish_reason})")
            else:
                rec.update(base, status="ok")
            with lock:
                if rec["status"] == "ok":
                    summary.ok += 1
                else:
                    summary.failed += 1
                summary.spent += cost - est
        except ProviderError as e:
            rec.update({"status": "error", "error": str(e)[:400], "latency_s": round(time.time() - t0, 2),
                        "cost_usd": 0.0})
            with lock:
                summary.failed += 1
                summary.spent -= est
        with lock:
            with open(store, "a", encoding="utf8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"  {spec.key:22s} {item['id']:24s} rep{rep} {rec['status']:5s} "
                f"${rec.get('cost_usd', 0):.4f}  {rec.get('latency_s', 0):.1f}s "
                f"finish={rec.get('finish_reason') or '-'}")

    if dry_run:
        for j in jobs:
            work(j)
        return summary
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(work, jobs))
    return summary
