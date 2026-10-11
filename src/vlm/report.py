"""
Review outputs: leaderboard tables and an HTML page with, for every chart and model, the model's
levels drawn over the same frame as the human gold levels.

  blue dashed = human gold level           green solid = model level that matched a gold level
  red solid   = model level with no match  orange tag  = gold level the model missed
"""

import html
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.research.chart_png import level, render_chart_png
from src.research.config_io import make_detector
from src.research.data import load_daily
from src.research.labels import match_levels
from src.vlm.fewshot import load_split_items
from src.vlm.packets import DEFAULT_EXPLANATIONS_ROOT, DEFAULT_ROOT, load_manifest
from src.vlm.runner import load_runs, results_dir
from src.vlm.scoring import PRIMARY_TOL, leaderboard, score_runs

GOLD, HIT, MISS, MISSED = "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"


def overlay_levels(gold: List[float], predicted: List[float], price: float, atr: float, tol_atr: float) -> List[Dict]:
    g = np.array(sorted(x for x in gold if x > price))
    p = np.array(sorted(x for x in predicted if x > price))
    pairs, gold_left, pred_left = match_levels(g, p, tol_atr * atr)
    out = [level(x, GOLD, "--", 1.2, f"gold {x:.2f}") for x in g]
    out += [level(p[j], HIT, "-", 2.0, f"ok {p[j]:.2f}") for _, j in pairs]
    out += [level(p[j], MISS, "-", 1.8, f"x {p[j]:.2f}") for j in pred_left]
    out += [level(g[i], MISSED, "--", 1.4, f"missed {g[i]:.2f}") for i in gold_left]
    return out


def _formatted_table(df: pd.DataFrame) -> str:
    shown = df.copy()
    for c in shown.columns:
        if shown[c].dtype.kind == "f":
            shown[c] = shown[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return shown.to_html(index=False, border=0)


def _page(title: str, body: str) -> str:
    return ("<html><head><meta charset='utf8'><title>" + html.escape(title) + "</title><style>"
            "body{font-family:sans-serif;margin:20px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px 8px;font-size:13px}"
            ".row{display:flex;flex-wrap:wrap;gap:12px}.cell{font-size:13px}</style></head><body>" + body + "</body></html>")


def _split_report(root: str, splits: Sequence[str], dataset_root: str, primary: float) -> Dict[str, str]:
    items = {}
    for split in splits:
        for item in load_split_items(split, dataset_root):
            items[item["id"]] = item
    runs = [r for r in load_runs(root) if r.get("split") in splits and r["item"] in items]
    if not runs:
        raise SystemExit(f"No runs for split(s) {', '.join(splits)}. Use `vlm_benchmark.py run --split` first.")
    scores = score_runs(runs, items)
    board = leaderboard(scores, primary)

    tag = "_".join(splits)
    out_dir = os.path.join(results_dir(root), "review")
    os.makedirs(out_dir, exist_ok=True)
    paths = {"leaderboard_csv": os.path.join(results_dir(root), f"leaderboard_{tag}.csv"),
             "per_run_csv": os.path.join(results_dir(root), f"per_run_scores_{tag}.csv"),
             "html": os.path.join(out_dir, f"index_{tag}.html")}
    board.to_csv(paths["leaderboard_csv"], index=False)
    scores.to_csv(paths["per_run_csv"], index=False)

    shown_scores = scores.copy()
    shown_scores["raw"] = ""
    run_lookup = {(r["item"], r["model"], r.get("repeat", 0), r.get("split") or "manifest",
                   r.get("shots") or "", r.get("prompt") or ""): r for r in runs}
    for idx, row in shown_scores.iterrows():
        run = run_lookup.get((row["item"], row["model"], row.get("repeat", 0), row["split"],
                              row["shots"], row["prompt"]))
        if run:
            shown_scores.at[idx, "raw"] = f"<a href='../raw/{run['key']}.txt'>raw</a>"
    body = (f"<h2>VLM resistance-level benchmark — {html.escape(tag)}</h2>"
            f"<p>Match tolerance {primary:g} ATR14. Rows are grouped by model, split, shots and prompt.</p>"
            f"{_formatted_table(board)}<h3>Per-chart scores</h3>{shown_scores.to_html(index=False, border=0, escape=False)}")
    with open(paths["html"], "w", encoding="utf8") as f:
        f.write(_page(f"VLM resistance benchmark {tag}", body))
    return paths


def build_report(root: str = DEFAULT_ROOT, primary: float = PRIMARY_TOL, render: bool = True,
                 splits: Optional[Sequence[str]] = None,
                 dataset_root: str = DEFAULT_EXPLANATIONS_ROOT) -> Dict[str, str]:
    if splits:
        return _split_report(root, tuple(splits), dataset_root, primary)

    manifest = load_manifest(root)
    items = {i["id"]: i for i in manifest["items"]}
    runs = [r for r in load_runs(root) if r["item"] in items and (r.get("split") or "manifest") == "manifest"]
    if not runs:
        raise SystemExit("No manifest runs yet. Use `vlm_benchmark.py run` first.")
    scores = score_runs(runs, items)
    board = leaderboard(scores, primary)

    out_dir = os.path.join(results_dir(root), "review")
    os.makedirs(out_dir, exist_ok=True)
    paths = {"leaderboard_csv": os.path.join(results_dir(root), "leaderboard.csv"),
             "per_run_csv": os.path.join(results_dir(root), "per_run_scores.csv"),
             "html": os.path.join(out_dir, "index.html")}
    board.to_csv(paths["leaderboard_csv"], index=False)
    scores.to_csv(paths["per_run_csv"], index=False)

    best = {}
    for r in runs:
        if r.get("status") == "ok":
            key = (r["model"], r.get("shots") or "", r.get("prompt") or "", r["item"])
            best.setdefault(key, r)

    det = make_detector("atr_pivot", None)
    sections = []
    for item in manifest["items"]:
        cells = []
        series = None
        matching = sorted(k for k in best if k[3] == item["id"])
        for model, shots, prompt, _ in matching:
            r = best[(model, shots, prompt, item["id"])]
            preds = [lv["price"] for lv in r.get("levels", [])]
            label = model + (f" <small>shots={html.escape(shots)} prompt={html.escape(prompt)}</small>"
                             if shots or prompt else "")
            img = os.path.join(out_dir, item["id"], f"{model}_{r['key']}.png")
            if render:
                if series is None:
                    daily = load_daily([item["symbol"]], item["asof"], item["asof"])[item["symbol"]]
                    p = det.prepare(daily)
                    series = (p, det.known_through_on(p, item["asof"]))
                os.makedirs(os.path.dirname(img), exist_ok=True)
                render_chart_png(series[0], series[1], img, item["symbol"], lookback=manifest["bars"], size=(12, 7.5), dpi=90,
                                 levels=overlay_levels(item["gold_levels"], preds, item["price"], item["atr14"], primary),
                                 title=f"{model} on {item['id']}")
            row = scores[(scores["model"] == model) & (scores["item"] == item["id"]) &
                         (scores["repeat"] == r["repeat"]) & (scores["shots"] == (r.get("shots") or ""))]
            t = f"{primary:g}"
            stats = (f"P {row[f'precision@{t}'].iloc[0]:.2f}  R {row[f'recall@{t}'].iloc[0]:.2f}  "
                     f"F1 {row[f'f1@{t}'].iloc[0]:.2f}  levels {int(row['n_pred'].iloc[0])}  ${r.get('cost_usd', 0):.3f}"
                     if len(row) and f"precision@{t}" in row else "no score")
            raw = os.path.join("..", "raw", r["key"] + ".txt").replace("\\", "/")
            image_html = (f"<a href='{item['id']}/{model}_{r['key']}.png'>"
                          f"<img src='{item['id']}/{model}_{r['key']}.png' width='430'></a><br>" if render else "")
            cells.append(f"<div class='cell'><b>{label}</b><br><small>{stats}</small><br>"
                         f"{image_html}<small><a href='{raw}'>raw answer</a></small></div>")
        gold_img = f"../../{item['chart_labeled']}"
        sections.append(f"<h3>{html.escape(item['id'])} (gold: {', '.join(f'{g:.2f}' for g in item['gold_levels'])})</h3>"
                        f"<div class='row'><div class='cell'><b>human gold</b><br><a href='{gold_img}'>"
                        f"<img src='{gold_img}' width='430'></a></div>{''.join(cells)}</div>")

    t = f"{primary:g}"
    body = (f"<h2>VLM resistance-level benchmark</h2><p>Match tolerance {primary:g} ATR14 (headline F1@{t}). "
            "Blue dashed = human gold; green = matched model level; red = unmatched model level; orange = missed gold.</p>"
            f"{_formatted_table(board)}{''.join(sections)}")
    with open(paths["html"], "w", encoding="utf8") as f:
        f.write(_page("VLM resistance benchmark", body))
    return paths
