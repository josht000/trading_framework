"""
Score a model's resistance levels against the human gold levels.

Scope: levels above the last close (the task asked only for those), gold likewise.
A predicted level matches a gold level when they are within tol_atr * ATR14 (greedy
one-to-one nearest matching, same rule as the detector benchmark). Three tolerances are
reported because reading a price off an axis is naturally a little loose; the primary one
is used for the headline F1.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.research.labels import match_levels

TOLERANCES = (0.25, 0.5, 1.0)
PRIMARY_TOL = 0.5


def score_levels(gold: Sequence[float], predicted: Sequence[float], price: float, atr: float,
                 tol_atrs: Sequence[float] = TOLERANCES) -> Dict:
    gold = np.array(sorted(g for g in gold if g > price), dtype=float)
    pred_all = np.array(sorted(predicted), dtype=float)
    pred = pred_all[pred_all > price]
    out = {"n_gold": len(gold), "n_pred": len(pred), "n_pred_below_price": int(len(pred_all) - len(pred))}
    for tol in tol_atrs:
        pairs, gold_left, pred_left = match_levels(gold, pred, tol * atr)
        tp = len(pairs)
        prec = tp / len(pred) if len(pred) else 0.0
        rec = tp / len(gold) if len(gold) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        t = f"{tol:g}"
        out.update({f"tp@{t}": tp, f"precision@{t}": prec, f"recall@{t}": rec, f"f1@{t}": f1})
        if tol == PRIMARY_TOL:
            out["mae_atr"] = float(np.mean([abs(gold[i] - pred[j]) / atr for i, j in pairs])) if pairs else np.nan
            out["matched_pred_idx"] = [int(j) for _, j in pairs]
            out["missed_gold"] = [float(gold[i]) for i in gold_left]
            out["extra_pred"] = [float(pred[j]) for j in pred_left]
    return out


def score_runs(runs: List[Dict], items: Dict[str, Dict], tol_atrs: Sequence[float] = TOLERANCES) -> pd.DataFrame:
    """One row per run with metrics; failed or unparseable runs score zero and are flagged."""
    rows = []
    for r in runs:
        item = items.get(r["item"])
        if item is None:
            continue
        levels = [lv["price"] for lv in (r.get("levels") or [])]
        parsed_ok = r.get("status") == "ok" and not r.get("parse_failed", False)
        base = {"model": r["model"], "split": r.get("split") or "manifest",
                "shots": r.get("shots") or "", "n_shots": r.get("n_shots", 0),
                "prompt": r.get("prompt") or "", "item": r["item"], "repeat": r.get("repeat", 0),
                "status": r.get("status"), "parse_failed": bool(r.get("parse_failed", False)),
                "finish_reason": r.get("finish_reason") or "", "error": r.get("error") or "",
                "cost_usd": r.get("cost_usd") or 0.0, "latency_s": r.get("latency_s"),
                "in_tokens": r.get("input_tokens"), "out_tokens": r.get("output_tokens"),
                "reasoning_tokens": r.get("reasoning_tokens")}
        if r.get("status") != "ok":
            base.update({"n_gold": len([g for g in item["gold_levels"] if g > item["price"]])})
            rows.append(base)
            continue
        m = score_levels(item["gold_levels"], levels if parsed_ok else [], item["price"], item["atr14"], tol_atrs)
        rows.append({**base, **{k: v for k, v in m.items() if not isinstance(v, list)}})
    return pd.DataFrame(rows)


def leaderboard(scores: pd.DataFrame, primary: float = PRIMARY_TOL) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    t = f"{primary:g}"
    group_cols = ["model", "split", "shots", "prompt"]
    rows = []
    for keys, g in scores.groupby(group_cols, dropna=False):
        model, split, shots, prompt = keys
        gk = g[g["status"] == "ok"]
        rows.append({
            "model": model, "split": split, "shots": shots, "prompt": prompt,
            "runs": len(g), "ok": len(gk),
            "api_fail": int((g["status"] != "ok").sum()), "parse_fail": int(g["parse_failed"].sum()),
            f"precision@{t}": gk[f"precision@{t}"].mean() if len(gk) else np.nan,
            f"recall@{t}": gk[f"recall@{t}"].mean() if len(gk) else np.nan,
            f"f1@{t}": gk[f"f1@{t}"].mean() if len(gk) else np.nan,
            "f1@0.25": gk["f1@0.25"].mean() if len(gk) and "f1@0.25" in gk else np.nan,
            "f1@1": gk["f1@1"].mean() if len(gk) and "f1@1" in gk else np.nan,
            "f1_std": gk[f"f1@{t}"].std() if len(gk) > 1 else np.nan,
            "mae_atr": gk["mae_atr"].mean() if len(gk) else np.nan,
            "avg_levels": gk["n_pred"].mean() if len(gk) else np.nan,
            "avg_gold": gk["n_gold"].mean() if len(gk) else np.nan,
            "cost_usd": g["cost_usd"].sum(), "usd_per_run": g["cost_usd"].mean(),
            "avg_latency_s": g["latency_s"].mean(),
            "reasoning_tokens": g["reasoning_tokens"].sum() if "reasoning_tokens" in g else np.nan,
        })
    return pd.DataFrame(rows).sort_values(f"f1@{t}", ascending=False).reset_index(drop=True)
