"""
Hand-labeled resistance/support levels for benchmarking the detector.

The label file is a plain CSV you can edit in a text editor or Excel
(default: data/labels/resistance_labels.csv). One row per level per (symbol, asof):

  symbol,asof,id,level,status,source,score,kind,note
  TSLA,2024-11-05,1,273.54,reject,atr_pivot,3.1,swing,weak single touch
  TSLA,2024-11-05,2,271.00,keep,atr_pivot,8.6,swing,clean prior high
  TSLA,2024-11-05,3,283.50,add,manual,,,missed July high

`id` numbers the levels of one (symbol, asof) from the highest price down; the chart draws
the same id next to each line so you can match a row to a line. `source` is the detector
that produced the row (or manual).

Manual labeling needs only three columns: symbol,asof,level (status defaults to add and
source to manual), e.g.  TSLA,2024-11-05,271.00  (an optional note column is fine).
Rows with a blank level are placeholders and are ignored; a non-numeric level is an error.

status:
  auto    exported by the detector, not reviewed yet (ignored when scoring)
  keep    detector level you agree with
  reject  detector level you disagree with (counts as a false positive if it is still detected)
  add     a level the detector missed (counts toward recall)

Workflow: export candidates, edit statuses/add rows, score, tune the detector, score again.
Scoring recomputes the detector's levels with whatever config you pass, so the same
labels act as a fixed benchmark across tuning runs.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.screening.resistance import ResistanceDetector

DEFAULT_LABEL_FILE = os.path.join("data", "labels", "resistance_labels.csv")
LABEL_COLUMNS = ["symbol", "asof", "id", "level", "status", "source", "score", "kind", "note"]
REQUIRED_COLUMNS = ["symbol", "asof", "level"]
STATUSES = {"auto", "keep", "reject", "add"}


def load_labels(path: str = DEFAULT_LABEL_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(path, dtype={"symbol": str, "note": str, "source": str, "kind": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for col, default in (("id", np.nan), ("status", "add"), ("source", "manual"),
                         ("score", np.nan), ("kind", ""), ("note", "")):
        if col not in df.columns:
            df[col] = default
    raw_level = df["level"]
    level = pd.to_numeric(raw_level, errors="coerce")
    invalid = raw_level.notna() & level.isna() & (raw_level.astype(str).str.strip() != "")
    if invalid.any():
        raise ValueError(f"Non-numeric level in {path} (rows {list(df.index[invalid] + 2)}): "
                         f"{raw_level[invalid].tolist()}")
    df["level"] = level
    df = df[df["level"].notna()].reset_index(drop=True)
    if df.empty:
        return df[LABEL_COLUMNS]
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["asof"] = pd.to_datetime(df["asof"]).dt.normalize()
    df["status"] = df["status"].fillna("add").astype(str).str.strip().str.lower().replace("", "add")
    df["source"] = df["source"].fillna("manual")
    bad = df[~df["status"].isin(STATUSES)]
    if not bad.empty:
        raise ValueError(f"Invalid status in {path} (allowed: {sorted(STATUSES)}):\n{bad.to_string()}")
    return df[LABEL_COLUMNS]


def save_labels(df: pd.DataFrame, path: str = DEFAULT_LABEL_FILE) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df.copy()
    out["asof"] = pd.to_datetime(out["asof"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["symbol", "asof", "level"], ascending=[True, True, False]).reset_index(drop=True)
    out["id"] = out["id"].astype("Int64")
    out[LABEL_COLUMNS].to_csv(path, index=False, float_format="%.4f")


def labels_for(df: pd.DataFrame, symbol: str, asof) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["symbol"] == symbol.upper()) & (df["asof"] == pd.Timestamp(asof).normalize())]


def export_levels(detector, daily: pd.DataFrame, symbol: str, asof,
                  path: str = DEFAULT_LABEL_FILE, max_dist_atr: float = 10.0,
                  source: str = "detector") -> int:
    """Append the detector's current levels for (symbol, asof) as status=auto rows.
    Levels already present within one cent are skipped, so re-exporting after tuning
    only adds the new candidates for you to review. New rows get ids continuing from the
    group's current maximum, highest price first."""
    p = detector.prepare(daily)
    k = detector.known_through_on(p, asof)
    lv = detector.levels(p, k)
    if lv.empty:
        return 0
    lv = lv[(lv["center"] - p.close[k]).abs() <= max_dist_atr * p.atr[k]]

    df = load_labels(path)
    group = labels_for(df, symbol, pd.Timestamp(p.dates[k]))
    existing = group["level"].to_numpy()
    next_id = int(group["id"].max()) + 1 if group["id"].notna().any() else 1
    rows = []
    for r in lv.sort_values("upper", ascending=False).itertuples(index=False):
        level = round(float(r.upper), 4)
        if len(existing) and np.min(np.abs(existing - level)) <= 0.01:
            continue
        rows.append({"symbol": symbol.upper(), "asof": pd.Timestamp(p.dates[k]), "id": next_id,
                     "level": level, "status": "auto", "source": source,
                     "score": round(float(r.score), 2), "kind": r.kind, "note": ""})
        next_id += 1
    if rows:
        save_labels(pd.concat([df, pd.DataFrame(rows)], ignore_index=True), path)
    return len(rows)


def match_levels(gold: np.ndarray, detected: np.ndarray, tol: float):
    """Greedy one-to-one nearest matching within tol. Returns (pairs, gold_left, det_left)."""
    pairs, used_g, used_d = [], set(), set()
    cand = sorted((abs(g - d), i, j) for i, g in enumerate(gold) for j, d in enumerate(detected))
    for diff, i, j in cand:
        if diff > tol:
            break
        if i in used_g or j in used_d:
            continue
        pairs.append((i, j))
        used_g.add(i)
        used_d.add(j)
    return pairs, [i for i in range(len(gold)) if i not in used_g], [j for j in range(len(detected)) if j not in used_d]


def score_labels(labels: pd.DataFrame, daily_by_symbol: Dict[str, pd.DataFrame],
                 detector: ResistanceDetector, tol_atr: float = 0.3, max_dist_atr: float = 10.0) -> dict:
    rows, misses, false_pos, unlabeled = [], [], [], []
    skipped = 0
    for (symbol, asof), g in labels.groupby(["symbol", "asof"]):
        reviewed = g[g["status"] != "auto"]
        if reviewed.empty or symbol not in daily_by_symbol:
            skipped += 1
            continue
        p = detector.prepare(daily_by_symbol[symbol])
        k = detector.known_through_on(p, asof)
        if k < 0 or not np.isfinite(p.atr[k]):
            skipped += 1
            continue
        price, atr = float(p.close[k]), float(p.atr[k])
        tol = tol_atr * atr
        in_scope = lambda x: abs(x - price) <= max_dist_atr * atr

        gold = reviewed[reviewed["status"].isin(["keep", "add"])]["level"].to_numpy()
        gold = gold[[in_scope(x) for x in gold]] if len(gold) else gold
        rejected = reviewed[reviewed["status"] == "reject"]["level"].to_numpy()

        lv = detector.levels(p, k)
        det = lv["upper"].to_numpy() if not lv.empty else np.array([])
        det = det[[in_scope(x) for x in det]] if len(det) else det

        tp_pairs, gold_left, det_left = match_levels(gold, det, tol)
        det_left_levels = det[det_left] if len(det_left) else np.array([])
        fp_pairs, _, unl_left = match_levels(rejected, det_left_levels, tol)

        tp, fn, fp, unl = len(tp_pairs), len(gold_left), len(fp_pairs), len(unl_left)
        rows.append({"symbol": symbol, "asof": asof.date(), "price": price, "gold": len(gold),
                     "detected": len(det), "tp": tp, "fn": fn, "fp_rejected": fp, "unlabeled": unl,
                     "recall": tp / len(gold) if len(gold) else np.nan,
                     "precision_reviewed": tp / (tp + fp) if (tp + fp) else np.nan,
                     "precision_all": tp / len(det) if len(det) else np.nan})
        misses += [(symbol, asof.date(), gold[i]) for i in gold_left]
        false_pos += [(symbol, asof.date(), det_left_levels[j]) for _, j in fp_pairs]
        unlabeled += [(symbol, asof.date(), det_left_levels[j]) for j in unl_left]

    per = pd.DataFrame(rows)
    out = {"per_asof": per, "missed": misses, "false_positives": false_pos,
           "unlabeled": unlabeled, "skipped_groups": skipped}
    if per.empty:
        return out
    tp, fn, fp, unl, det_n = (int(per[c].sum()) for c in ("tp", "fn", "fp_rejected", "unlabeled", "detected"))
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    prec_rev = tp / (tp + fp) if (tp + fp) else np.nan
    prec_all = tp / det_n if det_n else np.nan
    f1 = lambda p_, r_: 2 * p_ * r_ / (p_ + r_) if p_ and r_ and not np.isnan(p_ + r_) else np.nan
    out["overall"] = {"asofs_scored": len(per), "tp": tp, "fn": fn, "fp_rejected": fp, "unlabeled": unl,
                      "recall": recall, "precision_reviewed": prec_rev, "precision_all": prec_all,
                      "f1_reviewed": f1(prec_rev, recall), "f1_all": f1(prec_all, recall)}
    return out


def format_score(result: dict) -> str:
    if result["per_asof"].empty:
        return ("Nothing to score. Edit status in the label file to keep/reject/add for at least one "
                "(symbol, asof) group.")
    pd.set_option("display.width", 200)
    parts = [result["per_asof"].to_string(index=False, float_format=lambda x: f"{x:.2f}")]
    o = result["overall"]
    parts.append(
        f"\nOVERALL over {o['asofs_scored']} charts: TP={o['tp']} missed={o['fn']} "
        f"rejected-but-detected={o['fp_rejected']} unlabeled={o['unlabeled']}\n"
        f"  recall={o['recall']:.1%}  precision(reviewed)={o['precision_reviewed']:.1%}  "
        f"precision(all detected)={o['precision_all']:.1%}\n"
        f"  F1(reviewed)={o['f1_reviewed']:.3f}  F1(all)={o['f1_all']:.3f}")
    for title, key in (("MISSED (gold levels not detected)", "missed"),
                       ("FALSE POSITIVES (rejected levels still detected)", "false_positives"),
                       ("UNLABELED (detected, never reviewed; run `labels export` to add them)", "unlabeled")):
        if result[key]:
            parts.append(f"\n{title}:")
            parts += [f"  {s} {d}  {lvl:.2f}" for s, d, lvl in result[key]]
    if result["skipped_groups"]:
        parts.append(f"\n{result['skipped_groups']} (symbol, asof) groups skipped (no reviewed rows or no data).")
    return "\n".join(parts)
