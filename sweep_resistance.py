"""Grid-search the dataset resistance detectors on train+val and save the winners' scores.

Every parameter combination is run over the tune splits (default train+val) of
data/resistance_dataset and scored with score.py's tolerance rule (+-$0.10 of a label,
or +-1% of the label when the chart's last close is >= $50). Configs are ranked by
micro-F1 summed over the tune splits; the winner per detector family is then run on
every split and saved the usual way: <split>/predictions_<detector>.csv plus
<split>/eval_results_<detector>.txt (score.py output). The full sweep table goes to
data/resistance_dataset/sweep_results.csv.

  python sweep_resistance.py                     # sweep all detectors, save best scores
  python sweep_resistance.py --detector pivot    # one family only
  python sweep_resistance.py --no-save           # leaderboard only, write nothing
"""

import argparse
import csv
import glob
import itertools
import json
import os
import subprocess
import sys
from datetime import date

import pandas as pd

from eval_pivot_resistance import load_levels, make_detector

DATASET = os.path.join("data", "resistance_dataset")
SPLITS = ("train", "val", "test")
SWEEP_CSV = os.path.join(DATASET, "sweep_results.csv")

SRC_FILES = {
    "pivot": "src/screening/pivot_resistance.py",
    "atr_pivot": "src/screening/resistance.py",
    "zigzag": "src/screening/cluster_levels.py",
    "raw_price": "src/screening/cluster_levels.py",
    "volume_profile": "src/screening/volume_profile.py",
    "deriv_extrema": "src/screening/sr_finder.py",
    "volume_levels": "src/screening/sr_finder.py",
    "meanshift": "src/screening/sr_finder.py",
    "floor_pivots": "src/screening/sr_finder.py",
}

GRIDS = {
    "pivot": {
        "pivot_bars": [2, 3, 4, 5, 6, 8],
        "merge_tol": [0.05, 0.10, 0.15, 0.20, 0.30],
        "edge_pivots": [True, False],
        "max_levels": [None, 5, 8],
    },
    "atr_pivot": {
        "pivot_left": [3, 5, 7],
        "pivot_right": [2, 3],
        "min_prominence_atr": [0.5, 1.0, 1.5],
        "merge_atr": [0.3, 0.5, 0.8],
        "max_zone_atr": [1.5, 2.5],
        "lookback_bars": [120, 252],
        "include_ranges": [False, True],
        "include_swing_lows": [False, True],
    },
    "zigzag": [
        {"series": series, "peak_percent_delta": delta, "peaks": peaks,
         "min_bars_between_peaks": mb, "level_selector": sel, **merge}
        for series, delta, peaks, mb, sel, merge in itertools.product(
            ("high", "close"), (1, 2, 3, 4, 5, 7.5, 10), ("High", "All"),
            (0, 5, 10), ("mean", "median"),
            [{"merge_percent": v} for v in (0.5, 1.0, 2.0, 3.0, 5.0)]
            + [{"merge_distance": v} for v in (0.10, 0.25, 0.50)])
    ],
    "raw_price": [
        {"series": series, "bars_for_peak": bars, "level_selector": sel, **merge}
        for series, bars, sel, merge in itertools.product(
            ("high", "close"), (5, 11, 21, 31, 41, 51), ("mean", "median"),
            [{"merge_percent": v} for v in (0.5, 1.0, 2.0, 3.0, 5.0)]
            + [{"merge_distance": v} for v in (0.10, 0.25, 0.50)])
    ],
    "volume_profile": {
        "max_above": [3, 5, 8],
        "min_gap_atr": [0.5, 0.75, 1.0],
        "bin_atr": [0.10, 0.20],
        "min_prominence_frac": [0.03, 0.05, 0.10],
        "recency_halflife_bars": [63, 126, 252],
    },
    "deriv_extrema": (
        [{"series": s, "smooth": sm, "mode": "cross", "emit": e, "merge_tol": m}
         for s, sm, e, m in itertools.product(
             ("high", "close"), (True, False), ("raw", "smooth"), (0.10, 0.20))]
        + [{"series": s, "smooth": sm, "mode": "flat", "threshold": t, "relative": rel,
            "emit": e, "merge_tol": m}
           for s, sm, t, rel, e, m in itertools.product(
               ("high", "close"), (True, False), (0.002, 0.005, 0.02), (False, True),
               ("raw", "smooth"), (0.10, 0.20))]
    ),
    "volume_levels": {
        "series": ["high", "close"],
        "smooth": [True, False],
        "threshold": [0.005, 0.02, 0.08],
        "relative": [False, True],
        "percentile": [50, 75],
        "emit": ["raw", "smooth"],
        "merge_tol": [0.05, 0.10, 0.20],
    },
    "meanshift": {
        "series": ["high", "close"],
        "smooth": [False, True],
        "quantile": [0.05, 0.1, 0.2, 0.3],
        "emit": ["edges", "max", "center", "mean"],
        "min_cluster_size": [1, 3],
    },
    "floor_pivots": {
        "lookback": [1, 5, 20, 60],
        "variant": ["classic", "camarilla", "fibonacci"],
        "which": ["r12", "r123", "pp_r123", "all"],
    },
}


def load_split(split):
    """-> (frames, labels, closes): frames maps (symbol, asof) -> (chart_num, DataFrame)."""
    split_dir = os.path.join(DATASET, split)
    frames = {}
    for path in sorted(glob.glob(os.path.join(split_dir, "price_data", "*.csv"))):
        num, sym, asof = os.path.basename(path)[:-4].split("_")
        frames[(sym, asof)] = (num, pd.read_csv(path, parse_dates=["date"], index_col="date"))
    labels = load_levels(os.path.join(split_dir, "labels.csv"))
    closes = {k: float(df["close"].iloc[-1]) for k, (_, df) in frames.items()}
    return frames, labels, closes


def score_counts(labels, preds, closes):
    """score.py's matching: greedy nearest-pair assignment, tol $0.10 or 1% when close >= $50.
    -> (TP, FP, FN)."""
    tp = fp = fn = 0
    for key in labels:
        lab, prd = sorted(labels[key]), sorted(preds.get(key, []))
        tol = [0.01 * l if closes[key] >= 50 else 0.10 for l in lab]
        pairs = sorted((abs(p - l), i, j) for i, p in enumerate(prd) for j, l in enumerate(lab))
        up, ul, hits = set(), set(), 0
        for d, i, j in pairs:
            if d <= tol[j] + 1e-9 and i not in up and j not in ul:
                up.add(i)
                ul.add(j)
                hits += 1
        tp += hits
        fp += len(prd) - hits
        fn += len(lab) - hits
    return tp, fp, fn


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _fmt_val(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _fmt_params(params):
    return " ".join(f"{k}={_fmt_val(v)}" for k, v in params.items())


def _print_table(headers, table, indent="   "):
    """Right-aligned columns, two-space gutters."""
    widths = [max([len(h)] + [len(t[i]) for t in table]) for i, h in enumerate(headers)]
    print(indent + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for t in table:
        print(indent + "  ".join(c.rjust(w) for c, w in zip(t, widths)))


def run_grid(name, tune_data):
    """Score every grid combo on each tune split -> [{params, per_split}]."""
    grid = GRIDS[name]
    if isinstance(grid, dict):
        keys = list(grid)
        combos = [dict(zip(keys, c)) for c in itertools.product(*(grid[k] for k in keys))]
    else:
        combos = grid
    rows = []
    for params in combos:
        detect = make_detector(name, params)
        per_split = {}
        for split, (frames, labels, closes) in tune_data.items():
            preds = {}
            for key, (_, df) in frames.items():
                try:
                    preds[key] = detect(df)
                except Exception as e:
                    print(f"  warn: {name} {params} failed on {key}: {e}")
                    preds[key] = []
            per_split[split] = score_counts(labels, preds, closes)
        rows.append({"params": params, "per_split": per_split})
    return rows, len(combos)


def combined(row):
    tp = fp = fn = 0
    for tp_s, fp_s, fn_s in row["per_split"].values():
        tp, fp, fn = tp + tp_s, fp + fp_s, fn + fn_s
    return tp, fp, fn


def print_leaderboard(name, rows, top):
    """Aligned table: one column per param that varies inside the top-N shown;
    params constant across the top-N are hoisted to a `constant:` line."""
    ranked = sorted(rows, key=lambda r: metrics(*combined(r))[2], reverse=True)
    splits = list(rows[0]["per_split"])
    show = ranked[:top]
    keys = sorted({k for r in show for k in r["params"]})
    varying = [k for k in keys if len({json.dumps(r["params"].get(k)) for r in show}) > 1]
    const = _fmt_params({k: show[0]["params"][k] for k in keys if k not in varying})
    tune = "+".join(splits)

    print(f"\n== {name}: top {len(show)} of {len(ranked)} configs, ranked by {tune} F1 ==")
    if const:
        print(f"   constant: {const}")
    table = []
    for i, r in enumerate(show, 1):
        row = [str(i), f"{metrics(*combined(r))[2]:.3f}"]
        row += [f"{metrics(*r['per_split'][s])[2]:.3f}" for s in splits]
        row += [_fmt_val(r["params"].get(k)) for k in varying]
        table.append(row)
    _print_table(["#", "combF1"] + [f"{s}F1" for s in splits] + varying, table)
    return ranked


def save_csv(all_rows, tune_splits, path):
    cols = ["detector", "params", "combined_tp", "combined_fp", "combined_fn",
            "combined_precision", "combined_recall", "combined_f1"] + \
           [f"{s}_{m}" for s in tune_splits for m in ("tp", "fp", "fn", "precision", "recall", "f1")]
    carried = []
    if os.path.exists(path):
        with open(path) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames == cols:
                carried = [r for r in rd if r["detector"] not in all_rows]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, rows in all_rows.items():
            for r in rows:
                tp, fp, fn = combined(r)
                p, rec, f1 = metrics(tp, fp, fn)
                line = [name, json.dumps(r["params"]), tp, fp, fn, f"{p:.4f}", f"{rec:.4f}", f"{f1:.4f}"]
                for s in tune_splits:
                    ts, fs, ns = r["per_split"][s]
                    ps, rs, f1s = metrics(ts, fs, ns)
                    line += [ts, fs, ns, f"{ps:.4f}", f"{rs:.4f}", f"{f1s:.4f}"]
                w.writerow(line)
        for r in carried:
            w.writerow([r[c] for c in cols])
    print(f"\nsweep table -> {path}")


def save_best(name, params, n_configs, splits_data):
    """Write predictions_<name>.csv + eval_results_<name>.txt per split.
    Returns {split: (tp, fp, fn)} scored with the score.py rule."""
    results = {}
    for split, (frames, labels, closes) in splits_data.items():
        detect = make_detector(name, dict(params))
        split_dir = os.path.join(DATASET, split)
        pred_path = os.path.join(split_dir, f"predictions_{name}.csv")
        tag = f"pivot_bars={params.get('pivot_bars')}" if name == "pivot" else name
        preds, n_levels = {}, 0
        with open(pred_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "asof", "level", "note"])
            for (sym, asof), (num, df) in sorted(frames.items(), key=lambda kv: int(kv[1][0])):
                preds[(sym, asof)] = [round(lv, 2) for lv in detect(df)]
                for lv in preds[(sym, asof)]:
                    w.writerow([sym, asof, f"{lv:.2f}", f"chart {num} {tag}"])
                    n_levels += 1
        out = subprocess.run(
            [sys.executable, os.path.join(DATASET, "score.py"), split, os.path.abspath(pred_path)],
            capture_output=True, text=True)
        header = (f"{name} detector ({SRC_FILES[name]}, best of {n_configs} configs swept on "
                  f"train+val {date.today().isoformat()})\nparams: {json.dumps(params)}\n"
                  f"preds: {split}/predictions_{name}.csv ({n_levels} levels). Scored by score.py.\n\n")
        eval_path = os.path.join(split_dir, f"eval_results_{name}.txt")
        with open(eval_path, "w") as f:
            f.write(header + out.stdout)
        results[split] = score_counts(labels, preds, closes)
        print(f"  {split}: {pred_path} + {eval_path}")
    return results


def print_summary(summary, splits):
    """Final table: best config per detector with per-split F1s."""
    print("\n== best configs ==")
    table = []
    for name, row, results in summary:
        line = [name, f"{metrics(*combined(row))[2]:.3f}"]
        line += [f"{metrics(*results[s])[2]:.3f}" for s in splits]
        line.append(_fmt_params(row["params"]))
        table.append(line)
    _print_table(["detector", "combF1"] + [f"{s}F1" for s in splits] + ["params"], table)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", action="append", choices=list(GRIDS),
                    help="family to sweep; repeatable, default all")
    ap.add_argument("--tune", default="train,val", help="comma splits used to pick configs")
    ap.add_argument("--top", type=int, default=10, help="leaderboard rows per family")
    ap.add_argument("--no-save", action="store_true", help="skip writing predictions/eval files")
    args = ap.parse_args()

    tune_splits = args.tune.split(",")
    tune_data = {s: load_split(s) for s in tune_splits}
    names = args.detector or list(GRIDS)

    all_rows, best = {}, {}
    for name in names:
        rows, n = run_grid(name, tune_data)
        all_rows[name] = rows
        ranked = print_leaderboard(name, rows, args.top)
        best[name] = (ranked[0], n)

    save_csv(all_rows, tune_splits, SWEEP_CSV)

    if args.no_save:
        return
    splits_data = {s: tune_data.get(s) or load_split(s) for s in SPLITS}
    print("\n== saving best configs ==")
    summary = []
    for name, (row, n) in best.items():
        print(f"{name}:")
        results = save_best(name, row["params"], n, splits_data)
        summary.append((name, row, results))
    print_summary(summary, SPLITS)


if __name__ == "__main__":
    main()
