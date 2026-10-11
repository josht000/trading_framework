"""
Run the pivot-high resistance detector on a resistance_dataset split and score it against labels.

  python eval_pivot_resistance.py data/resistance_dataset/test
  python eval_pivot_resistance.py data/resistance_dataset/train --pivot-bars 4 --tol 0.10

Split layout: <split>/price_data/NN_SYMBOL_ASOF.csv (date,open,high,low,close,volume) and
<split>/labels.csv (symbol,asof,level,note; header optional). Writes <split>/predictions_pivot.csv
before it reads the labels, then prints per-chart and total TP/FP/FN, precision, recall, F1, accuracy.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

from src.screening.pivot_resistance import PivotResistanceConfig, PivotResistanceDetector


def load_bars(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [float(r["high"]) for r in rows], [float(r["close"]) for r in rows]


def load_levels(path):
    out = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if r and r[0] != "symbol":
                out[(r[0], r[1])].append(float(r[2]))
    return out


def match(preds, labels, tol):
    """Greedy one-to-one matching of each prediction to its nearest unused label within tol."""
    used, tp = set(), 0
    for p in sorted(preds):
        cand = [(abs(p - l), i) for i, l in enumerate(labels) if i not in used and abs(p - l) <= tol]
        if cand:
            used.add(min(cand)[1])
            tp += 1
    return tp, len(preds) - tp, len(labels) - tp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split_dir")
    ap.add_argument("--pivot-bars", type=int, default=PivotResistanceConfig.pivot_bars)
    ap.add_argument("--merge-tol", type=float, default=PivotResistanceConfig.merge_tol)
    ap.add_argument("--tol", type=float, default=0.10, help="hit tolerance in dollars")
    ap.add_argument("--no-edge-pivots", action="store_true")
    args = ap.parse_args()

    det = PivotResistanceDetector(PivotResistanceConfig(
        pivot_bars=args.pivot_bars, merge_tol=args.merge_tol, edge_pivots=not args.no_edge_pivots))

    preds = {}
    out_path = os.path.join(args.split_dir, "predictions_pivot.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "asof", "level", "note"])
        for path in sorted(glob.glob(os.path.join(args.split_dir, "price_data", "*.csv"))):
            num, sym, asof = os.path.basename(path)[:-4].split("_")
            highs, closes = load_bars(path)
            preds[(sym, asof)] = det.detect(highs, closes[-1])
            for lv in preds[(sym, asof)]:
                w.writerow([sym, asof, f"{lv:.2f}", f"chart {num} pivot_bars={args.pivot_bars}"])

    labels = load_levels(os.path.join(args.split_dir, "labels.csv"))
    TP = FP = FN = 0
    print(f"predictions -> {out_path}   hit tolerance +-${args.tol:.2f}\n")
    print(f"{'chart':<18}{'TP':>4}{'FP':>4}{'FN':>4}")
    for key in sorted(labels):
        tp, fp, fn = match(preds.get(key, []), sorted(labels[key]), args.tol)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        print(f"{key[0] + ' ' + key[1]:<18}{tp:>4}{fp:>4}{fn:>4}")
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = TP / (TP + FP + FN) if TP + FP + FN else 0.0
    print(f"\nTOTAL TP={TP} FP={FP} FN={FN}")
    print(f"precision={prec:.3f} recall={rec:.3f} F1={f1:.3f} accuracy(TP/(TP+FP+FN))={acc:.3f}")


if __name__ == "__main__":
    main()
