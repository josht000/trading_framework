"""
Run a resistance detector on a resistance_dataset split and score it against labels.

  python eval_pivot_resistance.py data/resistance_dataset/test
  python eval_pivot_resistance.py data/resistance_dataset/test --detector atr_pivot
  python eval_pivot_resistance.py data/resistance_dataset/train --pivot-bars 4 --tol 0.10
  python eval_pivot_resistance.py data/resistance_dataset/val --detector zigzag --set peak_percent_delta=2
  python eval_pivot_resistance.py data/resistance_dataset/val --detector atr_pivot --set include_ranges=false

Split layout: <split>/price_data/NN_SYMBOL_ASOF.csv (date,open,high,low,close,volume) and
<split>/labels.csv (symbol,asof,level,note; header optional). Writes <split>/predictions_<detector>.csv
before it reads the labels, then prints per-chart and total TP/FP/FN, precision, recall, F1, accuracy.

Detectors (config dataclass keyed by --set):
  pivot          simple swing-high ladder (src/screening/pivot_resistance.py)
  atr_pivot      ATR-scaled pivot zone detector (src/screening/resistance.py); each resistance
                 zone is emitted at its collapsed level (highest member pivot high)
  zigzag         ZigZag pivots + agglomerative clustering (src/screening/cluster_levels.py)
  raw_price      rolling-extreme bars + agglomerative clustering (same file)
  volume_profile high-volume-node detector (src/screening/volume_profile.py)
  deriv_extrema  smoothed-derivative local maxima (src/screening/sr_finder.py)
  volume_levels  low-slope + high-volume prices (same file)
  meanshift      MeanShift clusters over the price series (same file)
  floor_pivots   classic/camarilla/fibonacci pivot-point formulas (same file)

`series` is an extra --set key for zigzag/raw_price picking the input column (high|close).
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from dataclasses import fields

from src.screening.pivot_resistance import PivotResistanceConfig, PivotResistanceDetector

SR_FINDERS = ("deriv_extrema", "volume_levels", "meanshift", "floor_pivots")
DETECTORS = ("pivot", "atr_pivot", "zigzag", "raw_price", "volume_profile") + SR_FINDERS


def load_bars(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [float(r["high"]) for r in rows], [float(r["close"]) for r in rows]


def load_frame(path):
    import pandas as pd
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def config_cls(name):
    """The dataclass holding a detector's tunable parameters."""
    if name == "pivot":
        return PivotResistanceConfig
    if name == "atr_pivot":
        from src.screening.resistance import ResistanceConfig
        return ResistanceConfig
    if name == "volume_profile":
        from src.screening.volume_profile import VolumeProfileConfig
        return VolumeProfileConfig
    if name == "zigzag":
        from src.screening.cluster_levels import ZigZagLevelsConfig
        return ZigZagLevelsConfig
    if name == "raw_price":
        from src.screening.cluster_levels import RawPriceLevelsConfig
        return RawPriceLevelsConfig
    from src.screening.sr_finder import CONFIGS
    return CONFIGS[name]


def make_detector(name, params):
    """detect(src) -> sorted resistance levels above the last close. `src` is a
    price_data CSV path or a loaded DataFrame (date index, ohlcv columns)."""
    params = dict(params)
    if name == "atr_pivot":
        from src.screening.resistance import ResistanceDetector
        det = ResistanceDetector(config_cls(name)(**params))
    elif name == "volume_profile":
        from src.screening.volume_profile import VolumeProfileDetector
        det = VolumeProfileDetector(config_cls(name)(**params))
    elif name in ("zigzag", "raw_price"):
        from src.screening.cluster_levels import ClusterLevelsDetector
        series = params.pop("series", "high")
        det = ClusterLevelsDetector(kind=name, series=series, config=config_cls(name)(**params))
    elif name in SR_FINDERS:
        from src.screening.sr_finder import DETECTORS as FINDER_CLS
        det = FINDER_CLS[name](config_cls(name)(**params))
    else:
        det = PivotResistanceDetector(config_cls(name)(**params))

    def detect(src):
        df = src if hasattr(src, "columns") else load_frame(src)
        if name in ("atr_pivot", "volume_profile"):
            p = det.prepare(df)
            lad = det.ladder(p, len(p.close) - 1)
            return sorted(lad["upper"].tolist())
        if name in ("zigzag", "raw_price") + SR_FINDERS:
            return det.detect_bars(df)
        return det.detect(df["high"].tolist(), float(df["close"].iloc[-1]))

    return detect


def _coerce(raw, default):
    """Coerce a --set KEY=VALUE string to the config field's type."""
    s = raw.strip()
    if s.lower() in ("none", "null"):
        return None
    if isinstance(default, bool):
        return s.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(s)
    if isinstance(default, float):
        return float(s)
    if isinstance(default, tuple):
        return tuple(float(x) for x in s.split(","))
    if default is None:
        for typ in (int, float):
            try:
                return typ(s)
            except ValueError:
                pass
    return s


def parse_overrides(name, pairs):
    """--set KEY=VALUE list -> params dict, checked and coerced against the config fields."""
    cls = config_cls(name)
    defaults = {f.name: f.default for f in fields(cls)}
    if name in ("zigzag", "raw_price"):
        defaults["series"] = "high"
    params = {}
    for kv in pairs:
        key, sep, raw = kv.partition("=")
        if not sep or key not in defaults:
            raise ValueError(f"bad --set {kv!r}; valid keys for {name}: {sorted(defaults)}")
        params[key] = _coerce(raw, defaults[key])
    return params


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
    ap.add_argument("--detector", choices=DETECTORS, default="pivot")
    ap.add_argument("--pivot-bars", type=int, default=None)
    ap.add_argument("--merge-tol", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.10, help="hit tolerance in dollars")
    ap.add_argument("--no-edge-pivots", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config field override; repeatable")
    args = ap.parse_args()

    params = parse_overrides(args.detector, args.overrides)
    legacy = {"pivot_bars": args.pivot_bars, "merge_tol": args.merge_tol}
    if args.no_edge_pivots:
        legacy["edge_pivots"] = False
    defaults = {f.name for f in fields(config_cls(args.detector))}
    params.update({k: v for k, v in legacy.items() if v is not None and k in defaults})
    detect = make_detector(args.detector, params)

    preds = {}
    out_path = os.path.join(args.split_dir, f"predictions_{args.detector}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "asof", "level", "note"])
        for path in sorted(glob.glob(os.path.join(args.split_dir, "price_data", "*.csv"))):
            num, sym, asof = os.path.basename(path)[:-4].split("_")
            preds[(sym, asof)] = detect(path)
            for lv in preds[(sym, asof)]:
                note = f"chart {num} pivot_bars={params.get('pivot_bars', PivotResistanceConfig.pivot_bars)}" \
                    if args.detector == "pivot" else f"chart {num} {args.detector}"
                w.writerow([sym, asof, f"{lv:.2f}", note])

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
