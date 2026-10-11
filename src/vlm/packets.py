"""
Build the benchmark data set from the hand-labeled charts.

Layout (default root data/vlm):
  charts_without_labels/NN_SYM_date.png   what the VLM sees (blind: no levels, no detector)
  charts_with_labels/NN_SYM_date.png      same frame with the human gold levels drawn (for review)
  ohlcv/NN_SYM_date.csv                   daily OHLCV for exactly the bars on the chart
  manifest.json                           per-chart metadata and gold levels (never sent to a VLM)

Only charts with at least one labeled level are included.
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from src.research.chart_png import level, render_chart_png
from src.research.config_io import make_detector
from src.research.data import load_daily
from src.research.labels import labels_for, load_labels

DEFAULT_ROOT = os.path.join("data", "vlm")
DEFAULT_EXPLANATIONS_ROOT = os.path.join("data", "resistance_dataset")
DIRS = {"blind": "charts_without_labels", "labeled": "charts_with_labels", "ohlcv": "ohlcv"}


def load_explanation(item: Dict, root: str = DEFAULT_EXPLANATIONS_ROOT) -> str:
    name = item["id"] + ".txt"
    for split in ("train", "val"):
        path = os.path.join(root, split, "explainations", name)
        if os.path.isfile(path):
            with open(path, encoding="utf8") as f:
                return f.read().strip()
    return ""


def build_packets(chart_set_csv: str, labels_csv: str, root: str = DEFAULT_ROOT, bars: int = 160,
                  size=(16, 10), dpi: int = 100) -> List[Dict]:
    chart_set = pd.read_csv(chart_set_csv)
    labels = load_labels(labels_csv)
    for d in DIRS.values():
        os.makedirs(os.path.join(root, d), exist_ok=True)

    det = make_detector("atr_pivot", None)
    items = []
    for i, row in enumerate(chart_set.itertuples(index=False), 1):
        gold = labels_for(labels, row.symbol, row.asof)
        gold = gold[gold["status"].isin(["add", "keep"])]
        if gold.empty:
            continue
        daily = load_daily([row.symbol], row.asof, row.asof)[row.symbol]
        p = det.prepare(daily)
        k = det.known_through_on(p, row.asof)
        if k < 0:
            continue

        stem = f"{i:02d}_{row.symbol}_{row.asof}"
        blind_path = os.path.join(root, DIRS["blind"], stem + ".png")
        labeled_path = os.path.join(root, DIRS["labeled"], stem + ".png")
        csv_path = os.path.join(root, DIRS["ohlcv"], stem + ".csv")

        blind_title = f"Daily chart, last bar {pd.Timestamp(p.dates[k]).date()}  last close={p.close[k]:.2f}"
        render_chart_png(p, k, blind_path, row.symbol, lookback=bars, size=size, dpi=dpi, title=blind_title)
        gold_levels = sorted(float(x) for x in gold["level"])
        render_chart_png(p, k, labeled_path, row.symbol, lookback=bars, size=size, dpi=dpi,
                         levels=[level(x) for x in gold_levels])

        lo = max(0, k - bars)
        frame = pd.DataFrame({
            "date": pd.to_datetime(p.dates[lo:k + 1]).strftime("%Y-%m-%d"),
            "open": p.open[lo:k + 1], "high": p.high[lo:k + 1], "low": p.low[lo:k + 1],
            "close": p.close[lo:k + 1], "volume": p.volume[lo:k + 1].astype("int64"),
        })
        frame.to_csv(csv_path, index=False, float_format="%.4f")

        items.append({
            "id": stem, "number": i, "symbol": row.symbol, "asof": str(row.asof),
            "price": float(p.close[k]), "atr14": float(p.atr[k]), "n_bars": int(len(frame)),
            "first_date": frame["date"].iloc[0], "last_date": frame["date"].iloc[-1],
            "chart_blind": os.path.relpath(blind_path, root).replace("\\", "/"),
            "chart_labeled": os.path.relpath(labeled_path, root).replace("\\", "/"),
            "ohlcv": os.path.relpath(csv_path, root).replace("\\", "/"),
            "gold_levels": gold_levels,
        })

    manifest = {"bars": bars, "image_px": [int(size[0] * dpi), int(size[1] * dpi)], "items": items}
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2)
    return items


def load_manifest(root: str = DEFAULT_ROOT) -> Dict:
    with open(os.path.join(root, "manifest.json"), encoding="utf8") as f:
        return json.load(f)
