"""
Few-shot support: load train/val/test chart splits straight from data/resistance_dataset and turn the
labelled splits into worked examples (user turn: chart + OHLCV, assistant turn: analyst explanation + gold JSON).

Layout per split: <split>/charts/<id>.png, <split>/price_data/<id>.csv, <split>/labels.csv and, for the
labelled training splits only, <split>/explainations/<id>.txt.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.screening.resistance import _wilder_atr
from src.vlm.packets import DEFAULT_EXPLANATIONS_ROOT, load_explanation
from src.vlm.prompt import packet_text

SPLITS = ("train", "val", "test")


def _read_labels(split_dir: str) -> Dict[tuple, List[float]]:
    labels = defaultdict(list)
    with open(os.path.join(split_dir, "labels.csv"), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[(row["symbol"], row["asof"])].append(float(row["level"]))
    return labels


def load_split_items(split: str, dataset_root: str = DEFAULT_EXPLANATIONS_ROOT) -> List[Dict]:
    """Manifest-style items for one split, built from the OHLCV csv + labels (no market-data download needed)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    split_dir = os.path.join(dataset_root, split)
    labels = _read_labels(split_dir)
    items = []
    price_dir = os.path.join(split_dir, "price_data")
    for name in sorted(os.listdir(price_dir)):
        if not name.endswith(".csv"):
            continue
        stem = name[:-4]
        number, symbol, asof = stem.split("_", 2)
        frame = pd.read_csv(os.path.join(price_dir, name))
        atr = _wilder_atr(frame["high"].to_numpy(float), frame["low"].to_numpy(float),
                          frame["close"].to_numpy(float), 14)
        items.append({
            "id": stem, "number": int(number), "symbol": symbol, "asof": asof, "split": split,
            "price": float(frame["close"].iloc[-1]), "atr14": float(atr[-1]) if np.isfinite(atr[-1]) else float("nan"),
            "n_bars": int(len(frame)), "first_date": str(frame["date"].iloc[0]), "last_date": str(frame["date"].iloc[-1]),
            "chart_path": os.path.join(split_dir, "charts", stem + ".png"),
            "ohlcv_path": os.path.join(price_dir, name),
            "gold_levels": sorted(labels.get((symbol, asof), [])),
        })
    return sorted(items, key=lambda it: it["number"])


def shot_answer(item: Dict, explanation: str) -> str:
    """Assistant turn for a worked example: the analyst's reasoning, then the gold levels above the last close."""
    levels = [{"price": round(g, 4)} for g in item["gold_levels"] if g > item["price"]]
    answer = json.dumps({"resistance_levels": levels})
    return (explanation.strip() + "\n\n" if explanation.strip() else "") + answer


def load_shots(splits: Sequence[str], dataset_root: str = DEFAULT_EXPLANATIONS_ROOT) -> List[Dict]:
    """Worked examples in a fixed order (split order, then chart number) so the prompt prefix is byte-stable."""
    shots = []
    for split in splits:
        if split == "test":
            raise ValueError("the test split must never be used as few-shot examples")
        for item in load_split_items(split, dataset_root):
            with open(item["ohlcv_path"], encoding="utf8") as f:
                ohlcv = f.read()
            with open(item["chart_path"], "rb") as f:
                png = f.read()
            explanation = load_explanation(item, dataset_root)
            shots.append({"item": item, "png": png, "text": packet_text(item, ohlcv),
                          "answer": shot_answer(item, explanation)})
    return shots


def shots_fingerprint(shots: Sequence[Dict]) -> str:
    import hashlib
    h = hashlib.sha1()
    for s in shots:
        h.update(s["item"]["id"].encode())
        h.update(hashlib.sha1(s["png"]).digest())
        h.update(s["text"].encode())
        h.update(s["answer"].encode())
    return h.hexdigest()
