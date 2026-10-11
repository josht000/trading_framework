import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metrics(tp, fp, fn):
    return (
        tp / (tp + fp + fn) if tp + fp + fn else None,
        tp / (tp + fp) if tp + fp else None,
        tp / (tp + fn) if tp + fn else None,
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )


def match(pred, labels, large):
    pairs = sorted((abs(p - l), i, j) for i, p in enumerate(pred) for j, l in enumerate(labels))
    used_p, used_l, hits = set(), set(), []
    for distance, i, j in pairs:
        tolerance = labels[j] * 0.01 if large else 0.10
        if distance <= tolerance + 1e-9 and i not in used_p and j not in used_l:
            used_p.add(i)
            used_l.add(j)
            hits.append((pred[i], labels[j]))
    return hits, [p for i, p in enumerate(pred) if i not in used_p], [l for j, l in enumerate(labels) if j not in used_l]


def fmt(values):
    return " | ".join("N/A" if value is None else f"{value:.4f}" for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2] / "resistance_dataset" / args.split
    charts = {}
    for path in sorted((base / "price_data").glob("*.csv")):
        number, symbol, asof = path.stem.split("_")
        rows = load(path)
        assert rows[-1]["date"] == asof, path
        assert all(row["date"] <= asof for row in rows), path
        assert (base / "charts" / (path.stem + ".png")).is_file(), path
        charts[(symbol, asof)] = (number, float(rows[-1]["close"]), rows)
    preds = defaultdict(list)
    for row in load(args.predictions):
        key = row["symbol"], row["asof"]
        assert key in charts, ("Unknown prediction chart", key)
        level = float(row["level"])
        assert math.isfinite(level) and level > charts[key][1], (key, level)
        assert level not in preds[key], ("Duplicate prediction", key, level)
        fields = ("high", "low") if "support" in row["note"].lower() else ("high",)
        assert min(abs(float(r[field]) - level) for r in charts[key][2] for field in fields) <= 0.0051, ("No price anchor", key, level)
        preds[key].append(level)
    print(f"Prediction SHA256: {hashlib.sha256(args.predictions.read_bytes()).hexdigest()}")
    labels = defaultdict(list)
    with (base / "labels.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0] == "symbol":
                continue
            key = row[0], row[1]
            assert key in charts, ("Label without input chart", key)
            level = float(row[2])
            assert math.isfinite(level), row
            labels[key].append(level)
    print(f"Coverage: {len(charts)} charts; {sum(map(len, preds.values()))} predictions; {sum(map(len, labels.values()))} labels; {len(charts.keys() - labels.keys())} zero-label charts.")
    for key, levels in labels.items():
        if len(levels) != len(set(levels)):
            print(f"WARNING: duplicate labels retained for {key}")
        invalid = [level for level in levels if level <= charts[key][1]]
        if invalid:
            print(f"WARNING: labels at/below last close retained for {key}: {invalid}; close={charts[key][1]}")
    for policy in ["flat_0.10", "large_cap_1pct"]:
        total = [0, 0, 0]
        details = []
        print(f"\n### {args.split}: {policy}\n")
        print("| Chart | TP | FP | FN | Set accuracy | Precision | Recall | F1 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key, (number, close, _) in charts.items():
            hits, extra, missed = match(sorted(preds[key]), sorted(labels[key]), policy == "large_cap_1pct" and close >= 50)
            counts = len(hits), len(extra), len(missed)
            total = [a + b for a, b in zip(total, counts)]
            print(f"| {number} {key[0]} | {counts[0]} | {counts[1]} | {counts[2]} | {fmt(metrics(*counts))} |")
            details.append(f"{number} {key[0]}: hits={hits}; FP={extra}; FN={missed}")
        print(f"| TOTAL | {total[0]} | {total[1]} | {total[2]} | {fmt(metrics(*total))} |")
        print("\n" + "\n".join(details))


if __name__ == "__main__":
    main()
