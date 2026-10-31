"""Consolidate every study's partition metrics into one review CSV.

Scans results/<study>/(<variant>__)partition_{dev,test}.csv (baseline files have no
variant prefix) and concatenates them with study/variant columns.

  python tools/collect_results.py                      # all studies -> results/all_test_metrics.csv
  python tools/collect_results.py --studies a,b --out x.csv
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", default=None, help="comma-separated; default = every results/<study>/ dir")
    parser.add_argument("--out", default=str(ROOT / "results" / "all_test_metrics.csv"))
    args = parser.parse_args()
    studies = args.studies.split(",") if args.studies else sorted(
        p.name for p in (ROOT / "results").iterdir() if p.is_dir() and p.name != "timing")
    frames = []
    for s in studies:
        d = ROOT / "results" / s
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*partition_*.csv")):
            variant = f.name.split("__")[0] if "__" in f.name else "baseline"
            t = pd.read_csv(f)
            t.insert(0, "variant", variant)
            t.insert(0, "study", s)
            frames.append(t)
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["study", "variant", "fill", "period"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} rows ({out.study.nunique()} studies, {out.variant.nunique()} variants) -> {args.out}")


if __name__ == "__main__":
    main()
