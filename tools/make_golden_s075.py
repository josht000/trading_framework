"""Copy S075 study outputs into results/s075_residual_reversal/golden and update MANIFEST.json.

  python tools/make_golden_s075.py                 # baseline files
  python tools/make_golden_s075.py --variant big10 # that variant's <variant>__ files (added to the same manifest)
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "s075_residual_reversal"
NAMES = ("trades_close.csv", "trades_next_open.csv", "trade_summary.csv", "partition_dev.csv", "partition_test.csv", "partition_report.md")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline")
    args = parser.parse_args()
    prefix = "" if args.variant == "baseline" else f"{args.variant}__"
    out = ROOT / "results" / STUDY / "golden"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"study": STUDY, "files": []}
    new_files = []
    for name in NAMES:
        source = f"results/{STUDY}/{prefix}{name}"
        src = ROOT / source
        shutil.copyfile(src, out / src.name)
        rows = sum(1 for _ in src.open(encoding="utf-8")) - 1
        new_files.append({"file": src.name, "source": source, "sha256": sha256(src), "rows": rows, "deterministic": True})
    names = {f["file"] for f in new_files}
    manifest["files"] = [f for f in manifest["files"] if f["file"] not in names] + new_files
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(new_files)} golden files for {args.variant} to {out}")


if __name__ == "__main__":
    main()
