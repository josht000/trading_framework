"""Copy a study's outputs into results/<study>/golden and update MANIFEST.json.

  python tools/make_golden.py --study s001_rsi2_connors                 # baseline files
  python tools/make_golden.py --study s001_rsi2_connors --variant rsi5  # that variant's <variant>__ files
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("trades_close.csv", "trades_next_open.csv", "trade_summary.csv",
         "partition_dev.csv", "partition_test.csv", "partition_report.md")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_golden(study, variant="baseline", root=ROOT):
    prefix = "" if variant == "baseline" else f"{variant}__"
    out = Path(root) / "results" / study / "golden"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"study": study, "files": []}
    new_files = []
    for name in NAMES:
        src = Path(root) / "results" / study / f"{prefix}{name}"
        if not src.exists():
            continue
        shutil.copyfile(src, out / src.name)
        rows = sum(1 for _ in src.open(encoding="utf-8")) - 1
        new_files.append({"file": src.name, "source": f"results/{study}/{prefix}{name}",
                          "sha256": sha256(src), "rows": rows, "deterministic": True})
    if not new_files:
        raise FileNotFoundError(f"no outputs found for {study}:{variant} under results/{study}")
    names = {f["file"] for f in new_files}
    manifest["files"] = [f for f in manifest["files"] if f["file"] not in names] + new_files
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(new_files)} golden files for {study}:{variant} to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--variant", default="baseline")
    args = parser.parse_args()
    make_golden(args.study, args.variant)


if __name__ == "__main__":
    main()
