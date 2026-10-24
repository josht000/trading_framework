"""Compare study outputs with the recorded golden manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def current_path(root, relative):
    path = Path(relative)
    if path.parts[0] == "holdout":
        return root / "results" / Path(*path.parts)
    if path.parts[0] == "partition":
        return root / "reports" / Path(*path.parts)
    if path.parts[0] == "ranking":
        return root / "reports" / Path(*path.parts)
    if path.name == "ranked_table.parquet":
        return root / "reports" / path.name
    if path.name == "ranking_weights.json":
        return root / "results" / path.name
    return root / "results" / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(manifest_path):
    manifest_path = Path(manifest_path)
    root = manifest_path.parents[3]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identical, differing, nondeterministic = [], [], []
    for item in manifest["files"]:
        if not item.get("deterministic", False):
            nondeterministic.append(item["file"])
            continue
        actual = current_path(root, item["file"])
        if actual.exists() and sha256(actual) == item["sha256"]:
            identical.append(item["file"])
        else:
            differing.append(item["file"])
    print(f"deterministic identical: {len(identical)}")
    print(f"deterministic differing: {len(differing)}")
    for name in differing:
        print(f"DIFF {name}")
    print(f"non-deterministic reported only: {len(nondeterministic)}")
    for name in nondeterministic:
        print(f"NONDET {name}")
    return not differing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/golden/atr_pivot_lb13/MANIFEST.json")
    args = parser.parse_args()
    raise SystemExit(0 if check(args.manifest) else 1)


if __name__ == "__main__":
    main()
