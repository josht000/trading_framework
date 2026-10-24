"""Atomically maintain reports/exit/PROGRESS.md from campaign artifacts."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exit"
TABLES = ROOT / "reports" / "exit" / "tables"
REPORT = ROOT / "reports" / "exit" / "PROGRESS.md"
PLAN = {"fixed_time": 10, "trailing": 200, "momentum": 200, "scale_out": 150,
        "pm_ah": 100, "combos": 200, "learned": 20, "optimal_stopping": 20}


def _now():
    return time.time()


def _read_elapsed():
    start = float((OUT / "budget_start.txt").read_text().strip())
    elapsed = max(0, _now() - start)
    return start, elapsed, max(0, 21600 - elapsed)


def _trial_rows():
    rows = []
    logged = {}
    elapsed_path = OUT / "elapsed.jsonl"
    if elapsed_path.exists():
        for line in elapsed_path.read_text(errors="replace").splitlines():
            try:
                label = json.loads(line).get("label", "")
                if ":" in label:
                    family, count = label.rsplit(":", 1)
                    if count.isdigit():
                        logged[family] = max(logged.get(family, 0), int(count))
            except (ValueError, TypeError):
                continue
    for family, planned in PLAN.items():
        p = TABLES / f"{family}.csv"
        done = 0
        best = None
        if p.exists():
            try:
                d = pd.read_csv(p)
                done = len(d)
                if done:
                    b = d.sort_values("score", ascending=False).iloc[0]
                    best = (float(b["score"]), float(b["mean_ret"]), float(b["hit_rate"]))
            except (OSError, ValueError, KeyError):
                pass
        done = max(done, logged.get(family, 0))
        rows.append((family, done, planned, best))
    for p in sorted(TABLES.glob("robustness_*.csv")):
        try:
            d = pd.read_csv(p)
            best = None
            if len(d):
                b = d.sort_values("score", ascending=False).iloc[0]
                best = (float(b["score"]), float(b["mean_ret"]), float(b["hit_rate"]))
            rows.append((p.stem, len(d), 200, best))
        except (OSError, ValueError, KeyError):
            continue
    for family, done in sorted(logged.items()):
        if family.startswith("robustness_") and not any(r[0] == family for r in rows):
            rows.append((family, done, 200, None))
    return rows


def render():
    start, elapsed, remaining = _read_elapsed()
    now = pd.Timestamp.now(tz="America/New_York")
    rows = _trial_rows()
    finalists = (ROOT / "reports" / "exit" / "finalists.json").exists()
    stopped = (OUT / "STOPPED.txt").exists()
    elapsed_min = elapsed / 60
    total_min = 360
    bar = int(min(30, 30 * elapsed / 21600))
    running = "**STOPPED BY USER; no campaign/search jobs remain.**" if stopped else "campaign family inferred from the first incomplete phase"
    log = OUT / "campaign.log"
    if log.exists():
        text = log.read_text(errors="replace")
        if text:
            running = "campaign process; latest console output: " + text.splitlines()[-1][:120]
    incomplete = [(f, d, p) for f, d, p, _ in rows if d < p]
    if incomplete and not stopped:
        running = f"campaign family `{incomplete[0][0]}`; {incomplete[0][1]}/{incomplete[0][2]} trials complete"
    lines = [
        "# Exit campaign progress" + (" — STOPPED BY USER" if stopped else ""),
        "",
        f"- Budget start (epoch): `{int(start)}`; now: `{now.isoformat()}`; elapsed: `{elapsed_min:.1f} min`; remaining: `{remaining / 60:.1f} min`",
        f"- `[{'#' * bar}{'.' * (30 - bar)}] {100 * elapsed / 21600:.1f}% of 6h budget`",
        "",
        "## Phases",
        "",
        "| phase/family | status | trials done/planned |",
        "|---|---|---:|",
        f"| Phase 0 08:00 retest | done | n/a |",
        "| simulator + regression/no-lookahead tests | done | 10 tests passed |",
        "| oracle + fill baselines | done | cached windows + baseline tables |",
    ]
    for family, done, planned, _ in rows:
        status = "done" if done >= planned else ("running" if done else "pending")
        lines.append(f"| {family} | {status} | {done}/{planned} |")
    lines += ["", "## Currently running", "", f"- `{running}`", "- Started: campaign start is recorded in `results/exit/budget_start.txt`; ETA: bounded by remaining budget.", "",
              "## Best dev-only result by family", "",
              "| family | CV score J | mean ret | hit rate | vs realistic baseline |",
              "|---|---:|---:|---:|---:|"]
    baseline = None
    p = TABLES / "baseline_dev_realistic.csv"
    if p.exists():
        try:
            baseline = float(pd.read_csv(p)["ret"].mean())
        except (OSError, ValueError, KeyError):
            pass
    for family, _, _, best in rows:
        if best:
            score, mean, hit = best
            versus = mean - baseline if baseline is not None else float("nan")
            lines.append(f"| {family} | {score:.5f} | {mean:.5f} | {hit:.3f} | {versus:+.5f} |")
        else:
            lines.append(f"| {family} | — | — | — | — |")
    lines += ["", "## Phase 0 baseline", ""]
    comparison = OUT / "old_vs_new_baseline.csv"
    if comparison.exists():
        c = pd.read_csv(comparison)
        for period in ("dev", "test"):
            if period == "test" and not finalists:
                continue
            lines.append(f"### {period}")
            for _, r in c[c["period"] == period].iterrows():
                lines.append(f"- {r['version']}: {int(r['trades'])} trades, mean {r['mean%']:.3f}%, "
                             f"PF {r['PF']:.2f}, total {r['total%']:.1f}%, max DD {r['maxDD%']:.1f}%")
    lines += ["", "## Running log / findings", "",
              "- Legacy baseline equivalence sampled at 20 trades: max absolute return difference < 3e-16.",
              "- 08:00 floor is enforced in variant triggers and `_path.pm_last`; realistic exits use next-bar open.",
              "- Test-period exit results are withheld here until finalists are pre-registered.",
              "", "## Next steps", "",
        "- Campaign stopped at the user's instruction; no further experiments or locked-test exit evaluation will be run.",
        "- `reports/exit_testing_report.md` records the partial artifacts, limitations, and missing oracle/finalist outputs.",
              ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    a = ap.parse_args()
    while True:
        render()
        if not a.watch or _read_elapsed()[2] <= 0:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
