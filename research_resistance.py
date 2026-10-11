"""
Resistance-detector research CLI.

  python research_resistance.py ladder SOFI 2024-03-05
  python research_resistance.py chart  SOFI 2024-03-05 [--forward 12]
  python research_resistance.py eval   SOFI,PLUG,OPEN --start 2023-01-01 --end 2025-06-30 --split 2024-07-01

Manual benchmark (labels live in a CSV you edit by hand, see src/research/labels.py):
  python research_resistance.py labels export TSLA 2024-11-05     # append detector levels as status=auto
  (edit data/labels/resistance_labels.csv: keep / reject / add)
  python research_resistance.py labels score                      # precision / recall of the detector vs your labels

Detector settings live in a JSON file you can edit:
  python research_resistance.py config dump                       # writes data/labels/detector_config.json
  python research_resistance.py labels score --config data/labels/detector_config.json
Any command accepts --config; individual flags (--left, --merge, ...) override the file.
"""

import argparse
import os
import sys
import webbrowser

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.research.chart_levels import plot_levels
from src.research.config_io import DETECTORS, dump_config, make_detector
from src.research.data import load_daily
from src.research.labels import (DEFAULT_LABEL_FILE, export_levels, format_score, labels_for,
                                 load_labels, score_labels)
from src.research.pick_charts import PickConfig, pick_dates
from src.research.resistance_eval import EvalConfig, SOURCES, format_report, run_evaluation, split_by_date


def _detector(args):
    if args.detector == "volume_profile":
        return make_detector("volume_profile", args.config)
    return make_detector(
        "atr_pivot", args.config,
        pivot_left=args.left, pivot_right=args.right,
        min_prominence_atr=args.prominence, merge_atr=args.merge,
        include_moving_averages=args.ma, include_swing_lows=False if args.no_lows else None,
        include_ranges=False if args.no_ranges else None,
        collapse_to_high=False if args.band else None,
    )


def _asof_daily(symbol, asof):
    return load_daily([symbol], asof, asof)[symbol]


def cmd_ladder(args):
    daily = _asof_daily(args.symbol, args.asof)
    det = _detector(args)
    p = det.prepare(daily)
    k = det.known_through_on(p, args.asof)
    print(f"{args.symbol} as of {pd.Timestamp(p.dates[k]).date()}  price={p.close[k]:.2f}  ATR={p.atr[k]:.3f}")
    lv = det.levels(p, k)
    if lv.empty:
        print("no levels")
        return
    cols = ["role", "kind", "upper", "distance_atr", "distance_pct", "score", "n_pivots",
            "touches", "volume_ratio", "is_round", "ma_confluence", "last_touch_age"]
    collapsed = getattr(det.config, "collapse_to_high", True)
    if not collapsed:
        cols.insert(2, "lower")
    out = lv.sort_values("center", ascending=False)[cols].rename(columns={"upper": "level"} if collapsed else {})
    out.insert(0, "id", range(1, len(out) + 1))
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if lv[lv["role"] == "resistance"].empty:
        print("\nNo overhead resistance in lookback window (blue sky).")


def cmd_chart(args):
    daily = _asof_daily(args.symbol, args.asof)
    path = args.out or os.path.join("data", "charts", f"{args.symbol}_{args.asof}_{args.detector}.html")
    labels = labels_for(load_labels(args.labels), args.symbol, args.asof)
    plot_levels(daily, args.asof, _detector(args), symbol=args.symbol,
                lookback=args.lookback, forward=args.forward, path=path, labels=labels,
                live=args.live, detector_name=args.detector, blank=args.blank)
    print(f"Wrote {path}" + (f" (drawing {len(labels)} rows from {args.labels})" if len(labels) else ""))
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(path))


def cmd_eval(args):
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    daily = load_daily(symbols, args.start, args.end)
    print(f"Loaded {len(daily)} symbols: {', '.join(daily)}")
    results = run_evaluation(daily, eval_config=EvalConfig(), sources=args.sources.split(","),
                             detector=_detector(args))

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    for key in ("reactions", "breakouts", "uncond"):
        if not results[key].empty:
            d = results[key]["date"]
            results[key] = results[key][(d >= start) & (d <= end)].reset_index(drop=True)

    if args.split:
        parts = split_by_date(results, args.split)
        print(format_report(parts["tune"], f"TUNE  {args.start} .. {args.split}"))
        print()
        print(format_report(parts["holdout"], f"HOLDOUT  {args.split} .. {args.end}"))
    else:
        print(format_report(results, f"{args.start} .. {args.end}"))

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        for key, df in results.items():
            df.to_csv(f"{args.save}_{key}.csv", index=False)
        print(f"\nSaved events to {args.save}_*.csv")


def cmd_labels_export(args):
    daily = _asof_daily(args.symbol, args.asof)
    n = export_levels(_detector(args), daily, args.symbol, args.asof, args.labels, args.max_dist_atr,
                      source=args.detector)
    print(f"Added {n} candidate levels (status=auto, source={args.detector}) to {args.labels}")


def cmd_labels_score(args):
    labels = load_labels(args.labels)
    if labels.empty:
        print(f"No labels in {args.labels}. Run `labels export SYMBOL ASOF` first.")
        return
    daily = {}
    for symbol, g in labels.groupby("symbol"):
        daily.update(load_daily([symbol], g["asof"].min().strftime("%Y-%m-%d"), g["asof"].max().strftime("%Y-%m-%d")))
    result = score_labels(labels, daily, _detector(args), args.tol_atr, args.max_dist_atr)
    print(format_score(result))


def cmd_pick(args):
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    daily = load_daily(symbols, args.start, args.end)
    cfg = PickConfig(per_symbol=args.per_symbol, seed=args.seed, min_ratio=args.min_ratio,
                     max_ratio=args.max_ratio, min_ret20=args.min_ret20, max_price=args.max_price,
                     max_total=args.max_total)
    exclude = None
    if args.exclude:
        frames = [pd.read_csv(path, usecols=["symbol", "asof"]) for path in args.exclude]
        exclude = pd.concat(frames, ignore_index=True).drop_duplicates()
        print(f"Excluding {len(exclude)} previously used (symbol, date) pairs from {len(args.exclude)} files")
    picks = pick_dates(daily, args.start, args.end, cfg, exclude)
    if picks.empty:
        print("No eligible dates. Loosen --min-ratio/--max-ratio/--min-ret20 or widen the date range.")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    picks.assign(asof=picks["asof"].dt.strftime("%Y-%m-%d")).to_csv(args.out, index=False)
    print(picks.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote {len(picks)} chart dates to {args.out} (chosen without running any detector)")


def cmd_labels_build_set(args):
    chart_set = pd.read_csv(args.set)
    for row in chart_set.itertuples(index=False):
        daily = _asof_daily(row.symbol, row.asof)
        counts = {}
        for name in args.detectors.split(","):
            det = make_detector(name, None)
            counts[name] = export_levels(det, daily, row.symbol, row.asof, args.labels, args.max_dist_atr, source=name)
        labels = labels_for(load_labels(args.labels), row.symbol, row.asof)
        path = os.path.join("data", "charts", f"{row.symbol}_{row.asof}_set.html")
        plot_levels(daily, row.asof, make_detector("atr_pivot", None), symbol=row.symbol,
                    lookback=args.lookback, forward=args.forward, path=path, labels=labels,
                    detector_name="+".join(counts))
        print(f"{row.symbol} {row.asof}: added {counts} -> {path}")


MANUAL_HEADER = "symbol,asof,id,level,status,source,score,kind,note\n"


def cmd_labels_manual_set(args):
    chart_set = pd.read_csv(args.set)
    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.labels):
        os.makedirs(os.path.dirname(args.labels) or ".", exist_ok=True)
        with open(args.labels, "w", encoding="utf8") as f:
            f.write(MANUAL_HEADER)
    all_labels = load_labels(args.labels)
    det = make_detector("atr_pivot", None)

    rows = []
    for i, r in enumerate(chart_set.itertuples(index=False), 1):
        daily = _asof_daily(r.symbol, r.asof)
        labels = labels_for(all_labels, r.symbol, r.asof)
        name = f"{i:02d}_{r.symbol}_{r.asof}.html"
        plot_levels(daily, r.asof, det, symbol=r.symbol, lookback=args.lookback, forward=0,
                    path=os.path.join(args.out_dir, name), labels=labels, blank=True)
        p = det.prepare(daily)
        rows.append((i, r.symbol, r.asof, float(p.close[det.known_through_on(p, r.asof)]), name, len(labels)))

    body = "\n".join(
        f'<tr><td>{i}</td><td>{s}</td><td>{a}</td><td>{px:.2f}</td><td>{n}</td><td><a href="{f}">open</a></td></tr>'
        for i, s, a, px, f, n in rows)
    html = (
        "<html><head><title>Manual labeling set</title></head><body style='font-family:sans-serif'>"
        f"<h2>Manual labeling set ({len(rows)} charts)</h2>"
        f"<p>Add one row per level to <code>{args.labels}</code>: <code>symbol,asof,level</code> "
        "(optional <code>note</code>). Charts show only data up to the as-of date. Re-run "
        "<code>labels manual-set</code> to see your labels drawn as dashed lines.</p>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        "<tr><th>#</th><th>symbol</th><th>as of</th><th>price</th><th>levels labeled</th><th>chart</th></tr>"
        f"{body}</table></body></html>")
    index = os.path.join(args.out_dir, "index.html")
    with open(index, "w", encoding="utf8") as f:
        f.write(html)
    print(f"Wrote {len(rows)} blind charts and {index}")
    print(f"Label file: {args.labels}  (columns: symbol,asof,level; status defaults to add)")


def cmd_config_dump(args):
    path = dump_config(args.out, _detector(args).config)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--detector", choices=sorted(DETECTORS), default="atr_pivot",
                       help="which level detector to use (default atr_pivot)")
        p.add_argument("--config", help="detector settings JSON (see `config dump`)")
        p.add_argument("--left", type=int, help="pivot bars on the left")
        p.add_argument("--right", type=int, help="pivot confirmation bars on the right")
        p.add_argument("--prominence", type=float, help="min pivot prominence in ATR")
        p.add_argument("--merge", type=float, help="merge pivots within this many ATR")
        p.add_argument("--ma", action="store_const", const=True, help="include SMA20/50/200 as dynamic levels")
        p.add_argument("--no-lows", action="store_true", help="ignore swing lows")
        p.add_argument("--no-ranges", action="store_true", help="ignore 5-10 day consolidation ranges")
        p.add_argument("--band", action="store_true", help="keep zones as price bands instead of a single level")

    p = sub.add_parser("ladder"); p.add_argument("symbol"); p.add_argument("asof"); common(p)
    p.set_defaults(func=cmd_ladder)

    p = sub.add_parser("chart"); p.add_argument("symbol"); p.add_argument("asof"); common(p)
    p.add_argument("--lookback", type=int, default=160); p.add_argument("--forward", type=int, default=15)
    p.add_argument("--out"); p.add_argument("--no-open", action="store_true")
    p.add_argument("--live", action="store_true", help="also draw the detector's current levels over the label rows")
    p.add_argument("--blank", action="store_true", help="blind chart: never run or draw the detector")
    p.add_argument("--labels", default=DEFAULT_LABEL_FILE)
    p.set_defaults(func=cmd_chart)

    p = sub.add_parser("eval"); p.add_argument("symbols"); common(p)
    p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--split", help="tune/holdout split date")
    p.add_argument("--sources", default=",".join(SOURCES))
    p.add_argument("--save", help="path prefix for event CSVs")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("labels"); lsub = p.add_subparsers(dest="labels_cmd", required=True)
    q = lsub.add_parser("export"); q.add_argument("symbol"); q.add_argument("asof"); common(q)
    q.add_argument("--labels", default=DEFAULT_LABEL_FILE); q.add_argument("--max-dist-atr", type=float, default=10.0)
    q.set_defaults(func=cmd_labels_export)
    q = lsub.add_parser("score"); common(q)
    q.add_argument("--labels", default=DEFAULT_LABEL_FILE)
    q.add_argument("--tol-atr", type=float, default=0.3, help="match tolerance in ATR")
    q.add_argument("--max-dist-atr", type=float, default=10.0)
    q.set_defaults(func=cmd_labels_score)

    q = lsub.add_parser("build-set", help="export candidates from several detectors for a chart set and build charts")
    q.add_argument("--set", default=os.path.join("data", "labels", "chart_set.csv"))
    q.add_argument("--detectors", default="atr_pivot,volume_profile")
    q.add_argument("--labels", default=DEFAULT_LABEL_FILE); q.add_argument("--max-dist-atr", type=float, default=10.0)
    q.add_argument("--lookback", type=int, default=160); q.add_argument("--forward", type=int, default=15)
    q.set_defaults(func=cmd_labels_build_set)

    q = lsub.add_parser("manual-set", help="build blind charts (no detector, no future bars) for manual labeling")
    q.add_argument("--set", required=True, help="chart set CSV from `pick`")
    q.add_argument("--labels", default=os.path.join("data", "labels", "manual_labels.csv"))
    q.add_argument("--out-dir", default=os.path.join("data", "charts", "manual"))
    q.add_argument("--lookback", type=int, default=160)
    q.set_defaults(func=cmd_labels_manual_set)

    p = sub.add_parser("pick", help="choose chart dates by detector-independent rules")
    p.add_argument("symbols"); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--per-symbol", type=int, default=1); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-ratio", type=float, default=0.55, help="min close / prior 252-bar high")
    p.add_argument("--max-ratio", type=float, default=0.90, help="max close / prior 252-bar high")
    p.add_argument("--min-ret20", type=float, default=0.05, help="min 20-day return")
    p.add_argument("--max-price", type=float, default=20.0)
    p.add_argument("--max-total", type=int, default=0, help="randomly keep at most this many charts overall")
    p.add_argument("--exclude", nargs="*", default=[], help="CSVs with symbol,asof columns of charts already used")
    p.add_argument("--out", default=os.path.join("data", "labels", "chart_set.csv"))
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("config"); csub = p.add_subparsers(dest="config_cmd", required=True)
    q = csub.add_parser("dump"); common(q); q.add_argument("--out", help="default: per-detector file under data/labels/")
    q.set_defaults(func=cmd_config_dump)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
