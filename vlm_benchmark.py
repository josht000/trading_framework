"""
VLM benchmark for resistance-level reading.
python vlm_benchmark.py models
  python vlm_benchmark.py prepare                    # PNGs (with / without labels), OHLCV csvs, manifest
  python vlm_benchmark.py models                     # table of candidate VLMs: prices, vision signals, est. cost
  python vlm_benchmark.py run --dry-run              # one chart per request; cost preview only
  python vlm_benchmark.py run --split val --shots train --prompt learned    # dev: val charts, train worked examples
  python vlm_benchmark.py run --split test --shots train,val --prompt learned   # final: test charts, train+val examples
  python vlm_benchmark.py run --models deepseek-v4.1-flash --via openrouter --budget 3
  python vlm_benchmark.py report                     # leaderboard CSV + HTML review page

Keys (in .env, never printed): ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY.
--via openrouter sends every model through OpenRouter, so a single key covers all of them.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vlm.estimate import SCENARIOS, model_table, to_markdown
from src.vlm.fewshot import load_shots, load_split_items
from src.vlm.models import load_registry, select_models
from src.vlm.packets import DEFAULT_EXPLANATIONS_ROOT, DEFAULT_ROOT, build_packets, load_explanation, load_manifest
from src.vlm.prompt import build_system_prompt, packet_text
from src.vlm.providers import MOCK_BEHAVIOURS, mock_spec
from src.vlm.report import build_report
from src.vlm.runner import missing_keys, run_benchmark


def _models(args):
    if getattr(args, "mock", None):
        return [mock_spec(k) for k in args.mock.split(",")]
    reg = load_registry()
    models = select_models(reg, args.models, args.first_pass)
    return [m.via(args.via) for m in models]


def _read_user_text(path):
    if not path:
        return ""
    with open(path, encoding="utf8") as f:
        return f.read()


def _split_list(text):
    return tuple(x.strip() for x in text.split(",") if x.strip()) if text else ()


def _sizes(root, user_text="", explanations_root=DEFAULT_EXPLANATIONS_ROOT, split=None, shot_splits=(), prompt="rubric"):
    system_chars = len(build_system_prompt(prompt, user_text))
    shots = load_shots(shot_splits, explanations_root) if shot_splits else []
    shot_chars = sum(len(s["text"]) + len(s["answer"]) for s in shots)
    try:
        if split:
            items = load_split_items(split, explanations_root)
            size = (1600, 1000)
        else:
            mf = load_manifest(root)
            items, size = mf["items"], tuple(mf["image_px"])
    except FileNotFoundError:
        return (1600, 1000), system_chars + shot_chars, 7000, 0, len(shots)
    chars = []
    for it in items:
        ohlcv = it["ohlcv_path"] if split else os.path.join(root, it["ohlcv"])
        with open(ohlcv, encoding="utf8") as f:
            note = "" if split else load_explanation(it, explanations_root)
            chars.append(len(packet_text(it, f.read(), note)))
    return size, system_chars + shot_chars, int(sum(chars) / max(1, len(chars))), len(items), len(shots)


def cmd_prepare(args):
    items = build_packets(args.chart_set, args.labels, args.root, args.bars)
    print(f"Wrote {len(items)} packets to {args.root} ({args.bars + 1} daily bars each)")
    for it in items:
        print(f"  {it['id']:26s} {it['n_bars']} bars {it['first_date']}..{it['last_date']}  gold={it['gold_levels']}")


def cmd_models(args):
    reg = load_registry()
    user_text = _read_user_text(args.user_text_file)
    shot_splits = _split_list(args.shots)
    (w, h), system_chars, packet_chars, n, n_shots = _sizes(args.root, user_text, args.explanations_root, args.split,
                                                           shot_splits, "learned" if shot_splits else "rubric")
    n = n or 9
    models = select_models(reg, args.models, args.first_pass)
    df = model_table([m.via(args.via) for m in models], w, h, packet_chars, n, args.repeats,
                     args.scenario, system_chars=system_chars, n_images=1 + n_shots)
    df = df.sort_values(["tier", "in_$/M"], key=lambda s: s.map({"premium": 0, "mid": 1, "budget": 2}) if s.name == "tier" else s)
    print(f"Registry as of {reg['as_of']}.  Estimates: {n} charts x {args.repeats} repeats, {w}x{h}px chart, "
          f"one chart per request with {n_shots} worked examples, ~{system_chars} shared prompt chars "
          f"(system + examples) + ~{packet_chars} chars/chart, "
          f"{args.scenario} reasoning-token scenario.\n")
    print(to_markdown(df))
    if args.write:
        df.to_csv(os.path.join(args.root, "vlm_candidates.csv"), index=False)
        with open(os.path.join(args.root, "vlm_candidates.md"), "w", encoding="utf8") as f:
            f.write(to_markdown(df) + "\n\n" + "\n".join(f"- {x}" for x in reg["notes"]) + "\n")
        print(f"\nWrote {args.root}/vlm_candidates.csv and .md")
    for note in reg["notes"]:
        print(f"* {note}")


def cmd_run(args):
    models = _models(args)
    if not args.dry_run:
        miss = missing_keys(models)
        if miss:
            lines = [f"  {env}: needed by {', '.join(keys)}" for env, keys in miss.items()]
            raise SystemExit("Missing API keys (set them in .env):\n" + "\n".join(lines))
    try:
        s = run_benchmark(models, args.root, args.prompt or ("learned" if args.shots else "rubric"), args.repeats, args.budget, args.workers, args.limit,
                          args.max_tokens, args.dry_run, args.items.split(",") if args.items else None,
                          user_text=_read_user_text(args.user_text_file), explanations_root=args.explanations_root,
                          split=args.split, shot_splits=_split_list(args.shots))
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    label = "DRY RUN estimate" if args.dry_run else "spent (est+actual)"
    print(f"\nattempted_requests={s.attempted} cached_requests={s.cached} ok_requests={s.ok} failed_requests={s.failed} "
          f"skipped_over_budget={s.skipped_budget}  {label}: ${s.spent:.4f}")


def cmd_report(args):
    paths = build_report(args.root, args.tol, render=not args.no_images, splits=_split_list(args.split),
                         dataset_root=args.explanations_root)
    print("Wrote:\n  " + "\n  ".join(paths.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--chart-set", default=os.path.join("data", "labels", "chart_set_manual_1.csv"))
    p.add_argument("--labels", default=os.path.join("data", "labels", "manual_labels.csv"))
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--bars", type=int, default=160, help="lookback bars before the last bar (chart shows bars+1)")
    p.set_defaults(func=cmd_prepare)

    def selectors(q):
        q.add_argument("--models", help="comma-separated registry keys (default: all, or first-pass set)")
        q.add_argument("--first-pass", action="store_true", help="only the recommended first-pass models")
        q.add_argument("--via", choices=["native", "openrouter"], default="native")
        q.add_argument("--root", default=DEFAULT_ROOT)
        q.add_argument("--repeats", type=int, default=1)
        q.add_argument("--user-text-file", help="optional UTF-8 notes/examples included in the shared system prompt")
        q.add_argument("--explanations-root", default=DEFAULT_EXPLANATIONS_ROOT,
                       help="dataset root holding <split>/charts, price_data, labels.csv and explainations/")
        q.add_argument("--split", choices=["train", "val", "test"],
                       help="evaluate this dataset split (default: legacy data/vlm manifest)")
        q.add_argument("--shots", help="comma-separated labelled splits used as few-shot worked examples "
                                       "(e.g. train or train,val); never test")

    p = sub.add_parser("models"); selectors(p)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="base", help="assumed hidden-reasoning size")
    p.add_argument("--write", action="store_true", help="also write vlm_candidates.csv/.md")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("run"); selectors(p)
    p.add_argument("--mock", help=f"offline fake models for testing: {','.join(MOCK_BEHAVIOURS)}")
    p.add_argument("--prompt", choices=["rubric", "plain", "fewshot", "learned"],
                   help="default: learned when --shots is given, otherwise rubric")
    p.add_argument("--budget", type=float, default=5.0, help="USD cap for this invocation")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, help="run only the first N calls")
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="max output tokens incl. hidden reasoning; reasoning models can exhaust a small cap and return nothing")
    p.add_argument("--items", help="comma-separated substrings of chart ids to include")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--split", help="score these dataset splits instead of the manifest (comma-separated, e.g. val or test)")
    p.add_argument("--explanations-root", default=DEFAULT_EXPLANATIONS_ROOT)
    p.add_argument("--tol", type=float, default=0.5, help="match tolerance in ATR14 for the headline metrics")
    p.add_argument("--no-images", action="store_true")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
