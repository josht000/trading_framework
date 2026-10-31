"""Run a strategy study through the switchable framework."""

import argparse
import os
from pathlib import Path

from src.framework import RunContext, get_strategy
from src.framework.config import env_overrides, load_config
from src.framework.pipeline import ORDER, run
from src.framework.preflight import check_study_inputs
from src.utils import timing


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config")
    parser.add_argument("--stages", default=",".join(ORDER))
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="validate inputs and write a manifest without running the study")
    parser.add_argument("--variant", help="comma-separated names from the strategy config's variants: block (default: baseline)")
    parser.add_argument("--data-dir", help="folder of daily-bar parquet files (overrides data_dir in the strategy config)")
    args = parser.parse_args(argv)

    if args.profile:
        os.environ["PROFILE"] = "1"
    if args.rebuild_cache:
        os.environ["DATA_CACHE_REBUILD"] = "1"
    if not args.check_data:
        timing.install("run")
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    ctx = None
    base_env = dict(os.environ)
    for variant in [v.strip() for v in (args.variant or "").split(",") if v.strip()] or [None]:
        cfg = load_config(args.strategy, args.config, env=base_env, cli={"data_dir": args.data_dir}, variant=variant)
        manifest = check_study_inputs(
            args.strategy,
            cfg,
            root=Path.cwd(),
            results_dir=Path(os.path.abspath("results")),
        )
        checked = ", ".join(f"{key}={value.get('symbols', value.get('minute_files', 0))}" for key, value in manifest["inputs"].items())
        print(f"Input check {args.strategy}:{cfg['variant']}: {checked or 'no declared inputs'}")
        for warning in manifest["warnings"]:
            print(f"Input warning: {warning}")
        if args.check_data:
            continue
        for key, value in env_overrides(cfg).items():
            os.environ[key] = value
        ctx = RunContext(
            cfg=cfg,
            paths={"root": os.path.abspath("."), "results": os.path.abspath("results"), "reports": os.path.abspath("reports")},
            args=args,
            flags={"test": args.test, "rebuild": args.rebuild, "rebuild_cache": args.rebuild_cache, "profile": args.profile},
        )
        with timing.stage("load strategy"):
            strategy = get_strategy(args.strategy)
        with timing.stage(f"variant {cfg['variant']}"):
            run(ctx, strategy, stages)
    return ctx


if __name__ == "__main__":
    main()
