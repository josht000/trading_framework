"""Run a strategy study through the switchable framework."""

import argparse
import os

from src.framework import RunContext, get_strategy
from src.framework.config import env_overrides, load_config
from src.framework.pipeline import ORDER, run
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
    args = parser.parse_args(argv)

    if args.profile:
        os.environ["PROFILE"] = "1"
    if args.rebuild_cache:
        os.environ["DATA_CACHE_REBUILD"] = "1"
    cfg = load_config(args.strategy, args.config)
    for key, value in env_overrides(cfg).items():
        os.environ[key] = value
    timing.install("run")
    ctx = RunContext(
        cfg=cfg,
        paths={"root": os.path.abspath("."), "results": os.path.abspath("results"), "reports": os.path.abspath("reports")},
        args=args,
        flags={"test": args.test, "rebuild": args.rebuild, "rebuild_cache": args.rebuild_cache, "profile": args.profile},
    )
    with timing.stage("load strategy"):
        strategy = get_strategy(args.strategy)
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    run(ctx, strategy, stages)
    return ctx


if __name__ == "__main__":
    main()
