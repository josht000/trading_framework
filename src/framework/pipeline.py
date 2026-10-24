"""Ordered, dependency-aware study stages."""

from src.utils import timing

ORDER = ("trades", "table", "rank", "charts", "partition")
DEPENDENCIES = {
    "trades": (),
    "table": (),
    "rank": ("table",),
    "charts": ("table", "rank"),
    "partition": ("table", "rank"),
}


def _resolve(requested):
    selected = set()

    def add(name):
        for dependency in DEPENDENCIES[name]:
            add(dependency)
        selected.add(name)

    for stage in requested:
        if stage not in DEPENDENCIES:
            raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(ORDER)}")
        add(stage)
    return [stage for stage in ORDER if stage in selected]


def run(ctx, strategy, stages=None):
    requested = ORDER if stages is None else stages
    for name in _resolve(requested):
        try:
            with timing.stage(name):
                if name == "trades":
                    ctx.data["trades"] = strategy.build_trades(ctx)
                elif name == "table":
                    ctx.data["table"] = strategy.build_table(ctx, bool(ctx.flags.get("rebuild")))
                elif name == "rank":
                    ctx.data["setups"] = strategy.setups(ctx, ctx.data["table"])
                    strategy.rank_experiments(ctx, ctx.data["table"])
                elif name == "charts":
                    strategy.charts(ctx, ctx.data["table"], ctx.data["setups"])
                elif name == "partition":
                    from src.framework.evaluate import evaluate
                    ctx.data["evaluation"] = evaluate(ctx, ctx.data["table"], ctx.data["setups"])
        except Exception as exc:
            raise RuntimeError(f"study stage {name!r} failed") from exc
    return ctx
