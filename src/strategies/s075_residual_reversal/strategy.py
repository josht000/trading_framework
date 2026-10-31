"""S075 Stock-vs-Sector Residual Reversal: daily-study adapter."""

import numpy as np

from src.data.daily_panel import load_panel, load_sector_map
from src.framework import DailyStudy, daily_study
from src.framework.registry import register
from src.utils import timing

from .signals import build_trades, compute_signals

NAME = "s075_residual_reversal"


@register(NAME)
class S075ResidualReversal(DailyStudy):
    name = NAME
    TITLE = "S075 Stock-vs-Sector Residual Reversal"
    needs = ("daily",)
    exposure_window = {"entry": "close of signal session (or next open)", "exit": "residual recovery or 10 sessions"}

    def load(self, ctx, cfg):
        if not cfg.get("data_dir"):
            raise ValueError("no data dir: pass --data-dir or set data_dir in the strategy config")
        with timing.stage("sector map"):
            sector_map = load_sector_map()
        panel = load_panel(cfg["data_dir"], symbols=sector_map.index,
                           rebuild=bool(ctx.flags.get("rebuild_cache")))
        return panel, {"sector_map": sector_map}

    def signals(self, panel, aux, cfg):
        return compute_signals(panel, aux["sector_map"], cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "neg_resid5"

    def augment_trade_table(self, table):
        table["neg_resid5"] = -table["resid5"]
        return table

    def trade_stats(self, trades):
        stats = daily_study.period_stats(trades)
        returns = trades["ret"].to_numpy(float)
        stats["median_ret_pct"] = 100 * float(np.median(returns)) if len(returns) else np.nan
        stats["gross_mean_ret_pct"] = 100 * float(trades["ret_gross"].mean()) if len(trades) else np.nan
        stats.update({f"exit_{reason}": int(count) for reason, count in trades["exit_reason"].value_counts().items()})
        return stats

    def intro(self, cfg, state):
        return [
            f"- Universe: top {cfg['universe_top_n']} by 60d median dollar volume "
            f"(price >= {cfg['min_price']}, >= {cfg['min_history']} bars); peers: top {cfg['peer_top_n']} "
            f"pool, SimFin {cfg.get('peer_level', 'industry')} level (>= {cfg['min_peers']} peers).",
            f"- Signal: {cfg['resid_window']}d residual vs peers (beta {cfg['beta_window']}d through t-1), "
            f"bottom decile and close > SMA{cfg['sma_window']}; exit on residual recovery or {cfg['max_hold']} sessions.",
            f"- Long-only, research scale, {cfg['cost_bps_per_side']} bps/side; `close` fill is an upper bound and "
            "`next_open` is the robustness fill.",
            "- Limitations: SimFin sector map is a current snapshot covering mapped tickers; no earnings-window filter or delisting haircut.",
        ]

    def caption(self, cfg, state):
        dates = state["dates"]
        overlay = (cfg.get("variants") or {}).get(cfg["variant"], {}) if cfg["variant"] != "baseline" else {}
        return (
            f"variant={cfg['variant']} overlay={overlay or 'none'} | "
            f"universe top {cfg['universe_top_n']} (peers {cfg['peer_top_n']}, {cfg.get('peer_level', 'industry')}) | "
            f"signal {cfg.get('signal_mode', 'raw')} {cfg['resid_window']}d residual, decile {cfg['decile']}, "
            f"SMA{cfg['sma_window']}, skip {cfg.get('skip_days', 0)} | exit {cfg.get('exit_mode', 'resid')} "
            f"cap {cfg['max_hold']} stop {cfg.get('stop_loss', 0)} target {cfg.get('take_profit', 0)} | "
            f"{cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | {dates[0].date()}..{dates[-1].date()}"
        )
