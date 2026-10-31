"""S017 annual overnight-bias stock selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.data.liquid_universe import load_liquid_ranks
from src.data.market_universes import ETF_LIKE_SYMBOLS
from src.framework import report
from src.framework.daily_study import prefix
from src.framework.registry import register
from src.strategies._engines import align_ranks
from src.strategies._overnight import OvernightDailyStudy, join_panels, load_index_panel, load_raw_panel
from src.utils import timing

from .signals import build_trades, compute_signals, _formation_rows

NAME = "s017_night_trading"
TR_SUFFIX = "_1Day_tr.parquet"


def _formation_universe(rank_dates, rank_symbols, rank, cadence, top_n):
    rows = _formation_rows(rank_dates, cadence)
    selected = set()
    for row in rows:
        selected.update(rank_symbols[i] for i in np.flatnonzero(rank[row] <= int(top_n)))
    return rows, sorted(selected)


def _align_returns(panel, source):
    close = source["close"][:, 0]
    returns = np.zeros(len(close), dtype=float)
    valid = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[:-1] > 0)
    rows = np.flatnonzero(valid) + 1
    returns[rows] = close[rows] / close[rows - 1] - 1.0
    out = np.zeros(len(panel["dates"]), dtype=float)
    positions = pd.DatetimeIndex(panel["dates"]).get_indexer(pd.DatetimeIndex(source["dates"]))
    good = positions >= 0
    out[positions[good]] = returns[good]
    return out


@register(NAME)
class S017NightTrading(OvernightDailyStudy):
    name = NAME
    TITLE = "S017 Night Trading Overnight-Bias Stock Selection"

    def load(self, ctx, cfg):
        rank_dates, rank_symbols, rank_matrix = load_liquid_ranks(cfg.get("ranks_dir", "data/cache/liquid_ranks"))
        formation_rows, candidate_symbols = _formation_universe(
            rank_dates, rank_symbols, rank_matrix, cfg.get("formation_cadence", "annual"),
            int(cfg["universe_top_n"]),
        )
        raw_root = Path(cfg["raw_data_dir"])
        available = {p.name[:-len(cfg["raw_data_suffix"])] for p in raw_root.glob(f"*{cfg['raw_data_suffix']}")}
        stock_symbols = sorted((set(candidate_symbols) & available) - ETF_LIKE_SYMBOLS)
        missing = sorted(set(candidate_symbols) - available - ETF_LIKE_SYMBOLS)
        if missing:
            raise FileNotFoundError(f"raw overnight bars missing for {len(missing)} liquid names: {', '.join(missing[:10])}")
        if not stock_symbols:
            raise FileNotFoundError("no PIT liquid stocks have raw overnight bars")
        with timing.stage("raw stock bars and corporate actions"):
            stocks, stock_aux = load_raw_panel(
                raw_root, stock_symbols, cfg["corporate_actions_file"],
                rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        with timing.stage("SPY raw benchmark"):
            spy, spy_aux = load_index_panel(
                "SPY", raw_root, cfg["corporate_actions_file"], cfg["yahoo_data_dir"],
                start_date=cfg["start_date"], rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        panel, aux = join_panels(stocks, stock_aux, spy, spy_aux)
        rank = align_ranks(panel["dates"], panel["symbols"], rank_dates, rank_symbols, rank_matrix)
        aux["rank"] = rank
        aux["formation_rows"] = formation_rows
        aux["raw_stock_symbols"] = stock_symbols
        aux["formation_rows"] = _formation_rows(panel["dates"], cfg.get("formation_cadence", "annual"))
        with timing.stage("SPY total-return benchmark"):
            tr = load_panel(
                cfg["tr_data_dir"], symbols=["SPY"], suffix=cfg.get("tr_data_suffix", TR_SUFFIX),
                rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
            aux["spy_total_return"] = _align_returns(panel, tr)
        return panel, aux

    def signals(self, panel, aux, cfg):
        with timing.stage("252-session overnight bias and FDR screen"):
            return compute_signals(panel, aux, cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def book(self, state, cfg, fill):
        book = super().book(state, cfg, fill)
        mode = cfg.get("holding_mode", "overnight_only")
        if mode == "overnight_only":
            book["entry_mat"] = state["sig"]["overnight_entry"]
            book["exit_mat"] = state["sig"]["overnight_exit"]
        elif mode == "full_day":
            book["entry_mat"] = state["panel"]["open"]
            book["exit_mat"] = state["panel"]["close"]
        else:
            book["entry_mat"] = np.ones(state["panel"]["close"].shape, dtype=float)
            book["exit_mat"] = 1.0 + state["sig"]["spread_returns"]
        return book

    def book(self, state, cfg, fill):
        book = super().book(state, cfg, fill)
        mode = cfg.get("holding_mode", "overnight_only")
        if mode == "overnight_only":
            book["entry_mat"] = state["aux"]["overnight_entry"]
            book["exit_mat"] = state["aux"]["overnight_exit"]
        elif mode == "full_day":
            book["entry_mat"] = state["panel"]["open"]
            book["exit_mat"] = state["panel"]["close"]
        else:
            book["entry_mat"] = np.ones(state["panel"]["close"].shape, dtype=float)
            book["exit_mat"] = 1.0 + state["sig"]["spread_returns"]
        return book

    def rank_col(self):
        return "ret"

    def baselines(self, cfg, state, strat_r):
        return {
            "same_universe_unconditional_overnight": state["b_day"],
            "spy_overnight": state["aux"]["overnight_return"][:, -1],
            "spy_total_return": state["aux"]["spy_total_return"],
        }

    def intro(self, cfg, state):
        cohorts = state["sig"]["formation_summary"]
        return [
            f"- PIT liquid_top_{cfg['universe_top_n']} US common stocks; formation={cfg['formation_cadence']}, "
            f"252 overnight observations, t>{cfg['min_t_stat']}, BH q={cfg['fdr_q']:.0%}, and top {cfg['top_fraction']:.0%} by mean overnight return.",
            f"- {len(cohorts)} formation cohorts; {len(state['aux']['raw_stock_symbols'])} symbols have raw bars and corporate actions. "
            f"Positions hold {cfg['holding_mode']} and equal-weight the selected names.",
            f"- Auction-based close/open fills; {cfg['cost_bps_per_side']} bps/side, stock cost ladder and break-even cost required.",
            "- The source used 5% significance and found about one-fifth of stocks in its Overnight Bias Group. The catalog adds a top-quintile rule and BH FDR control.",
        ]

    def caption(self, cfg, state):
        return (
            f"variant={cfg['variant']} | top{cfg['universe_top_n']} formation={cfg['formation_cadence']} "
            f"t>{cfg['min_t_stat']} FDR={cfg['fdr_q']:.0%} hold={cfg['holding_mode']} | "
            f"{cfg['cost_bps_per_side']}bps/side | {state['dates'][0].date()}..{state['dates'][-1].date()}"
        )

    def extra_summary(self, ctx, table, out):
        state = self._state(ctx)
        pre = prefix(ctx.cfg)
        summary = pd.DataFrame(state["sig"]["formation_summary"])
        if len(summary):
            summary.drop(columns=["selected_symbols"], errors="ignore").to_csv(
                out / f"{pre}formation_cohorts.csv", index=False
            )
            members = state["sig"]["cohort_members"]
            overlap = []
            for i in range(1, len(members)):
                previous, current = set(members[i - 1]), set(members[i])
                union = previous | current
                overlap.append({
                    "from": summary.iloc[i - 1]["formation_date"],
                    "to": summary.iloc[i]["formation_date"],
                    "n_previous": len(previous),
                    "n_current": len(current),
                    "jaccard": len(previous & current) / len(union) if union else np.nan,
                })
            pd.DataFrame(overlap).to_csv(out / f"{pre}cohort_persistence.csv", index=False)
        report.section(f"{NAME} formation cohorts")
        report.table(summary.drop(columns=["selected_symbols"], errors="ignore"), floatfmt=".3f")
