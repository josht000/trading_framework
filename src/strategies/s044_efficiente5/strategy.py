"""S044 discrete replication of the JPM Efficiente 5 tactical ETF portfolio."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import daily_study as BS
from src.framework import report
from src.framework.daily_study import prefix
from src.framework.registry import register
from src.utils import timing

from .optimizer import ASSET_CAPS, CLASS_GROUPS, TOTAL_UNITS
from .signals import ASSETS, build_trades, compute_signals

NAME = "s044_efficiente5"
YAHOO_SUFFIX = "_1Day_yahoo.parquet"
TR_SUFFIX = "_1Day_tr.parquet"


def _load_yahoo_total_return(data_dir, symbols):
    frames = {}
    for symbol in symbols:
        frame = pd.read_parquet(Path(data_dir) / f"{symbol}{YAHOO_SUFFIX}")
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            index = index.tz_convert("America/New_York").tz_localize(None)
        index = index.normalize()
        raw_close = frame["close"].to_numpy(float)
        adjusted = frame["adj_close"].to_numpy(float)
        factor = np.divide(adjusted, raw_close, out=np.ones(len(frame)), where=raw_close > 0)
        frames[symbol] = pd.DataFrame(
            {
                "open": frame["open"].to_numpy(float) * factor,
                "close": adjusted,
                "volume": frame["volume"].to_numpy(float),
            },
            index=index,
        )
    dates = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in frames.values()))))
    panel = {
        "dates": dates,
        "symbols": list(symbols),
        "open": np.full((len(dates), len(symbols)), np.nan),
        "close": np.full((len(dates), len(symbols)), np.nan),
        "volume": np.full((len(dates), len(symbols)), np.nan),
    }
    row_map = {date: i for i, date in enumerate(dates)}
    for j, symbol in enumerate(symbols):
        frame = frames[symbol]
        rows = np.array([row_map[date] for date in frame.index], dtype=int)
        for field in ("open", "close", "volume"):
            panel[field][rows, j] = frame[field].to_numpy(float)
    return panel


def _monthly_returns(daily_returns, dates):
    frame = pd.DataFrame({"return": daily_returns}, index=pd.DatetimeIndex(dates))
    return frame.groupby(frame.index.to_period("M"))["return"].apply(
        lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0)
    )


def _sharpe(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    sd = values.std(ddof=1)
    return float(values.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan


def _purged_cpcv(monthly, n_groups=6, n_test_groups=2, purge_months=1):
    values = np.asarray(monthly, dtype=float)
    groups = [x for x in np.array_split(np.arange(len(values)), n_groups) if len(x)]
    rows = []
    for fold, test_groups in enumerate(combinations(range(len(groups)), n_test_groups), 1):
        test_idx = np.concatenate([groups[i] for i in test_groups])
        test_mask = np.zeros(len(values), dtype=bool)
        test_mask[test_idx] = True
        purge = test_mask.copy()
        for idx in test_idx:
            lo, hi = max(0, idx - purge_months), min(len(values), idx + purge_months + 1)
            purge[lo:hi] = True
        train_idx = np.flatnonzero(~purge)
        test = values[test_idx]
        train = values[train_idx]
        rows.append({
            "fold": fold,
            "test_groups": ",".join(str(i) for i in test_groups),
            "train_months": len(train),
            "test_months": len(test),
            "train_sharpe": _sharpe(train),
            "test_sharpe": _sharpe(test),
            "test_total_log": float(np.log1p(test).sum()),
        })
    return pd.DataFrame(rows)


@register(NAME)
class S044Efficiente5(BS.DailyStudy):
    name = NAME
    TITLE = "S044 Efficiente 5 MPT Replication"

    def load(self, ctx, cfg):
        mode = cfg.get("mode", "replication")
        with timing.stage("ETF total-return panel"):
            if mode == "replication":
                panel = _load_yahoo_total_return(cfg["data_dir"], ASSETS)
            else:
                panel = load_panel(
                    cfg["data_dir"], symbols=ASSETS, suffix=cfg.get("data_suffix", TR_SUFFIX),
                    rebuild=bool(ctx.flags.get("rebuild_cache")),
                )
        if set(panel["symbols"]) != set(ASSETS):
            raise ValueError("S044 requires all 13 catalog ETF series")
        if list(panel["symbols"]) != list(ASSETS):
            cols = [panel["symbols"].index(s) for s in ASSETS]
            panel = {**panel, "symbols": list(ASSETS),
                     **{field: panel[field][:, cols] for field in ("open", "close", "volume")}}
        return panel, {"mode": mode}

    def signals(self, panel, aux, cfg):
        with timing.stage("integer portfolio optimization"):
            return compute_signals(panel, cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret"

    def baselines(self, cfg, state, strat_r):
        return {
            "equal_weight_excess_over_BIL": state["sig"]["equal_weight_excess"],
            "SPY_total_return": state["sig"]["spy_return"],
            "sixty_forty_SPY_TLT": state["sig"]["sixty_forty"],
            "dev_vol_matched_equal_weight": BS.vol_matched(
                state["sig"]["equal_weight_excess"], strat_r, state["masks"]["dev"]
            ),
        }

    def intro(self, cfg, state):
        return [
            f"- Data mode: {cfg['mode']}; 13 ETFs, {cfg['lookback']} daily total-return observations, "
            f"{cfg['rebalance']} rebalance, 5%-grid weights, target volatility {cfg['target_vol']:.0%}.",
            "- Exact discrete search enforces per-asset and asset-class caps; target volatility escalates by 1% if infeasible. "
            "Reported return is portfolio total return minus BIL.",
            f"- Costs={cfg['cost_bps_per_side']} bps/side; one-way weight turnover and monthly weight constraints are in "
            "`weights_by_rebalance.csv`. Close is an upper bound; `next_open` is primary.",
            "- Benchmarks: equal-weight same ETF universe, SPY, 60/40 SPY-TLT, and a dev-vol-matched equal-weight sleeve.",
            "- The published replication target is Sharpe 0.50-0.60, not Sharpe >2; source acceptance also checks yearly average weights and 5%-grid constraints.",
        ]

    def caption(self, cfg, state):
        return (
            f"variant={cfg['variant']} | mode={cfg['mode']} rebalance={cfg['rebalance']} "
            f"lookback={cfg['lookback']} target={cfg['target_vol']:.0%} | "
            f"{cfg['cost_bps_per_side']}bps/side | {state['dates'][0].date()}..{state['dates'][-1].date()}"
        )

    def extra_summary(self, ctx, table, out):
        state = self._state(ctx)
        cfg = ctx.cfg
        pre = prefix(cfg)
        diagnostics = pd.DataFrame(state["sig"]["optimization_diagnostics"])
        if len(diagnostics):
            weights = pd.DataFrame(diagnostics.pop("weights").tolist(), columns=ASSETS)
            monthly = pd.concat([diagnostics.reset_index(drop=True), weights], axis=1)
            monthly.to_csv(out / f"{pre}weights_by_rebalance.csv", index=False)
            weight_rows = []
            for year, idx in monthly.groupby(pd.to_datetime(monthly["signal_date"]).dt.year).groups.items():
                mean_weights = weights.loc[idx].mean()
                weight_rows.append({"year": year, **mean_weights.to_dict()})
            pd.DataFrame(weight_rows).to_csv(out / f"{pre}average_weights_by_year.csv", index=False)
            turnover = np.abs(np.diff(np.vstack([np.zeros((1, len(ASSETS))), weights.to_numpy()]), axis=0)).sum(1)
            pd.DataFrame({
                "signal_date": monthly["signal_date"],
                "one_way_turnover": turnover,
            }).to_csv(out / f"{pre}monthly_turnover.csv", index=False)

        trades = state["trades"][BS.fills_of(cfg)[0]]
        book = self.book(state, cfg, BS.fills_of(cfg)[0])
        daily, _ = BS._account(trades, book)
        split = pd.Timestamp(cfg["partition_split_date"])
        dates = pd.DatetimeIndex(state["dates"])
        dev = dates < split
        monthly_returns = _monthly_returns(daily[dev], dates[dev])
        cpcv = _purged_cpcv(monthly_returns.to_numpy())
        cpcv.to_csv(out / f"{pre}cpcv.csv", index=False)
        report.section(f"{NAME} monthly constraints and purged CPCV")
        report.table(cpcv, floatfmt=".3f")
