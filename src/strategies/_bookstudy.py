"""Shared tail for catalog daily equal-weight book studies (see s075_residual_reversal).

Covers the mechanics every catalog study repeats: fill-model evaluation through
src.framework.daily_partition, partition csvs, trial registry, cost ladder,
extra baselines (buy-and-hold / SMA200 timing / vol-matched EW / SPY / 60-40 /
exposure-matched random timing), daily returns, dashboard PNG, the
variants_comparison.csv upsert and the markdown partition report.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import partition as P
from src.framework import daily_partition as DP
from src.framework import dashboard, report
from src.framework.base import Setup, Strategy
from src.utils import timing

REPORT_COLS = ["trades", "days", "total_log", "trend", "trend_beta", "beta", "random_pick_universe", "random_pick_pool",
               "random_p_universe", "random_p_pool", "bias", "skill", "skill_vs_universe", "skill_vs_pool", "skill_pct",
               "skill_lo", "skill_hi", "skill_p", "skill_ex5", "win_pct", "mean_ret_pct", "pf", "cagr_pct", "sharpe",
               "max_dd_pct", "avg_hold", "avg_positions", "avg_fraction_invested", "turnover_annual"]
MIN_DEV_TRADES, MIN_TEST_TRADES = 100, 30


def out_dir(ctx, name):
    out = Path(ctx.paths.get("results", "results")) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def prefix(cfg):
    return f"{cfg['results_tag']}__" if cfg.get("results_tag") else ""


def fills_of(cfg):
    return [x.strip() for x in str(cfg["fills"]).split(",") if x.strip()]


def md_table(frame, cols):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("n/a" if isinstance(v, (float, np.floating)) else str(v))

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for _, row in frame.iterrows()]
    return "\n".join(lines)


def update_comparison(out, frame, cfg, masks):
    """Upsert this variant into variants_comparison.csv and add the cross-variant selection haircut (dev, per fill)."""
    keep = ["trades", "total_log", "trend", "random_pick_pool", "bias", "skill", "skill_vs_pool", "skill_pct", "skill_p", "sharpe",
            "cagr_pct", "max_dd_pct", "mean_ret_pct", "win_pct", "avg_hold", "avg_positions"]
    rows = frame[["fill", "period"] + keep].copy()
    rows.insert(0, "variant", cfg["variant"])
    path = out / "variants_comparison.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["variant"] != cfg["variant"]].drop(columns=[c for c in ("selection_haircut", "skill_after_haircut") if c in old])
    table = pd.concat([old, rows], ignore_index=True)
    daily = {}
    for f in out.glob("*daily_returns.csv"):
        name = f.name[: -len("daily_returns.csv")].rstrip("_") or "baseline"
        d = pd.read_csv(f, parse_dates=["date"])
        daily[name] = d[d["date"] < pd.Timestamp(cfg["partition_split_date"])]
    table["selection_haircut"] = np.nan
    for fill in fills_of(cfg):
        col = f"{fill}_r_day"
        series = {k: np.log1p(v[col].to_numpy()) for k, v in daily.items() if col in v and len(v) == len(next(iter(daily.values())))}
        if len(series) < 2:
            continue
        mat = np.vstack(list(series.values()))
        corr = np.corrcoef(mat)
        rho = float((corr.sum() - len(series)) / (len(series) * (len(series) - 1)))
        n_eff = max(1.0, 1.0 + (len(series) - 1) * (1.0 - rho))
        sd = float(mat.std(axis=1, ddof=1).mean() * np.sqrt(mat.shape[1]))
        table.loc[(table["fill"] == fill) & (table["period"] == "dev"), "selection_haircut"] = P.expected_max_noise(sd, n_eff)
    table["skill_after_haircut"] = np.where(table["period"] == "dev", table["skill"] - table["selection_haircut"].fillna(0.0), np.nan)
    table.to_csv(path, index=False)


def _account(trades, book):
    return DP.account_series(DP.expand_positions(trades["entry_idx"], trades["exit_idx"]),
                             trades["sym_idx"].to_numpy(np.int64),
                             trades["entry_px"].to_numpy(float), trades["exit_px"].to_numpy(float),
                             book["close_ff"], book["cost"], book["n_rows"], int(book.get("slots", 1)))


def cost_ladder(trades, book, masks, multipliers=(0.5, 1.0, 2.0, 3.0)):
    """Exact account totals/sharpe per period at cost x multiplier for one fill's trades."""
    expansion = DP.expand_positions(trades["entry_idx"], trades["exit_idx"])
    cols = trades["sym_idx"].to_numpy(np.int64)
    entry, exit_ = trades["entry_px"].to_numpy(float), trades["exit_px"].to_numpy(float)
    rows = []
    for mult in multipliers:
        r_day, _ = DP.account_series(expansion, cols, entry, exit_, book["close_ff"], book["cost"] * mult,
                                     book["n_rows"], int(book.get("slots", 1)))
        for name, mask in masks.items():
            r = r_day[mask]
            sd = r.std(ddof=1) if len(r) > 1 else np.nan
            rows.append({"cost_mult": mult, "cost_bps_side": book["cost"] * mult * 1e4, "period": name,
                         "total_log": float(np.log1p(r).sum()),
                         "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan})
    return pd.DataFrame(rows)


def random_timing(trades, book, masks, first_row, last_row_by_col, n_draws=500, seed=DP.SEED):
    """Exposure-matched random-timing null: same instruments, same hold lengths, uniform random entry rows.

    Returns (totals per draw per period, mean per-session log return across draws).
    """
    rng = np.random.default_rng(seed + 31)
    sym = trades["sym_idx"].to_numpy(np.int64)
    hold = (trades["exit_idx"] - trades["entry_idx"]).to_numpy(np.int64)
    n = len(trades)
    n_rows = book["n_rows"]
    totals = np.zeros((n_draws, len(masks)))
    entry_mat, exit_mat, close_ff = book["entry_mat"], book["exit_mat"], book["close_ff"]
    mask_list = [masks[k] for k in masks]
    groups = {s: np.flatnonzero(sym == s) for s in np.unique(sym)}
    for k in range(n_draws):
        keep_sym, keep_e, keep_x = [], [], []
        for s, grp in groups.items():
            placed = []
            for i in grp:
                h = int(hold[i])
                last = last_row_by_col[s]
                for _attempt in range(10):
                    e = int(rng.integers(first_row, max(n_rows - h, first_row + 1)))
                    x = min(e + h, last)
                    if x < e or e > last or not np.isfinite(entry_mat[e, s]) or not np.isfinite(exit_mat[x, s]):
                        continue
                    if any(e < px and x > pe for pe, px in placed):
                        continue
                    placed.append((e, x))
                    keep_sym.append(s)
                    keep_e.append(e)
                    keep_x.append(x)
                    break
        if not keep_e:
            continue
        expansion = DP.expand_positions(np.asarray(keep_e), np.asarray(keep_x))
        r_day, _ = DP.account_series(expansion, np.asarray(keep_sym), entry_mat[keep_e, keep_sym],
                                     exit_mat[keep_x, keep_sym], close_ff, book["cost"], n_rows, int(book.get("slots", 1)))
        log_r = np.log1p(r_day)
        for m, mask in enumerate(mask_list):
            totals[k, m] = log_r[mask].sum()
    return totals


def baseline_stats(name, r_day, masks):
    rows = []
    for period, mask in masks.items():
        r = np.asarray(r_day)[mask]
        log_r = np.log1p(r)
        eq = np.cumprod(1 + r)
        sd = r.std(ddof=1) if len(r) > 1 else np.nan
        rows.append({"baseline": name, "period": period, "total_log": float(log_r.sum()),
                     "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan,
                     "cagr_pct": 100 * float(np.expm1(log_r.sum() * 252 / max(len(r), 1))),
                     "max_dd_pct": 100 * float((eq / np.maximum.accumulate(eq) - 1).min()) if len(r) else np.nan})
    return rows


def vol_matched(bench_r, strat_r, fit_mask):
    """Scale benchmark daily returns so its dev volatility matches the strategy's dev volatility."""
    bs, ss = np.asarray(bench_r)[fit_mask], np.asarray(strat_r)[fit_mask]
    sd_b, sd_s = bs.std(ddof=1), ss.std(ddof=1)
    w = float(sd_s / sd_b) if sd_b and sd_b > 0 else 1.0
    return w * np.asarray(bench_r)


def evaluate_study(ctx, name, title, state, cfg, *, trades, books, masks, universe, pool,
                   caption, report_intro, extra_baselines=None, timing_null=False, notes=""):
    """Full evaluation tail: fills -> partition -> registry -> report/dashboard/comparison. Returns the results frame."""
    fills = fills_of(cfg)
    split_row = state["split_row"]
    dev_mask = masks["dev"]
    logs = []
    for fill in fills:
        r_day, _ = _account(trades[fill], books[fill])
        logs.append(np.log1p(r_day[dev_mask]))
    if len(logs) > 1:
        rho = float(np.corrcoef(logs)[0, 1])
    else:
        rho = 0.0
    n_eff = max(1.0, 1.0 + (len(fills) - 1) * (1.0 - rho))
    bias_note = f"{len(fills)} fill models, daily corr {rho:.2f}"

    by_fill, series = {}, {}
    for fill in fills:
        by_fill[fill], series[fill] = DP.evaluate_fill(trades[fill], books[fill], masks, universe, pool,
                                                      n_eff, cfg, seed=int(cfg.get("seed", DP.SEED)), bias_note=bias_note)
    frame = DP.results_frame(by_fill)
    out = out_dir(ctx, name)
    pre = prefix(cfg)
    for period in ("dev", "test"):
        frame[frame["period"] == period].drop(columns=["bias_src"]).to_csv(out / f"{pre}partition_{period}.csv", index=False)

    primary = fills[0]
    with timing.stage("cost ladder"):
        ladder = cost_ladder(trades[primary], books[primary], masks)
    ladder.insert(0, "fill", primary)
    ladder.to_csv(out / f"{pre}cost_ladder.csv", index=False)

    base_rows = []
    for label, arr in (extra_baselines or {}).items():
        base_rows += baseline_stats(label, arr, masks)
    if timing_null:
        with timing.stage("random timing"):
            close_ff = books[primary]["close_ff"]
            last_by_col = np.where(np.isfinite(close_ff), np.arange(close_ff.shape[0])[:, None], -1).max(0)
            totals = random_timing(trades[primary], books[primary], masks, state["first_row"], last_by_col,
                                   n_draws=int(cfg.get("n_timing_draws", 500)), seed=int(cfg.get("seed", DP.SEED)))
        for m, period in enumerate(masks):
            base_rows.append({"baseline": "random_timing", "period": period,
                              "total_log": float(totals[:, m].mean()),
                              "sharpe": np.nan, "cagr_pct": np.nan, "max_dd_pct": np.nan,
                              "p_total_ge_strategy": float((np.sum(totals[:, m] >= frame[(frame["fill"] == primary) & (frame["period"] == period)]["total_log"].iloc[0]) + 1) / (totals.shape[0] + 1))})
    if base_rows:
        pd.DataFrame(base_rows).to_csv(out / f"{pre}baselines.csv", index=False)

    dev = frame[frame["period"] == "dev"]
    P.log_trials(name, len(fills), per_trial_total_log=dev["total_log"].tolist(), days=int(dev_mask.sum()),
                 notes=f"variant {cfg['variant']}: fill models; variant-level selection is tracked in variants_comparison.csv{'; ' + notes if notes else ''}",
                 source=f"{name} daily partition")

    _daily_returns(out, state, series, cfg, pre)
    _write_report(out, name, title, frame, state, cfg, pre, report_intro, ladder, base_rows)
    dashboard.daily_dashboard(out / f"{pre}summary.png", f"{title}: {cfg['variant']}", caption, state["dates"], series, frame,
                              trades, state["b_day"], state["first_row"], split_row,
                              n_trials_note="Dev bias = selection haircut (fills); the cross-variant haircut is in variants_comparison.csv. Test bias = 0 by construction.")
    update_comparison(out, frame, cfg, masks)

    flags = _flags(frame, primary)
    for f in flags:
        report.section(f"{name}: {f}")
    report.section(f"{name} partition test (log returns, research scale, net of costs)")
    short = {"period": "per", "trades": "n", "total_log": "total", "random_pick_universe": "rnd_univ", "random_pick_pool": "rnd_pool",
             "skill_pct": "skill%", "skill_p": "p"}
    report.table(frame[["fill", "period", "trades", "total_log", "trend", "random_pick_universe", "random_pick_pool", "bias", "skill",
                        "skill_pct", "skill_p", "sharpe", "pf", "win_pct"]].rename(columns=short), floatfmt=".3f")
    return {"results": frame, "flags": flags}


def daily_state(ctx, name, cfg, load_fn, signals_fn, trades_fn, rebuild=False):
    """Run-local study state: panel -> signals -> masks -> per-fill trades (cached on ctx.data)."""
    if name in ctx.data:
        return ctx.data[name]
    with timing.stage("load panel"):
        panel, aux = load_fn(ctx, cfg)
    with timing.stage("signals"):
        sig = signals_fn(panel, aux, cfg)
    n_rows = len(panel["dates"])
    split_row = int(np.searchsorted(panel["dates"].values, pd.Timestamp(cfg["partition_split_date"]).to_datetime64()))
    tradable = sig["tradable"]
    first = int(np.argmax(tradable.any(1))) if tradable.any() else 0
    rows = np.arange(n_rows)
    masks = {"dev": (rows >= first) & (rows < split_row), "test": rows >= split_row}
    ret = np.full(panel["close"].shape, np.nan)
    ret[1:] = panel["close"][1:] / panel["close"][:-1] - 1.0
    live = tradable & np.isfinite(ret)
    b_day = np.where(live, np.nan_to_num(ret), 0.0).sum(1) / np.maximum(live.sum(1), 1)
    trades = {}
    with timing.stage("trades"):
        for fill in fills_of(cfg):
            trades[fill] = trades_fn(panel, sig, cfg, fill)
    state = {"panel": panel, "sig": sig, "b_day": b_day,
             "close_ff": pd.DataFrame(panel["close"]).ffill().to_numpy(),
             "masks": masks, "trades": trades, "first_row": first, "split_row": split_row, "dates": panel["dates"], "aux": aux}
    ctx.data[name] = state
    return state


def period_stats(t):
    r = t["ret"].to_numpy(float)
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    return {"trades": len(t), "win_pct": 100 * float((r > 0).mean()) if len(r) else np.nan,
            "mean_ret_pct": 100 * float(r.mean()) if len(r) else np.nan,
            "pf": float(wins / losses) if losses > 0 else np.nan,
            "avg_hold": float(t["hold"].mean()) if len(t) else np.nan}


def trades_table(state, cfg):
    split = pd.Timestamp(cfg["partition_split_date"])
    frames = []
    for fill, trades in state["trades"].items():
        frame = trades.copy()
        frame["day"] = frame["entry_date"]
        frame["period"] = np.where(frame["entry_date"] >= split, "test", "dev")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class DailyStudy(Strategy):
    """Adapter shell shared by the catalog daily-book studies (see s001_rsi2_connors)."""

    TITLE = ""
    timing_null = False

    def load(self, ctx, cfg):
        """-> (panel, aux dict)."""
        raise NotImplementedError

    def signals(self, panel, aux, cfg):
        """-> dict with tradable, pool + strategy matrices."""
        raise NotImplementedError

    def trades_for(self, panel, sig, cfg, fill):
        raise NotImplementedError

    def book(self, state, cfg, fill):
        panel = state["panel"]
        mat = panel["open"] if fill == "next_open" else panel["close"]
        return {"close_ff": state["close_ff"], "entry_mat": mat, "exit_mat": mat,
                "cost": float(cfg["cost_bps_per_side"]) / 1e4, "n_rows": len(panel["dates"]),
                "b_day": state["b_day"], "slots": int(cfg["book_slots"])}

    def rank_col(self):
        return "ret_gross"

    def intro(self, cfg, state):
        return []

    def baselines(self, cfg, state, strat_r):
        return {}

    def caption(self, cfg, state):
        return f"variant={cfg['variant']} | {cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | {cfg['data_dir']} {state['dates'][0].date()}..{state['dates'][-1].date()}"

    def extra_summary(self, ctx, table, out):
        return None

    def _state(self, ctx, rebuild=False):
        return daily_state(ctx, self.name, ctx.cfg, self.load, self.signals, self.trades_for, rebuild)

    def build_trades(self, ctx):
        state = self._state(ctx, bool(ctx.flags.get("rebuild_cache")))
        out = out_dir(ctx, self.name)
        for fill, trades in state["trades"].items():
            trades.drop(columns=["sym_idx", "signal_idx", "entry_idx", "exit_idx"]).to_csv(
                out / f"{prefix(ctx.cfg)}trades_{fill}.csv", index=False)
        return trades_table(state, ctx.cfg)

    def build_table(self, ctx, rebuild):
        return trades_table(self._state(ctx, rebuild or bool(ctx.flags.get("rebuild_cache"))), ctx.cfg)

    def setups(self, ctx, table):
        return {fill: Setup(label=fill, select=lambda frame, fill=fill: frame[frame["fill"] == fill],
                            rank_col=self.rank_col()) for fill in fills_of(ctx.cfg)}

    def rank_experiments(self, ctx, table):
        out = out_dir(ctx, self.name)
        summary = table.groupby(["fill", "period"]).apply(lambda t: pd.Series(period_stats(t)), include_groups=False).reset_index()
        summary.to_csv(out / f"{prefix(ctx.cfg)}trade_summary.csv", index=False)
        report.section(f"{self.name} trade summary (net of costs, research scale)")
        report.table(summary, floatfmt=".3f")
        self.extra_summary(ctx, table, out)
        return summary

    def charts(self, ctx, table, setups):
        return None

    def evaluate(self, ctx, table, setups):
        state, cfg = self._state(ctx), ctx.cfg
        fills = fills_of(cfg)
        books = {fill: self.book(state, cfg, fill) for fill in fills}
        strat_r = _account(state["trades"][fills[0]], books[fills[0]])[0]
        return evaluate_study(ctx, self.name, self.TITLE, state, cfg,
                              trades=state["trades"], books=books, masks=state["masks"],
                              universe=state["sig"]["tradable"], pool=state["sig"]["pool"],
                              caption=self.caption(cfg, state), report_intro=self.intro(cfg, state),
                              extra_baselines=self.baselines(cfg, state, strat_r), timing_null=self.timing_null)


def _flags(frame, primary):
    out = []
    p = frame[frame["fill"] == primary].set_index("period")
    if "dev" in p.index and p.loc["dev", "trades"] < MIN_DEV_TRADES:
        out.append(f"insufficient evidence: {int(p.loc['dev', 'trades'])} dev trades < {MIN_DEV_TRADES}")
    if "test" in p.index and p.loc["test", "trades"] < MIN_TEST_TRADES:
        out.append(f"insufficient evidence: {int(p.loc['test', 'trades'])} test trades < {MIN_TEST_TRADES}")
    if {"dev", "test"}.issubset(p.index) and np.isfinite(p.loc["dev", "skill"]) and np.isfinite(p.loc["test", "skill"]):
        if np.sign(p.loc["dev", "skill"]) != np.sign(p.loc["test", "skill"]) and p.loc["dev", "skill"] != 0:
            out.append("DECAYED: test skill sign differs from dev")
    return out


def _daily_returns(out, state, series, cfg, pre):
    dates = state["dates"]
    data = {"date": dates, "b_day": state["b_day"]}
    for fill, s in series.items():
        data[f"{fill}_r_day"], data[f"{fill}_positions"] = s["r_day"], s["count"]
    pd.DataFrame(data).iloc[state["first_row"]:].to_csv(out / f"{pre}daily_returns.csv", index=False)


def _write_report(out, name, title, frame, state, cfg, pre, report_intro, ladder, base_rows):
    dates = state["dates"]
    first, split = dates[state["first_row"]], dates[min(state["split_row"], len(dates) - 1)]
    lines = [f"# {title}: partition report", ""]
    if cfg["variant"] != "baseline":
        lines += [f"- Variant: `{cfg['variant']}` overlay {(cfg.get('variants') or {}).get(cfg['variant'])}", ""]
    lines += [f"- Data: `{cfg['data_dir']}`; sessions {dates[0].date()} .. {dates[-1].date()}; evaluation starts {first.date()}.",
              f"- Split: dev < {split.date()} <= test (locked; parameters are the catalog baseline or registered variants).",
              "- Trend = exposure-weighted equal-weight tradable-universe return; skill = total - trend - bias; units are log returns."]
    lines += report_intro
    lines += [""]
    primary = frame["fill"].iloc[0]
    for f in _flags(frame, primary):
        lines += [f"**{f}**", ""]
    for period in ("dev", "test"):
        lines += [f"## {period}", "", md_table(frame[frame["period"] == period], ["fill"] + REPORT_COLS), ""]
        lines += [f"Bias source ({fill}): {src}" for fill, src in frame[frame["period"] == period][["fill", "bias_src"]].itertuples(index=False)] + [""]
    lines += ["## Cost ladder (primary fill)", "", md_table(ladder, ["fill", "cost_mult", "cost_bps_side", "period", "total_log", "sharpe"]), ""]
    if base_rows:
        lines += ["## Baselines", "", md_table(pd.DataFrame(base_rows), list(pd.DataFrame(base_rows).columns)), ""]
    (out / f"{pre}partition_report.md").write_text("\n".join(lines), encoding="utf-8")
