"""S075 Stock-vs-Sector Residual Reversal: framework adapter (daily bars, research scale, long-only)."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import partition as P
from src.data.daily_panel import load_panel, load_sector_map
from src.framework import daily_partition as DP
from src.framework import dashboard, report
from src.framework.base import Setup, Strategy
from src.framework.registry import register
from src.utils import timing

from .signals import build_trades, compute_signals

NAME = "s075_residual_reversal"
SUMMARY_COLS = ["fill", "period", "trades", "win_pct", "mean_ret_pct", "median_ret_pct", "pf", "avg_hold"]
REPORT_COLS = ["trades", "days", "total_log", "trend", "trend_beta", "beta", "random_pick_universe", "random_pick_pool",
               "random_p_universe", "random_p_pool", "bias", "skill", "skill_vs_universe", "skill_vs_pool", "skill_pct",
               "skill_lo", "skill_hi", "skill_p", "skill_ex5", "win_pct", "mean_ret_pct", "pf", "cagr_pct", "sharpe",
               "max_dd_pct", "avg_hold", "avg_positions", "avg_fraction_invested", "turnover_annual"]


def _out_dir(ctx):
    """Every output of this study lives in results/<name>/."""
    out = Path(ctx.paths.get("results", "results")) / NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prefix(cfg):
    """Variant runs write `<variant>__<file>` next to the baseline files (baseline has no prefix)."""
    return f"{cfg['results_tag']}__" if cfg.get("results_tag") else ""


def _fills(cfg):
    return [x.strip() for x in str(cfg["fills"]).split(",") if x.strip()]


def _state(ctx, rebuild=False):
    """Load the panel, compute signals and trades once per run (kept in ctx.data)."""
    if NAME in ctx.data:
        return ctx.data[NAME]
    cfg = ctx.cfg
    if not cfg.get("data_dir"):
        raise ValueError("no data dir: pass --data-dir or set data_dir in the strategy config")
    with timing.stage("sector map"):
        sector_map = load_sector_map()
    panel = load_panel(cfg["data_dir"], symbols=sector_map.index, rebuild=rebuild)
    with timing.stage("signals"):
        sig = compute_signals(panel, sector_map, cfg)
    close = panel["close"]
    n_rows = len(panel["dates"])
    ret, tradable = sig["ret"], sig["tradable"]
    with timing.stage("benchmark"):
        b_day = sig["b_day"]
        close_ff = pd.DataFrame(close).ffill().to_numpy()
    split = pd.Timestamp(cfg["partition_split_date"])
    first = int(np.argmax(tradable.any(1)))
    split_row = int(np.searchsorted(panel["dates"].values, split.to_datetime64()))
    rows = np.arange(n_rows)
    masks = {"dev": (rows >= first) & (rows < split_row), "test": rows >= split_row}
    trades = {}
    with timing.stage("trades"):
        for fill in _fills(cfg):
            trades[fill] = build_trades(panel, sig, cfg, fill)
    state = {"panel": panel, "sig": sig, "b_day": b_day, "close_ff": close_ff, "masks": masks, "trades": trades,
             "first_row": first, "split_row": split_row}
    ctx.data[NAME] = state
    return state


def _table(state, cfg):
    split = pd.Timestamp(cfg["partition_split_date"])
    frames = []
    for fill, trades in state["trades"].items():
        frame = trades.copy()
        frame["day"] = frame["entry_date"]
        frame["period"] = np.where(frame["entry_date"] >= split, "test", "dev")
        frame["neg_resid5"] = -frame["resid5"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _book(state, cfg, fill):
    panel = state["panel"]
    mat = panel["open"] if fill == "next_open" else panel["close"]
    return {"close_ff": state["close_ff"], "entry_mat": mat, "exit_mat": mat, "cost": float(cfg["cost_bps_per_side"]) / 1e4,
            "n_rows": len(panel["dates"]), "b_day": state["b_day"], "slots": int(cfg["book_slots"])}


def _fill_account(state, cfg, fill):
    trades, book = state["trades"][fill], _book(state, cfg, fill)
    return DP.account_series(DP.expand_positions(trades["entry_idx"], trades["exit_idx"]), trades["sym_idx"].to_numpy(np.int64),
                             trades["entry_px"].to_numpy(float), trades["exit_px"].to_numpy(float), book["close_ff"],
                             book["cost"], book["n_rows"], book["slots"])


def _md_table(frame, cols):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("n/a" if isinstance(v, (float, np.floating)) else str(v))

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for _, row in frame.iterrows()]
    return "\n".join(lines)


def update_comparison(out, frame, cfg, state):
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
    dev_mask = state["masks"]["dev"]
    daily = {}
    for f in out.glob("*daily_returns.csv"):
        name = f.name[: -len("daily_returns.csv")].rstrip("_") or "baseline"
        d = pd.read_csv(f, parse_dates=["date"])
        daily[name] = d[d["date"] < pd.Timestamp(cfg["partition_split_date"])]
    table["selection_haircut"] = np.nan
    for fill in _fills(cfg):
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


@register(NAME)
class S075ResidualReversal(Strategy):
    name = NAME
    needs = ("daily",)
    exposure_window = {"entry": "close of signal session (or next open)", "exit": "residual recovery or 10 sessions"}

    def build_trades(self, ctx):
        state = _state(ctx, bool(ctx.flags.get("rebuild_cache")))
        results_dir = _out_dir(ctx)
        for fill, trades in state["trades"].items():
            trades.drop(columns=["sym_idx", "signal_idx", "entry_idx", "exit_idx"]).to_csv(results_dir / f"{_prefix(ctx.cfg)}trades_{fill}.csv", index=False)
        return _table(state, ctx.cfg)

    def build_table(self, ctx, rebuild):
        return _table(_state(ctx, rebuild or bool(ctx.flags.get("rebuild_cache"))), ctx.cfg)

    def setups(self, ctx, table):
        return {fill: Setup(label=fill, select=lambda frame, fill=fill: frame[frame["fill"] == fill], rank_col="neg_resid5")
                for fill in _fills(ctx.cfg)}

    def rank_experiments(self, ctx, table):
        rows = []
        for fill in _fills(ctx.cfg):
            for period in ("dev", "test"):
                t = table[(table["fill"] == fill) & (table["period"] == period)]
                r = t["ret"].to_numpy(float)
                wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
                rows.append({"fill": fill, "period": period, "trades": len(t), "win_pct": 100 * float((r > 0).mean()) if len(r) else np.nan,
                             "mean_ret_pct": 100 * float(r.mean()) if len(r) else np.nan,
                             "median_ret_pct": 100 * float(np.median(r)) if len(r) else np.nan,
                             "pf": float(wins / losses) if losses > 0 else np.nan,
                             "avg_hold": float(t["hold"].mean()) if len(t) else np.nan,
                             "gross_mean_ret_pct": 100 * float(t["ret_gross"].mean()) if len(t) else np.nan,
                             **{f"exit_{k}": int(v) for k, v in t["exit_reason"].value_counts().items()}})
        summary = pd.DataFrame(rows)
        out = _out_dir(ctx)
        summary.to_csv(out / f"{_prefix(ctx.cfg)}trade_summary.csv", index=False)
        report.section("S075 trade summary (net of costs, research scale)")
        report.table(summary[SUMMARY_COLS].rename(columns={"mean_ret_pct": "mean%", "median_ret_pct": "med%", "win_pct": "win%", "avg_hold": "hold"}), floatfmt=".3f")
        return summary

    def charts(self, ctx, table, setups):
        """The single dashboard PNG is written by evaluate() (it needs the partition results)."""
        return None

    def evaluate(self, ctx, table, setups):
        state, cfg = _state(ctx), ctx.cfg
        fills = _fills(cfg)
        panel, sig = state["panel"], state["sig"]
        dev = state["masks"]["dev"]
        logs = [np.log1p(_fill_account(state, cfg, fill)[0][dev]) for fill in fills]
        rho = float(np.corrcoef(logs)[0, 1]) if len(logs) > 1 else 0.0
        n_eff = max(1.0, 1.0 + (len(fills) - 1) * (1.0 - rho))
        note = f"{len(fills)} fill models, daily corr {rho:.2f}"
        by_fill, series = {}, {}
        for fill in fills:
            by_fill[fill], series[fill] = DP.evaluate_fill(state["trades"][fill], _book(state, cfg, fill), state["masks"], sig["tradable"], sig["pool"],
                                                n_eff, cfg, seed=int(cfg.get("seed", DP.SEED)), bias_note=note)
        frame = DP.results_frame(by_fill)
        out = _out_dir(ctx)
        pre = _prefix(cfg)
        for period in ("dev", "test"):
            frame[frame["period"] == period].drop(columns=["bias_src"]).to_csv(out / f"{pre}partition_{period}.csv", index=False)
        dev = frame[frame["period"] == "dev"]
        P.log_trials(NAME, len(fills), per_trial_total_log=dev["total_log"].tolist(), days=int(state["masks"]["dev"].sum()),
                     notes=f"variant {cfg['variant']}: fill models; variant-level selection is tracked in variants_comparison.csv", source="s075 daily partition")
        self._write_report(out, frame, state, cfg)
        self._daily_returns(out, state, series, cfg)
        self._dashboard(out, frame, state, series, cfg)
        update_comparison(out, frame, cfg, state)
        report.section("S075 partition test (log returns, research scale, net of costs)")
        short = {"period": "per", "trades": "n", "total_log": "total", "random_pick_universe": "rnd_univ", "random_pick_pool": "rnd_pool",
                 "skill_pct": "skill%", "skill_p": "p"}
        report.table(frame[["fill", "period", "trades", "total_log", "trend", "random_pick_universe", "random_pick_pool", "bias", "skill",
                            "skill_pct", "skill_p", "sharpe"]].rename(columns=short), floatfmt=".3f")
        return {"results": frame, "setups": setups}

    @staticmethod
    def _daily_returns(out, state, series, cfg):
        dates = state["panel"]["dates"]
        data = {"date": dates, "b_day": state["b_day"]}
        for fill, s in series.items():
            data[f"{fill}_r_day"], data[f"{fill}_positions"] = s["r_day"], s["count"]
        pd.DataFrame(data).iloc[state["first_row"]:].to_csv(out / f"{_prefix(cfg)}daily_returns.csv", index=False)

    @staticmethod
    def _dashboard(out, frame, state, series, cfg):
        dates = state["panel"]["dates"]
        overlay = (cfg.get("variants") or {}).get(cfg["variant"], {}) if cfg["variant"] != "baseline" else {}
        caption = (f"variant={cfg['variant']}  overlay={overlay or 'none (catalog baseline)'}\n"
                   f"universe top {cfg['universe_top_n']} liquid (peers top {cfg['peer_top_n']}, SimFin industry) | signal {cfg.get('signal_mode', 'raw')} "
                   f"{cfg['resid_window']}d residual, decile {cfg['decile']}, SMA{cfg['sma_window']}, skip {cfg.get('skip_days', 0)} | exit {cfg.get('exit_mode', 'resid')} "
                   f"cap {cfg['max_hold']} stop {cfg.get('stop_loss', 0)} target {cfg.get('take_profit', 0)} | regime {cfg.get('regime_mode', 'none')} | "
                   f"{cfg['cost_bps_per_side']} bps/side | slots {cfg['book_slots']} | data {cfg['data_dir']} {dates[0].date()}..{dates[-1].date()}")
        dashboard.daily_dashboard(out / f"{_prefix(cfg)}summary.png", f"S075 Stock-vs-Sector Residual Reversal: {cfg['variant']}", caption, dates, series, frame,
                                  state["trades"], state["b_day"], state["first_row"], state["split_row"],
                                  n_trials_note="Dev bias = selection haircut (fills); the cross-variant haircut is in variants_comparison.csv. Test bias = 0 by construction.")

    @staticmethod
    def _write_report(out, frame, state, cfg):
        dates = state["panel"]["dates"]
        first, split = dates[state["first_row"]], dates[min(state["split_row"], len(dates) - 1)]
        lines = [
            "# S075 Stock-vs-Sector Residual Reversal: partition report", ""] + ([f"- Variant: `{cfg['variant']}` overlay {(cfg.get('variants') or {}).get(cfg['variant'])}", ""] if cfg["variant"] != "baseline" else []) + [
            f"- Data: `{cfg['data_dir']}`; sessions {dates[0].date()} .. {dates[-1].date()}; evaluation starts {first.date()}.",
            f"- Split: dev < {split.date()} <= test (locked; no parameters tuned).",
            f"- Universe: top {cfg['universe_top_n']} by 60d median dollar volume (price >= {cfg['min_price']}, >= {cfg['min_history']} bars); "
            f"peers: top {cfg['peer_top_n']} pool, same SimFin industry (>= {cfg['min_peers']} peers, else sector).",
            f"- Signal: {cfg['resid_window']}d residual vs peers (beta {cfg['beta_window']}d through t-1), bottom decile and close > SMA{cfg['sma_window']}; "
            f"exit when residual > 0 or {cfg['max_hold']} sessions. Long-only, each position weighs 1/max(open, {cfg['book_slots']}) of capital (rest cash), research scale, {cfg['cost_bps_per_side']} bps per side.",
            "- `close` fill is an upper bound (signal and fill at the same close); `next_open` is the robustness variant.",
            "- Trend = exposure-weighted equal-weight tradable-universe return; skill = total - trend - bias; units are log returns.",
            "- Limitations: SimFin sector map is a current snapshot covering only mapped tickers; no earnings-window filter; no delisting haircut.", ""]
        for period in ("dev", "test"):
            lines += [f"## {period}", "", _md_table(frame[frame["period"] == period], ["fill"] + REPORT_COLS), ""]
            lines += [f"Bias source ({fill}): {src}" for fill, src in frame[frame["period"] == period][["fill", "bias_src"]].itertuples(index=False)] + [""]
        (out / f"{_prefix(cfg)}partition_report.md").write_text("\n".join(lines), encoding="utf-8")
