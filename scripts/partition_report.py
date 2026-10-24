"""Partition test (total = trend + skill + bias) for the ranked strategies or any trades CSV.

  python scripts/partition_report.py                                   # base, S1, H3, H2, R3, S1/H3 + tuned score; dev and locked test
  python scripts/partition_report.py --names base S1 R3                # subset (names: summary-table labels' first word, 'S1+tuned', R2/R3/RIDGE)
  python scripts/partition_report.py --csv results/pit_trades_atr_pivot_lb13_pm_target25_v2.csv --keys above_sma200 "iwm_day>=0" --name mine

Writes reports/partition/partition_<name>_<period>.csv + .md and reports/partition/figures/partition_<name>_<period>.png.
Definitions: see src/backtest/partition.py. Test-period bias is 0 by construction; dev bias is estimated on dev only.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd

import rank_final as RF
from rank_test import apply_keys, build_table
from src.backtest import partition as P
from src.backtest import kfold as K
from src.utils import timing

OUT = Path("reports") / "partition"
DEFAULT = ["base", "S1", "H3", "H2", "R3", "S1+tuned", "H3+tuned", "F30-M200-H20"]
LABEL = {"base": "base (rel. volume rank, no hygiene)", "S1": "S1 (rel. volume rank, no hygiene)", "H3": "H3 (rel. volume rank, no hygiene)",
         "H2": "H2 (rel. volume rank, no hygiene)", "base+tuned": "base + hygiene, tuned score", "S1+tuned": "S1 + hygiene, tuned score",
         "H3+tuned": "H3 + hygiene, tuned score",
         "F30-M200-H20": "F30-M200-H20 (rel. volume rank, no hygiene)",
         "H1": "H1 (rel. volume rank, no hygiene)",
         "F30-H20": "F30-H20 (rel. volume rank, no hygiene)",
         "M200": "M200 (rel. volume rank, no hygiene)", "M200-H20": "M200-H20 (rel. volume rank, no hygiene)"}


def compute(T, names):
    cases = RF.summary_cases()
    defs = None
    res = {}
    for n in names:
        if n in LABEL:
            r = RF.partition_cases(T, [LABEL[n]], cases)
            for per in ("dev", "test"):
                res[(n, per)] = r[(LABEL[n], per)]
                sel, days, rank, universe = RF.case_selection(T, LABEL[n], per, cases, return_pools=True)
                kr, ks = K.kfold_stability(sel, days, rank=rank, pool=sel, label=f"{n}|{per}")
                K.write_kfold(kr, ks, OUT, n, per)
        else:
            defs, prep, ridge = defs or RF.strategies(T)
            for per in ("dev", "test"):
                s, rank = RF.selected(T, n, per, defs, prep, ridge)
                days = pd.bdate_range(T[T["period"] == per]["day"].min(), T[T["period"] == per]["day"].max())
                base = T[T["period"] == per]
                universe = base[RF.hygiene(base, 1)]
                res[(n, per)] = P.run_partition(s, days, rank, pool=s, universe=universe, areas=["signal", "exit", "entry_filters", "ranking"] if n in ("R2", "R3", "RIDGE", "S1+R2") else ["signal", "exit"],
                                                oos=per == "test")
                kr, ks = K.kfold_stability(s, days, rank=rank, pool=s, label=f"{n}|{per}", tuned=n.endswith("+tuned"))
                K.write_kfold(kr, ks, OUT, n.replace("+", "_"), per)
    return res


def write(res, name_of=lambda k: k, tag=""):
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for per in ("dev", "test"):
        with timing.stage("write/plots"):
            keys = [k for k in res if k[1] == per]
            rows = [{"setup": k[0], **P.flat(res[k])} for k in keys]
            df = pd.DataFrame(rows).set_index("setup")
            for k, row in zip(keys, rows):
                pd.DataFrame([row]).to_csv(OUT / f"partition_{k[0].replace('+', '_').replace(' ', '_')}_{per}.csv", index=False)
            df.to_csv(OUT / f"partition_all{tag}_{per}.csv")
            cols = ["total_log", "trend_iwm", "trend_spy", "trend_beta_iwm", "beta_iwm", "beta_t_iwm",
                    "random_pick_universe", "random_pick_pool", "random_p_universe", "random_p_pool",
                    "bias_A", "bias_B", "bias", "bias_signed", "bias_sd", "bias_N_eff", "bias_unreliable",
                    "skill", "skill_vs_universe", "skill_vs_pool", "skill_pct", "skill_p",
                    "days_no_choice_pool", "days_no_choice_universe", "skill_ex5", "cost_est"]
            show = df[cols].copy()
            for c in ("total_log", "trend_iwm", "trend_spy", "trend_beta_iwm", "random_pick_universe",
                      "random_pick_pool", "bias_A", "bias_B", "bias", "bias_signed", "bias_sd", "skill",
                      "skill_vs_universe", "skill_vs_pool", "skill_pct", "skill_ex5", "cost_est"):
                show[c] = 100 * show[c].astype(float)
            md = [f"## Partition test, {per} (log-return percent points; skill_pct in %)", "",
                  "Skill is a residual: its interpretation depends on the trend baseline and on whether the random comparison is the universe or strategy-filtered pool. Bias-B is a pooled comparable-trial estimate; unreliable estimates exceed observed total.", "",
                  show.to_markdown(floatfmt=".2f"), "", "Bias source per row:", ""] + [f"- {k}: {v}" for k, v in df["bias_src"].items()]
            (OUT / f"partition_all{tag}_{per}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
            main = [k for k in keys if k[0] in ("base", "S1", "H3", "R3")]
            if main:
                P.plot_partition([res[k] for k in main], figs / f"partition_all{tag}_{per}.png", [f"{k[0]} ({per})" for k in main])
            for k in keys:
                P.plot_partition([res[k]], figs / f"partition_{k[0].replace('+', '_')}_{per}.png", [f"{k[0]} ({per})"])
        with timing.stage("print_summary"):
            P.print_summary([(k[0], res[k]) for k in keys], f"Partition test, {per}")


def main():
    timing.install("partition_report")
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=DEFAULT)
    ap.add_argument("--csv")
    ap.add_argument("--keys", nargs="*", default=["all"])
    ap.add_argument("--name", default="custom")
    ap.add_argument("--rank", default="rel_vol")
    a = ap.parse_args()
    if a.csv:
        from src.backtest.market_regime import attach_regime
        with timing.stage("load CSV"):
            t = pd.read_csv(a.csv)
            t["entry_time"] = t["entry_time"].astype(str)
            t["day"] = pd.to_datetime(t["entry_time"].str[:10])
            t = attach_regime(t)
            sel = apply_keys(t, a.keys)
        with timing.stage("compute"):
            res = {}
            for per, m in (("dev", sel["day"] < "2026-06-16"), ("test", sel["day"] >= "2026-06-16")):
                x = sel[m]
                if len(x):
                    days = pd.bdate_range(x["day"].min(), x["day"].max())
                    res[(a.name, per)] = P.run_partition(x, days, a.rank, pool=x, areas=["signal", "exit"], oos=per == "test")
        write(res)
        return
    with timing.stage("build_table"):
        T = build_table()
    with timing.stage("compute"):
        res = compute(T, a.names)
    write(res, tag="" if a.names == DEFAULT else "_subset")


if __name__ == "__main__":
    main()
