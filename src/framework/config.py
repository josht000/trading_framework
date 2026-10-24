"""Configuration loading with YAML < environment < CLI precedence."""

from pathlib import Path
import os

import yaml


DEFAULTS = {
    "results_tag": "",
    "lookback_bars": 0,
    "max_positions_per_day": 5,
    "exit_variants": "v2",
    "processes": None,
    "data_cache": True,
    "ladder_cache": True,
    "size_usd": 1250.0,
    "participation": 0.10,
    "slip_rth": 0.0015,
    "slip_ext": 0.004,
    "min_fill": 0.25,
    "test_months": 4,
    "backtest_set": "dev",
    "backtest_start_date": "2024-10-18",
    "backtest_end_date": "2100-01-01",
    "backtest_commission": 0.001,
    "resistance_method": "atr_pivot",
    "require_news": False,
    "require_market_cap": False,
    "news_lookback_days": 3,
    "min_news_articles": 1,
    "min_breakout_atr": 0.0,
    "backtest_mode": "pit",
    "backtest_initial_cash": 100000.0,
    "backtest_symbols": "SOFI",
    "partition_split_date": "2026-06-16",
    "iters": 700,
}


def _coerce(value, template):
    if template is None:
        if isinstance(value, str) and value.lower() in {"auto", "none", "null", ""}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(template, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return str(value)


def load_config(strategy, path=None, env=None, cli=None):
    """Load one strategy config, applying environment and CLI overrides."""
    env = os.environ if env is None else env
    cli = {} if cli is None else {k: v for k, v in cli.items() if v is not None}
    path = Path(path or Path("configs") / f"{strategy}.yaml")
    yaml_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = dict(DEFAULTS)
    cfg.update(yaml_cfg)
    for key, default in DEFAULTS.items():
        env_key = key.upper()
        if env_key in env:
            cfg[key] = _coerce(env[env_key], cfg.get(key, default))
    for key, value in cli.items():
        if key in cfg:
            cfg[key] = _coerce(value, cfg.get(key))
        else:
            cfg[key] = value
    cfg["strategy"] = strategy
    return cfg


def env_overrides(cfg):
    """Return config values in the environment names consumed by legacy code."""
    out = {}
    for key, value in cfg.items():
        if key in {"strategy", "partition_split_date", "iters"} or value is None:
            continue
        env_key = key.upper()
        if isinstance(value, bool):
            out[env_key] = "1" if value else "0"
        else:
            out[env_key] = str(value)
    return out
