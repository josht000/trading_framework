import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "live"


@dataclass(frozen=True)
class SizingConfig:
    sleeve_start: float = 5000.0
    max_positions: int = 5
    position_frac: float = 0.20
    participation: float = 0.10
    min_fill: float = 0.25


@dataclass(frozen=True)
class ExitConfig:
    ah_window_min: int = 45
    ah_stop: float = 0.12
    ah_target: float = 0.50
    pm_target: float = 0.25


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    threshold: float
    weights: dict
    headroom_min: float = 0.20
    above_sma200: bool = True
    hygiene_level: int = 1
    entry_time: str = "15:55"
    sizing: SizingConfig = field(default_factory=SizingConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)


@dataclass(frozen=True)
class RiskConfig:
    max_positions: int = 5
    max_position_frac: float = 0.20
    max_gross_frac: float = 1.0
    max_limit_deviation: float = 0.02
    max_spread_frac: float = 0.03
    max_quote_age_sec: int = 60
    daily_loss_halt: float = -0.05
    drawdown_halt: float = -0.15
    max_order_rejects: int = 5
    kill_file: str = "data/live/KILL"


def _yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_strategy(name="overnight_h3") -> StrategyConfig:
    d = _yaml(CONFIG_DIR / f"{name}.yaml")
    return StrategyConfig(
        name=d["name"], threshold=d["score"]["threshold"], weights=d["score"]["weights"],
        headroom_min=d["filters"]["headroom_min"], above_sma200=d["filters"]["above_sma200"],
        hygiene_level=d["hygiene_level"], entry_time=d["entry"]["time"],
        sizing=SizingConfig(**d["sizing"]), exits=ExitConfig(**d["exits"]),
    )


def load_risk() -> RiskConfig:
    return RiskConfig(**_yaml(CONFIG_DIR / "risk.yaml"))


@dataclass(frozen=True)
class AlpacaSettings:
    key: str
    secret: str
    paper: bool = True


def load_alpaca() -> AlpacaSettings:
    load_dotenv(ROOT / ".env")
    url = os.getenv("ALPACA_BASE_URL", "")
    if "paper" not in url or os.getenv("ALPACA_PAPER", "true").lower() != "true":
        raise RuntimeError(f"refusing to load Alpaca settings: not a paper configuration ({url!r})")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing")
    return AlpacaSettings(key, secret)
