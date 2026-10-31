"""S003 RSI(4) 25/75 (Connors): same signal machinery as S001, different parameters."""

from src.strategies.s001_rsi2_connors.signals import compute_signals as _compute


def compute_signals(panel, cfg):
    return _compute(panel, cfg)


def build_trades(panel, sig, cfg, fill):
    from src.strategies._engines import swing_trades

    return swing_trades(panel, sig["signal"], sig["exit_now"], cfg, fill, extras={"rsi4": sig["rsi"]})
