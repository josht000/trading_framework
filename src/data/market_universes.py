"""Shared symbol lists for daily-history fetch and universe builders."""

ETF_SYMBOLS = (
    "SPY", "QQQ", "IWM", "DIA", "RSP", "IVV", "VOO", "TQQQ", "SQQQ",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "TLT", "IEF", "SHY", "BIL", "AGG", "LQD", "HYG",
    "GDX", "GDXJ", "EFA", "ACWX", "EEM", "EWJ", "XHB",
)

ETF_LIKE_SYMBOLS = frozenset(ETF_SYMBOLS) | frozenset({
    "GLD", "SLV", "USO", "CPER", "GSG", "EMB", "IYR", "TIP", "UUP",
    "DBC", "VNQ",
})
