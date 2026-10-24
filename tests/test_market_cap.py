import pandas as pd

from src.data.market_cap import merge_market_cap


def test_market_cap_merge_priority_sources_and_suspect(tmp_path):
    enrichment = tmp_path / "massive.parquet"
    e = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "day": pd.to_datetime(["2025-01-02"] * 3),
        "market_cap": [100.0, 200.0, 300.0],
        "weighted_shares": [10.0, 1.0, 1.0],
        "share_class_shares": [10.0, 10.0, 1.0],
        "validation_only": [False, False, False],
    })
    e = pd.concat([e, pd.DataFrame({
        "symbol": ["B"], "day": pd.to_datetime(["2025-01-02"]),
        "market_cap": [999.0], "weighted_shares": [1.0],
        "share_class_shares": [1.0], "validation_only": [True],
    })], ignore_index=True)
    e.to_parquet(enrichment, index=False)
    trades = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "day": pd.to_datetime(["2025-01-02"] * 4),
        "mcap_pit": [999.0, None, None, None],
    })
    out = merge_market_cap(trades, enrichment)
    assert out["mcap_final"].iloc[:3].tolist() == [999.0, 200.0, 300.0]
    assert pd.isna(out["mcap_final"].iloc[3])
    assert out["mcap_src"].tolist() == ["simfin", "massive", "massive", "missing"]
    assert out["mcap_suspect"].tolist() == [False, True, False, False]
    assert out["mcap_pit"].tolist()[0] == 999.0
