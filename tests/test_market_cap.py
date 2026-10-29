import pandas as pd

import src.data.market_cap as mc
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


def test_stale_simfin_blanks_and_uses_massive_shares_times_price(tmp_path, monkeypatch):
    simfin = tmp_path / "us-shareprices-daily.csv"
    pd.DataFrame({"Ticker": ["A", "B"], "Date": ["2025-10-19", "2025-10-01"]}).to_csv(simfin, index=False)
    monkeypatch.setattr(mc, "SIMFIN_PRICES", simfin)
    enrich = tmp_path / "massive.parquet"
    pd.DataFrame({
        "symbol": ["A", "B", "B"], "day": pd.to_datetime(["2025-10-20"] * 3),
        "market_cap": [None, 999.0, 777.0], "weighted_shares": [20.0, 10.0, 99.0],
        "share_class_shares": [None, 10.0, 99.0], "validation_only": [False, False, True],
    }).to_parquet(enrich, index=False)
    t = pd.DataFrame({"symbol": ["A", "B"], "day": pd.to_datetime(["2025-10-20"] * 2),
                      "mcap_pit": [100.0, 200.0], "shares_pit": [10.0, 20.0],
                      "price": [6.0, 7.0]})
    out = merge_market_cap(t, enrich)
    assert out["mcap_stale"].tolist() == [False, True]
    assert out["mcap_src"].tolist() == ["simfin", "massive_sharesxprice"]
    assert out["mcap_final"].tolist() == [100.0, 70.0]
    assert out["mcap_pit"].iloc[0] == 100.0 and pd.isna(out["mcap_pit"].iloc[1])
