import numpy as np
import pandas as pd

from src.features import (
    AUTOREGRESSIVE_FEATURES,
    TARGET_COLUMN,
    build_modelling_table,
    build_latest_forecast_row,
)
from src.train import chronological_split, build_selected_model


def make_master(n=24):
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    price = np.arange(n, dtype=float) * 1000 + 200000
    return pd.DataFrame({
        "date": dates,
        "average_house_price_gbp": price,
        "sales_volume": np.arange(n) + 50000,
        "average_weekly_earnings_gbp": np.arange(n) + 500,
        "unemployment_rate_pct": np.full(n, 4.0),
        "cpi_annual_rate_pct": np.full(n, 2.0),
        "gdp_monthly_growth_pct": np.zeros(n),
        "public_sector_net_borrowing_ex_banks_gbp_mn": np.arange(n) + 10000,
        "bank_rate_month_end_pct": np.full(n, 0.5),
    })


def test_target_is_one_month_ahead():
    model = build_modelling_table(make_master())
    gap = (
        model["target_date"].dt.to_period("M")
        - model["forecast_origin"].dt.to_period("M")
    )
    assert all(x.n == 1 for x in gap)


def test_lag_1_uses_forecast_origin_price():
    master = make_master()
    model = build_modelling_table(master)
    lookup = master.set_index("date")["average_house_price_gbp"]
    expected = model["forecast_origin"].map(lookup)
    assert np.allclose(model["house_price_lag_1m_gbp"], expected)


def test_latest_forecast_row_has_all_selected_features():
    row = build_latest_forecast_row(make_master())
    assert set(AUTOREGRESSIVE_FEATURES).issubset(row.columns)
    assert not row[AUTOREGRESSIVE_FEATURES].isna().any().any()


def test_model_pipeline_can_fit():
    X = pd.DataFrame(
        np.random.default_rng(42).normal(size=(40, len(AUTOREGRESSIVE_FEATURES))),
        columns=AUTOREGRESSIVE_FEATURES,
    )
    y = pd.Series(np.random.default_rng(7).normal(size=40))
    model = build_selected_model()
    model.fit(X, y)
    assert len(model.predict(X.iloc[:2])) == 2


def test_documented_split_sizes():
    df = pd.DataFrame({TARGET_COLUMN: np.arange(160)})
    train, validation, test = chronological_split(df)
    assert len(train) == 112
    assert len(validation) == 24
    assert len(test) == 24
