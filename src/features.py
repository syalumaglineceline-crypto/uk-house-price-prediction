"""Leakage-aware feature engineering matched to the current repository."""

from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_COLUMN = "target_house_price_next_month_gbp"

AUTOREGRESSIVE_FEATURES = [
    "house_price_lag_1m_gbp",
    "house_price_lag_3m_gbp",
    "house_price_lag_12m_gbp",
    "house_price_rolling_3m_mean_gbp",
    "house_price_rolling_12m_mean_gbp",
    "target_month_sin",
    "target_month_cos",
]

REQUIRED_MASTER_COLUMNS = {
    "date",
    "average_house_price_gbp",
    "sales_volume",
    "average_weekly_earnings_gbp",
    "unemployment_rate_pct",
    "cpi_annual_rate_pct",
    "gdp_monthly_growth_pct",
    "public_sector_net_borrowing_ex_banks_gbp_mn",
    "bank_rate_month_end_pct",
}

MODEL_COLUMNS = [
    "forecast_origin",
    "target_date",
    TARGET_COLUMN,
    "house_price_lag_1m_gbp",
    "house_price_lag_3m_gbp",
    "house_price_lag_12m_gbp",
    "house_price_rolling_3m_mean_gbp",
    "house_price_rolling_12m_mean_gbp",
    "sales_volume_lag_1m",
    "average_weekly_earnings_lag_1m_gbp",
    "unemployment_rate_lag_1m_pct",
    "cpi_annual_rate_lag_1m_pct",
    "gdp_monthly_growth_lag_1m_pct",
    "public_sector_net_borrowing_lag_1m_gbp_mn",
    "bank_rate_lag_1m_pct",
    "target_month_sin",
    "target_month_cos",
]


def _prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_MASTER_COLUMNS.difference(master.columns)
    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )
    df = master.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in master dataset.")
    return df


def build_modelling_table(master: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the one-month-ahead modelling table used in the project."""
    df = _prepare_master(master)

    df["target_date"] = df["date"].shift(-1)
    df[TARGET_COLUMN] = df["average_house_price_gbp"].shift(-1)

    # Lags are expressed relative to the target month t+1.
    df["house_price_lag_1m_gbp"] = df["average_house_price_gbp"]
    df["house_price_lag_3m_gbp"] = df["average_house_price_gbp"].shift(2)
    df["house_price_lag_12m_gbp"] = df["average_house_price_gbp"].shift(11)

    df["house_price_rolling_3m_mean_gbp"] = (
        df["average_house_price_gbp"].rolling(3, min_periods=3).mean()
    )
    df["house_price_rolling_12m_mean_gbp"] = (
        df["average_house_price_gbp"].rolling(12, min_periods=12).mean()
    )

    # Exogenous values are known at forecast origin t.
    df["sales_volume_lag_1m"] = df["sales_volume"]
    df["average_weekly_earnings_lag_1m_gbp"] = df["average_weekly_earnings_gbp"]
    df["unemployment_rate_lag_1m_pct"] = df["unemployment_rate_pct"]
    df["cpi_annual_rate_lag_1m_pct"] = df["cpi_annual_rate_pct"]
    df["gdp_monthly_growth_lag_1m_pct"] = df["gdp_monthly_growth_pct"]
    df["public_sector_net_borrowing_lag_1m_gbp_mn"] = (
        df["public_sector_net_borrowing_ex_banks_gbp_mn"]
    )
    df["bank_rate_lag_1m_pct"] = df["bank_rate_month_end_pct"]

    month = df["target_date"].dt.month
    df["target_month_sin"] = np.sin(2 * np.pi * month / 12)
    df["target_month_cos"] = np.cos(2 * np.pi * month / 12)

    model = (
        df[
            [
                "date", "target_date", TARGET_COLUMN,
                "house_price_lag_1m_gbp",
                "house_price_lag_3m_gbp",
                "house_price_lag_12m_gbp",
                "house_price_rolling_3m_mean_gbp",
                "house_price_rolling_12m_mean_gbp",
                "sales_volume_lag_1m",
                "average_weekly_earnings_lag_1m_gbp",
                "unemployment_rate_lag_1m_pct",
                "cpi_annual_rate_lag_1m_pct",
                "gdp_monthly_growth_lag_1m_pct",
                "public_sector_net_borrowing_lag_1m_gbp_mn",
                "bank_rate_lag_1m_pct",
                "target_month_sin", "target_month_cos",
            ]
        ]
        .rename(columns={"date": "forecast_origin"})
        .dropna()
        .reset_index(drop=True)
    )
    validate_modelling_table(model)
    return model


def build_latest_forecast_row(master: pd.DataFrame) -> pd.DataFrame:
    """
    Create the latest feature row for a genuine one-month-ahead forecast.

    Unlike build_modelling_table(), this does not require the future target
    to already exist. It uses the most recent observed house-price history.
    """
    df = _prepare_master(master)
    if len(df) < 12:
        raise ValueError("At least 12 monthly observations are required.")

    latest = df.iloc[-1]
    target_date = pd.Timestamp(latest["date"]) + pd.offsets.MonthBegin(1)
    target_month = target_date.month

    row = {
        "forecast_origin": pd.Timestamp(latest["date"]),
        "target_date": target_date,
        "house_price_lag_1m_gbp": float(df["average_house_price_gbp"].iloc[-1]),
        "house_price_lag_3m_gbp": float(df["average_house_price_gbp"].iloc[-3]),
        "house_price_lag_12m_gbp": float(df["average_house_price_gbp"].iloc[-12]),
        "house_price_rolling_3m_mean_gbp": float(
            df["average_house_price_gbp"].iloc[-3:].mean()
        ),
        "house_price_rolling_12m_mean_gbp": float(
            df["average_house_price_gbp"].iloc[-12:].mean()
        ),
        "target_month_sin": float(np.sin(2 * np.pi * target_month / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * target_month / 12)),
    }
    return pd.DataFrame([row])


def validate_modelling_table(model: pd.DataFrame) -> None:
    missing = set(MODEL_COLUMNS).difference(model.columns)
    if missing:
        raise ValueError("Missing modelling columns: " + ", ".join(sorted(missing)))

    if model[MODEL_COLUMNS].isna().any().any():
        raise ValueError("Missing values remain in modelling table.")

    origin = pd.to_datetime(model["forecast_origin"])
    target = pd.to_datetime(model["target_date"])

    if not origin.is_monotonic_increasing or not target.is_monotonic_increasing:
        raise ValueError("Forecast rows are not chronologically ordered.")

    gaps = target.dt.to_period("M") - origin.dt.to_period("M")
    if not all(gap.n == 1 for gap in gaps):
        raise ValueError("Every target_date must be exactly one month ahead.")
