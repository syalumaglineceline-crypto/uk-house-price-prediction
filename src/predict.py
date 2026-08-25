"""Generate a genuine next-month forecast from the most recent master-data row."""

from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

try:
    from .features import AUTOREGRESSIVE_FEATURES, build_latest_forecast_row
    from .data_pipeline import load_master
except ImportError:
    from features import AUTOREGRESSIVE_FEATURES, build_latest_forecast_row
    from data_pipeline import load_master


MODEL_PATH = Path("models/final_lasso_full_history.joblib")


def forecast_next_month(
    master_path: str | Path = "data/processed/uk_housing_master_2009_2023.csv",
    model_path: str | Path = MODEL_PATH,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    master = load_master(master_path)
    row = build_latest_forecast_row(master)
    forecast = float(model.predict(row[AUTOREGRESSIVE_FEATURES])[0])

    return pd.DataFrame({
        "forecast_origin": row["forecast_origin"],
        "target_date": row["target_date"],
        "forecast_house_price_gbp": [forecast],
    })


if __name__ == "__main__":
    result = forecast_next_month()
    Path("results").mkdir(exist_ok=True)
    result.to_csv("results/latest_next_month_forecast.csv", index=False)
    print(result.to_string(index=False))
