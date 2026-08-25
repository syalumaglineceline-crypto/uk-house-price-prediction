"""Final untouched-test evaluation for the selected forecasting model."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .features import AUTOREGRESSIVE_FEATURES, TARGET_COLUMN
    from .train import load_modelling_data, train_evaluation_model
except ImportError:
    from features import AUTOREGRESSIVE_FEATURES, TARGET_COLUMN
    from train import load_modelling_data, train_evaluation_model


def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_final_test(df: pd.DataFrame):
    model, test = train_evaluation_model(df)
    pred = model.predict(test[AUTOREGRESSIVE_FEATURES])
    metrics = calculate_metrics(test[TARGET_COLUMN], pred)

    out = test[["forecast_origin", "target_date", TARGET_COLUMN]].copy()
    out["prediction_gbp"] = pred
    out["residual_gbp"] = out[TARGET_COLUMN] - out["prediction_gbp"]
    out["absolute_error_gbp"] = out["residual_gbp"].abs()
    return metrics, out


if __name__ == "__main__":
    df = load_modelling_data()
    metrics, predictions = evaluate_final_test(df)

    Path("results").mkdir(exist_ok=True)
    predictions.to_csv("results/pipeline_final_test_predictions.csv", index=False)
    Path("results/pipeline_final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Final untouched-test metrics")
    print(f"MAE:  £{metrics['MAE']:,.2f}")
    print(f"RMSE: £{metrics['RMSE']:,.2f}")
    print(f"MAPE: {metrics['MAPE_pct']:.3f}%")
    print(f"R²:   {metrics['R2']:.4f}")
