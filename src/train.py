"""Training utilities matched to the selected Autoregressive Lasso model."""

from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .features import AUTOREGRESSIVE_FEATURES, TARGET_COLUMN
except ImportError:
    from features import AUTOREGRESSIVE_FEATURES, TARGET_COLUMN


TRAIN_SIZE = 112
VALIDATION_SIZE = 24
TEST_SIZE = 24
SELECTED_ALPHA = 10
MAX_ITER = 50_000


def load_modelling_data(
    path: str | Path = "data/processed/uk_housing_modelling_base.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["forecast_origin", "target_date"])
    df = df.sort_values("forecast_origin").reset_index(drop=True)
    required = {
        "forecast_origin", "target_date", TARGET_COLUMN, *AUTOREGRESSIVE_FEATURES
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if df[list(required)].isna().any().any():
        raise ValueError("Missing values detected in modelling inputs.")
    return df


def chronological_split(df: pd.DataFrame):
    expected = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
    if len(df) != expected:
        raise ValueError(
            f"Expected {expected} rows for the documented 112/24/24 split; "
            f"found {len(df)}."
        )
    a = TRAIN_SIZE
    b = TRAIN_SIZE + VALIDATION_SIZE
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def build_selected_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=SELECTED_ALPHA, max_iter=MAX_ITER)),
    ])


def train_evaluation_model(df: pd.DataFrame):
    """
    Train on train+validation only. Keep the final 24 observations untouched
    for the documented final out-of-time evaluation.
    """
    train, validation, test = chronological_split(df)
    development = pd.concat([train, validation], ignore_index=True)
    model = build_selected_model()
    model.fit(
        development[AUTOREGRESSIVE_FEATURES],
        development[TARGET_COLUMN],
    )
    return model, test


def train_deployment_model(df: pd.DataFrame):
    """
    Retrain the already-selected model on all available labelled history.

    Use only after final model selection/evaluation is complete. This is the
    model intended for the next unseen month, not for reporting test metrics.
    """
    model = build_selected_model()
    model.fit(df[AUTOREGRESSIVE_FEATURES], df[TARGET_COLUMN])
    return model


def save_model(model, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output


if __name__ == "__main__":
    data = load_modelling_data()

    evaluation_model, _ = train_evaluation_model(data)
    save_model(evaluation_model, "models/final_lasso_evaluation.joblib")

    deployment_model = train_deployment_model(data)
    save_model(deployment_model, "models/final_lasso_full_history.joblib")

    print("Saved evaluation and full-history deployment models.")
