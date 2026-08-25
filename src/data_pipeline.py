"""Data loading, validation and modelling-table reconstruction."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    from .features import build_modelling_table
except ImportError:
    from features import build_modelling_table


MASTER_PATH = Path("data/processed/uk_housing_master_2009_2023.csv")
MODELLING_PATH = Path("data/processed/uk_housing_modelling_base.csv")


def load_master(path: str | Path = MASTER_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Master dataset is empty.")
    return df.sort_values("date").reset_index(drop=True)


def rebuild_modelling_table(
    master_path: str | Path = MASTER_PATH,
    output_path: str | Path = MODELLING_PATH,
) -> pd.DataFrame:
    master = load_master(master_path)
    model = build_modelling_table(master)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(output, index=False)
    return model


if __name__ == "__main__":
    model = rebuild_modelling_table()
    print(f"Rebuilt modelling table with {len(model)} rows.")
