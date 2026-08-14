# Data

This project reconstructs a monthly UK housing-market dataset for January 2009 to April 2023 using authoritative public data sources and a reproducible, time-aware preparation workflow.

## Target

- **Average UK House Price**

## Master dataset

`processed/uk_housing_master_2009_2023.csv` contains the aligned monthly series used for analysis and feature construction:

- Average UK House Price
- 12-month House Price Change (descriptive only)
- Sales Volume
- Average Weekly Earnings
- Unemployment Rate
- CPI Annual Inflation Rate
- Monthly GDP/GVA Growth
- Public Sector Net Borrowing excluding Public Sector Banks
- Bank Rate at Month End

## Modelling dataset

`processed/uk_housing_modelling_base.csv` is a leakage-aware one-month-ahead forecasting table. Each row predicts the next month's average UK house price using information dated at the forecast origin or earlier.

The feature set includes:

- lagged house prices
- rolling house-price statistics
- lagged housing-market and macroeconomic indicators
- Bank Rate
- calendar seasonality features

Target-derived same-month variables are not used as predictors in the forecasting table.

## Data sources

The reconstruction uses authoritative UK sources including:

- HM Land Registry / UK House Price Index
- Office for National Statistics (ONS)
- Bank of England

Exact source series, coverage and reconstruction use are documented in `source_manifest.csv`.

## Data quality and provenance

The preparation workflow:

- aligns all observations to a monthly time index
- validates chronology and duplicate dates
- records source provenance for backfilled observations
- checks missing values before modelling
- keeps target-derived descriptive fields separate from predictive features
- creates lagged and rolling features using historical information only

`quality_report.txt` records the principal automated quality checks for the reconstructed dataset.

## Reproducibility

The modelling table is regenerated in `../notebooks/01_data_preparation.ipynb` from the master dataset. Transformations are implemented programmatically rather than through manual spreadsheet editing.

See `../RECONSTRUCTION_NOTES.md` for the forecasting-design and feature-selection rationale.
