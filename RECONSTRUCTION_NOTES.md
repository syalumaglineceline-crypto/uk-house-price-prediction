# Reconstruction Notes

## Purpose

This repository reconstructs and extends the data pipeline used in the MSc industry project *Forecasting Future Trends in the UK Housing Market*. The reconstruction prioritises reproducibility, clear provenance and leakage-aware forecasting design.

## Data window

The master dataset covers January 2009 to April 2023, matching the original project window. UK HPI prices and annual changes are retained from the April 2023 HPI vintage. Two unavailable SalesVolume observations at the end of that vintage are filled from a later official HPI release and explicitly flagged in the data.

## Source-vintage strategy

For ONS variables, historical series files close to the original 2023 submission period are used where available. This avoids silently mixing modern revisions into a reconstruction of a historical project. The source manifest records the series IDs and publisher pages.

## Forecasting design

The modelling table is deliberately different from a simple contemporaneous regression. Each row predicts the next month's UK average house price using information dated at the forecast origin or earlier. Lagged house prices, rolling price statistics, macroeconomic indicators and calendar seasonality are therefore available before the target month.

This design prevents target-month information from entering the feature matrix. In a production deployment, publication lags for individual ONS/HPI releases would also be modelled explicitly.

## Feature rationalisation

Some variables from the original academic analysis are not used as contemporaneous predictors in the reconstructed forecasting table:

- **12-month house-price change** remains in the master dataset for descriptive analysis, but is excluded from modelling because it is calculated from the target series itself.
- **House-price-to-earnings ratio** is not recreated as a monthly model feature because a same-month ratio embeds the target house price and would introduce leakage.
- **Population** is not interpolated from annual estimates for the model, avoiding artificial monthly precision.
- **Incomplete mortgage-rate series** are not filled or fabricated. Instead, the reconstruction adds the complete official Bank Rate history as a reproducible interest-rate indicator. Exact Bank of England mortgage-rate series can be added later if their full official download is stored in the repository.

## Validation plan

Model development will use chronological validation only. The final notebooks will compare each model with a naive one-step-ahead baseline, use expanding/rolling time-series cross-validation for model selection, and reserve the latest period as an untouched holdout test set.

## Reproducibility

Raw-source files, the master dataset and the modelling table are kept conceptually separate. The final notebook will regenerate the modelling table from the master data and document all transformations rather than relying on manual spreadsheet edits.
