# Model Card — UK House Price Forecasting

## Model Overview

**Final selected model:** Autoregressive Lasso Regression  
**Forecast horizon:** One month ahead  
**Target:** Average UK house price (£)  
**Study period:** January 2009 to April 2023  
**Final untouched test period:** May 2021 to April 2023

This model is part of a portfolio reconstruction and extension of an MSc industry project completed with Asset Dreams. The reconstruction focuses on reproducibility, chronological validation, leakage prevention, explainability and responsible model documentation.

## Intended Use

The model is intended as a technical forecasting demonstration for national-level UK average house prices.

It is **not** intended for:

- individual property valuation;
- mortgage or lending decisions;
- automated investment decisions;
- regional or property-level forecasting;
- use as a substitute for professional market judgement.

## Model Inputs

The final model uses leakage-aware autoregressive features available before the target month:

- 1-month lagged average house price
- 3-month lagged average house price
- 12-month lagged average house price
- 3-month rolling mean of house prices
- 12-month rolling mean of house prices
- sine-encoded target month
- cosine-encoded target month

Lasso regularisation reduced several highly correlated price-history features to zero coefficients in the final fit.

## Evaluation Design

Model development followed a chronological design:

- Training: 112 observations
- Validation: 24 observations
- Final untouched test: 24 observations

Hyperparameter tuning and model selection were completed before the final test period was evaluated.

The selected model was also compared with:

- persistence baseline
- annual-drift baseline
- 12-month seasonal baseline
- Artificial Neural Network
- other classical machine-learning models

## Final Test Performance

| Metric | Result |
|---|---:|
| MAE | £3,255 |
| RMSE | £4,956 |
| MAPE | 1.21% |
| R² | 0.844 |

Approximate moving-block bootstrap intervals for the 24-month test window:

- MAE 95% interval: approximately **£1,502 to £5,671**
- RMSE 95% interval: approximately **£1,931 to £7,474**

These intervals describe uncertainty in the observed test-window error and are not formal prediction intervals for future house prices.

## Explainability

The final model is interpretable through both regularised coefficients and SHAP values.

Global SHAP analysis shows that the most recent observed house price is the dominant driver of the one-month-ahead prediction. Month-of-year terms and the 12-month lag make smaller contributions.

SHAP values explain the fitted model's mathematical behaviour; they should not be interpreted as causal effects.

## Responsible Modelling Considerations

- The modelling dataset contains only 160 monthly forecasting observations.
- The target is a UK national average and does not represent regional or property-level variation.
- Official statistical series may be revised after their first publication.
- The reconstructed dataset is aligned by reference month. A production implementation would need to model the actual publication lag of each data source.
- House-price history dominates the short-horizon forecast. This does not imply that macroeconomic variables are unimportant to the housing market.
- The reconstruction does not fabricate unavailable mortgage-rate history. Official Bank Rate is retained as a reproducible interest-rate indicator in the broader dataset.
- Economic shocks are retained rather than removed as outliers, but future market regimes may differ from the historical sample.
- Model outputs should be reviewed alongside current market information and domain expertise.

## Model Selection Decision

Autoregressive Lasso Regression was selected because it produced the strongest out-of-time performance among the locked candidate models and baselines while remaining simple, transparent and reproducible.

The Artificial Neural Network was evaluated under the same chronological test framework but did not outperform the regularised Lasso model. That negative result is retained transparently rather than selecting a more complex algorithm for presentation purposes.

## Reproducibility

The complete workflow is documented across the repository notebooks:

1. `01_data_preparation.ipynb`
2. `02_exploratory_analysis.ipynb`
3. `03_machine_learning.ipynb`
4. `04_neural_network.ipynb`
5. `05_model_evaluation.ipynb`

Supporting evaluation outputs are stored in the `results/` directory.
