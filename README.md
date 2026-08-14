# Forecasting Future Trends in the UK Housing Market

Machine-learning and neural-network forecasting of UK house prices, developed from an MSc Business Analytics industry project completed with Asset Dreams and subsequently reconstructed as a reproducible technical portfolio project.

## Project Overview

This project investigates one-month-ahead forecasting of the average UK house price using historical housing-market and economic data.

The original MSc industry project explored a range of regression, ensemble and neural-network methods for UK house-price prediction. The portfolio reconstruction extends that work with a stricter forecasting design focused on chronological validation, leakage-aware feature engineering, reproducibility, explainability and responsible model evaluation.

The final workflow covers the full analytical lifecycle from data reconstruction and preprocessing through exploratory analysis, classical machine learning, deep learning, final model comparison and SHAP-based explainability.

## Forecasting Objective

The modelling task is to predict the **next month's average UK house price** using information available at the forecast origin.

The reconstructed dataset covers monthly observations from **January 2009 to April 2023**.

The forecasting workflow deliberately avoids random train/test splitting. Instead, observations remain in chronological order so that model evaluation more closely reflects how a forecasting system would operate in practice.

## Data

The project combines UK housing-market and macroeconomic indicators from authoritative public sources, including:

- UK House Price Index / HM Land Registry
- Office for National Statistics
- Bank of England
- UK public-sector finance statistics

The repository includes a reconstructed monthly master dataset, a leakage-aware modelling table, a data dictionary, a source manifest and documented data-quality checks.

## Modelling Approach

### Forecasting baselines

- Persistence baseline
- Annual-drift baseline
- 12-month seasonal baseline

### Classical machine learning

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Hyperparameter tuning uses **expanding-window time-series cross-validation**.

### Artificial Neural Network

A feed-forward neural network is implemented in **PyTorch** and evaluated under the same chronological framework as the classical models.

The ANN workflow includes training-only scaling, ReLU activation, Adam optimisation, mean-squared-error loss, validation-based architecture selection, early stopping, deterministic random seeds and an untouched final test.

## Chronological Evaluation Design

The final modelling table contains **160 complete forecasting observations**.

- **Training:** 112 observations
- **Validation:** 24 observations
- **Final untouched test:** 24 observations

The final test period covers **May 2021 to April 2023** and is not used for model or hyperparameter selection.

## Final Model

The final selected model is **Autoregressive Lasso Regression**.

It uses lagged and rolling house-price information together with month-of-year seasonality.

### Final Test Performance

| Metric | Result |
|---|---:|
| MAE | ~£3,255 |
| RMSE | ~£4,956 |
| MAPE | ~1.21% |
| R² | ~0.844 |

The selected Lasso model outperformed the neural network and the simple persistence and drift baselines on the untouched test period.

An important finding is that adding the full set of macroeconomic indicators did **not automatically improve one-month-ahead forecasting accuracy**. Recent house-price history provided the strongest short-horizon signal.

## Explainability

The final model is interpreted using:

- standardised Lasso coefficients;
- SHAP values.

Global SHAP analysis shows that the most recent lagged house price is the dominant contributor to the one-month-ahead forecast. Local SHAP analysis is also used to explain a difficult final-test prediction.

SHAP values explain the model's mathematical behaviour and should not be interpreted as causal effects.

## Responsible Modelling

The project explicitly documents key limitations:

- the modelling sample contains only 160 monthly forecasting observations;
- the target is a UK national average rather than a regional or property-level value;
- official statistical series may be revised after publication;
- a production system would need to account for actual data-publication lags;
- unusual economic shocks are retained rather than removed as outliers;
- model outputs should not be used as automated mortgage, valuation or investment decisions.

See `MODEL_CARD.md` and `05_model_evaluation.ipynb` for the full assessment.

## Repository Structure

```text
uk-house-price-prediction/
├── README.md
├── RECONSTRUCTION_NOTES.md
├── MODEL_CARD.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── data_dictionary.csv
│   ├── quality_report.txt
│   ├── source_manifest.csv
│   └── processed/
│       ├── uk_housing_master_2009_2023.csv
│       └── uk_housing_modelling_base.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_machine_learning.ipynb
│   ├── 04_neural_network.ipynb
│   └── 05_model_evaluation.ipynb
└── results/
    ├── final_test_predictions.csv
    ├── final_model_comparison.csv
    ├── shap_global_importance.csv
    └── lasso_coefficients.csv
```

## Notebook Workflow

### `01_data_preparation.ipynb`
Reconstructs and validates the modelling dataset, creates leakage-aware lagged features and prepares the one-month-ahead forecasting table.

### `02_exploratory_analysis.ipynb`
Explores house-price trends, economic indicators, seasonality, lag relationships, multicollinearity and the difference between price-level and price-change relationships.

### `03_machine_learning.ipynb`
Builds forecasting baselines and classical machine-learning models using chronological validation and time-series cross-validation.

### `04_neural_network.ipynb`
Implements and evaluates a PyTorch Artificial Neural Network under the same chronological forecasting framework.

### `05_model_evaluation.ipynb`
Consolidates model results, evaluates residuals and uncertainty, performs SHAP explainability and documents the final model-selection decision.

## Technologies

Python, pandas, NumPy, scikit-learn, PyTorch, SHAP, statsmodels, Matplotlib, Jupyter Notebook, time-series cross-validation, machine learning, deep learning and explainable AI.

## Key Findings

1. Chronological validation can produce very different conclusions from random train/test splitting.
2. Recent house-price history is highly informative for one-month-ahead national forecasting.
3. More features do not automatically produce a better model.
4. Tree ensembles can struggle to extrapolate a strongly trending target beyond the range observed during training.
5. A neural network can be evaluated rigorously without assuming that deep learning must outperform simpler approaches.
6. Regularised Lasso provided the strongest final out-of-time performance.
7. Explainability, uncertainty and limitations are treated as part of model quality rather than optional additions.

## Project Context

The original work was completed as an MSc Business Analytics industry project at Aston University in collaboration with Asset Dreams, a UK real-estate company.

This repository is a reconstructed and extended technical portfolio version of that work. The extensions focus on stronger forecasting methodology, reproducibility, chronological validation, deep-learning evaluation, explainability and responsible modelling.
