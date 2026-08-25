# Forecasting Future Trends in the UK Housing Market

A one-month-ahead UK house-price forecasting project developed from my MSc Business Analytics industry work with Asset Dreams and later rebuilt as a reproducible portfolio project.

The main focus is simple: use historical housing-market and economic data to estimate the **next month's average UK house price**, while keeping the modelling process realistic, leakage-aware and easy to reproduce.

> **Final model:** Autoregressive Lasso Regression  
> **Forecast horizon:** One month ahead  
> **Final test MAPE:** ~1.21%  
> **Validation style:** Chronological / out-of-time

## What this project covers

The original MSc project explored regression, ensemble and neural-network methods for UK house-price prediction.

For the portfolio version, I rebuilt the workflow around a stricter forecasting setup, with emphasis on:

- chronological validation instead of random splitting;
- lagged and rolling time-series features;
- baseline forecasting models;
- classical machine learning;
- neural-network evaluation;
- SHAP-based explainability;
- reproducible pipeline code;
- lightweight MLOps and decision-support components.

The reconstructed dataset covers monthly observations from **January 2009 to April 2023**.

## Data

The project combines housing-market and macroeconomic indicators from public UK sources, including:

- UK House Price Index / HM Land Registry
- Office for National Statistics
- Bank of England
- UK public-sector finance statistics

The repository includes a reconstructed monthly master dataset, a modelling table, a source manifest, a data dictionary and basic data-quality checks.

## Forecasting setup

The target is the **next month's average UK house price**.

The modelling workflow keeps observations in time order so the validation process reflects how a real forecasting system would behave.

### Baselines

- Persistence baseline
- Annual-drift baseline
- 12-month seasonal baseline

### Machine-learning models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Hyperparameter tuning uses **expanding-window time-series cross-validation**.

### Neural network

A feed-forward neural network is implemented in **PyTorch** and evaluated under the same chronological setup as the classical models.

The ANN workflow includes training-only scaling, ReLU activation, Adam optimisation, mean-squared-error loss, validation-based architecture selection, early stopping, deterministic random seeds and an untouched final test set.

## Evaluation design

The final modelling table contains **160 complete forecasting observations**.

- **Training:** 112 observations
- **Validation:** 24 observations
- **Final untouched test:** 24 observations

The final test period covers **May 2021 to April 2023** and is not used for model or hyperparameter selection.

## Final model and results

The final selected model is **Autoregressive Lasso Regression**.

It uses recent house-price history, rolling-price information and month-of-year seasonality.

| Metric | Result |
|---|---:|
| MAE | ~£3,255 |
| RMSE | ~£4,956 |
| MAPE | ~1.21% |
| R² | ~0.844 |

The Lasso model outperformed the neural network and the simple persistence and drift baselines on the untouched test period.

One useful result from the project was that adding more macroeconomic variables did **not automatically improve short-horizon accuracy**. Recent house-price history remained the strongest signal for the one-month-ahead forecast.

## Explainability

The final model is interpreted using:

- standardised Lasso coefficients;
- SHAP values.

Global SHAP analysis shows that the most recent lagged house price is the main contributor to the one-month-ahead forecast. Local SHAP analysis is also used to explain a difficult final-test prediction.

SHAP is used here to explain model behaviour, not to claim causal relationships.

<details>
<summary><strong>Production-style forecasting pipeline</strong></summary>

The notebook workflow is also wrapped in reusable Python modules so the project is not limited to exploratory notebooks.

The pipeline covers:

- data loading and validation;
- lagged, rolling and seasonal feature generation;
- model training;
- final out-of-time evaluation;
- next-month prediction;
- automated tests;
- GitHub Actions checks;
- a lightweight Streamlit decision-support interface.

The evaluation and deployment stages are kept separate. The evaluation model preserves the final 24 observations as an untouched test set. After model selection, the deployment version retrains the selected specification on all available labelled history before forecasting the next unseen month.

This extension is intended to demonstrate reproducibility, basic MLOps practice and how an experimental forecasting model can be turned into a reusable analytical workflow.

</details>

## Repository structure

```text
uk-house-price-prediction/
├── README.md
├── RECONSTRUCTION_NOTES.md
├── MODEL_CARD.md
├── requirements.txt
├── config.yaml
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_pipeline.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── README.md
│   ├── data_dictionary.csv
│   ├── quality_report.txt
│   ├── source_manifest.csv
│   └── processed/
│       ├── uk_housing_master_2009_2023.csv
│       └── uk_housing_modelling_base.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_machine_learning.ipynb
│   ├── 04_neural_network.ipynb
│   └── 05_model_evaluation.ipynb
│
└── results/
    ├── final_test_predictions.csv
    ├── final_model_comparison.csv
    ├── shap_global_importance.csv
    └── lasso_coefficients.csv
```

## Notebook workflow

### `01_data_preparation.ipynb`

Reconstructs and validates the modelling dataset, creates leakage-aware lagged features and prepares the one-month-ahead forecasting table.

### `02_exploratory_analysis.ipynb`

Explores house-price trends, economic indicators, seasonality, lag relationships, multicollinearity and the difference between price-level and price-change relationships.

### `03_machine_learning.ipynb`

Builds forecasting baselines and classical machine-learning models using chronological validation and time-series cross-validation.

### `04_neural_network.ipynb`

Implements and evaluates a PyTorch neural network under the same chronological forecasting framework.

### `05_model_evaluation.ipynb`

Consolidates model results, evaluates residuals and uncertainty, performs SHAP explainability and documents the final model-selection decision.

## Responsible modelling

A few limitations are worth keeping visible:

- the modelling sample contains only 160 monthly forecasting observations;
- the target is a UK national average rather than a regional or property-level value;
- official statistical series may be revised after publication;
- a production system would need to reflect actual data-publication lags;
- unusual economic shocks are retained rather than removed as outliers;
- model outputs should not be used as automated mortgage, valuation or investment decisions.

See `MODEL_CARD.md` and `05_model_evaluation.ipynb` for the fuller assessment.

## Technologies

Python, pandas, NumPy, scikit-learn, PyTorch, SHAP, statsmodels, Matplotlib, Jupyter Notebook, Streamlit, GitHub Actions, joblib, time-series cross-validation, machine learning, deep learning and explainable AI.

## Main takeaways

- Chronological validation can lead to very different conclusions from random splitting.
- Recent house-price history is highly informative for short-horizon national forecasting.
- More features do not automatically mean a better model.
- Tree ensembles can struggle to extrapolate strongly trending targets.
- Deep learning does not automatically outperform simpler models.
- Regularised Lasso gave the strongest final out-of-time performance.
- Explainability and limitations are part of the modelling process, not optional extras.
- Reusable code, automated checks and a simple decision-support layer make the forecasting workflow more practical.

## Project context

The original work was completed as an MSc Business Analytics industry project at Aston University in collaboration with Asset Dreams, a UK real-estate company.

This repository is a reconstructed and extended technical portfolio version of that work. The later extensions focus on forecasting methodology, reproducibility, chronological validation, explainability, reusable pipeline design, basic MLOps practices and responsible modelling.
