# Data

This project uses monthly UK housing-market and macroeconomic indicators to model and forecast movements in average UK house prices.

The dataset brings together housing, mortgage, labour-market and wider economic indicators from authoritative UK public sources.

## Target Variable

- Average UK House Price

## Predictor Variables

- Average House Price Annual Change
- Transaction Volume
- Number of Mortgage Approvals
- Average House Price to Earnings Ratio
- Average Quoted 2-Year Fixed Mortgage Rate (75% LTV)
- Average Quoted 3-Year Fixed Mortgage Rate (75% LTV)
- Gross Domestic Product (GDP)
- Average Weekly Earnings
- Unemployment Rate
- Inflation Rate
- Government Borrowing
- Population
- Year
- Month

## Data Sources

The project uses publicly available data from authoritative UK sources, including:

- UK Government housing statistics
- Office for National Statistics (ONS)
- Bank of England

A detailed data-source table will document the source, frequency, coverage and definition of each variable used in the final modelling dataset.

## Data Preparation

The reconstruction follows a reproducible and time-aware data preparation process, including:

- aligning variables to a consistent monthly time index
- validating data types and ranges
- identifying duplicate and missing observations
- applying appropriate missing-value treatment
- checking consistency across source datasets
- creating derived features where analytically justified
- maintaining chronological ordering throughout the modelling workflow

Preprocessing decisions are designed to minimise information leakage between historical training data and future observations.

## Data Organisation

Raw and processed data are kept separate:

```text
data/
├── raw/
└── processed/
