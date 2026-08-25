"""Simple decision-support dashboard for the UK house-price forecasting project."""

from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UK House Price Forecasting", layout="wide")
st.title("UK House Price Forecasting Decision Support")
st.caption(
    "One-month-ahead forecasting with chronological validation and an "
    "Autoregressive Lasso model."
)

pred_path = Path("results/pipeline_final_test_predictions.csv")
metrics_path = Path("results/pipeline_final_test_metrics.json")
latest_path = Path("results/latest_next_month_forecast.csv")

if not pred_path.exists() or not metrics_path.exists():
    st.info(
        "Run `python -m src.evaluate` first to generate final-test outputs."
    )
    st.stop()

pred = pd.read_csv(pred_path, parse_dates=["target_date", "forecast_origin"])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final-test MAE", f"£{metrics['MAE']:,.0f}")
c2.metric("Final-test RMSE", f"£{metrics['RMSE']:,.0f}")
c3.metric("Final-test MAPE", f"{metrics['MAPE_pct']:.2f}%")
c4.metric("Final-test R²", f"{metrics['R2']:.3f}")

st.subheader("Actual vs predicted house price")
chart = pred.set_index("target_date")[
    ["target_house_price_next_month_gbp", "prediction_gbp"]
].rename(columns={
    "target_house_price_next_month_gbp": "Actual",
    "prediction_gbp": "Predicted",
})
st.line_chart(chart)

if latest_path.exists():
    latest = pd.read_csv(latest_path, parse_dates=["forecast_origin", "target_date"])
    row = latest.iloc[0]
    st.subheader("Latest one-month-ahead forecast")
    st.metric(
        f"Forecast for {row['target_date'].strftime('%B %Y')}",
        f"£{row['forecast_house_price_gbp']:,.0f}",
    )
    st.caption(
        f"Forecast origin: {row['forecast_origin'].strftime('%B %Y')}. "
        "This is a portfolio decision-support demonstration, not investment advice."
    )

st.subheader("Largest final-test errors")
errors = pred.nlargest(5, "absolute_error_gbp")[
    ["target_date", "target_house_price_next_month_gbp",
     "prediction_gbp", "absolute_error_gbp"]
]
st.dataframe(errors, use_container_width=True)
