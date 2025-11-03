
import streamlit as st
import pandas as pd
from model_core import ensure_ready, predict_range, TZ

st.set_page_config(page_title="Prediction Model App", layout="wide")
st.title("Prediction Model (Ready-to-Use)")
st.caption("Baseline Ridge with aligned features - No GPU required")

with st.spinner("Loading model & data..."):
    ensure_ready()

col1, col2 = st.columns([2,2])
with col1:
    today_local = pd.Timestamp.now(tz=TZ).normalize()
    start_date = st.date_input("Start date (local)", today_local.date())
    days = st.slider("Days to forecast", 1, 7, 3)
with col2:
    start_hour = st.slider("Start hour (0-23 local)", 0, 23, 0)

start_local = pd.Timestamp(start_date, tz=TZ) + pd.Timedelta(hours=start_hour)
end_local = start_local + pd.Timedelta(days=days) - pd.Timedelta(hours=1)

start_utc = start_local.tz_convert("UTC")
end_utc = end_local.tz_convert("UTC")

with st.spinner("Predicting..."):
    df = predict_range(start_utc, end_utc)

st.line_chart(df.rename(columns={"pred_demand_mw": "Predicted Demand (MW)"}))
st.dataframe(df.tail(24))

st.markdown("Deploy: push this folder to GitHub, keep runtime.txt at repo root with python-3.12, and set main file to Prediction-Model/app.py on Streamlit Cloud.")
