import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import ta
import io

# === PAGE CONFIG ===
st.set_page_config(page_title="Pro Pattern Detector", layout="wide")
st.title("Advanced Stock & Crypto Pattern Detector")
st.markdown("*Head & Shoulders • Bull/Bear Flags • Breakouts • Hourly/Daily/Weekly/Monthly*")

# === SIDEBAR ===
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="TSLA", help="e.g., AAPL, BTC-USD, NVDA")

# Timeframe mapping
timeframe_map = {
    "Hourly (1h)": ("1h", "60d"),
    "Hourly (2h)": ("2h", "90d"),
    "Hourly (4h)": ("4h", "180d"),
    "Daily": ("1d", "2y"),
    "Weekly": ("1wk", "5y"),
    "Monthly": ("1mo", "10y"),
    "1 Month": ("1d", "1mo"),
    "3 Months": ("1d", "3mo"),
    "6 Months": ("1d", "6mo"),
    "1 Year": ("1d", "1y"),
    "2 Years": ("1d", "2y"),
    "5 Years": ("1d", "5y")
}

selected_label = st.sidebar.selectbox(
    "Timeframe",
    options=list(timeframe_map.keys()),
    index=3  # Default: Daily
)

if st.sidebar.button("Analyze"):
    with st.spinner("Fetching data & detecting patterns..."):
        interval, period = timeframe_map[selected_label]
        order = 8 if "h" in interval else 5  # Smoother swings on hourly

        # === DOWNLOAD DATA ===
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
       
