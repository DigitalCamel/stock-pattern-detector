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
        if data.empty:
            st.error("No data found. Try another ticker or timeframe.")
            st.stop()

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.dropna(inplace=True)

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data.reset_index(inplace=True)
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)

        # === INDICATORS ===
        close = data['Close'].squeeze()
        data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        data['RSI'] = ta.momentum.rsi(close, window=14)
        data_clean = data.dropna().copy()

        if len(data_clean) < 50:
            st.error("Not enough data. Try a longer timeframe.")
            st.stop()

        # === SWING POINTS ===
        high_idx = argrelextrema(data_clean['High'].values, np.greater, order=order)[0]
        low_idx = argrelextrema(data_clean['Low'].values, np.less, order=order)[0]

        # === PATTERN DETECTION ===
        patterns = []

        # 1. Head & Shoulders (Bearish)
        highs = data_clean.iloc[high_idx]
        for i in range(2, len(highs) - 2):
            ls, hd, rs = highs['High'].iloc[i-2:i+1]
            if hd > ls and hd > rs and abs(ls - rs) / hd < 0.07:
                neck_lows = data_clean.loc[highs.index[i-2]:highs.index[i+1], 'Low']
                neckline = neck_lows.min()
                patterns.append({
                    "type": "Head & Shoulders",
                    "date": highs.index[i],
                    "price": hd,
                    "neckline": neckline,
                    "color": "purple",
                    "target": neckline - (hd - neckline),
                    "signal": "Bearish"
                })

        # 2. Bull Flag
        lows = data_clean.iloc[low_idx]
        for i in range(1, len(lows) - 12):
            pole_low = lows['Low'].iloc[i]
            pole_high = data_clean.loc[lows.index[i]:lows.index[i+12], 'High'].max()
            if pole_high > pole_low * 1.2:
                flag_high = data_clean.loc[lows.index[i+6]:lows.index[i+12], 'High'].max()
                flag_low = data_clean.loc[lows.index[i+6]:lows.index[i+12], 'Low'].min()
                if (flag_high - flag_low) < (pole_high - pole_low) * 0.5:
                    breakout = data_clean['Close'].iloc[-1] > flag_high
                    patterns.append({
                        "type": "Bull Flag",
                        "date": lows.index[i+9],
                        "price": flag_high,
                        "color": "lime",
                        "target": flag_high + (pole_high - pole_low),
                        "signal": "Bullish",
                        "breakout": breakout
                    })

        # 3. Bear Flag
        for i in range(1, len(highs) - 12):
            pole_high = highs['High'].iloc[i]
            pole_low = data_clean.loc[highs.index[i]:highs.index[i+12], 'Low'].min()
            if pole_high > pole_low * 1.2:
                flag_high = data_clean.loc[highs.index[i+6]:highs.index[i+12], 'High'].max()
                flag_low = data_clean.loc[highs.index[i+6]:highs.index[i+12], 'Low'].min()
                if (flag_high - flag_low) < (pole_high - pole_low) * 0.5:
                    breakdown = data_clean['Close'].iloc[-1] < flag_low
                    patterns.append({
                        "type": "Bear Flag",
                        "date": highs.index[i+9],
                        "price": flag_low,
                        "color": "red",
                        "target": flag_low - (pole_high - pole_low),
                        "signal": "Bearish",
                        "breakout": breakdown
                    })

        # 4. Volume Breakout
        avg_vol = data_clean['Volume'].rolling(20).mean().iloc[-1]
        if (data_clean['Volume'].iloc[-1] > avg_vol * 2 and
            data_clean['Close'].iloc[-1] > data_clean['High'].iloc[-10:-1].max()):
            patterns.append({
                "type": "Volume Breakout",
                "date": data_clean.index[-1],
                "price": data_clean['Close'].iloc[-1],
                "color": "gold",
                "signal": "Bullish"
                # No target here — will be skipped safely
            })

        # === DISPLAY RESULTS ===
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Points", len(data_clean))
            st.metric("Patterns Found", len(patterns))
        with col2:
            current_price = data_clean['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
            active = [p for p in patterns if "breakout" not in p or p["breakout"]]
            st.metric("Active Signals", len(active))

        # === PLOT CHART ===
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data_clean.index, data_clean['Close'], label='Close', color='black', linewidth=1)
        ax.plot(data_clean['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
        ax.plot(data_clean['SMA_50'], label='SMA 50', color='blue', alpha=0.7)

        # Mark patterns
        for p in patterns:
            ax.scatter(p['date'], p['price'], color=p['color'], s=150, zorder=6, edgecolors='black', linewidth=1)
            status = "LIVE" if "breakout" not in p or p["breakout"] else "Pending"
            ax.text(p['date'], p['price'], f" {p['type']}\n{status}", 
                    fontsize=9, color=p['color'], weight='bold', ha='center', va='bottom')

            # Target line
            if 'target' in p:
                ax.hlines(p['target'], p['date'], data_clean.index[-1],
