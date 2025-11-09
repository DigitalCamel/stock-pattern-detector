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
    with st.spinner("Fetching data..."):
        interval, period = timeframe_map[selected_label]
        order = 8 if "h" in interval else 5

        st.write(f"Downloading {ticker} | {selected_label} | {interval}, {period}")

        # === DOWNLOAD DATA ===
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        
        if data.empty:
            st.error("No data from Yahoo Finance.")
            st.stop()
        st.write(f"Downloaded {len(data)} rows.")

        # === FIX: Handle MultiIndex + Date ===
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]  # Flatten
        if isinstance(data.index, pd.MultiIndex):
            data = data.copy()
            data.reset_index(inplace=True)
            if 'Date' in data.columns:
                data.set_index('Date', inplace=True)
            else:
                data.index = data.index.get_level_values(0)
                data = data.copy()
        else:
            data.reset_index(inplace=True)

        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
        elif data.index.name is None or 'date' not in str(data.index.name).lower():
            st.error("Could not set Date index.")
            st.stop()

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.dropna(inplace=True)

        # === INDICATORS ===
        close = data['Close'].squeeze()
        data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        data['RSI'] = ta.momentum.rsi(close, window=14)
        data_clean = data.dropna().copy()

        if len(data_clean) < 50:
            st.error(f"Only {len(data_clean)} rows after indicators. Need 50+.")
            st.stop()

        st.write(f"After indicators: {len(data_clean)} rows. Price: ${data_clean['Close'].iloc[-1]:.2f}")

        # === SWING POINTS ===
        high_idx = argrelextrema(data_clean['High'].values, np.greater, order=order)[0]
        low_idx = argrelextrema(data_clean['Low'].values, np.less, order=order)[0]
        st.write(f"Swing highs: {len(high_idx)}, lows: {len(low_idx)}")

        # === PATTERN DETECTION ===
        patterns = []
        highs = data_clean.iloc[high_idx]
        lows = data_clean.iloc[low_idx]

        # Head & Shoulders
        if len(highs) > 4:
            for i in range(2, len(highs) - 2):
                ls, hd, rs = highs['High'].iloc[i-2:i+1]
                if hd > ls and hd > rs and abs(ls - rs) / hd < 0.07:
                    neckline = data_clean.loc[highs.index[i-2]:highs.index[i+1], 'Low'].min()
                    patterns.append({
                        "type": "Head & Shoulders", "date": highs.index[i], "price": hd,
                        "color": "purple", "target": neckline - (hd - neckline), "signal": "Bearish"
                    })

        # Bull Flag
        if len(lows) > 12:
            for i in range(1, len(lows) - 12):
                pole_low = lows['Low'].iloc[i]
                pole_high = data_clean.loc[lows.index[i]:lows.index[i+12], 'High'].max()
                if pole_high > pole_low * 1.2:
                    flag_high = data_clean.loc[lows.index[i+6]:lows.index[i+12], 'High'].max()
                    flag_low = data_clean.loc[lows.index[i+6]:lows.index[i+12], 'Low'].min()
                    if (flag_high - flag_low) < (pole_high - pole_low) * 0.5:
                        breakout = data_clean['Close'].iloc[-1] > flag_high
                        patterns.append({
                            "type": "Bull Flag", "date": lows.index[i+9], "price": flag_high,
                            "color": "lime", "target": flag_high + (pole_high - pole_low),
                            "signal": "Bullish", "breakout": breakout
                        })

        # Bear Flag
        if len(highs) > 12:
            for i in range(1, len(highs) - 12):
                pole_high = highs['High'].iloc[i]
                pole_low = data_clean.loc[highs.index[i]:highs.index[i+12], 'Low'].min()
                if pole_high > pole_low * 1.2:
                    flag_high = data_clean.loc[highs.index[i+6]:highs.index[i+12], 'High'].max()
                    flag_low = data_clean.loc[highs.index[i+6]:highs.index[i+12], 'Low'].min()
                    if (flag_high - flag_low) < (pole_high - pole_low) * 0.5:
                        breakdown = data_clean['Close'].iloc[-1] < flag_low
                        patterns.append({
                            "type": "Bear Flag", "date": highs.index[i+9], "price": flag_low,
                            "color": "red", "target": flag_low - (pole_high - pole_low),
                            "signal": "Bearish", "breakout": breakdown
                        })

        # Volume Breakout
        avg_vol = data_clean['Volume'].rolling(20).mean().iloc[-1]
        if data_clean['Volume'].iloc[-1] > avg_vol * 2 and data_clean['Close'].iloc[-1] > data_clean['High'].iloc[-10:-1].max():
            patterns.append({
                "type": "Volume Breakout", "date": data_clean.index[-1],
                "price": data_clean['Close'].iloc[-1], "color": "gold", "signal": "Bullish"
            })

        # === OUTPUT ===
        st.success(f"Found {len(patterns)} pattern(s)!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patterns", len(patterns))
            st.metric("Data Points", len(data_clean))
        with col2:
            st.metric("Price", f"${data_clean['Close'].iloc[-1]:.2f}")
            active = [p for p in patterns if "breakout" not in p or p["breakout"]]
            st.metric("Active", len(active))

        # === PLOT ===
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data_clean.index, data_clean['Close'], label='Close', color='black')
        ax.plot(data_clean['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
        ax.plot(data_clean['SMA_50'], label='SMA 50', color='blue', alpha=0.7)

        for p in patterns:
            ax.scatter(p['date'], p['price'], color=p['color'], s=150, zorder=6, edgecolors='black')
            status = "LIVE" if "breakout" not in p or p["breakout"] else "Pending"
            ax.text(p['date'], p['price'], f" {p['type']}\n{status}", fontsize=9, color=p['color'], weight='bold', ha='center')
            if 'target' in p:
                ax.hlines(p['target'], p['date'], data_clean.index[-1], color=p['color'], linestyle='--', alpha=0.7)

        ax.set_title(f"{ticker} • {selected_label} • {len(patterns)} Patterns", fontsize=16)
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        st.pyplot(fig)
        st.download_button("Download Chart", buf.getvalue(), f"{ticker}_{selected_label}_chart.png", "image/png")

        # === LIST ===
        if patterns:
            st.subheader("Patterns")
            for p in patterns:
                status = "LIVE" if "breakout" not in p or p["breakout"] else "Pending"
                target = f" → ${p['target']:.2f}" if 'target' in p else ""
                st.write(f"**{p['type']}** • {p['date'].date()} • ${p['price']:.2f}{target} • *{status}*")
        else:
            st.info("No patterns. Try `TSLA` or `BTC-USD`.")

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.markdown("Pro Pattern Detector v4.0")
