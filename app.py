import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import ta
import io

# === PAGE CONFIG ===
st.set_page_config(page_title="Pattern Detector Pro", layout="wide")
st.title("Pattern Detector Pro")
st.markdown("*H&S • Bull/Bear Flags • Breakouts • All Timeframes*")

# === SIDEBAR ===
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="TSLA", help="e.g., AAPL, BTC-USD")
timeframe_map = {
    "Hourly (1h)": ("1h", "60d"),
    "Hourly (2h)": ("2h", "90d"),
    "Hourly (4h)": ("4h", "180d"),
    "Daily": ("1d", "2y"),
    "Weekly": ("1wk", "5y"),
    "Monthly": ("1mo", "10y"),
    "1mo": ("1d", "1mo"),
    "3mo": ("1d", "3mo"),
    "6mo": ("1d", "6mo"),
    "1y": ("1d", "1y"),
    "2y": ("1d", "2y"),
    "5y": ("1d", "5y")
}
selected_label = st.sidebar.selectbox("Timeframe", list(timeframe_map.keys()), index=3)

if st.sidebar.button("Analyze"):
    with st.spinner("Loading..."):
        interval, period = timeframe_map[selected_label]
        order = 8 if "h" in interval else 5

        # === DOWNLOAD ===
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            st.error("No data. Try another ticker.")
            st.stop()

        # === ULTIMATE DATE FIX ===
        data = data.copy()

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Reset index to access Date
        data = data.reset_index()

        # Find Date column (any variation)
        date_col = None
        for col in data.columns:
            if str(col).lower() in ['date', 'datetime', 'time', 'index']:
                date_col = col
                break

        if date_col is None:
            st.error("No Date column found. Data structure unknown.")
            st.stop()

        # Rename and set index
        data = data.rename(columns={date_col: 'Date'})
        data.set_index('Date', inplace=True)

        # Keep only OHLCV
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.dropna(inplace=True)

        # === INDICATORS ===
        close = data['Close']
        data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        data['RSI'] = ta.momentum.rsi(close, window=14)
        data_clean = data.dropna().copy()

        if len(data_clean) < 50:
            st.error(f"Need 50+ rows. Got {len(data_clean)}.")
            st.stop()

        # === SWINGS ===
        high_idx = argrelextrema(data_clean['High'].values, np.greater, order=order)[0]
        low_idx = argrelextrema(data_clean['Low'].values, np.less, order=order)[0]

        # === PATTERNS ===
        patterns = []
        highs = data_clean.iloc[high_idx]
        lows = data_clean.iloc[low_idx]

        # Head & Shoulders
        if len(highs) >= 5:
            for i in range(2, len(highs) - 2):
                ls, hd, rs = highs['High'].iloc[i-2:i+1]
                if hd > ls * 1.01 and hd > rs * 1.01 and abs(ls - rs) / hd < 0.1:
                    neckline = data_clean.loc[highs.index[i-2]:highs.index[i+1], 'Low'].min()
                    patterns.append({
                        "type": "H&S", "date": highs.index[i], "price": hd,
                        "color": "purple", "target": neckline - (hd - neckline)
                    })

        # Bull Flag
        if len(lows) >= 12:
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
                            "breakout": breakout
                        })

        # Volume Breakout
        vol_avg = data_clean['Volume'].rolling(20).mean().iloc[-1]
        if data_clean['Volume'].iloc[-1] > vol_avg * 2 and data_clean['Close'].iloc[-1] > data_clean['High'].iloc[-10:-1].max():
            patterns.append({
                "type": "Breakout", "date": data_clean.index[-1], "price": data_clean['Close'].iloc[-1],
                "color": "gold"
            })

        # === OUTPUT ===
        st.success(f"Found {len(patterns)} pattern(s)!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patterns", len(patterns))
            st.metric("Data", len(data_clean))
        with col2:
            st.metric("Price", f"${data_clean['Close'].iloc[-1]:.2f}")
            active = len([p for p in patterns if "breakout" not in p or p["breakout"]])
            st.metric("Live", active)

        # === PLOT ===
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data_clean.index, data_clean['Close'], label='Close', color='black')
        ax.plot(data_clean['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
        ax.plot(data_clean['SMA_50'], label='SMA 50', color='blue', alpha=0.7)

        for p in patterns:
            ax.scatter(p['date'], p['price'], color=p['color'], s=150, zorder=6)
            status = "LIVE" if "breakout" not in p or p["breakout"] else "Pending"
            ax.text(p['date'], p['price'], f" {p['type']}\n{status}", fontsize=9, color=p['color'], weight='bold', ha='center')
            if 'target' in p:
                ax.hlines(p['target'], p['date'], data_clean.index[-1], color=p['color'], linestyle='--', alpha=0.7)

        ax.set_title(f"{ticker} • {selected_label} • {len(patterns)} Patterns")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        st.pyplot(fig)
        st.download_button("Download Chart", buf.getvalue(), f"{ticker}_{selected_label}.png", "image/png")

        if patterns:
            st.subheader("Patterns")
            for p in patterns:
                t = f" → ${p['target']:.2f}" if 'target' in p else ""
                s = "LIVE" if "breakout" not in p or p["breakout"] else "Pending"
                st.write(f"**{p['type']}** • {p['date'].date()} • ${p['price']:.2f}{t} • *{s}*")
        else:
            st.info("No patterns. Try `TSLA` or `BTC-USD`.")

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.markdown("Pro Pattern Detector v7.0")
