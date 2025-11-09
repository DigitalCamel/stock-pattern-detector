import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import ta
import io

# Page config
st.set_page_config(page_title="Stock Pattern Detector", layout="wide")
st.title("Stock Technical Analysis - Head & Shoulders Detector")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="NVDA", help="e.g., AAPL, TSLA, BTC-USD")
period = st.sidebar.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4)

if st.sidebar.button("Analyze Stock"):
    with st.spinner("Fetching data..."):
        # Download data
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
        if data.empty:
            st.error("No data found. Try another ticker!")
            st.stop()
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.dropna(inplace=True)
        
        # Flatten columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        data.reset_index(inplace=True)
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
        
        # Add indicators
        close_series = data['Close'].squeeze()
        data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
        data['RSI'] = ta.momentum.rsi(close_series, window=14)
        data_clean = data.dropna(subset=['SMA_20', 'RSI']).copy()
        
        if data_clean.empty:
            st.error("Not enough data for indicators. Try a longer timeframe!")
            st.stop()
        
        # Find swings
        order = 5
        high_idx = argrelextrema(data_clean['High'].values, np.greater, order=order)[0]
        low_idx = argrelextrema(data_clean['Low'].values, np.less, order=order)[0]
        
        # Detect H&S
        patterns = []
        highs = data_clean.iloc[high_idx]
        dates = highs.index
        
        for i in range(2, len(highs) - 2):
            ls = highs['High'].iloc[i-2]
            hd = highs['High'].iloc[i]
            rs = highs['High'].iloc[i+1]
            
            if hd > ls and hd > rs and abs(ls - rs) / hd < 0.05:
                neck_lows = data_clean.loc[dates[i-2]:dates[i+1], 'Low']
                neckline = neck_lows.min()
                
                patterns.append({
                    "head_date": dates[i],
                    "head_price": hd,
                    "neckline": neckline,
                    "left_shoulder": dates[i-2],
                    "right_shoulder": dates[i+1]
                })
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Points", len(data_clean))
            st.metric("Swing Highs Found", len(high_idx))
            st.metric("H&S Patterns", len(patterns))
        
        with col2:
            if patterns:
                latest_pattern = patterns[-1]
                target = latest_pattern['neckline'] - (latest_pattern['head_price'] - latest_pattern['neckline'])
                st.metric("Latest H&S Head Price", f"${latest_pattern['head_price']:.2f}")
                st.metric("Neckline", f"${latest_pattern['neckline']:.2f}")
                st.metric("Bearish Target", f"${target:.2f}")
                st.info("**Bearish signal!** Wait for neckline break to sell/short.")
            else:
                st.warning("No H&S patterns found. Try a volatile stock like TSLA!")
        
        # Plot chart
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data_clean.index, data_clean['Close'], label='Close', color='black', linewidth=1)
        ax.plot(data_clean['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
        
        # Swing highs
        ax.scatter(data_clean.iloc[high_idx].index, data_clean.iloc[high_idx]['High'],
                   color='red', marker='v', s=80, label='Swing High', zorder=5)
        
        # H&S markers
        for p in patterns:
            ax.scatter([p['left_shoulder'], p['head_date'], p['right_shoulder']],
                       [data_clean.loc[p['left_shoulder'],'High'],
                        data_clean.loc[p['head_date'],'High'],
                        data_clean.loc[p['right_shoulder'],'High']],
                       color='purple', s=120, marker='^', zorder=6)
            
            # Fixed line: Added missing )
            ax.text(p['head_date'], p['head_price'] + (data_clean['Close'].max() * 0.02),
                    'H&S', fontsize=14, color='purple', weight='bold', ha='center')
            
            # Neckline
            ax.hlines(p['neckline'], p['left_shoulder'], p['right_shoulder'],
                      color='purple', linestyle='--', linewidth=2, label='Neckline' if p == patterns[0] else "")
        
        ax.set_title(f"{ticker} – Head & Shoulders Detection ({period})", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to bytes for download
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.pyplot(fig)
        st.download_button("Download Chart PNG", buf.getvalue(), f"{ticker}_HS_{period}.png", "image/png")
        
        # List patterns
        if patterns:
            st.subheader("Detected Patterns")
            for p in patterns:
                st.write(f"**Head:** {p['head_date'].date()} | Price: ${p['head_price']:.2f} | Neckline: ${p['neckline']:.2f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
