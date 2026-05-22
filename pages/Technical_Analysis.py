import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📈 Technical Analysis")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

data = yf.download(stock, period="2y")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Candlestick
st.subheader("📊 Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
))

fig.update_layout(
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# RSI
st.subheader("📉 RSI Indicator")

delta = data['Close'].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss

rsi = 100 - (100 / (1 + rs))

fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
    x=data.index,
    y=rsi,
    name='RSI'
))

fig_rsi.add_hline(y=70)
fig_rsi.add_hline(y=30)

fig_rsi.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_rsi, use_container_width=True)

# MACD
st.subheader("📈 MACD Indicator")

ema12 = data['Close'].ewm(span=12).mean()
ema26 = data['Close'].ewm(span=26).mean()

macd = ema12 - ema26
signal = macd.ewm(span=9).mean()

fig_macd = go.Figure()

fig_macd.add_trace(go.Scatter(
    x=data.index,
    y=macd,
    name='MACD'
))

fig_macd.add_trace(go.Scatter(
    x=data.index,
    y=signal,
    name='Signal'
))

fig_macd.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_macd, use_container_width=True)

# Bollinger Bands
st.subheader("📊 Bollinger Bands")

ma20 = data['Close'].rolling(20).mean()
std20 = data['Close'].rolling(20).std()

upper = ma20 + 2 * std20
lower = ma20 - 2 * std20

fig_bb = go.Figure()

fig_bb.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name='Close'
))

fig_bb.add_trace(go.Scatter(
    x=data.index,
    y=upper,
    name='Upper Band'
))

fig_bb.add_trace(go.Scatter(
    x=data.index,
    y=lower,
    name='Lower Band'
))

fig_bb.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig_bb, use_container_width=True)