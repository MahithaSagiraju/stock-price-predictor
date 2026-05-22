import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 Stock Dashboard")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

data = yf.download(stock, period="1y")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Price",
    f"${round(float(data['Close'].iloc[-1]),2)}"
)

col2.metric(
    "Highest Price",
    f"${round(float(data['High'].max()),2)}"
)

col3.metric(
    "Lowest Price",
    f"${round(float(data['Low'].min()),2)}"
)

change = data['Close'].iloc[-1] - data['Close'].iloc[-2]

col4.metric(
    "Daily Change",
    f"{round(float(change),2)}"
)

st.subheader("📈 Stock Price Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Close Price'
))

fig.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Latest Data")

st.write(data.tail())