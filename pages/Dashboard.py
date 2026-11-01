import streamlit as st
import plotly.graph_objects as go

from data_utils import download_stock_data

st.set_page_config(page_icon="📊", layout="wide")

st.title("📊 Stock Dashboard")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

data = download_stock_data(stock, period="1y")

if data is None:
    st.stop()

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

st.plotly_chart(fig, width="stretch")

st.subheader("📋 Latest Data")

st.write(data.tail())