import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 Multi Stock Comparison")

stocks = st.multiselect(
    "Select Stocks",
    [
        'AAPL',
        'TSLA',
        'MSFT',
        'GOOGL',
        'AMZN',
        'META',
        'TCS.NS',
        'INFY.NS',
        'RELIANCE.NS'
    ],
    default=['AAPL','TSLA']
)

fig = go.Figure()

for stock in stocks:

    data = yf.download(
        stock,
        period="1y"
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name=stock
    ))

fig.update_layout(
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig, use_container_width=True)