import streamlit as st
import plotly.graph_objects as go

from data_utils import download_stock_data

st.set_page_config(page_icon="📊", layout="wide")

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

failed = []

for stock in stocks:

    data = download_stock_data(
        stock,
        period="1y",
        show_error=False,
    )

    if data is None:
        failed.append(stock)
        continue

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name=stock
    ))

if failed:
    st.warning(
        "Could not load data for: " + ", ".join(failed)
        + ". Yahoo Finance may be rate limiting requests; try again shortly."
    )

if not stocks:
    st.info("Select at least one stock to compare.")

fig.update_layout(
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig, width="stretch")