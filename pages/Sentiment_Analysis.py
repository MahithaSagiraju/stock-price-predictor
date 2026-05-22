import streamlit as st
from textblob import TextBlob

st.set_page_config(layout="wide")

st.title("📰 Sentiment Analysis")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

news_headlines = [
    f"{stock} stock shows strong quarterly earnings growth",
    f"Analysts predict positive future for {stock}",
    f"{stock} market demand increases globally",
    f"{stock} faces market competition",
    f"{stock} stock volatility increases"
]

for headline in news_headlines:

    analysis = TextBlob(headline)

    polarity = analysis.sentiment.polarity

    st.subheader(headline)

    if polarity > 0:

        st.success(
            f"Positive Sentiment ({polarity:.2f})"
        )

    elif polarity < 0:

        st.error(
            f"Negative Sentiment ({polarity:.2f})"
        )

    else:

        st.info(
            f"Neutral Sentiment ({polarity:.2f})"
        )