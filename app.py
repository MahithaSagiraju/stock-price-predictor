import streamlit as st

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: white;
    }

    h1 {
        color: white;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 AI Stock Predictor")

st.markdown(
    """
    ## Welcome to AI Stock Analytics Platform

    ### Features
    - 📊 Dashboard
    - 📈 Technical Analysis
    - 🤖 AI Prediction
    - 📰 Sentiment Analysis
    - 📉 Multi Stock Comparison

    Use the sidebar to navigate.
    """
)

st.image("assets/stock-graph.svg", use_container_width=True)
