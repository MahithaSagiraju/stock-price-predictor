"""Shared helpers for the AI Stock Predictor app."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


@st.cache_data(ttl=900, show_spinner=False)
def _download(symbol: str, period: str) -> pd.DataFrame:
    return yf.download(
        symbol,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=False,
    )


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.loc[:, ~data.columns.duplicated()]


def download_stock_data(
    symbol: str,
    period: str = "1y",
    retries: int = 3,
    pause: float = 1.5,
    show_error: bool = True,
) -> pd.DataFrame | None:
    """Download OHLCV data for ``symbol``, retrying transient failures.

    Yahoo Finance occasionally returns an empty frame, or raises
    ``ValueError: No objects to concatenate``, for valid symbols. Retrying a
    few times keeps a single bad response from crashing a page. Returns
    ``None`` when the data could not be loaded.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        if show_error:
            st.error("Please enter a stock symbol.")
        return None

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            data = _download(symbol, period)
        except Exception as exc:  # noqa: BLE001 - surface any yfinance failure
            last_error = exc
            data = None

        if data is not None and not data.empty:
            data = _flatten_columns(data)
            if all(column in data.columns for column in REQUIRED_COLUMNS):
                return data
            last_error = ValueError(f"Missing price columns for '{symbol}'.")

        # Do not let an empty/failed response stay cached for the TTL.
        _download.clear()
        if attempt < retries:
            time.sleep(pause)

    if show_error:
        st.error(
            f"Could not load price data for **{symbol}**. Yahoo Finance may be "
            "rate limiting requests, or the symbol may be invalid. "
            "Please try again in a moment."
        )
        if last_error is not None:
            st.caption(f"Details: {last_error}")
    return None
