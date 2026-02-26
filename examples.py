"""
Factor extraction from S&P 500 monthly returns using PPCA.

Stocks enter and leave the index over time, so a returns matrix starting
from 2000 is naturally sparse—roughly 30-40% missing for 100 names.
PPCA handles this directly without any upfront imputation.

Requires: yfinance, pandas  (pip install yfinance pandas)
"""

import io

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ppca import PPCA


def sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (compatible; research/examples)"},
        timeout=10,
    )
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def monthly_returns(
    tickers: list[str],
    start: str = "2000-01-01",
    top_n: int | None = None,
    liquidity_lookback_days: int = 252,
) -> pd.DataFrame:
    """Download adjusted closes and resample to month-end log returns.

    If *top_n* is given, only the most liquid stocks (by average daily
    dollar volume over the last *liquidity_lookback_days* trading days)
    are kept.
    """
    fields = ["Close", "Volume"] if top_n else ["Close"]
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)[fields]

    if top_n:
        close = data["Close"]
        volume = data["Volume"]
        dollar_vol = (close * volume).iloc[-liquidity_lookback_days:]
        top = dollar_vol.mean().nlargest(top_n).index
        px = close[top]
    else:
        px = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data

    px = px.dropna(how="all", axis=1)
    return px.resample("ME").last().pipe(np.log).diff().iloc[1:] * 100.0


if __name__ == "__main__":
    tickers = sp500_tickers()
    n_stocks = 250
    print(f"Downloading data for {len(tickers)} S&P 500 stocks, keeping top {n_stocks} by liquidity ...")

    ret = monthly_returns(tickers, top_n=n_stocks)
    X = ret.values
    mask = ret.isna().values

    print(f"{X.shape[0]} months × {X.shape[1]} stocks  |  {100 * mask.mean():.1f}% missing\n")

    # Twelve latent factors loosely correspond to broad market and sector drivers.
    # The first loading is constrained positive so that factor 1 behaves like
    # a "market" factor (all stocks load in the same direction).
    n_components = 12
    signs = [1] + [0] * (n_components - 1)
    ppca = PPCA(n_components=n_components, max_iterations=300, tol=1e-3, loading_signs=signs)
    factors = ppca.fit_transform(X, missing_mask=mask)

    print(f"Latent factors:        {factors.shape}")
    print(f"Explained variance:    {ppca.explained_variance_ratio_.round(3)}")
    print(f"Total explained var:    {ppca.explained_variance_ratio_.sum():.3f}")
    print(f"Reconstruction error:  {ppca.reconstruction_error(X, missing_mask=mask):.5f}")

    # inverse_transform fills the gaps—each missing return is estimated
    # from the low-rank factor structure learned on the observed data.
    X_filled = ppca.inverse_transform(factors)
    print(f"\nImputed {mask.sum()} missing monthly returns via rank-5 model.")
