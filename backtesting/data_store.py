"""Data-access layer for the strategy backtester.

Provides a single `DataStore.load_universe(...)` entry point so strategies
and the engine never touch data-source details directly. Today this wraps
yfinance; migrating to a local Parquet mirror or a SQL backend later only
requires touching this file.

Base currency: native per ticker (USD for US listings, etc). This
deliberately differs from the model-risk workstream, which translates
everything to EUR for regulatory reporting. Strategy research typically
runs in listing currency so as not to mix in FX noise; if an EUR-based
backtest is ever needed, add an optional currency-translation step here.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


class DataStore:
    def __init__(self, source: str = "yfinance"):
        if source != "yfinance":
            raise NotImplementedError(
                f"Only the 'yfinance' source is supported in v1; got {source!r}."
            )
        self.source = source

    def load_universe(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        freq: str = "1D",
    ) -> pd.DataFrame:
        """Daily adjusted close prices. Rows = dates, columns = tickers."""
        if freq != "1D":
            raise NotImplementedError(
                f"Only daily frequency is supported in v1; got {freq!r}."
            )
        import yfinance as yf

        tickers = list(tickers)
        raw = yf.download(
            tickers, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        if isinstance(raw.columns, pd.MultiIndex):
            closes = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
        else:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
        return closes.dropna(how="all").sort_index()
