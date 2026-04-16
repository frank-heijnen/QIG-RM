"""Strategy protocol used by the config-driven backtest engine.

A strategy is a Python class with:
  - `__init__(**parameters)` taking only its own hyper-parameters;
  - `generate_signals(prices) -> pd.DataFrame` returning a target-weight
    (or signal) frame aligned to `prices.index` and `prices.columns`.

The engine is responsible for data loading, rebalance cadence, position
sizing / leverage caps, trading costs, and accounting -- none of that
leaks into the strategy class. This separation is the whole point of
the config-driven architecture described in
`backtesting/report/backtesting_report.tex`.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        prices
            Daily close prices, rows = dates, columns = tickers.

        Returns
        -------
        DataFrame with the same index and columns as `prices`. Values are
        per-asset signals / target weights. Positive = long, negative =
        short, zero = flat. The engine normalises rows to respect the
        configured max leverage, so strategies do not need to.
        """
        ...
