"""Cross-sectional momentum -- long top quantile, short bottom quantile.

Signals are raw +1 / -1 / 0 per asset per day; the engine normalises rows
to respect `risk.max_leverage` and imposes the configured rebalance
cadence, so this class stays small and testable.
"""
from __future__ import annotations

import pandas as pd


class MomentumStrategy:
    def __init__(self, lookback: int, top_quantile: float):
        if not 0.0 < top_quantile < 0.5:
            raise ValueError("top_quantile must lie in (0, 0.5).")
        self.lookback = int(lookback)
        self.top_quantile = float(top_quantile)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change(self.lookback)
        ranks = returns.rank(axis=1, pct=True)
        longs = (ranks >= 1.0 - self.top_quantile).astype(int)
        shorts = (ranks <= self.top_quantile).astype(int) * -1
        return longs + shorts
