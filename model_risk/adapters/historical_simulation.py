"""Historical simulation VaR -- the simplest concrete adapter.

Used as the first end-to-end smoke test of the harness. Portfolio-level:
expects the history as a one-column DataFrame of EUR daily P&L and ignores
positions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from model_risk.protocols import VaRForecast


class HistoricalSimulationVaR:
    """
    VaR as the (1-alpha) empirical quantile of the trailing window,
    expressed as a positive EUR loss. ES is the mean of the losses beyond
    that quantile.
    """

    name = "historical_simulation"

    def __init__(self, window: int = 500):
        self.window = window
        self._tail: Optional[np.ndarray] = None

    def calibrate(self, history: pd.DataFrame) -> None:
        if history.shape[1] != 1:
            raise ValueError(
                "HistoricalSimulationVaR expects a one-column portfolio P&L "
                f"history; got {history.shape[1]} columns."
            )
        series = history.iloc[:, 0].dropna()
        if len(series) < self.window:
            raise ValueError(
                f"Need at least {self.window} observations; got {len(series)}."
            )
        self._tail = series.iloc[-self.window:].to_numpy()

    def forecast(
        self,
        positions: Optional[pd.Series],
        horizon: int = 1,
        alpha: float = 0.99,
    ) -> VaRForecast:
        if self._tail is None:
            raise RuntimeError("Model not calibrated.")
        if horizon != 1:
            raise NotImplementedError("Only 1-day horizon is supported in v1.")

        q = float(np.quantile(self._tail, 1.0 - alpha))
        var = -q if q < 0 else 0.0
        tail_losses = self._tail[self._tail <= q]
        es = float(-tail_losses.mean()) if len(tail_losses) > 0 else var
        return VaRForecast(var=var, es=es, meta={"window": self.window})
