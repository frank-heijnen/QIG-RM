"""Types and protocol every VaR model must satisfy to plug into the runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import pandas as pd


@dataclass
class VaRForecast:
    """A single day's VaR / ES forecast expressed as positive EUR losses."""
    var: float
    es: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass
class TestResult:
    """Outcome of a statistical coverage / independence test."""
    name: str
    statistic: float
    pvalue: float
    reject_5pct: bool
    verdict: str


class VaRModel(Protocol):
    """
    Contract every adapter must implement.

    The runner calls `calibrate` periodically (daily / weekly / monthly) with
    a history slice that contains no future information, then calls
    `forecast` every day with the latest positions. Splitting the two lets
    expensive estimations (GARCH, bootstraps) run on the chosen cadence
    while forecasts stay cheap.
    """

    name: str

    def calibrate(self, history: pd.DataFrame) -> None:
        """Re-estimate parameters from the supplied history (strictly lagged)."""

    def forecast(
        self,
        positions: Optional[pd.Series],
        horizon: int = 1,
        alpha: float = 0.99,
    ) -> VaRForecast:
        """Return a VaR / ES forecast for the given positions."""
