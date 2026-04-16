"""Walk-forward engine for VaR model backtests.

The runner is deliberately thin: it owns the no-lookahead slicing of
history, the refit cadence, and the breach determination. Everything
model-specific lives behind the VaRModel protocol.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from model_risk.protocols import VaRModel


RefitCadence = Union[str, int]


def run_backtest(
    model: VaRModel,
    history: pd.DataFrame,
    positions: Optional[pd.DataFrame],
    hpl: pd.Series,
    window: tuple[str, str],
    alpha: float = 0.99,
    horizon: int = 1,
    refit_every: RefitCadence = "W",
) -> pd.DataFrame:
    """
    Run a walk-forward VaR backtest.

    Parameters
    ----------
    model
        Anything implementing the VaRModel protocol.
    history
        Data the model needs for calibration (prices, returns, factors).
        The model is only ever shown rows strictly before the forecast date.
    positions
        Daily positions keyed by date x ticker. Pass None for models that
        work on portfolio-level P&L and do not need asset-level weights.
    hpl
        Realised hypothetical P&L in EUR indexed by date. Build this with
        pnl.compute_hpl_eur or supply a pre-computed series.
    window
        Inclusive (start, end) date range for the backtest.
    alpha
        Confidence level, e.g. 0.99 for 99% VaR.
    horizon
        Forecast horizon in trading days.
    refit_every
        'D' = every day, 'W' = first day of each ISO week, 'M' = first day
        of each calendar month, or an int = every N trading days.

    Returns
    -------
    DataFrame indexed by date with columns: var, es, hpl, breach.
    """
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    dates = history.loc[start:end].index

    records = []
    last_refit: Optional[pd.Timestamp] = None

    for t in dates:
        if _should_refit(t, last_refit, refit_every, dates):
            model.calibrate(history.loc[history.index < t])
            last_refit = t

        pos_tm1 = None
        if positions is not None:
            lagged = positions.shift(1)
            if t in lagged.index:
                pos_tm1 = lagged.loc[t]

        forecast = model.forecast(pos_tm1, horizon=horizon, alpha=alpha)

        realized = hpl.get(t, np.nan)
        breach = (realized < -forecast.var) if pd.notna(realized) else np.nan

        records.append({
            "date": t,
            "var": forecast.var,
            "es": forecast.es,
            "hpl": realized,
            "breach": breach,
        })

    return pd.DataFrame(records).set_index("date")


def _should_refit(
    t: pd.Timestamp,
    last_refit: Optional[pd.Timestamp],
    cadence: RefitCadence,
    dates: pd.DatetimeIndex,
) -> bool:
    if last_refit is None:
        return True
    if isinstance(cadence, int):
        return (dates.get_loc(t) - dates.get_loc(last_refit)) >= cadence
    if cadence == "D":
        return True
    if cadence == "W":
        return t.isocalendar().week != last_refit.isocalendar().week \
            or t.isocalendar().year != last_refit.isocalendar().year
    if cadence == "M":
        return (t.year, t.month) != (last_refit.year, last_refit.month)
    raise ValueError(f"Unrecognized refit cadence: {cadence!r}")
