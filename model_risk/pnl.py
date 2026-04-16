"""P&L construction utilities.

Two responsibilities:

  1. `compute_hpl_eur` -- turn a (date x ticker) quantities frame and a
     (date x ticker) EUR prices frame into a daily hypothetical P&L
     series. Separated from the runner so the same HPL can be reused
     across models.

  2. `constant_book_frames` -- broadcast today's portfolio backward as
     if it had been held unchanged over the historical window. This is
     what we use when no true position history is available: it lets us
     run the harness on real market data against the book we hold now,
     with the understanding that the resulting backtest is a model
     sensitivity check rather than a historically faithful Basel run.
"""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd


def compute_hpl_eur(
    positions: pd.DataFrame,
    prices_eur: pd.DataFrame,
) -> pd.Series:
    """
    Hypothetical daily P&L in EUR under frozen positions.

    Positions are lagged one day: the position at the close of t-1 is held
    through day t and revalued at t's close.

        hpl_t = sum_i positions_{t-1, i} * (prices_{t, i} - prices_{t-1, i})

    Both inputs must already be in EUR. FX translation (quantities and
    prices in foreign currencies) is the caller's responsibility -- see
    `macro.fx_risk` for the currency classification logic.
    """
    common = prices_eur.columns.intersection(positions.columns)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between positions and prices.")
    price_diff = prices_eur[common].diff()
    frozen_positions = positions[common].shift(1)
    return (frozen_positions * price_diff).sum(axis=1)


def constant_book_frames(
    positions_eur_now: Mapping[str, float],
    eur_prices: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Broadcast today's EUR-valued book backwards under constant quantities.

    Parameters
    ----------
    positions_eur_now
        Mapping ticker -> EUR value as of `anchor_date` (defaults to the
        last date in `eur_prices`).
    eur_prices
        Date x ticker frame of EUR-denominated prices (stock price * FX
        spot for non-EUR listings; stock price for EUR listings).
    anchor_date
        Date at which the supplied EUR values apply. Must be present in
        eur_prices.index. Defaults to eur_prices.index[-1].

    Returns
    -------
    quantities, eur_values : (DataFrame, DataFrame)
        Both are date x ticker frames. `quantities` holds the implied
        constant share count (EUR value at anchor / EUR price at anchor),
        broadcast across all dates. `eur_values` holds quantity * price
        per day, i.e. the book's mark-to-market EUR value through time.

    The caller typically feeds `quantities` into `compute_hpl_eur` to
    produce realised HPL, and `eur_values` to the runner as the
    `positions` argument for models that consume EUR-value-per-ticker
    (e.g. the parametric FX adapter).
    """
    common = [t for t in positions_eur_now if t in eur_prices.columns]
    missing = [t for t in positions_eur_now if t not in eur_prices.columns]
    if missing:
        raise ValueError(f"Tickers not in eur_prices: {missing}")

    anchor = anchor_date if anchor_date is not None else eur_prices.index[-1]
    if anchor not in eur_prices.index:
        raise ValueError(f"anchor_date {anchor} is not in eur_prices.index")

    anchor_prices = eur_prices.loc[anchor, common]
    qty = pd.Series(
        {t: positions_eur_now[t] / float(anchor_prices[t]) for t in common},
        dtype=float,
    )

    quantities = pd.DataFrame(
        [qty.values] * len(eur_prices.index),
        index=eur_prices.index,
        columns=common,
    )
    eur_values = eur_prices[common].multiply(qty, axis=1)
    return quantities, eur_values
