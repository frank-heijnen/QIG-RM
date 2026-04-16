"""VaRModel adapter wrapping the parametric FX VaR module (macro/fx_risk.py).

Contract
--------
At calibrate() time the model is handed a DataFrame with MultiIndex columns
(level 0 in {'stock', 'fx'}, level 1 = ticker or currency). It estimates
FX vol / covariance and per-stock FX betas from that history.

At forecast() time `positions` is a pd.Series keyed by ticker of
EUR-denominated values. Cash per currency is pulled via the
`cash_by_ccy_fn` callable supplied at construction, which can be a
static dict (today's snapshot) or a per-date lookup if cash ever
becomes available as a time series.

Assumes a Normal FX return distribution; ES is left for a follow-up.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import pandas as pd

from macro.fx_risk import (
    effective_notional_by_currency,
    estimate_fx_betas,
    fx_vol_and_covariance,
    parametric_fx_var,
)
from model_risk.protocols import VaRForecast


CashLookup = Callable[[Optional[pd.Timestamp]], Mapping[str, float]]


class ParametricFxVaR:
    name = "parametric_fx"

    def __init__(
        self,
        currency_map: Mapping[str, str],
        cash_by_ccy_fn: CashLookup,
    ):
        self.currency_map = dict(currency_map)
        self.cash_by_ccy_fn = cash_by_ccy_fn
        self._betas: Optional[pd.DataFrame] = None
        self._sigma: Optional[pd.Series] = None
        self._cov: Optional[pd.DataFrame] = None

    def calibrate(self, history: pd.DataFrame) -> None:
        if not isinstance(history.columns, pd.MultiIndex):
            raise ValueError(
                "ParametricFxVaR expects MultiIndex columns with level 0 "
                "in {'stock', 'fx'}."
            )
        stock_prices = history.xs("stock", axis=1, level=0)
        fx_prices = history.xs("fx", axis=1, level=0)
        self._sigma, self._cov = fx_vol_and_covariance(fx_prices)
        self._betas = estimate_fx_betas(stock_prices, fx_prices, self.currency_map)

    def forecast(
        self,
        positions: Optional[pd.Series],
        horizon: int = 1,
        alpha: float = 0.99,
    ) -> VaRForecast:
        if self._betas is None:
            raise RuntimeError("Model not calibrated.")
        if positions is None:
            raise ValueError(
                "ParametricFxVaR needs positions (EUR values per ticker)."
            )

        positions_eur = positions.dropna().to_dict()
        date = positions.name if isinstance(positions.name, pd.Timestamp) else None
        cash = dict(self.cash_by_ccy_fn(date))

        notional = effective_notional_by_currency(
            positions_eur=positions_eur,
            currency_map=self.currency_map,
            cash_by_ccy=cash,
            betas=self._betas,
        )
        nav_eur = sum(positions_eur.values()) + sum(cash.values())
        horizon_years = horizon / 252.0

        result = parametric_fx_var(
            notional_by_ccy=notional,
            cov_fx_ann=self._cov,
            sigma_fx_ann=self._sigma,
            nav_eur=nav_eur,
            alpha=alpha,
            horizon_years=horizon_years,
        )

        return VaRForecast(
            var=float(result.portfolio_var_eur),
            es=None,
            meta={
                "per_currency": result.per_currency.to_dict(orient="index"),
                "nav_eur": float(nav_eur),
                "z": result.z,
            },
        )
