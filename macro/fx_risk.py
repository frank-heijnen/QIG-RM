"""Parametric FX VaR in EUR base currency.

Extracted from macro/Exchange_rate_risk.ipynb so the model-risk backtest
harness can wrap it as an adapter, and so the monthly FX VaR number can
be produced from a script or a notebook interchangeably.

Conventions
-----------
  - Base currency = EUR. Positions and cash are expressed in EUR.
  - FX tickers use the Yahoo convention "<ccy>EUR=X" -> EUR per <ccy>, so
    a positive log-return means the foreign currency strengthened vs EUR.
  - Volatilities and covariances are annualized with sqrt(252).
  - Horizon is expressed as a fraction of one year (1/12 = one month,
    1/252 = one trading day).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data fetching (thin wrappers around yfinance; kept inside functions so
# the module imports cleanly offline and stays mockable in tests)
# ---------------------------------------------------------------------------

def download_fx_history(
    currencies: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Daily EUR/<ccy> close prices. One column per non-EUR currency."""
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for ccy in currencies:
        raw = yf.download(f"{ccy}EUR=X", start=start, end=end,
                          interval="1d", auto_adjust=True, progress=False)
        s = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        out[ccy] = s.squeeze().dropna()
    return pd.concat(out, axis=1).dropna()


def download_stock_history(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Adjusted close prices per ticker in the stock's listing currency."""
    import yfinance as yf

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    return raw[["Close"]].rename(columns={"Close": tickers[0]})


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def fx_vol_and_covariance(
    fx_prices: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized vol vector and covariance matrix of EUR/<ccy> log-returns."""
    r = np.log(fx_prices).diff().dropna()
    sigma_ann = r.std(ddof=1) * math.sqrt(trading_days)
    cov_ann = r.cov(ddof=1) * trading_days
    return sigma_ann, cov_ann


def estimate_fx_betas(
    stock_prices: pd.DataFrame,
    fx_prices: pd.DataFrame,
    currency_map: Mapping[str, str],
) -> pd.DataFrame:
    """
    For each stock in `currency_map` whose currency != 'EUR', regress its
    local-currency log-return on its EUR/<ccy> FX log-return and record the
    beta.

        r_{i,t}^{ccy} = alpha_i + beta_i * r_{FX,t} + eps_{i,t}

    Returns a DataFrame indexed by ticker with columns ['ccy', 'beta_fx',
    'fx_factor'] where fx_factor = 1 + beta_fx is the loading applied to
    the stock's EUR value when it enters the FX covariance form.
    """
    r_stock = np.log(stock_prices).diff().dropna(how="all")
    r_fx = np.log(fx_prices).diff().dropna()
    rows = []
    for ticker, ccy in currency_map.items():
        if ccy == "EUR":
            continue
        if ticker not in r_stock.columns or ccy not in r_fx.columns:
            continue
        aligned = pd.concat(
            [r_stock[ticker], r_fx[ccy].rename("r_fx")], axis=1
        ).dropna()
        if aligned.empty:
            continue
        r_i = aligned[ticker].to_numpy()
        r_f = aligned["r_fx"].to_numpy()
        var_fx = r_f.var(ddof=1)
        beta = float(np.cov(r_i, r_f, ddof=1)[0, 1] / var_fx) if var_fx > 0 else 0.0
        rows.append({
            "ticker": ticker,
            "ccy": ccy,
            "beta_fx": beta,
            "fx_factor": 1.0 + beta,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(
        columns=["ccy", "beta_fx", "fx_factor"]
    )


def effective_notional_by_currency(
    positions_eur: Mapping[str, float],
    currency_map: Mapping[str, str],
    cash_by_ccy: Mapping[str, float],
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-currency gross and effective notionals in EUR.

        gross_c = sum_{i in c} V_i              +  cash_c
        eff_c   = sum_{i in c} (1 + beta_i) V_i +  cash_c

    Tickers without a fitted beta default to factor = 1 (no natural hedge).
    """
    non_eur = sorted(
        {c for c in currency_map.values() if c != "EUR"}
        | {c for c in cash_by_ccy if c != "EUR"}
    )
    rows = []
    for c in non_eur:
        tickers_in_c = [t for t, cc in currency_map.items() if cc == c]
        gross_stock = sum(positions_eur.get(t, 0.0) for t in tickers_in_c)
        eff_stock = 0.0
        for t in tickers_in_c:
            v = positions_eur.get(t, 0.0)
            factor = float(betas.loc[t, "fx_factor"]) if t in betas.index else 1.0
            eff_stock += factor * v
        cash_c = float(cash_by_ccy.get(c, 0.0))
        rows.append({
            "ccy": c,
            "gross_eur": gross_stock + cash_c,
            "effective_eur": eff_stock + cash_c,
        })
    return pd.DataFrame(rows).set_index("ccy")


# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------

@dataclass
class FxVaRResult:
    per_currency: pd.DataFrame
    portfolio_var_eur: float
    portfolio_var_pct: float
    horizon_years: float
    z: float


def parametric_fx_var(
    notional_by_ccy: pd.DataFrame,
    cov_fx_ann: pd.DataFrame,
    sigma_fx_ann: pd.Series,
    nav_eur: float,
    alpha: float = 0.95,
    horizon_years: float = 1.0 / 12.0,
) -> FxVaRResult:
    """
    Per-currency and portfolio-level parametric FX VaR in EUR using
    effective notionals and the full FX covariance.

        VaR_c   = z_alpha * |N^eff_c| * sigma_c * sqrt(horizon)
        VaR_pf  = z_alpha * sqrt(n^T Sigma_ann n * horizon)
    """
    z = float(norm.ppf(alpha))
    scale = math.sqrt(horizon_years)

    per = notional_by_ccy.copy()
    per["sigma_ann"] = [float(sigma_fx_ann[c]) for c in per.index]
    per["var_eur"] = z * per["effective_eur"].abs() * per["sigma_ann"] * scale

    non_eur = list(per.index)
    n = per["effective_eur"].to_numpy()
    cov = cov_fx_ann.loc[non_eur, non_eur].to_numpy()
    port_var_quad = float(n @ cov @ n)
    sigma_port = math.sqrt(max(port_var_quad, 0.0) * horizon_years)
    var_port = z * sigma_port

    return FxVaRResult(
        per_currency=per[["effective_eur", "sigma_ann", "var_eur"]],
        portfolio_var_eur=var_port,
        portfolio_var_pct=(var_port / nav_eur * 100.0) if nav_eur else 0.0,
        horizon_years=horizon_years,
        z=z,
    )


def compute_fx_var(
    positions_eur: Mapping[str, float],
    currency_map: Mapping[str, str],
    cash_by_ccy: Mapping[str, float],
    stock_prices: pd.DataFrame,
    fx_prices: pd.DataFrame,
    alpha: float = 0.95,
    horizon_years: float = 1.0 / 12.0,
) -> FxVaRResult:
    """One-shot helper -- equivalent to cells 5, 7, 9, 11 of the notebook."""
    sigma_ann, cov_ann = fx_vol_and_covariance(fx_prices)
    betas = estimate_fx_betas(stock_prices, fx_prices, currency_map)
    notional = effective_notional_by_currency(
        positions_eur, currency_map, cash_by_ccy, betas
    )
    nav_eur = sum(positions_eur.values()) + sum(cash_by_ccy.values())
    return parametric_fx_var(
        notional_by_ccy=notional,
        cov_fx_ann=cov_ann,
        sigma_fx_ann=sigma_ann,
        nav_eur=nav_eur,
        alpha=alpha,
        horizon_years=horizon_years,
    )
