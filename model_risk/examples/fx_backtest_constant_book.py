"""End-to-end parametric FX VaR backtest on the current book.

Since we have no historical position snapshots, we run the harness under
a constant-book assumption: today's positions and cash are taken as
fixed and we ask -- given historical market data over the backtest
window, how would the parametric FX VaR model have called risk on this
book, and how often would the realised hypothetical P&L (from FX moves)
have broken through the VaR line.

This is a model-sensitivity check, not a Basel-compliant backtest: the
"portfolio held in 2023" is counter-factual. Coverage tests still run
and are informative about model mechanics, but a breach cluster here
could reflect either model weakness or the fact that we simply did not
hold these names then.

Run:
    python -m model_risk.examples.fx_backtest_constant_book
"""
from __future__ import annotations

import pandas as pd

from macro.fx_risk import download_fx_history, download_stock_history
from model_risk.adapters.parametric_fx import ParametricFxVaR
from model_risk.diagnostics.coverage import (
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
    rolling_traffic_light,
)
from model_risk.pnl import compute_hpl_eur, constant_book_frames
from model_risk.runner import run_backtest


# ---------------------------------------------------------------------------
# 1. Current book -- lifted verbatim from macro/Exchange_rate_risk.ipynb
# ---------------------------------------------------------------------------
POSITIONS_EUR_NOW = {
    "APP": 693.76,
    "GOOG": 373.60,
    "GOOGL": 375.87,
    "HUM": 337.04,
    "LLY": 309.38,
    "LVS": 359.10,
    "NEM": 467.26,
    "NOW": 283.64,
    "SYF": 403.50,
    "TTD": 476.73,
    "UNH": 377.79,
    "LIGHT.AS": 523.56,
    "PNDORA.CO": 409.94,
}
CURRENCY_MAP = {
    "APP": "USD", "GOOG": "USD", "GOOGL": "USD", "HUM": "USD", "LLY": "USD",
    "LVS": "USD", "NEM": "USD", "NOW": "USD", "SYF": "USD", "TTD": "USD",
    "UNH": "USD",
    "LIGHT.AS": "EUR",
    "PNDORA.CO": "DKK",
}
CASH_BY_CCY_NOW = {
    "EUR": 1066.24,
    "USD": 1274.09,
    "CHF":  488.87,
    "DKK":   10.10,
}

HIST_START = "2023-01-01"
HIST_END   = "2025-10-16"
BACKTEST_START = "2024-06-03"
BACKTEST_END   = "2025-10-15"
ALPHA = 0.99
REFIT = "M"


def main() -> None:
    # 2. Download prices
    non_eur_ccys = sorted({c for c in CURRENCY_MAP.values() if c != "EUR"}
                         | {c for c in CASH_BY_CCY_NOW if c != "EUR"})
    fx_prices = download_fx_history(non_eur_ccys, HIST_START, HIST_END)
    stock_prices = download_stock_history(list(CURRENCY_MAP), HIST_START, HIST_END)

    # 3. Build EUR-denominated prices per ticker (stock price * spot for
    #    non-EUR listings; stock price alone for EUR listings).
    common_dates = stock_prices.index.intersection(fx_prices.index)
    stock_prices = stock_prices.loc[common_dates]
    fx_prices = fx_prices.loc[common_dates]

    eur_prices = pd.DataFrame(index=common_dates)
    for ticker, ccy in CURRENCY_MAP.items():
        if ticker not in stock_prices.columns:
            continue
        if ccy == "EUR":
            eur_prices[ticker] = stock_prices[ticker]
        else:
            eur_prices[ticker] = stock_prices[ticker] * fx_prices[ccy]
    eur_prices = eur_prices.dropna()

    # 4. Broadcast today's book backwards under constant quantities.
    quantities, eur_values = constant_book_frames(
        positions_eur_now=POSITIONS_EUR_NOW,
        eur_prices=eur_prices,
        anchor_date=eur_prices.index[-1],
    )

    # 5. Realised hypothetical P&L in EUR (all risk, not just FX).
    hpl = compute_hpl_eur(quantities, eur_prices)

    # 6. MultiIndex history for the FX adapter.
    history = pd.concat(
        {"stock": stock_prices[list(CURRENCY_MAP)], "fx": fx_prices},
        axis=1,
    ).dropna()

    # 7. Run the backtest.
    model = ParametricFxVaR(
        currency_map=CURRENCY_MAP,
        cash_by_ccy_fn=lambda date: CASH_BY_CCY_NOW,
    )
    result = run_backtest(
        model=model,
        history=history,
        positions=eur_values,
        hpl=hpl,
        window=(BACKTEST_START, BACKTEST_END),
        alpha=ALPHA,
        refit_every=REFIT,
    )

    # 8. Report.
    n = len(result)
    breaches = int(result["breach"].sum())
    print(f"Backtest window   : {BACKTEST_START} -> {BACKTEST_END}  ({n} days)")
    print(f"Model             : {model.name}, alpha={ALPHA}, refit={REFIT}")
    print(f"Breaches          : {breaches}  ({breaches/n:.2%})")
    print(f"VaR (EUR)         : min={result['var'].min():.2f}  "
          f"mean={result['var'].mean():.2f}  max={result['var'].max():.2f}")
    print(f"HPL (EUR)         : min={result['hpl'].min():.2f}  "
          f"max={result['hpl'].max():.2f}")
    print()
    print("Coverage tests")
    print("  Kupiec POF        :", kupiec_pof(result["breach"], alpha=ALPHA))
    print("  Christoffersen ind:", christoffersen_independence(result["breach"]))
    print("  Christoffersen CC :", christoffersen_cc(result["breach"], alpha=ALPHA))
    print()
    print("Rolling traffic light (tail):")
    print(rolling_traffic_light(result["breach"]).tail())
    print()
    print("Caveat: HPL here reflects total P&L on the book (equity + FX),")
    print("while the parametric FX VaR only forecasts the FX component.")
    print("The breach count over-states true FX-model failure until HPL is")
    print("decomposed into FX vs equity contributions (next step).")


if __name__ == "__main__":
    main()
