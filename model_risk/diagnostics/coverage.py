"""Basel traffic-light, Kupiec POF, Christoffersen independence and CC tests.

All tests take the per-day `breach` column produced by runner.run_backtest
(1 = loss exceeded VaR, 0 = did not, NaN = unknown / no P&L that day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from model_risk.protocols import TestResult


def _safe_log(a: float) -> float:
    return float(np.log(a)) if a > 0 else 0.0


def kupiec_pof(breaches: pd.Series, alpha: float) -> TestResult:
    """
    Kupiec proportion-of-failures test (unconditional coverage).

    H0: P(breach) == 1 - alpha.  LR ~ chi2(1).
    """
    b = breaches.dropna().astype(int)
    n = len(b)
    x = int(b.sum())
    p = 1.0 - alpha
    phat = x / n if n > 0 else 0.0

    log_l0 = x * _safe_log(p) + (n - x) * _safe_log(1 - p)
    log_l1 = x * _safe_log(phat) + (n - x) * _safe_log(1 - phat)
    lr = -2.0 * (log_l0 - log_l1)
    pvalue = float(1.0 - chi2.cdf(lr, df=1))
    reject = pvalue < 0.05
    verdict = f"observed breach rate {phat:.2%} vs expected {p:.2%}"
    return TestResult("kupiec_pof", float(lr), pvalue, reject, verdict)


def christoffersen_independence(breaches: pd.Series) -> TestResult:
    """
    Christoffersen (1998) independence test.

    H0: breaches form an iid sequence (no day-after-day clustering).
    LR ~ chi2(1).
    """
    b = breaches.dropna().astype(int).to_numpy()
    if len(b) < 2:
        return TestResult("christoffersen_ind", float("nan"), float("nan"),
                          False, "not enough data")

    n00 = int(((b[:-1] == 0) & (b[1:] == 0)).sum())
    n01 = int(((b[:-1] == 0) & (b[1:] == 1)).sum())
    n10 = int(((b[:-1] == 1) & (b[1:] == 0)).sum())
    n11 = int(((b[:-1] == 1) & (b[1:] == 1)).sum())

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    log_l_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    log_l_alt = (
        n00 * _safe_log(1 - pi0) + n01 * _safe_log(pi0)
        + n10 * _safe_log(1 - pi1) + n11 * _safe_log(pi1)
    )
    lr = -2.0 * (log_l_null - log_l_alt)
    pvalue = float(1.0 - chi2.cdf(lr, df=1))
    reject = pvalue < 0.05
    verdict = f"P(breach|prev breach)={pi1:.2%}, P(breach|no prev)={pi0:.2%}"
    return TestResult("christoffersen_ind", float(lr), pvalue, reject, verdict)


def christoffersen_cc(breaches: pd.Series, alpha: float) -> TestResult:
    """
    Christoffersen conditional coverage test.

    Combines Kupiec (unconditional) and the independence test:
        LR_cc = LR_uc + LR_ind ~ chi2(2).
    """
    uc = kupiec_pof(breaches, alpha)
    ind = christoffersen_independence(breaches)
    lr = uc.statistic + ind.statistic
    pvalue = float(1.0 - chi2.cdf(lr, df=2))
    reject = pvalue < 0.05
    return TestResult(
        "christoffersen_cc", float(lr), pvalue, reject,
        f"{uc.verdict}; {ind.verdict}",
    )


_BASEL_ADDONS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def traffic_light(breaches_last_250: int) -> str:
    """Basel zone for the most recent 250 trading days of 99% VaR breaches."""
    if breaches_last_250 <= 4:
        return "green"
    if breaches_last_250 <= 9:
        return "yellow"
    return "red"


def basel_multiplier(breaches_last_250: int) -> float:
    """
    Basel VaR multiplier for market-risk capital.
    Base 3.0 in the green zone, scaling to 4.0 in red.
    """
    if breaches_last_250 <= 4:
        return 3.0
    if breaches_last_250 <= 9:
        return 3.0 + _BASEL_ADDONS[breaches_last_250]
    return 4.0


def rolling_traffic_light(breaches: pd.Series, window: int = 250) -> pd.DataFrame:
    """Rolling breach count, zone and Basel multiplier."""
    b = breaches.dropna().astype(int)
    count = b.rolling(window).sum()
    df = pd.DataFrame({"breaches_250": count})
    df["zone"] = df["breaches_250"].apply(
        lambda x: traffic_light(int(x)) if pd.notna(x) else None
    )
    df["multiplier"] = df["breaches_250"].apply(
        lambda x: basel_multiplier(int(x)) if pd.notna(x) else None
    )
    return df
