
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Iterable
import matplotlib.pyplot as plt



def _weighted_quantile(x, q, w=None):
    x = np.asarray(x, float)
    if w is None:
        return np.quantile(x, q)
    w = np.asarray(w, float)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cutoff = q * cw[-1]
    i = np.searchsorted(cw, cutoff)
    i = min(max(i, 0), len(x)-1)
    return float(x[i])

def fan_stats(V_paths: np.ndarray, weights=None, qs=(0.01,0.05,0.25,0.5,0.75,0.95,0.99)):
    """
    Per-time weighted quantiles for a fan chart.
    Returns dict: {'t': range(T+1), 'q': array(len(qs), T+1)}
    """
    Tp1, M = V_paths.shape
    qvals = np.zeros((len(qs), Tp1))
    for t in range(Tp1):
        for i,q in enumerate(qs):
            qvals[i,t] = _weighted_quantile(V_paths[t,:], q, weights)
    return {'t': np.arange(Tp1), 'qs': np.array(qs), 'q': qvals}

def tail_set(V_paths: np.ndarray, horizon: int, alpha: float, weights=None):
    """
    Indices of tail scenarios at 'horizon' (in steps) using VaR_{alpha}.
    """
    V0 = V_paths[0,:]
    Vh = V_paths[horizon,:]
    losses = V0 - Vh
    var = _weighted_quantile(losses, alpha/100.0, weights)
    if weights is None:
        idx = np.where(losses >= var - 1e-12)[0]
    else:
        # include all with loss >= VaR; keep order by loss asc
        idx = np.argsort(losses)
        idx = idx[losses[idx] >= var - 1e-12]
    return idx, var

def es_path(V_paths: np.ndarray, tail_idx: np.ndarray, weights=None):
    """
    Expected Shortfall path: average path across tail_idx.
    """
    if len(tail_idx)==0:
        return V_paths[:, :1].mean(axis=1)
    if weights is None:
        return V_paths[:, tail_idx].mean(axis=1)
    w = np.asarray(weights, float)[tail_idx]
    w = w / (w.sum() + 1e-16)
    return (V_paths[:, tail_idx] * w[None,:]).sum(axis=1)

def drawdown_matrix(V_paths: np.ndarray):
    """
    Return drawdown matrix (T+1, M) in % from running peak.
    """
    peak = np.maximum.accumulate(V_paths, axis=0)
    dd = 1.0 - (V_paths / peak)
    return dd
# --- Robust PSD projection tweak (safe to replace the old one) ---
def _nearest_pd(A: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """Project covariance to nearest PSD; scrub non-finite and add tiny jitter if needed."""
    B = (A.values + A.values.T) / 2
    B = np.where(np.isfinite(B), B, 0.0)  # scrub NaN/inf
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        B = B + eps * np.eye(B.shape[0])
        eigvals, eigvecs = np.linalg.eigh(B)
    eigvals_clipped = np.clip(eigvals, eps, None)
    B_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
    B_psd = (B_psd + B_psd.T) / 2
    return pd.DataFrame(B_psd, index=A.index, columns=A.columns)


# --- Internal helper: align & clean inputs for a given portfolio ticker set ---
def _prepare_inputs_for_tail(historic_prices: pd.DataFrame,
                             mu_ann: pd.Series,
                             Sigma_ann: pd.DataFrame,
                             portfolio_tickers) -> tuple:
    """
    Returns (S0_clean, mu_clean, Sigma_clean, dropped) aligned to 'portfolio_tickers'.
    Drops tickers with non-finite entries; adds tiny jitter for numerical stability.
    """
    tickers = pd.Index(portfolio_tickers)
    common = tickers.intersection(historic_prices.columns).intersection(mu_ann.index).intersection(Sigma_ann.index)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between portfolio and fitted parameters.")

    S0 = historic_prices.iloc[-1].reindex(common)
    mu = mu_ann.reindex(common)
    Sigma = Sigma_ann.reindex(index=common, columns=common)

    good = pd.Index([t for t in common
                     if np.isfinite(S0.loc[t]) and np.isfinite(mu.loc[t])
                     and np.isfinite(Sigma.loc[t]).all()])

    dropped = [t for t in common if t not in good]

    S0_clean = S0.reindex(good)
    mu_clean = mu.reindex(good)
    Sigma_clean = Sigma.reindex(index=good, columns=good)

    # tiny jitter to avoid numerical issues on near-singular Σ
    if len(good) > 0:
        Sigma_clean = Sigma_clean + 1e-12 * np.eye(len(good))

    return S0_clean, mu_clean, Sigma_clean, dropped


def plot_worst_paths(t, V_paths, k=20):
    final_vals = V_paths[-1, :]
    idx = np.argsort(final_vals)[:k]
    for i in idx:
        plt.plot(t, V_paths[:, i], alpha=0.7)
    plt.title(f"Worst {k} portfolio paths")
# --- Public convenience: one-call rare-event run for a portfolio ---
def simulate_portfolio_tail(historic_prices: pd.DataFrame,
                            portfolio_tickers,
                            *,
                            T: float,
                            M: int,
                            N: int = 252,
                            # one of these two:
                            shares: np.ndarray | None = None,
                            weights: np.ndarray | pd.Series | dict | None = None,
                            budget: float | None = None,
                            # rare-event knobs:
                            nu: float = 5.0,
                            p_sys: float = 0.01, mu_sys: float = 0, sigma_sys: float = 0.06,
                            p_idio: float = 0.003, mu_idio: float = -0.06, sigma_idio: float = 0.12,
                            importance_tilt: float | None = 0.05) -> dict:
    """
    End-to-end:
    - fit μ, Σ from historic_prices
    - clean & align to portfolio_tickers
    - simulate heavy-tail+jumps with importance sampling
    - aggregate to portfolio via shares OR weights+budget
    - compute VaR/ES and drawdown stats

    Returns dict with:
      'tickers', 'dropped', 'sim', 'V_paths', 'metrics', 'dd_stats'
    """
    # fit params
    mu_ann, Sigma_ann, _ = fit_mv_params(
    historic_prices,
    N=N,
    focus_tickers=portfolio_tickers,
    min_valid_obs=126   # tune if you want stricter/looser column survival
    )

    # align & clean
    S0, mu_c, Sigma_c, dropped = _prepare_inputs_for_tail(historic_prices, mu_ann, Sigma_ann, portfolio_tickers)
    if len(S0) == 0:
        raise ValueError("After cleaning, no valid tickers remain for rare-event simulation.")

    # simulate
    sim = simulate_heavy_tail_paths(
        S0=S0, mu_ann=mu_c, Sigma_ann=Sigma_c,
        T=T, M=M, N=N, nu=nu,
        p_sys=p_sys, mu_sys=mu_sys, sigma_sys=sigma_sys,
        p_idio=p_idio, mu_idio=mu_idio, sigma_idio=sigma_idio,
        importance_tilt=importance_tilt
    )

    # aggregate to portfolio
    if shares is not None:
        # ensure shares line up with the cleaned ticker order
        if isinstance(shares, (pd.Series, pd.DataFrame)):
            shares_vec = np.asarray(pd.Series(shares).reindex(S0.index).values, dtype=float)
        else:
            # assume original order matched 'portfolio_tickers'
            order_map = pd.Index(portfolio_tickers)
            shares_map = pd.Series(np.asarray(shares, dtype=float), index=order_map[:len(shares)])
            shares_vec = shares_map.reindex(S0.index).values
        V_paths = portfolio_paths_from_shares(sim["S_paths"], shares_vec)
    else:
        if weights is None or budget is None:
            raise ValueError("Provide either 'shares' or ('weights' and 'budget').")
        if hasattr(weights, "reindex"):
            w_vec = np.asarray(pd.Series(weights).reindex(S0.index).values, dtype=float)
        elif isinstance(weights, dict):
            w_vec = np.asarray([weights.get(t, 0.0) for t in S0.index], dtype=float)
        else:
            w_vec = np.asarray(weights, dtype=float)
            if len(w_vec) != len(S0):
                raise ValueError("Length of weights does not match number of cleaned tickers.")
        V_paths = portfolio_paths_from_S(sim["S_paths"], w_vec, budget)

    metrics = tail_metrics(V_paths, alphas=(95.0, 99.0, 99.5), horizon_days=(1, 10, 252), weights=sim["weights"])
    dd_stats = max_drawdown_stats(V_paths, weights=sim["weights"])

    return {
        "tickers": list(S0.index),
        "dropped": dropped,
        "sim": sim,
        "V_paths": V_paths,
        "metrics": metrics,
        "dd_stats": dd_stats,
    }


def _annualize_from_logrets(log_ret: pd.DataFrame, N:int=252) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Return annualized drift (mu) and covariance matrix (Sigma) from log returns.
    """
    mu_daily = log_ret.mean()            # per day
    cov_daily = log_ret.cov()            # per day
    mu_ann = mu_daily * N
    Sigma_ann = cov_daily * N
    mu = pd.Series(mu_ann, index=log_ret.columns)
    Sigma = pd.DataFrame(Sigma_ann, index=log_ret.columns, columns=log_ret.columns)
    return mu, Sigma

def fit_mv_params(
    historic_prices: pd.DataFrame,
    N: int = 252,
    *,
    focus_tickers: Optional[Iterable[str]] = None,
    min_valid_obs: int = 126  # ~6 months of daily data
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Estimate annualized mu, Sigma and correlation from historical close prices.
    Robust to missing data:
      - optionally subset to focus_tickers first
      - require at least `min_valid_obs` non-NaN returns per column
      - drop remaining rows with any NaNs only after filtering columns
    """
    if focus_tickers is not None:
        cols = [c for c in focus_tickers if c in historic_prices.columns]
        if len(cols) == 0:
            raise ValueError("None of the focus_tickers are present in historic_prices.")
        prices = historic_prices[cols]
    else:
        prices = historic_prices

    # log returns
    log_ret = np.log(prices).diff()

    # keep columns with enough data
    keep = [c for c in log_ret.columns if log_ret[c].count() >= min_valid_obs]
    if len(keep) == 0:
        # helpful message for debugging
        counts = log_ret.count().sort_values(ascending=False)
        raise ValueError(
            f"No columns have >= {min_valid_obs} valid return observations. "
            f"Top counts:\n{counts.head(10)}"
        )

    log_ret = log_ret[keep].dropna(how="any")  # now safe to drop rows with residual NaNs
    if log_ret.empty:
        raise ValueError("After filtering and dropping NaNs, no overlapping dates remain.")

    # daily -> annual
    mu_daily = log_ret.mean()
    cov_daily = log_ret.cov()
    mu_ann = mu_daily * N
    Sigma_ann = cov_daily * N

    # PSD + corr
    Sigma_ann = _nearest_pd(Sigma_ann)
    corr = _cov_to_corr(Sigma_ann)

    # Return as pandas with sensible indexes
    mu_ann = pd.Series(mu_ann.values, index=log_ret.columns)
    Sigma_ann = pd.DataFrame(Sigma_ann.values, index=log_ret.columns, columns=log_ret.columns)
    corr = pd.DataFrame(corr.values, index=log_ret.columns, columns=log_ret.columns)
    return mu_ann, Sigma_ann, corr


def _cov_to_corr(Sigma: pd.DataFrame) -> pd.DataFrame:
    s = np.sqrt(np.diag(Sigma.values))
    S_inv = np.diag(1.0 / s)
    C = S_inv @ Sigma.values @ S_inv
    return pd.DataFrame(C, index=Sigma.index, columns=Sigma.columns)

def _nearest_pd(A: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """Project a covariance matrix to the nearest positive semidefinite matrix."""
    B = (A.values + A.values.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals_clipped = np.clip(eigvals, eps, None)
    B_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
    B_psd = (B_psd + B_psd.T) / 2
    return pd.DataFrame(B_psd, index=A.index, columns=A.columns)

def _mv_student_t(nu: float, dim: int, size: int) -> np.ndarray:
    """
    Draw 'size' samples from a standard multivariate Student-t with df=nu and identity scale.
    Returns array of shape (size, dim).
    """
    z = np.random.normal(size=(size, dim))
    g = np.random.chisquare(df=nu, size=size)
    t = z / np.sqrt(g[:, None] / nu)
    return t

def simulate_heavy_tail_paths(
    S0: pd.Series,
    mu_ann: pd.Series,
    Sigma_ann: pd.DataFrame,
    T: float,
    M: int,
    N: int = 252,
    nu: float = 5.0,
    p_sys: float = 0.02,          # daily probability of a systemic jump
    mu_sys: float = 0,        # mean systemic jump (log-return)
    sigma_sys: float = 0.05,      # std of systemic jump (log-return)
    p_idio: float = 0.005,        # daily idiosyncratic jump probability per asset
    mu_idio: float = -0.05,
    sigma_idio: float = 0.10,
    importance_tilt: Optional[float] = None  # if set, increases p_sys to tilt tail sampling; use for weights
) -> Dict[str, np.ndarray]:
    """
    Simulate correlated asset price paths with Student-t innovations + (systemic + idiosyncratic) jumps.
    Returns dict with 't', 'S_paths' (T*N+1, M, d), 'weights' (M,) if importance_tilt is used else None.
    """
    tickers = list(S0.index)
    d = len(tickers)
    steps = int(T * N)
    dt = 1.0 / N

    # Daily covariance
    Sigma_daily = Sigma_ann.values / N
    Sigma_daily = _nearest_pd(pd.DataFrame(Sigma_daily, index=tickers, columns=tickers)).values
    try:
        L = np.linalg.cholesky(Sigma_daily)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma_daily + 1e-10*np.eye(d))

    # Drift per step in log space: (mu - 0.5*diag(Sigma)) * dt
    mu_log_ann = mu_ann.values - 0.5 * np.diag(Sigma_ann.values)
    mu_step = mu_log_ann * dt

    # Importance sampling on systemic jump probability
    p_sys_base = p_sys
    p_sys_sim = min(0.999, max(0.0, importance_tilt)) if importance_tilt is not None else p_sys_base

    S_paths = np.zeros((steps + 1, M, d))
    S_paths[0, :, :] = S0.values[None, None, :]
    log_w = np.zeros(M)  # log-weights per path

    for t in range(1, steps + 1):
        # Heavy-tail diffusive shock (already daily scaled via L)
        t_shocks = _mv_student_t(nu, d, M)            # (M, d)
        diffusive = t_shocks @ L                       # daily covariance

        # Systemic jump (same across assets)
        sys_happened = np.random.rand(M) < p_sys_sim
        sys_jump = np.where(sys_happened, np.random.normal(mu_sys, sigma_sys, size=M), 0.0)

        # Importance reweighting (Bernoulli LR)
        if importance_tilt is not None:
            p0, p1 = p_sys_base, p_sys_sim
            lr = np.where(sys_happened, (p0 / p1), ((1 - p0) / (1 - p1)))
            log_w += np.log(lr)

        # Idiosyncratic jumps per asset
        idio_happened = (np.random.rand(M, d) < p_idio)
        idio_jump = np.where(idio_happened, np.random.normal(mu_idio, sigma_idio, size=(M, d)), 0.0)

        # Total log-return increment
        incr = mu_step[None, :] + diffusive + idio_jump + sys_jump[:, None]

        # Update prices
        S_paths[t, :, :] = S_paths[t - 1, :, :] * np.exp(incr)

    weights = None
    if importance_tilt is not None:
        w = np.exp(log_w)
        weights = w / (w.mean() + 1e-12)

    return {
        "t": np.linspace(0.0, T, steps + 1),
        "S_paths": S_paths,
        "weights": weights,
        "params": {
            "nu": nu, "p_sys": p_sys_base, "p_sys_sim": p_sys_sim,
            "mu_sys": mu_sys, "sigma_sys": sigma_sys,
            "p_idio": p_idio, "mu_idio": mu_idio, "sigma_idio": sigma_idio
        }
    }

def portfolio_paths_from_S(S_paths: np.ndarray, weights: np.ndarray, budget: float) -> np.ndarray:
    """
    Convert asset price paths (steps+1, M, d) into portfolio value paths (steps+1, M)
    using buy-and-hold shares implied by 'weights' and initial prices.
    """
    steps_plus, M, d = S_paths.shape
    S0 = S_paths[0, 0, :]  # initial prices per asset
    alloc = weights * budget
    shares = alloc / S0
    V = (S_paths * shares[None, None, :]).sum(axis=2)
    return V

def portfolio_paths_from_shares(S_paths: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Convert asset price paths (steps+1, M, d) into portfolio value paths using fixed 'shares' (1D length d).
    """
    return (S_paths * shares[None, None, :]).sum(axis=2)

def weighted_percentile(a: np.ndarray, q: float, w: Optional[np.ndarray]=None) -> float:
    """
    Weighted percentile of a 1D array 'a' at quantile q in [0,100].
    """
    a = np.asarray(a, dtype=float)
    if w is None:
        return float(np.percentile(a, q))
    w = np.asarray(w, dtype=float)
    assert a.shape == w.shape
    idx = np.argsort(a)
    a_sorted = a[idx]
    w_sorted = w[idx]
    cum_w = np.cumsum(w_sorted)
    cutoff = q / 100.0 * cum_w[-1]
    i = np.searchsorted(cum_w, cutoff)
    i = min(max(i, 0), len(a_sorted)-1)
    return float(a_sorted[i])

def tail_metrics(V_paths: np.ndarray, alphas=(95.0, 99.0, 99.5), horizon_days=(1, 10, 252), weights: Optional[np.ndarray]=None) -> Dict[str, Dict[str, float]]:
    """
    Compute VaR/CVaR at multiple confidence levels and horizons (in trading days).
    V_paths is (steps+1, M).
    Returns dict like metrics['1d']['VaR_99']=..., metrics['10d']['ES_99.5']=...
    """
    steps_plus, M = V_paths.shape
    out = {}
    for h in horizon_days:
        if h >= steps_plus:
            continue
        V0 = V_paths[0, :]
        Vh = V_paths[h, :]
        losses = V0 - Vh  # positive = loss
        key = f"{h}d"
        out[key] = {}
        for a in alphas:
            var = weighted_percentile(losses, a, w=weights)
            if weights is None:
                mask = losses >= var - 1e-12
                es = float(losses[mask].mean()) if mask.any() else float(var)
            else:
                idx = np.argsort(losses)
                l_sorted = losses[idx]
                w_sorted = weights[idx]
                tail_mask = l_sorted >= var - 1e-12
                if tail_mask.any():
                    es = float(np.sum(l_sorted[tail_mask]*w_sorted[tail_mask]) / np.sum(w_sorted[tail_mask]))
                else:
                    es = float(var)
            out[key][f"VaR_{a}"] = var
            out[key][f"ES_{a}"]  = es
    return out

def max_drawdown_stats(V_paths: np.ndarray, weights: Optional[np.ndarray]=None) -> Dict[str, float]:
    """
    Compute distributional stats of max drawdown across scenarios. Returns mean, 95th, 99th percentiles (loss in currency units).
    """
    M = V_paths.shape[1]
    drawdowns = np.empty(M, dtype=float)
    for m in range(M):
        v = V_paths[:, m]
        peak = np.maximum.accumulate(v)
        dd = 1.0 - v/peak
        drawdowns[m] = dd.max() * V_paths[0, m]
    out = {}
    for q in [95.0, 99.0]:
        out[f"maxDD_p{q}"] = weighted_percentile(drawdowns, q, w=weights)
    out["maxDD_mean"] = float(np.average(drawdowns, weights=(weights if weights is not None else np.ones_like(drawdowns))))
    return out
