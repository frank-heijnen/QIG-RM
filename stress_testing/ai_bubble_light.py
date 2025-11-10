import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
from dataclasses import dataclass
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

# =======================
# CONFIG
# =======================
START = "2019-01-01"
END   = None  # None -> today
WIN_RHO = 24  # months for rolling rho on log-ratios
MIN_WIN_GSADF = 36  # minimum window for GSADF on log-ratios
STEP_GSADF = 3      # evaluate every 3 months for speed
BOOT_B = 100        # bootstrap replications (fast)
BOOT_BLOCK = 6      # months per resampled block

# Curated cohorts: tune to taste
ENABLERS = ["NVDA","AVGO","AMD","TSM","ASML","SMCI","MRVL","MU","VRT","ANET","EQIX"]
PLATFORMS = ["MSFT","GOOGL","META","ORCL","CRM","SNOW","PLTR","DDOG","MDB"]
APPS = ["IONQ","AI","SOUN","UPST","PATH","ESTC","SHOP","ABNB","ZS"]

GROUP_WEIGHTS = {"ENABLERS":0.50, "PLATFORMS":0.40, "APPS":0.10}  # CTA-style (optional)
ALL = sorted(set(ENABLERS+PLATFORMS+APPS))

# =======================
# DATA HELPERS
# =======================
@dataclass
class F:
    rev_ttm: float
    shares: float
    debt: float
    cash: float

def _safe_info(tk):
    try: return tk.get_info()
    except Exception:
        try: return tk.info
        except Exception: return {}

def fundamentals(tkr: str) -> F:
    tk = yf.Ticker(tkr)
    # TTM revenue
    rev_ttm = np.nan
    qf = tk.quarterly_financials
    if isinstance(qf, pd.DataFrame) and "Total Revenue" in qf.index:
        r = qf.loc["Total Revenue"].dropna()
        if len(r) >= 4: rev_ttm = float(r.iloc[:4].sum())
        elif len(r) > 0: rev_ttm = float(r.sum())
    info = _safe_info(tk)
    if np.isnan(rev_ttm):
        for k in ["totalRevenue","trailingAnnualRevenue","revenue"]:
            if k in info and pd.notna(info[k]): rev_ttm = float(info[k]); break
    # shares
    shares = np.nan
    try:
        fi = tk.fast_info
        so = fi.get("sharesOutstanding", None)
        if so is not None and pd.notna(so): shares = float(so)
    except Exception:
        pass
    if np.isnan(shares):
        for k in ["sharesOutstanding","floatShares"]:
            if k in info and pd.notna(info.get(k, np.nan)):
                shares = float(info[k]); break
    # debt / cash (best-effort snapshot)
    debt = float(info.get("totalDebt", np.nan)) if pd.notna(info.get("totalDebt", np.nan)) else np.nan
    cash = float(info.get("totalCash", np.nan)) if pd.notna(info.get("totalCash", np.nan)) else np.nan
    return F(rev_ttm, shares, debt, cash)

def prices_monthly(tickers, start=START, end=END):
    if end is None: end = dt.date.today().isoformat()
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    return px.resample("M").last().dropna(how="all")

# =======================
# RATIO CONSTRUCTION
# =======================
def basket_ps_evs(px_m: pd.DataFrame, cohort: list, fdict: dict):
    """
    Returns DataFrame with P/S and EV/S for an equal-weight cohort basket.
    Market cap is time-varying via price; debt/cash treated as snapshot (ok for signal).
    """
    tick = [t for t in cohort if (t in px_m.columns) and pd.notna(fdict[t].rev_ttm) and pd.notna(fdict[t].shares)]
    if not tick: return None
    # market caps path
    caps = px_m[tick].multiply(pd.Series({t: fdict[t].shares for t in tick}), axis=1)
    mc_ser = caps.sum(axis=1)
    # revenue TTM (snapshot sum)
    rev = sum(fdict[t].rev_ttm for t in tick if pd.notna(fdict[t].rev_ttm))
    # EV = MC + debt - cash (snapshot)
    debt = sum((fdict[t].debt for t in tick if pd.notna(fdict[t].debt))) if any(pd.notna(fdict[t].debt) for t in tick) else 0.0
    cash = sum((fdict[t].cash for t in tick if pd.notna(fdict[t].cash))) if any(pd.notna(fdict[t].cash) for t in tick) else 0.0
    ev_ser = mc_ser + debt - cash
    df = pd.DataFrame({"PtoS": mc_ser/rev, "EVtoS": ev_ser/rev}).dropna()
    return df

def cta_combo(en_df, pl_df, ap_df):
    """
    Combine group indices with CTA-style weights.
    We normalize each group to 1 at its first observation before weighting,
    then rescale back to a level to keep interpretability.
    """
    parts = []
    if en_df is not None: parts.append(("ENABLERS", en_df))
    if pl_df is not None: parts.append(("PLATFORMS", pl_df))
    if ap_df is not None: parts.append(("APPS", ap_df))
    if not parts: return None
    def norm(s): return s / s.iloc[0]
    ps = sum(GROUP_WEIGHTS[g]*norm(df["PtoS"]) for g,df in parts)
    evs = sum(GROUP_WEIGHTS[g]*norm(df["EVtoS"]) for g,df in parts)
    base_ps  = parts[0][1]["PtoS"].iloc[0]
    base_evs = parts[0][1]["EVtoS"].iloc[0]
    return pd.DataFrame({"PtoS": ps*base_ps, "EVtoS": evs*base_evs})

# =======================
# ROLLING RHO (fast)
# =======================
def rolling_rho_log(series: pd.Series, window=WIN_RHO, step=1) -> pd.Series:
    """
    Computes rho from Δy_t = α + ρ y_{t-1} + ε_t
    using cumulative sums (O(T)) over rolling windows on y = log(series).
    """
    y = np.log(series.dropna())
    if len(y) < window + 1: return pd.Series(dtype=float, name="rho")
    x = y.shift(1).iloc[1:]   # y_{t-1}
    z = y.diff().iloc[1:]     # Δy_t
    idx = z.index
    n = len(z)
    w = window
    cx, cz = x.cumsum(), z.cumsum()
    cx2, cxz = (x*x).cumsum(), (x*z).cumsum()
    def seg(c, i0, i1): return c.iloc[i1] - (c.iloc[i0-1] if i0>0 else 0.0)
    out, dates = [], []
    i = w-1
    while i < n:
        i0, i1 = i-(w-1), i
        Sx, Sz, Sx2, Sxz = seg(cx,i0,i1), seg(cz,i0,i1), seg(cx2,i0,i1), seg(cxz,i0,i1)
        num = Sxz - (Sx*Sz)/w
        den = Sx2 - (Sx*Sx)/w
        out.append(num/den if den!=0 else np.nan)
        dates.append(idx[i])
        i += step
    return pd.Series(out, index=pd.Index(dates, name="Date"), name="rho")

# =======================
# LIGHTWEIGHT GSADF
# =======================
def adf_right(y_arr):
    """
    Return (rho_hat, t_stat) for Δy_t on [1, y_{t-1}].
    """
    dy = np.diff(y_arr)
    y_1 = y_arr[:-1]
    X = add_constant(y_1)
    model = OLS(dy, X).fit()
    return float(model.params[1]), float(model.tvalues[1])

def gsadf_fast(y: pd.Series, min_window=MIN_WIN_GSADF, step=STEP_GSADF, B=BOOT_B, block_len=BOOT_BLOCK):
    """
    Right-tailed GSADF on y (log ratio). Returns:
    - DataFrame (index = end indices mapped to dates) with sup t-stat and its rho,
    - 95% bootstrap critical value.
    """
    y = pd.Series(y).dropna()
    yv = y.values
    T = len(yv)
    if T < min_window + 5:
        raise ValueError("Series too short for GSADF.")
    sup_t, sup_r, idx = [], [], []
    for r2 in range(min_window, T+1, step):
        t_best = -np.inf
        r_best = np.nan
        for r1 in range(0, r2 - min_window + 1, step):
            rho_hat, t = adf_right(yv[r1:r2])
            if t > t_best:
                t_best, r_best = t, rho_hat
        sup_t.append(t_best); sup_r.append(r_best); idx.append(r2-1)
    out = pd.DataFrame({"tstat": sup_t, "rho_at_sup": sup_r}, index=y.index[idx])

    # Bootstrap null: resample de-meaned Δy blocks; rebuild series
    dy = np.diff(yv)
    dy = dy - dy.mean()
    sup_boot = []
    for _ in range(B):
        res = []
        while len(res) < len(dy):
            i0 = np.random.randint(0, max(1, len(dy)-block_len+1))
            res.extend(dy[i0:i0+block_len].tolist())
        yb = np.cumsum(res[:len(dy)]) + yv[0]
        # compute sup t over the same grid
        tb = -np.inf
        for r2 in range(min_window, len(yb)+1, step):
            for r1 in range(0, r2 - min_window + 1, step):
                _, t = adf_right(yb[r1:r2])
                if t > tb: tb = t
        sup_boot.append(tb)
    crit95 = float(np.percentile(sup_boot, 95))
    return out, crit95

def bubble_episodes(tstat_series: pd.Series, crit: float):
    """
    Return list of (start_date, end_date) where tstat > crit.
    """
    sig = tstat_series > crit
    episodes, in_bubble, start = [], False, None
    for t, flag in sig.items():
        if flag and not in_bubble:
            in_bubble, start = True, t
        elif not flag and in_bubble:
            episodes.append((start, t))
            in_bubble = False
    if in_bubble:
        episodes.append((start, tstat_series.index[-1]))
    return episodes

# =======================
# PLOTTING
# =======================
def plot_all(name, ratio_series, rho_series, gsadf_df, crit95):
    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(ratio_series.index, ratio_series.values, label=name)
    ax.set_title(f"{name} (level)")
    ax.grid(True, alpha=0.3)
    ax.legend(); plt.tight_layout(); plt.show()

    fig2, ax2 = plt.subplots(figsize=(11,4))
    ax2.plot(rho_series.index, rho_series.values, label="rolling rho")
    ax2.axhline(0.0, linestyle="--")
    ax2.set_title(f"{name}: rolling ρ of log(ratio)  (window={WIN_RHO}m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(); plt.tight_layout(); plt.show()

    fig3, ax3 = plt.subplots(figsize=(11,4))
    ax3.plot(gsadf_df.index, gsadf_df["tstat"], label="GSADF sup t-stat")
    ax3.axhline(crit95, linestyle="--", label=f"95% crit ≈ {crit95:.2f}")
    ax3.set_title(f"{name}: GSADF (right-tailed) on log(ratio)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(); plt.tight_layout(); plt.show()

# =======================
# DRIVER
# =======================
def run():
    print("Downloading monthly prices...")
    px_m = prices_monthly(ALL, START, END)
    print("Fetching fundamentals (TTM revenue, shares, debt, cash)...")
    fdict = {t: fundamentals(t) for t in ALL}

    # Build baskets (equal-weight by construction of basket sums)
    en = basket_ps_evs(px_m, ENABLERS, fdict)
    pl = basket_ps_evs(px_m, PLATFORMS, fdict)
    ap = basket_ps_evs(px_m, APPS, fdict)
    combo = cta_combo(en, pl, ap)

    baskets = {
        "ENABLERS P/S": en["PtoS"] if en is not None else None,
        "ENABLERS EV/S": en["EVtoS"] if en is not None else None,
        "PLATFORMS P/S": pl["PtoS"] if pl is not None else None,
        "PLATFORMS EV/S": pl["EVtoS"] if pl is not None else None,
        "APPS P/S": ap["PtoS"] if ap is not None else None,
        "APPS EV/S": ap["EVtoS"] if ap is not None else None,
        "CTA_COMBO P/S": combo["PtoS"] if combo is not None else None,
        "CTA_COMBO EV/S": combo["EVtoS"] if combo is not None else None,
    }

    # Focus on the “frothiest” candidates first: APPS EV/S, then ENABLERS EV/S
    ordered = [k for k in ["APPS EV/S","ENABLERS EV/S","PLATFORMS EV/S","CTA_COMBO EV/S",
                           "APPS P/S","ENABLERS P/S","PLATFORMS P/S","CTA_COMBO P/S"]
               if baskets.get(k) is not None and not baskets[k].dropna().empty]

    summaries = []
    for name in ordered:
        ratio = baskets[name].dropna()
        y = np.log(ratio)
        # rolling rho (descriptive)
        rho = rolling_rho_log(ratio, window=WIN_RHO, step=1)
        # lightweight GSADF (inferential)
        gs_df, crit95 = gsadf_fast(y, min_window=MIN_WIN_GSADF, step=STEP_GSADF, B=BOOT_B, block_len=BOOT_BLOCK)
        episodes = bubble_episodes(gs_df["tstat"], crit95)

        print(f"\n===== {name} =====")
        print(f"Last rolling ρ: {rho.dropna().iloc[-3:].round(3).to_string() if not rho.dropna().empty else 'n/a'}")
        print(f"GSADF 95% critical ≈ {crit95:.2f}; last t-stat = {gs_df['tstat'].iloc[-1]:.2f}, rho@sup = {gs_df['rho_at_sup'].iloc[-1]:.3f}")
        if episodes:
            print("Bubble episodes (t > crit):")
            for s,e in episodes:
                print(f"  {s.date()}  →  {e.date()}")
        else:
            print("No bubble episodes detected at 95%.")

        summaries.append({
            "basket": name,
            "crit95": crit95,
            "last_t": gs_df["tstat"].iloc[-1],
            "last_rho_at_sup": gs_df["rho_at_sup"].iloc[-1],
            "episodes": len(episodes)
        })

        # Plots
        try:
            plot_all(name, ratio, rho, gs_df, crit95)
        except Exception:
            pass

    if summaries:
        df_sum = pd.DataFrame(summaries)
        print("\n==== SUMMARY ====")
        print(df_sum.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    run()
