## Visualiser ASR Portfolio Tracker

# Import relevant Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_loss_exceedance(V_paths, weights=None, horizon=252, title="Loss Exceedance (Survival)"):
    """
    Survival function of losses at horizon: P(Loss > x).
    Shows VaR/ES markers at 95%, 99%, 99.5%.
    """
    V0 = V_paths[0,:]
    Vh = V_paths[horizon,:]
    loss = V0 - Vh
    order = np.argsort(loss)
    x = loss[order]
    if weights is None:
        w = np.ones_like(x, float)
    else:
        w = np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    sf = 1.0 - cw / cw[-1]  # survival function

    plt.figure()
    plt.plot(x, sf)
    for a in (95.0, 99.0, 99.5):
        # weighted quantile
        cutoff = a/100.0 * cw[-1]
        i = np.searchsorted(cw, cutoff)
        i = min(max(i,0), len(x)-1)
        var = x[i]
        plt.axvline(var, linestyle="--")
        plt.text(var, 0.5, f"VaR {a:.1f}% = {var:,.0f}", rotation=90, va="center")
    plt.xlabel("Loss at horizon")
    plt.ylabel("P(Loss > x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_fan_chart(V_paths, weights=None, title="Portfolio Value: Fan Chart"):
    """
    Fan chart of weighted quantiles over time, plus median.
    """
    from extra.rare_events import fan_stats
    st = fan_stats(V_paths, weights)
    t = st['t']
    qs = st['qs']
    Q = st['q']  # shape (len(qs), T+1)
    plt.figure()
    # shaded bands between 1-99, 5-95, 25-75
    bands = [(0.01,0.99),(0.05,0.95),(0.25,0.75)]
    for lo,hi in bands:
        i_lo = np.where(np.isclose(qs, lo))[0][0]
        i_hi = np.where(np.isclose(qs, hi))[0][0]
        plt.fill_between(t, Q[i_lo], Q[i_hi], alpha=0.15)
    # median
    i_med = np.where(np.isclose(qs, 0.5))[0][0]
    plt.plot(t, Q[i_med], linewidth=1.5)
    plt.xlabel("Time step")
    plt.ylabel("Portfolio value")
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_es_path(V_paths, weights=None, horizon=252, alpha=99.5, title=None):
    """
    Expected Shortfall path: average trajectory of tail scenarios at horizon.
    """
    from extra.rare_events import tail_set, es_path
    tail_idx, var = tail_set(V_paths, horizon=horizon, alpha=alpha, weights=weights)
    esp = es_path(V_paths, tail_idx, weights)
    med = np.median(V_paths, axis=1)
    plt.figure()
    t = np.arange(V_paths.shape[0])
    plt.plot(t, med, label="Median path")
    plt.plot(t, esp, label=f"ES path (tail @ {alpha}%, H={horizon}d)")
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Portfolio value")
    if title is None:
        title = f"Median vs Expected-Shortfall Path (alpha={alpha}%)"
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_drawdown_heatmap(V_paths, weights=None, horizon=252, alpha=99.5, top_k=200, title=None):
    """
    Heatmap of drawdowns over time for the worst scenarios (ranked by final loss).
    Rows = scenarios, columns = time; darker = deeper drawdown.
    """
    V0 = V_paths[0,:]
    Vh = V_paths[horizon,:]
    losses = V0 - Vh
    order = np.argsort(losses)[::-1]  # worst first
    sel = order[:min(top_k, V_paths.shape[1])]
    DD = 100.0 * (1.0 - V_paths[:, sel] / np.maximum.accumulate(V_paths[:, sel], axis=0))
    plt.figure(figsize=(8, 6))
    plt.imshow(DD.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="Drawdown (%)")
    plt.ylabel(f"Worst {len(sel)} scenarios")
    plt.xlabel("Time step")
    if title is None:
        title = f"Drawdown Heatmap — worst {len(sel)} scenarios (ranked by loss @ H={horizon}d)"
    plt.title(title)
    
def plot_historical_prices(historic_prices, tickers = None, show_prices = False) -> None:
    r"""
    Plot past price trajectories for one or more tickers

    :params historic_prices: Dataframa object containing past close prices of stocks
    :params tickers: list of tickers of the stocks that are desired to be plotted

    :returns: a nice plot
    """

    if tickers is None:
        tickers = historic_prices.columns.tolist()
    data = historic_prices[tickers]

    if show_prices == True:
        print(data)

    figsize=(12, 6)
    plt.figure(figsize=figsize)
    sns.lineplot(data=data)
    plt.title(f"Historical Prices of {tickers}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(title="Ticker")
    plt.tight_layout()
    plt.show()

def correlation_analysis(historic_prices) -> None:
    """
    Correlation analysis of current stocks in our portfolio

    :param historic_prices: DataFrame of past close prices of stocks
    :returns: a nice seaborn heatmap
    """
    df = historic_prices
    returns = df.pct_change().dropna()
    corr = returns.corr()

    # Print top/bottom correlated pairs (same as before)…
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = corr.where(mask)
    strong_pairs = pairs.unstack().dropna().sort_values(ascending=False)
    print("\nTop 3 most strongly positively correlated pairs:")
    print(strong_pairs.head(3))
    print("\nTop 3 most strongly negatively correlated pairs:")
    print(strong_pairs.tail(3))

    # --- New seaborn heatmap code ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=~mask,               # show only one triangle if you like
        annot=True,               # numbers in each cell
        fmt=".2f",                # 2 decimal places
        cmap="vlag",              # blue ↔ red diverging map
        linewidths=0.5,           # lines between cells
        cbar_kws={"shrink": .75}  # smaller colorbar
    )
    plt.title("Return Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_current_prices(master_data, tickers = None) -> None:
    r"""
    Barplot of current prices for each specified ticker

    :params master_data: the master data of the downloaded stock data, which has a column with the current (last) closing price
    :params tickers: list of tickers of the stocks that are desired to be plotted

    :returns: a nice barplot
    """
    if tickers is None:
        tickers = master_data.index.tolist()

    df = master_data.loc[tickers]
    df = df.sort_values("current_price", ascending=False)

    figsize=(10, 5)
    plt.figure(figsize=figsize)
    sns.barplot(x=df.index, y=df["current_price"], palette="viridis")
    plt.title("Current Prices by Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_portfolio_trajectories(t, port_paths, n_paths = 10, show_mean = True ) -> None:
    r"""
    Plot simulated portfolio paths over time.

    :params t: time grid
    :params port_paths: Dataframe object of simulated portfolio paths
    :params n_paths: number of paths of the 100k to plot

    :returns: a nice plot of simulated portfolio paths
    """

     # Determine how many paths to plot
    total = port_paths.shape[1]
    if n_paths > total:
        n_plot = total
    else:
        n_plot = n_paths

    # Build DataFrame for Seaborn
    cols = [f"Path {i+1}" for i in range(n_plot)]
    df = pd.DataFrame(port_paths[:, :n_plot], index=t, columns=cols)

    figsize=(12, 6)
    plt.figure(figsize=figsize)
    sns.lineplot(data=df)

    # Plot also the mean of the paths
    if show_mean:
        # Make confidence bands
        lower, upper = np.percentile(port_paths, [5,95], axis=1)
        plt.fill_between(t, lower, upper, color="grey", alpha=0.3, label=f"Confidence Band {(round(float(lower[-1]),ndigits=2), round(float(upper[-1]), ndigits=2))}")
        #Show Mean
        mean_path = port_paths.mean(axis=1)
        plt.plot(t, mean_path, color="black", linewidth=2, label=f"Mean {round(mean_path[-1],ndigits=2)}")
        plt.legend(title="Simulation", loc="upper left")

    plt.title("Simulated Portfolio Value Paths")
    plt.xlabel("Time (years)")
    plt.yscale("log")
    plt.ylabel("Portfolio Value (log scale)")
    plt.tight_layout()
    plt.show()

def histogram_uncertainty(port_paths) -> None:
    r"""
    Plot histogram to see distribution of terminal portfolio values, includes calculating VaR, CVaR (tail risk) and

    :params port_paths: matrix of simulated portfolio paths

    :returns: a nice histogram including risk metrics
    """
    # Extract terminal portfolio values
    final_vals = port_paths[-1, :]

    # Time for plotting
    plt.figure(figsize=(10,6))
    sns.histplot(final_vals, bins=50, log_scale=(True,False)) # Draw x-axis logarithmically, because some terminal values explode
    plt.xlabel("Portfolio Value at T")
    plt.ylabel("Frequency")

    # Calculate VaR and CVaR
    alpha = 5
    VaR   = np.percentile(final_vals, alpha)
    CVaR  = final_vals[final_vals <= VaR].mean()

    plt.axvline(VaR,  color="red", linestyle="--", label=f"{alpha}% VaR = €{VaR:,.0f}")
    plt.axvline(CVaR, color="orange", linestyle="--", label=f"{alpha}% CVaR = €{CVaR:,.0f}")
    plt.legend()
    plt.title("Distribution of Simulated Portfolio Values at T with VaR/CVaR")
    plt.tight_layout()
    plt.show()

import numpy as np

def compute_solvency_capital_requirement(port_paths, days_per_year = 252, alpha = 99.5) -> float:
    r"""
    According to Solvency II, one must be able to cover a 1 in 200 shock event and still stay solvent.
    This formula calculates the required capital to cover this event based on an investment of "budget".
    Hence loss = initial budget - VaR(portfolio_value_1year, 0.5 percentile)

    :params port_paths: matrix of simulated portfolio paths

    :returns: the required buffer such that with alpha% confidence the one-year losses wil not exceed "scr"
    """
    # Extract starting portfolio value, which equals budget for all paths
    initial_value = port_paths[0, 0]  # assume all paths start at the same V0

    # Get simulated portfolio values within one year
    value_1year = port_paths[days_per_year, :]  # shape (M,)

    # Find the (100 - alpha)% percentile of V1y, i.e. bottom (100-alpha)% tail
    tail_pct = 100 - alpha
    floor_value = np.percentile(value_1year, tail_pct)
    print(initial_value)
    print(floor_value)
    # Capital buffer required
    scr = initial_value - floor_value
    return scr
