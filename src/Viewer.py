## Visualiser ASR Portfolio Tracker

# Import relevant Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_historical_prices(historic_prices, tickers = None, show_prices = False):
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

def plot_current_prices(master_data, tickers = None):
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

def plot_portfolio_trajectories(t, port_paths, n_paths = 10, show_mean = True ):
    r"""
    Plot simulated portfolio paths over time.

    :params t:
    :params port_paths: 
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
        mean_path = port_paths.mean(axis=1)
        plt.plot(t, mean_path, color="black", linewidth=2, label="Mean")
        plt.legend(title="Simulation", loc="upper left")

    plt.title("Simulated Portfolio Value Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.show()

def histogram_uncertainty(port_paths):
    r"""
    Plot histogram to see distribution of terminal portfolio values, includes calculating VaR and CVaR (tail risk)

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

    # Calculate 
    alpha = 5
    VaR   = np.percentile(final_vals, alpha)
    CVaR  = final_vals[final_vals <= VaR].mean()

    plt.axvline(VaR,  color="red", linestyle="--", label=f"{alpha}% VaR = €{VaR:,.0f}")
    plt.axvline(CVaR, color="orange", linestyle="--", label=f"{alpha}% CVaR = €{CVaR:,.0f}")
    plt.legend()
    plt.title("Distribution of Simulated Portfolio Values at T with VaR/CVaR")
    plt.tight_layout()
    plt.show()
