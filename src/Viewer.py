## Visualiser ASR Portfolio Tracker

# Import relevant Packages
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

