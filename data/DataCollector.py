## Gathering the data using Yahoo Finance API

# Import relevant packages
import yfinance as yf
import pandas as pd

def fetch_master_data(tickers) -> pd.DataFrame:
    r"""
    Fetch data of the specified stocks that does not change over time (masterdata), it consists out of:
        - ticker
        - sector
        - asset class
        - current price
        - market cap (for asset allocation)

    :params tickers: list of stock tickers
    
    :returns: DataFrame object of master data
    """
    multi_ticker = yf.Tickers(" ".join(tickers))

    rows = []
    for ticker in tickers:
        info = multi_ticker.tickers[ticker].info
        rows.append({
            "ticker": ticker,
            "sector": info.get("sector"),
            "asset_class": info.get("quoteType"),        
            "current_price": info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"), # For asset allocation
        })

    dataframe = pd.DataFrame(rows).set_index("ticker")

    return dataframe

def fetch_history(tickers, start = "2015-01-01", end = "2024-12-31", interval = "1d") -> pd.DataFrame:
    r"""
    Fetch the historic prices of specified stocks, use closing price of each day

    :params tickers: list of stock tickers

    :returns: DataFrame object of price histories
    """

    data = yf.download(tickers, start=start, end=end, interval=interval, group_by="ticker", auto_adjust=True, progress = False)

    # Below dataframe consists out of columns with different indexes: (characteristic, ticker), we only want characteristic "Close"
    price_dfs = []
    for t in tickers:
        price_dfs.append(data[(t,"Close")].rename(t))
    
    historic_prices = pd.concat(price_dfs, axis=1)

    return historic_prices

def fetch_features(tickers, historic_prices) -> pd.DataFrame:
    r"""
    Fetch the needed features from Yahoo Finance to make them ready for preparation, the following features are chosen:
        - ret_1: 1-day return
        - ret_5: 5-day (weekly) return
        - vol_20: 20-day rolling volatility of ret_1
        - mom_20: 20-day momentum = close / 20-day MA - 1
    
    These four capture the short-term return direction, the volatility over periods and medium-term momentum (if price above 20-day average, 
    could indicate bullish signal and vice versa).

    :params tickers: list of stock tickers
    :params historic_prices: Datafram object of historic prices

    :returns: Dataframe object containing the desired features with multi-index columns (level 0: ticker name, level 1: feature name)
    """
    feat_dfs = []
    for ticker in tickers:
        df = historic_prices[ticker].to_frame("close")
        # One day returns and five day returns
        df["ret_1"]  = df["close"].pct_change(1)
        df["ret_5"]  = df["close"].pct_change(5)
        # First create moving window, then apply std function
        df["vol_20"] = df["ret_1"].rolling(20).std()
        # First calculate moving average, use this to calculate momentum
        df["ma_20"]  = df["close"].rolling(20).mean()
        df["mom_20"] = df["close"] / df["ma_20"] - 1
        # Note that NaN values are generated because differences are taken, drop them
        df = df.dropna()

        # To create a multi-index dataframe object, give these columns a ticker name
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        feat_dfs.append(df)

    # Join the list of dataframes side-by-side for each ticker
    features = pd.concat(feat_dfs, axis=1)
    return features
