## Gathering the data using Yahoo Finance API

# Import relevant packages
import yfinance as yf
import pandas as pd

def fetch_master_data(tickers) -> pd.DataFrame:
    r"""
    Fetch data of the specified stocks (being a list) that does not change over time (masterdata), it consists out of:
        - ticker
        - sector
        - asset class
        - current price
        - market cap (for asset allocation)
    
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
    Fetch the historic prices of specified stocks (being again a list), use closing price of each day

    :returns: DataFrame object of price histories
    """

    data = yf.download(tickers, start=start, end=end, interval=interval, group_by="ticker", auto_adjust=True, progress = False)

    # Below dataframe consists out of columns with different indexes: (characteristic, ticker), we only want characteristic "Close"
    price_dfs = []
    for t in tickers:
        price_dfs.append(data[(t,"Close")].rename(t))
    
    historic_prices = pd.concat(price_dfs, axis=1)

    return historic_prices

