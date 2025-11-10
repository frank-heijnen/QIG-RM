# ## Gathering the data using Yahoo Finance API

# # Import relevant packages
# import yfinance as yf
# import pandas as pd
# from curl_cffi import requests

# def fetch_master_data(tickers) -> pd.DataFrame:
#     r"""
#     Fetch data of the specified stocks that does not change over time (masterdata), it consists out of:
#         - ticker
#         - sector
#         - asset class
#         - current price
#         - market cap (for asset allocation)

#     :params tickers: list of stock tickers
    
#     :returns: DataFrame object of master data
#     """
    
#     session = requests.Session(impersonate="chrome")
#     multi_ticker = yf.Tickers(" ".join(tickers), session = session)

#     rows = []
#     for ticker in tickers:
#         info = multi_ticker.tickers[ticker].info
#         rows.append({
#             "ticker": ticker,
#             "sector": info.get("sector"),
#             "asset_class": info.get("quoteType"),        
#             "current_price": info.get("regularMarketPrice"),
#             "market_cap": info.get("marketCap"),
#         })

#     dataframe = pd.DataFrame(rows).set_index("ticker")

#     return dataframe

# # End date is based on transaction report
# def fetch_history(tickers, start = "2015-01-01", end = "2025-10-16", interval = "1d") -> pd.DataFrame:
#     r"""
#     Fetch the historic prices of specified stocks, use closing price of each day

#     :params tickers: list of stock tickers

#     :returns: DataFrame object of price histories
#     """

#     data = yf.download(tickers, start=start, end=end, interval=interval, group_by="ticker", auto_adjust=True, progress = False)

#     # Below dataframe consists out of columns with different indexes: (characteristic, ticker), we only want characteristic "Close"
#     price_dfs = []
#     for t in tickers:
#         price_dfs.append(data[(t,"Close")].rename(t))
    
#     historic_prices = pd.concat(price_dfs, axis=1)

#     return historic_prices

# ============================================================
# Gathering the data using Yahoo Finance API (USD-adjusted)
# ============================================================

import yfinance as yf
import pandas as pd
from curl_cffi import requests

def fetch_master_data(tickers) -> pd.DataFrame:
    r"""
    Fetch master data for specified stocks, including:
        - ticker
        - sector
        - asset class
        - current price (in USD)
        - market cap (in USD)
        - currency of listing

    :params tickers: list of stock tickers
    :returns: DataFrame of master data (USD-denominated)
    """
    session = requests.Session(impersonate="chrome")
    multi_ticker = yf.Tickers(" ".join(tickers), session=session)

    # --- Step 1: gather raw info per ticker ---
    rows = []
    for ticker in tickers:
        info = multi_ticker.tickers[ticker].info or {}
        rows.append({
            "ticker": ticker,
            "sector": info.get("sector"),
            "asset_class": info.get("quoteType"),
            "current_price_native": info.get("regularMarketPrice"),
            "market_cap_native": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
        })

    df = pd.DataFrame(rows).set_index("ticker")

    # --- Step 2: identify non-USD currencies ---
    non_usd_currencies = [c for c in df["currency"].unique() if c != "USD"]
    fx_pairs = {ccy: f"{ccy}USD=X" for ccy in non_usd_currencies}

    # --- Step 3: download FX rates safely ---
    fx_rates = {}
    for ccy, pair in fx_pairs.items():
        fx_data = yf.download(pair, period="5d", interval="1d", progress=False)

        # robust selection of a numeric column
        if "Adj Close" in fx_data.columns:
            rate = fx_data["Adj Close"].dropna().iloc[-1]
        elif "Close" in fx_data.columns:
            rate = fx_data["Close"].dropna().iloc[-1]
        else:
            numeric_cols = fx_data.select_dtypes("number")
            rate = float(numeric_cols.iloc[-1, 0]) if not numeric_cols.empty else 1.0

        # ensure scalar float
        try:
            fx_rates[ccy] = float(rate)
        except Exception:
            fx_rates[ccy] = 1.0  # fallback to 1 if conversion fails

    # --- Step 4: apply conversion to USD ---
    df["fx_to_usd"] = df["currency"].apply(lambda c: 1.0 if c == "USD" else fx_rates.get(c, 1.0))
    df["current_price_usd"] = df["current_price_native"] * df["fx_to_usd"]
    df["market_cap_usd"] = df["market_cap_native"] * df["fx_to_usd"]

    # --- Step 5: clean up and return ---
    df = df[[
        "sector",
        "asset_class",
        "currency",
        "current_price_usd",
        "market_cap_usd"
    ]].rename(columns={
        "current_price_usd": "current_price",
        "market_cap_usd": "market_cap"
    })

    return df


def fetch_history(tickers, start="2015-01-01", end="2025-10-16", interval="1d") -> pd.DataFrame:
    r"""
    Fetch historic prices of specified stocks (daily close).
    Automatically converts non-USD tickers to USD using FX data.

    :params tickers: list of stock tickers
    :returns: DataFrame with all prices in USD
    """

    # --- Download prices
    data = yf.download(
        tickers, start=start, end=end, interval=interval,
        group_by="ticker", auto_adjust=True, progress=False
    )

    # --- Detect structure and extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-index structure: (ticker, 'Close')
        close_prices = []
        for t in tickers:
            try:
                close_prices.append(data[(t, "Close")].rename(t))
            except KeyError:
                # fallback if column is named differently
                sub_df = data[t]
                if "Close" in sub_df.columns:
                    close_prices.append(sub_df["Close"].rename(t))
                else:
                    close_prices.append(sub_df.iloc[:, 0].rename(t))
        historic_prices = pd.concat(close_prices, axis=1)

    else:
        # Flat column structure, e.g. direct prices
        if "Adj Close" in data.columns:
            historic_prices = data["Adj Close"]
        elif "Close" in data.columns:
            historic_prices = data["Close"]
        else:
            historic_prices = data

    # --- Detect currencies
    session = requests.Session(impersonate="chrome")
    multi_ticker = yf.Tickers(" ".join(tickers), session=session)
    currency_map = {t: multi_ticker.tickers[t].info.get("currency", "USD") for t in tickers}

    # --- Fetch FX data for non-USD currencies
    fx_pairs = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "CHF": "CHFUSD=X"}
    fx_data = {}
    for ccy, pair in fx_pairs.items():
        fx_data[ccy] = yf.download(pair, start=start, end=end, interval=interval, progress=False)
        fx_data[ccy] = fx_data[ccy].iloc[:, 0]  # use first column for robustness

    # --- Convert non-USD tickers
    for t in tickers:
        ccy = currency_map.get(t, "USD")
        if ccy in fx_data:
            rate = fx_data[ccy].reindex(historic_prices.index, method="ffill")
            historic_prices[t] = historic_prices[t] * rate

    return historic_prices



# For training XGBoost model for predicting stock prices
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
