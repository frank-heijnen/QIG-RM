## Model ASR Portfolio Tracker

# Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Seed for reproducability of simulated stock paths
np.random.seed(1234)

# Data on 10 years NEEDS TO BE ROBUST STILL!
T = 10

# Focuses on simulation of stock prices
@dataclass
class Stock:
    ticker: str
    sector: str
    asset_class: str
    S0: float
    current_price: float
    market_cap: float
    mu: float
    sigma: float

    def estimate_simulation_params(historic_prices, N):
        r"""
        Given a dataframe of historic stock prices, estimate mu and sigma of each stock to simulate stock paths later for portfolio simulation.
        Make use of log returns assuming a Geometric Brownian Motion structure for stock prices:
            log(S_t/S_{t-1}) = (mu - 0.5*sigma^2)*dt + sigma* sqrt(dt)* Z

        :returns: vector of mu and sigma with corresponding estimates
        """
        log_ret = np.log(historic_prices / historic_prices.shift(1)).dropna()

        # Calculate estimates
        mean = log_ret.mean()           
        stdev = log_ret.std(ddof=1)

        # Account for the trading days (periods) per year, because stock prices are going to be simulated daily. 
        periods_per_year = N/T
        sigma = stdev * np.sqrt(periods_per_year)
        mu = mean * periods_per_year + 0.5 * sigma**2

        return mu, sigma

    def simulate_stock(self, T, N, M):
        r"""
        Simulate stock price of stock 'ticker' according to Geometric Brownian Motion for simplicity:

        :param ticker: Name of the stock which is simulated
        :param S0: initial stock price
        :param mu: drift/expected return based on historical data
        :param sigma: volatility based on historical data
        :param T: total time in years to simulate
        :param N: number of time steps
        :param M: number of simulated paths

        :returns: time grid and M simulated stock prices for N periods, hence S is of shape (N,M)
        """
        dt = T/N
        t = np.linspace(0, T, N)

        # Generate st. normals Z ~ N(0,1)
        Z = np.random.normal(0, 1, size=(N-1,M))
        # Calculate the price changes for N-1 periods
        increments = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
        # Add zeros at the start to set log(S0)
        log_returns = np.vstack([np.zeros(M), increments])
        # Cumulative sum of the returns gives log(S)
        log_S = np.cumsum(log_returns, axis=0)
        S = self.S0 * np.exp(log_S)

        return t, S

# Focuses on allocation given a certain budget
class Portfolio:
    def __init__(self, budget, assets = None):
        self.assets = assets or []
        self.budget = budget
    
    def add_asset(self, asset: Stock):
        try:
            self.assets.append(asset)
        except:
            print(f"Not possible to add {asset.ticker} to portfolio since we do not have corresponding data")

    def delete_asset(self, asset: Stock):
        try:
            self.assets.remove(asset)
        except:
            print(f"{asset.ticker} is not in the portfolio, so cannot be deleted")
    
    def asset_allocation(self, method: str) -> pd.DataFrame:
        r"""
        Compute portfolio weights given a certain budget based on a particular method. For simplicity the following methods can be used:
            - "equal": equal weight across assets
            - "marketcap": weight determined by each assets' market cap
        
        :returns: Dataframe object of current portfolio, displaying each asset's name, sector, asset class, quantity,
                  purchase price and transaction value #and current value
        """

        # Dependent on which method is specified, corresponding weights are determined
        if method == "equal":
            weights = np.repeat(1 / len(self.assets), len(self.assets))
        elif method == "marketcap":
            caps = np.array([asset.market_cap for asset in self.assets], dtype=float)
            weights = caps / caps.sum()
        else:
            raise ValueError(f"Unknown method '{method}'")

        return weights
    
    def display_portfolio(self, weights):
        r"""
        Displaying the current portfolio including relevant info, with the possibility to see calculations for the total portfolio value and 
        the (relative) weights of each asset includingthe option to see the same per asset class and sector.

        :params: the weights determined by the asset_allocation function

        :returns: Dataframe object of current portfolio, displaying each asset's name, sector, asset class, quantity,
                  purchase price and transaction value #and current value
        """
        tickers = [asset.ticker for asset in self.assets]
        prices = np.array([asset.current_price for asset in self.assets])

        # Determine allocations
        allocations = weights * self.budget

        # Shares that are bought against last traded price
        shares = allocations / prices

        # Construct dataframe containing desired info
        df = pd.DataFrame({
            "sector": np.array(asset.sector for asset in self.assets),
            "asset class": np.array(asset.asset_class for asset in self.assets),
            "quantity": shares,
            "purchase price": prices,
            "weight": weights,
            "transaction value": allocations,
        }, index=tickers)

        # Show the allocation info
        print(f"=========== Current Portfolio Characteristics making use of the {5} method ===========")
        print("")
        print(df)

    def simulate_portfolio(self, T, N, M) -> dict:
        r"""
        Simulate the portfolio by simulating its individual stocks 
        """
        results = {}
        for asset in self.assets:
            t, S = asset.simulate_stock(T,N,M)
            results[asset.ticker] = (t, S)
        return results



