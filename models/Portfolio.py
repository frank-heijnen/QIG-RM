## Model ASR Portfolio Tracker

# Import relevant packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
from data.DataCollector   import fetch_features, fetch_history
from models.ML_Model      import prepare_training_data, train_GBR, predict_next_returns

# Seed for reproducability of simulated stock paths
np.random.seed(1234)

# Focuses on simulation of stock prices
@dataclass
class Stock:
    ticker: str
    sector: str
    asset_class: str
    S0: float
    transaction_price: float
    current_price: float
    market_cap: float
    mu: float
    sigma: float

    def estimate_simulation_params(historic_prices: pd.DataFrame, N=252) -> tuple:
        r"""
        Given a dataframe of historic stock prices, estimate mu and sigma of each stock, this is used to simulate stock paths for portfolio simulation.
        Make use of log returns assuming a Geometric Brownian Motion structure for stock prices:
            log(S_t/S_{t-1}) = (mu - 0.5*sigma^2)*dt + sigma* sqrt(dt)* Z

        :params historic_prices: Dataframe object of historic prices
        :params N: number of trading days

        :returns: vector of mu and sigma with corresponding estimates
        """
        # Calculate daily returns
        log_ret = np.log(historic_prices / historic_prices.shift(1)).dropna()

        # Calculate sample mean and standard deviation
        mean = log_ret.mean()           
        stdev = log_ret.std(ddof=1)

        # Annualize: Account for the trading days (periods) per year, because stock prices are going to be simulated daily. 
        sigma = stdev * np.sqrt(N)
        mu = mean * N + 0.5 * sigma**2 # Corretion term because of LogNormal distribution

        return mu, sigma

    def simulate_stock(self, T, M, N=252) -> tuple:
        r"""
        Simulate stock prices of Stock object according to Geometric Brownian Motion for simplicity:
 
        :param T: total time in years to simulate
        :param N: number of time steps
        :param M: number of simulated paths

        :returns: time grid and M simulated stock prices for N periods, hence S is of shape (N,M)
        """
        total_steps = T * N
        dt = 1/N
        t = np.linspace(0, T, total_steps+1)

        # Generate st. normals Z ~ N(0,1)
        Z = np.random.normal(0, 1, size=(total_steps, M))
        # Calculate the price changes for T * N periods
        increments = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
        # Add zeros at the start to set log(S0): S0 * exp(0) = S)
        log_returns = np.vstack([np.zeros(M), increments])
        # Cumulative sum of the returns gives log(S)
        log_S = np.cumsum(log_returns, axis=0)
        S = self.transaction_price * np.exp(log_S)

        return t, S

# Focuses on allocation given a certain budget
class Portfolio:
    def __init__(self, budget, assets = None):
        self.assets = assets or []
        self.budget = budget
    
    def add_asset(self, asset: Stock) -> None:
        r"""
        Helper function that adds asset object to portfolio
        """
        try:
            self.assets.append(asset)
        except:
            print(f"Not possible to add {asset.ticker} to portfolio since we do not have corresponding data")

    def delete_asset(self, asset: Stock) -> None:
        r"""
        Helper function that deletes asset object from of portfolio
        """
        try:
            self.assets.remove(asset)
        except:
            print(f"{asset.ticker} is not in the portfolio, so cannot be deleted")
    
    def asset_allocation(self, method: str) -> np.ndarray:
        r"""
        Compute portfolio weights given a certain budget based on a particular method. For simplicity the following methods can be used:
            - "equal": equal weight across assets
            - "marketcap": weights determined by each assets' market cap
            - "ml": machine-learning based weight allocation

        :params method: weight allocation method used
        
        :returns: Dataframe object of current portfolio, displaying each asset's name, sector, asset class, quantity,
                  purchase price and transaction value #and current value
        """
        # Dependent on which method is specified, corresponding weights are determined
        if method == "equal":
            weights = np.repeat(1 / len(self.assets), len(self.assets))
        elif method == "marketcap":
            caps = np.array([asset.market_cap for asset in self.assets], dtype=float)
            weights = caps / caps.sum()
        elif method== "ml":

            # Make next-day predictions using the trained model
            predictions = predict_next_returns(self.forecaster, self.features)

            # Negative predictions get value 0, also normalize weights such that sum = 1
            w_ml = predictions.apply(lambda x: max(x,0))
            if w_ml.sum() > 0:
                weights = w_ml.values / w_ml.sum()
            else: # If for example all predicted returns are negative, then do equal weighting
                weights = np.repeat(1 / len(self.assets), len(self.assets))

            # Get realized returns for 02-01-2025
            # Excuse me for the hard-coding...
            tickers = [asset.ticker for asset in self.assets]
            prices = fetch_history(tickers, start="2024-12-30", end  = "2025-01-03") # End date is exclusive
            realized_returns = prices.iloc[-1] / prices.iloc[-2] - 1                    

            table = pd.DataFrame({
                "Predicted next return": predictions,
                "Realized returns": realized_returns,
                "Weights": weights
            })
            print("")
            print(f"========== Comparing the predicted returns with realized returns, also showing portfolio weights ==========")
            print("")
            print(table)         

        else:
            raise ValueError(f"Unknown method '{method}'")

        return weights
    
    def display_portfolio(self, weights, method: str) -> pd.DataFrame:
        r"""
        Displaying the current portfolio including relevant info, with the possibility to see calculations for the total portfolio value and 
        the (relative) weights of each asset including the option to see the same per asset class and sector.

        :params weights: the weights determined by the asset_allocation function
        :params method: weight allocation method used

        :returns: Dataframe object of current portfolio, displaying each asset's name, sector, asset class, quantity,
                  purchase price transaction value and current value
        """
        tickers = [asset.ticker for asset in self.assets]
        transcation_prices = np.array([asset.transaction_price for asset in self.assets])
        current_prices = np.array([asset.current_price for asset in self.assets])

        # Determine allocations
        allocations = weights * self.budget

        # Shares that are bought against the transaction price, this is the price on 31-12-2024
        shares = allocations / transcation_prices

        # Calculate current value
        current_value = shares * current_prices

        # Construct dataframe containing desired info
        df = pd.DataFrame({
            "sector": np.array(asset.sector for asset in self.assets),
            "asset class": np.array(asset.asset_class for asset in self.assets),
            "quantity": shares,
            "purchase price": transcation_prices,
            "weight": weights,
            "transaction value": allocations,
            "current value": current_value,
        }, index=tickers)

        # Show the allocation info
        print("")
        print(f"=========== Current Portfolio Characteristics making use of the {method} method ===========")
        print("")
        print(df)
        print("")

        # Show how weights are calculated
        print(f"========== Calculation of portfolio value ==========")
        print("")
        if method == "equal":
            print("Portfolio weights are w_i = 1 / |assets| for asset i \nPortfolio value = sum_(i=1)^(#assets) w_i * budget")
        elif method == "marketcap":
            print("Portfolio weights are w_i = market_cap_i / sum_(i=1)^(#assets) market_cap_i\nPortfolio value = sum_(i=1)^(#assets) w_i * budget")
        else: 
            print("Portfolio weights are w_i = tilde(p)_i / sum_(i=1)^(#assets) tilde(p)_i where tilde(p)_i = max(forecasted_price_i, 0)\nPortfolio value = sum_(i=1)^(#assets) w_i * budget")

        return df

    def simulate_portfolio(self, df: pd.DataFrame, T, M, N=252) -> tuple:
        r"""
        Simulate the portfolio over T years, with N steps per year, M paths, by simulating its individual stocks 

        :params df: Dataframe object with portfolio information, including the hold shares per stock
        :params T: amount of years to simulate
        :params N: number of trading days
        :params M: amount of Monte Carlo paths to simulate 

        :returns: (T * N + 1, M) array of portfolio paths
        """
        total_steps = T * N
        t = np.linspace(0, T, total_steps+1)

        shares = df["quantity"]

        port_paths = np.zeros((total_steps+1, M))

        for asset in self.assets:
            # call the instance method on each Stock
            _, S = asset.simulate_stock(T, M)
            port_paths += S * shares.loc[asset.ticker]
        
        return t, port_paths
    
    def train_forecaster(self, historic_prices: pd.DataFrame, test_size: float = 0.2, random_state: int = 0) -> None:
        r"""
        Train a GradientBoostingRegressor on the constructed historical features to predict next-day returns.
        After calling this, self.forecaster will be the fitted XGBoost model which is used in the asset_allocation function.

        :params historic_prices: dataframe object with historic prices
        """
        # Fetch list of tickers
        tickers = [asset.ticker for asset in self.assets]

        # Fetch the features
        features = fetch_features(tickers, historic_prices)
        self.features = features

        # Prepare the training data
        X, y = prepare_training_data(features)

        # Train the model!
        self.forecaster = train_GBR(X, y, test_size=test_size, random_state=random_state)

            
        




