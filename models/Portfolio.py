# Import relevant packages
import numpy as np
import pandas as pd
from arch import arch_model
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

    def simulate_stock_GBM(self, T, M, N=252) -> tuple:
        r"""
        Simulate stock prices of Stock object according to Geometric Brownian Motion for simplicity:
 
        :param T: total time in years to simulate
        :param N: number of time steps
        :param M: number of simulated paths

        :returns: time grid and M simulated stock prices for N periods, hence S is of shape (N,M)
        """
        total_steps = int(T * N)
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
        elif method == "manual": # For manual input
            weights = np.array([
                            2.599,      # DIEb
                            3.7632,     # ENR
                            1.8535,     # HEId
                            6.3816,     # PRX
                            23.1614,    # UMI
                            0.859,      # AXON
                            7.6303,     # EQT
                            6.072,      # HOLX
                            2.4376,     # PAYC
                            2.5795,     # PLTR
                            13.934,     # SMCI
                            4.0667,     # TPR
                            9.4194,     # UAL
                            3.8177      # WYNN
                        ], dtype=float)

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
        allocations = weights * transcation_prices

        # Calculate current value
        current_value = weights * current_prices

        # Construct dataframe containing desired info
        df = pd.DataFrame({
            "sector": np.array(asset.sector for asset in self.assets),
            "asset class": np.array(asset.asset_class for asset in self.assets),
            "quantity": weights,
            "purchase price": transcation_prices,
            "weight": weights,
            "transaction value": allocations,
            "current value": current_value,
        }, index=tickers)

        # Show the allocation info
        print("")
        print("=========== Current Portfolio Characteristics on 21-11-2025 ===========")
        print("")
        print(f"Portfolio Value after Purchase in USD: ${np.sum(df["quantity"] * df["purchase price"]):,.2f}")
        print("")
        print(f"Portfolio Value currently in USD: ${np.sum(df["current value"]):,.2f}")
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
        elif method == "manual":
            print("Portfolio weights are based on trading strategies")
        else: 
            print("Portfolio weights are w_i = tilde(p)_i / sum_(i=1)^(#assets) tilde(p)_i where tilde(p)_i = max(forecasted_price_i, 0)\nPortfolio value = sum_(i=1)^(#assets) w_i * budget")

        df2 = (
            df['sector']
            .value_counts()
            .rename_axis('sector')
            .reset_index(name='count')
            .sort_values('count', ascending=False)
            .reset_index(drop=True)
        )
        # Calculate HHI based on sector current values
        total_value = df["current value"].sum()
        sector_values = df.groupby("sector")["current value"].sum()
        sector_weights = sector_values / total_value

        # Herfindahl–Hirschman Index (HHI)
        hhi = np.sum(sector_weights ** 2)

        # Append HHI as a separate row to df2
        df2 = pd.concat(
            [
                df2,
                pd.DataFrame({"sector": ["HHI (sector concentration)"], "count": [round(hhi*10000)]})
            ],
            ignore_index=True
        )

        # How many different sectors including sector count
        print("")
        print(f"========== Sector concentration ==========")
        print("")
        print(df2)
        print("")
        
        return df

    def simulate_portfolio_GBM(self, df: pd.DataFrame, T, M, N=252) -> tuple:
        r"""
        Simulate the portfolio over T years, with N steps per year, M paths, by simulating its individual stocks 

        :params df: Dataframe object with portfolio information, including the hold shares per stock
        :params T: amount of years to simulate
        :params N: number of trading days
        :params M: amount of Monte Carlo paths to simulate 

        :returns: (T * N + 1, M) array of portfolio paths
        """
        total_steps = int(T * N)
        t = np.linspace(0, T, total_steps+1)

        shares = df["quantity"]

        port_paths = np.zeros((total_steps+1, M))

        for asset in self.assets:
            # call the instance method on each Stock
            _, S = asset.simulate_stock_GBM(T, M)
            port_paths += S * shares.loc[asset.ticker]
        
        return t, port_paths

    # Helper function
    def _nearest_psd(self, A, eps=1e-12):
        """Project a symmetric matrix to the nearest PSD by flooring eigenvalues at 0."""
        B = (A + A.T) / 2.0
        vals, vecs = np.linalg.eigh(B)
        vals[vals < 0] = 0.0
        return (vecs * vals) @ vecs.T + eps * np.eye(B.shape[0])

    def simulate_portfolio_GBM_correlated(self, df: pd.DataFrame, prices: pd.DataFrame, T: float, M: int, N: int = 252):
        r"""
        Correlated GBM simulation for the whole portfolio using a Cholesky-based correlation structure.

        :params df: DataFrame with portfolio info; must include columns ['quantity', 'current value'] and be indexed by tickers matching `prices`.
        :params prices: DataFrame of adjusted close price history; columns = tickers (same order as df.index), rows = dates.
        :params T: Horizon in years (e.g., 1.0).
        :params N: Number of steps per year (trading days, default 252).
        :params M: Number of Monte Carlo paths.

        :returns: (t, port_paths) where
                - t is an np.ndarray of shape (steps + 1,) with the time grid in years,
                - port_paths is an np.ndarray of shape (steps + 1, M) with simulated portfolio value paths.
        """
        tickers = list(df.index)
        px = prices[tickers].dropna(how="any")

        # --- Estimate parameters from history (annualized) ---
        logret = np.log(px).diff().dropna()
        mu = (logret.mean() * 252.0).to_numpy()             # shape (A,)
        sigma = (logret.std() * np.sqrt(252.0)).to_numpy()  # shape (A,)
        Corr = logret.corr().to_numpy()

        # Ensure PSD and get Cholesky
        Corr_psd = self._nearest_psd(Corr)
        L = np.linalg.cholesky(Corr_psd)

        # --- Setup vectors/mats ---
        A = len(tickers)
        steps = int(T * N)
        dt = 1.0 / N

        shares = df["quantity"].to_numpy(dtype=float)                       # shape (A,)
        S0 = df["purchase price"].to_numpy(dtype=float)   # current price per asset

        drift = (mu - 0.5 * sigma**2)[:, None] * dt              # shape (A,1)
        vol_step = (sigma * np.sqrt(dt))[:, None]                 # shape (A,1)

        # Prices: shape (A, steps+1, M)
        S = np.empty((A, steps + 1, M), dtype=float)
        S[:, 0, :] = S0[:, None]

        # --- Simulate correlated shocks and evolve GBMs ---
        rng = np.random.default_rng()
        for t_idx in range(steps):
            Z = rng.standard_normal(size=(A, M))         # iid N(0,1)
            eps = L @ Z                                   # correlated shocks (A,M)
            S[:, t_idx + 1, :] = S[:, t_idx, :] * np.exp(drift + vol_step * eps)

        # Portfolio value paths: sum(shares * prices) over assets
        port_paths = (shares[:, None, None] * S).sum(axis=0)     # (steps+1, M)
        t = np.linspace(0.0, T, steps + 1)

        return t, port_paths
    
    def simulate_portfolio_GARCH11(self, df: pd.DataFrame, prices: pd.DataFrame, 
                                T: float, M: int, N: int = 252):
        """
        Simulate portfolio value paths using univariate GARCH(1,1) volatility dynamics per asset.

        :params df: DataFrame with portfolio info; must include 'quantity' and 'current value', indexed by tickers.
        :params prices: DataFrame of adjusted close prices (columns = tickers, rows = dates).
        :params T: Time horizon in years (e.g., 1.0).
        :params M: Number of Monte Carlo paths.
        :params N: Number of time steps per year (default 252).
        
        :returns: (t, port_paths)
                t -> np.ndarray of time steps (length = N+1)
                port_paths -> np.ndarray of simulated portfolio values (shape = (N+1, M))
        """

        tickers = list(df.index)
        px = prices[tickers].dropna(how="any")
        logret = np.log(px).diff().dropna()

        steps = int(T * N)
        dt = 1 / N

        # Precompute quantities and prices
        shares = df["quantity"].to_numpy(dtype=float)
        S0 = df["current value"].to_numpy(dtype=float) / shares

        # Storage for simulations
        port_paths = np.zeros((steps + 1, M))
        port_paths[0, :] = np.sum(shares * S0)

        # Simulate each asset's GARCH(1,1)
        simulated_paths = np.zeros((steps + 1, len(tickers), M))

        for i, ticker in enumerate(tickers):
            returns = logret[ticker].dropna()
            model = arch_model(returns, mean='constant', vol='GARCH', p=1, q=1, rescale=False)
            fitted = model.fit(disp='off')

            # Simulate one path at a time (arch >= 6.x doesn’t accept repetitions)
            sim_returns = np.zeros((steps, M))

            for m in range(M):
                sim = model.simulate(fitted.params, nobs=steps)
                sim_returns[:, m] = sim['data']

            simulated_prices = S0[i] * np.exp(np.cumsum(sim_returns, axis=0))
            simulated_paths[1:, i, :] = simulated_prices


        # Compute total portfolio value per simulation
        for m in range(M):
            port_paths[1:, m] = np.sum(simulated_paths[1:, :, m] * shares, axis=1)

        # Time vector
        t = np.linspace(0, T, steps + 1)

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

            
        




