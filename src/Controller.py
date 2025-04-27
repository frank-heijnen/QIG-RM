## Control Room of the portfolio tracker

# Import relevant functions
from Model import Stock, Portfolio
from data.DataCollector import *
from Viewer import *
import matplotlib.pyplot as plt

# Define the possible stocks to invest in
magnificent_seven_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

# Define total amount of years T, amount of trading days N and number of paths M to simulate
T, N, M = 15, 252, 100000

# Define budget
budget = 100000

def main():
    # Download the required Yahoo Finance stock data
    master_data = fetch_master_data(magnificent_seven_tickers)
    historic_prices = fetch_history(magnificent_seven_tickers)

    # Estimate the historic mean and standard deviation of each stock
    mu, sigma = Stock.estimate_simulation_params(historic_prices, N)

    # Create portfolio with initial budget, add and remove the assets you want
    portfolio = Portfolio(budget)
    portfolio.add_asset("AAPL")
    portfolio.delete_asset("AAPL")

    # Make stock objects which can be added to the portfolio
    assets = []
    for ticker, row in master_data.iterrows():
        assets.append(
            Stock(
                ticker = ticker,
                sector = row["sector"],
                asset_class = row["asset_class"],
                S0 = historic_prices[ticker].iloc[0],
                transaction_price = historic_prices[ticker].iloc[-1],
                current_price = row["current_price"],
                market_cap = master_data.loc[ticker, "market_cap"],
                mu = mu[ticker],          
                sigma = sigma[ticker],       
            )
        )

    # Question (1)
    # For now add all available assets
    # For input only the constructed asset object is necessary, characteristics like sector or asset class are specified in this object
    for asset in assets:
        portfolio.add_asset(asset)
    
    # Question (2)
    # Plot current and historic prices of particular stocks on the basis of the asset ticker, assets desired to be plotted need to be in list format 
    # If user wants to see historic prices set show_prices = True
    stocks_to_fetch = [magnificent_seven_tickers[0], magnificent_seven_tickers[1]] # For example "AAPL" and "MSFT"
    plot_historical_prices(historic_prices, stocks_to_fetch, show_prices = True)
    plot_current_prices(master_data, stocks_to_fetch)
    
    # Calculate corresponding portfolio weights using a specified method (equal weighted method or market cap method)
    method = "marketcap"
    weights = portfolio.asset_allocation(method=method)

    # Question (3), (4)
    df = portfolio.display_portfolio(weights, method=method) # Automatically prints portfolio info

    # Question (5)
    # Simulate 100.000 portfolio paths and plot the trajectories
    t, port_paths = portfolio.simulate_portfolio(df, T, N, M)
    plot_portfolio_trajectories(t, port_paths, n_paths = 20)

    # Demonstrate the impact of risk and uncertainty
    histogram_uncertainty(port_paths)

if __name__ == "__main__":
    main()
