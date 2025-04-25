## Control Room of the portfolio tracker

# Import relevant functions
from Model import Stock, Portfolio
from data.DataCollector import fetch_history, fetch_master_data
from Viewer import plot_historical_prices, plot_current_prices

# Define the possible stocks to invest in
magnificent_seven_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

# Define total amount of years T, amount of trading days N and number of paths M to simulate
T, N, M = 15, 252, 100000

# Define budget
budget = 100000

def main():
    master_data = fetch_master_data(magnificent_seven_tickers)

    # Estimate mu and sigma of each stock
    historic_prices = fetch_history(magnificent_seven_tickers)
    mu, sigma = Stock.estimate_simulation_params(historic_prices, N)

    # Question (2)
    # Plot current and historic prices of particular stocks on the basis of the asset ticker, assets desired to be plotted need to be in list format 
    # If user wants to see historic prices set show_prices = True
    stocks_to_fetch = [magnificent_seven_tickers[0], magnificent_seven_tickers[1]] # For example "AAPL" and "MSFT"
    plot_historical_prices(historic_prices, tickers = stocks_to_fetch, show_prices = True)
    plot_current_prices(master_data, tickers = stocks_to_fetch)
    
    # Make stock objects which can be added to the portfolio
    assets = []
    for ticker, row in master_data.iterrows():
        assets.append(
            Stock(
                ticker = ticker,
                sector = row["sector"],
                asset_class = row["asset_class"],
                S0 = historic_prices[ticker].iloc[0],
                current_price = row["current_price"],
                market_cap = master_data.loc[ticker, "market_cap"],
                mu = mu[ticker],          
                sigma = sigma[ticker],       
            )
        )
    
    # Make portfolio with initial budget, add and remove the assets you want
    portfolio = Portfolio(budget)

    # Question (1)
    # For now add all available assets
    # Only the constructed asset object is necessary as input, characteristics like sector or asset class ar specified in asset object
    for asset in assets:
        portfolio.add_asset(asset)

    # Calculate corresponding portfolio weights using a specified method (equal weighted method or market cap method)
    weights=portfolio.asset_allocation(method="equal")

    # Question (3), (4)
    portfolio.display_portfolio(weights)

    # Question (5)
    # Simulate 100.000 portfolio paths and plot

if __name__ == "__main__":
    main()
