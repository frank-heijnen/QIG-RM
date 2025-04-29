## Control Room of the portfolio tracker

# Import relevant functions
from models.Portfolio import Stock, Portfolio
from data.DataCollector import *
from models.Viewer import *

# Define the possible stocks to invest in, currently only the magnificent seven
stylized_economy = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

# Define total amount of years T and number of paths M to simulate. We assume daily (252) returns
T, M = 15, 100000

# Define budget
budget = 100000

# Specify weight allocation method (equal weighted method or market cap method, ML method comes at the end)
method = "marketcap"

def main():
    global method

    # Download the required Yahoo Finance stock data
    master_data = fetch_master_data(stylized_economy)
    historic_prices = fetch_history(stylized_economy)

    # Estimate the historic mean and standard deviation of each stock
    mu, sigma = Stock.estimate_simulation_params(historic_prices)

    # Make stock objects which can be added to the portfolio
    assets = {}
    for ticker, row in master_data.iterrows():
        assets[ticker] = Stock(
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
            
    # Create portfolio with initial budget
    portfolio = Portfolio(budget)
    
    # Ex (1)
    # Add and remove the assets you want in the following way
    # For input, only the constructed asset object is only necessary. Characteristics like sector/asset class/purchase price are directly specified in this object
    portfolio.add_asset(assets["AAPL"])
    portfolio.add_asset(assets["MSFT"])
    portfolio.add_asset(assets["GOOGL"]) # E.g. add Apple, Microsoft and Google

    portfolio.delete_asset(assets["AAPL"]) # Delete Apple

    # For now, add all available stocks to our portfolio, make new object
    portfolio = Portfolio(budget, [assets[key] for key,_ in assets.items()])
    
    # Ex (2)
    # Plot current - and historic prices of particular stocks on the basis of the asset tickers, assets desired to be plotted need to be in list format 
    # If user wants to see historic prices in terminal, set show_prices = True
    stocks_to_fetch = [stylized_economy[0], stylized_economy[1]] # For example "AAPL" and "MSFT"
    plot_current_prices(master_data, stocks_to_fetch)
    plot_historical_prices(historic_prices, stocks_to_fetch, show_prices = True)
    
    # Calculate corresponding portfolio weights using a specified method 
    weights = portfolio.asset_allocation(method)

    # Ex (3), (4)
    df = portfolio.display_portfolio(weights, method) # Automatically prints portfolio info

    # Ex (5)
    # Simulate 100.000 portfolio paths and plot a number of the trajectories which can be specified by n_paths
    # This will take a couple of minutes, please be patient... +- 3.5 min
    t, port_paths = portfolio.simulate_portfolio(df, T, M)

    # Trajectories are plotted in log-scale on y-axis because paths can explode
    n_paths_to_plot = 20
    plot_portfolio_trajectories(t, port_paths, n_paths = n_paths_to_plot)

    # Demonstrate the impact of risk and uncertainty, plots distribution of terminal portfolio values including VaR/CVaR
    histogram_uncertainty(port_paths)
    # Further calculate the 1 year 99.5 VaR, this is the required buffer used in Solvency II to stay solvent in case of a rare event happening.
    scr_1y = compute_solvency_capital_requirement(port_paths, days_per_year=252, alpha=99.5)
    print(f"SCR: 1y 99.5% VaR = €{scr_1y:,.0f}")

    # Ex (6) 
    # Implement ML, read corresponding section in README.md before continuing
    portfolio.train_forecaster(historic_prices)

    # Now calculate weights with the trained XGBoost model
    # This automatically displays the predicted returns, the realized returns for comparison and the corresponding weights
    method = "ml"
    weights_ml = portfolio.asset_allocation(method)

    # Display portfolio
    df_ml = portfolio.display_portfolio(weights_ml, method)

if __name__ == "__main__":
    main()
