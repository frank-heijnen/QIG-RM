## Control Room of the portfolio tracker

# Import relevant functions
from models.Portfolio import Stock, Portfolio
from data.DataCollector import *
from models.Viewer import *

# Define the tickers in our portfolio
current_portfolio = ["AIR.PA", "DJDA.BE", "ENR.DE", "ZAL.DE", "PODD", "SNA", "TPL", "TSLA", "ISRG", "BIDU", "CNC"]

# Define total amount of years T and number of paths M to simulate. We assume daily (252) returns
T, M = 1, 100000

# Define budget
budget = 100000

# Specify weight allocation method (equal weighted method or market cap method, ML method comes at the end)
method = "manual"

def main():
    global method

    # Download the required Yahoo Finance stock data
    master_data = fetch_master_data(current_portfolio)
    historic_prices = fetch_history(current_portfolio)

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
    portfolio = Portfolio(budget, [assets[key] for key,_ in assets.items()])

    # Look at var-covar matrix
    correlation_analysis(historic_prices)
    
    # Plot current - and historic prices of particular stocks on the basis of the asset tickers, assets desired to be plotted need to be in list format 
    # If user wants to see historic prices in terminal, set show_prices = True
    stocks_to_fetch = current_portfolio # For example index 0 and 1
    plot_current_prices(master_data, stocks_to_fetch)
    plot_historical_prices(historic_prices, stocks_to_fetch, show_prices = True)
    
    # Calculate corresponding portfolio weights using a specified method 
    weights = portfolio.asset_allocation(method)

    df = portfolio.display_portfolio(weights, method) # Automatically prints portfolio info

    # Simulate 100.000 portfolio paths and plot a number of the trajectories which can be specified by n_paths
    t, port_paths = portfolio.simulate_portfolio(df, T, M)

    # Trajectories are plotted in log-scale on y-axis because paths can explode
    n_paths_to_plot = 20
    plot_portfolio_trajectories(t, port_paths, n_paths = n_paths_to_plot)

    # Demonstrate the impact of risk and uncertainty, plots distribution of terminal portfolio values including VaR/CVaR
    histogram_uncertainty(port_paths)
    # Further calculate the 1 year 99.5 VaR, this is the required buffer used in Solvency II to stay solvent in case of a rare event happening.
    scr_1y = compute_solvency_capital_requirement(port_paths, days_per_year=252, alpha=99.5)
    print(f"SCR: 1y 99.5% VaR = €{scr_1y:,.0f}")

    # # Implement ML, read corresponding section in README.md before continuing
    # portfolio.train_forecaster(historic_prices)

    # # Now calculate weights with the trained XGBoost model
    # # This automatically displays the predicted returns, the realized returns for comparison and the corresponding weights
    # method = "ml"
    # weights_ml = portfolio.asset_allocation(method)

    # # Display portfolio
    # df_ml = portfolio.display_portfolio(weights_ml, method)

if __name__ == "__main__":
    main()

    # Scrap paper
    # weights = np.array([3.0927, 2.3174, 5.8055, 11.6446, 1.5286, 1.2794, 0.3141, 1.4641, 1.4286, 7.6673, 7.5068], dtype=float)
    # report_prices = np.array(
    #     [161.26,183.40,81.68,30.58,321,321.14,1278.90,341.04,542.67,84.07,58.07]
    # )
    # recent_prices = np.array([
    # 162.100006, 181.761581,  79.160004,  30.290001, 324.500000,
    # 322.119995, 1353.713013, 334.619995, 547.070007,  85.480003,
    #  60.740002
    # ])
    # holdings = weights * report_prices
    # holdings1 = weights * recent_prices
