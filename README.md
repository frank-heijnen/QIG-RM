# ASR-Portfolio-Tracker

This repository contains a command-line interface (CLI) application to track a simple investment portfolio of the magnificent seven, using VSCode as IDE. I assume that in our stylized economy it is only possible to invest in these stocks, but you could extend it to any universe of tickers. It is also assumed that the total budget is invested at 31-12-2024 (see Ex (1) for an elaboration).

This repository consists of the following components:

- **Model.py**  
  The main engine: defines a `Stock` class and a `Portfolio` class with all the simulation & allocation logic.

- **Viewer.py**  
  A set of plotting functions to show current and historical price charts.

- **DataCollector.py**  
  Pulls in historical prices and other meta-/masterdata from the Yahoo Finance API.

- **ML_Model.py**
  It was asked to implement ML into this assignment. This file contains functions used for feature engineering, training a gradient boosting regressor and predicting stock returns. These are used to construct portfolio weight in **Model.py**.

- **Controller.py**  
  The “control room” / CLI entry-point that ties together data fetching, modeling and viewing.

For the required packages and dependencies see `requirements.txt` (NumPy, pandas, yfinance, seaborn, etc.); run pip install -r requirements.txt in terminal.

## Notes on Exercises

In the control room **Controller.py** the stylized economy and some state variables are defined. If one wishes to be able to invest in more stocks, simply add corresponding ticker to the stylized_economy list (note that stock needs to exist the Yahoo Finance API). 

First, load the masterdata and historic prices of the stocks in our economy. Second, estimate the mean return and its standard deviation such that stock prices can be simulated later. Lastly, asset objects are made and a portfolio is constructed.

There are two possibilites for running my code:

- Running the example code in the control room file, by specifying the state variables you want

- Setting a breakpoint, at for example line 41 (portfolio initialization), and using the *Debug Console*

**Ex (1)**

In this question it is asked that the application should allow users to add assets to the portfolio by specifying (ticker, sector, asset class, quantity, purchase price). Specifying both purchase price and quantity is not a very realistic assumption, since you cannot buy a number of shares for the price you want. Also in Ex (3) it is asked to calculate portfolio weights; note that if I provided the option to buy a particular quantity of stocks, the portfolio weights already exist.
Therefore I assume that I have a budget to invest, choose in which stocks I want to invest, and determine portfolio weights on the basis of an allocation method.

Syntax adding asset: `portfolio.add_asset(assets[ticker])`
Syntax deleting asset: `portfolio.delete_asset(assets[ticker])`

where *assets* is a dictionary containing the asset objects of our economy, containing all desired info (ticker, sector, asset class, initial price, transaction price, current price, market cap, yearly mean, yearly stdev).

I just add all stocks in our economy to the portfolio, this can be done directly by specifying a list with Stock objects as second argument in Portfolio.

**Ex (2)**

For plotting historical - and current prices (for any combination of tickers), first specify the tickers in list format: *stocks_to_fetch*. If you want to see historic prices, set show_prices = True in the *plot_historical_prices*. See the syntax below.

Syntax seeing current prices: `plot_current_prices(master_data, stocks_to_fetch)`
Syntax plotting historical prices: `plot_historical_prices('dataframe of historic prices', stocks_to_fetch, show_prices = True/False)`

Also the portfolio weights are calculated in this part of the code by running `weights = portfolio.asset_allocation('weight allocation method')`.

**Ex (3), Ex (4)**

When calling `portfolio.display_portfolio(weights, 'weight allocation method')` the desired portfolio information is displayed including how the weights are calculated. I called this object *df*, which is used for portfolio simulation in next question.

**Ex (5)**

Portfolio paths are simulated by running `t, port_paths = portfolio.simulate_portfolio(df, 'years to simulate', '#MC simulations')`.

Afterwards specify how many paths you want to plot by *n_paths_to_plot* and run `plot_portfolio_trajectories(t, port_paths, n_paths = n_paths_to_plot)`.

To see the Value at Risk and Conditional Value at Risk including the distribution of portfolio values, run `histogram_uncertainty(port_paths)`.

**Ex (6): Implementing ML**
As an extra, I want to use ML to determine the portfolio weights. At 31-12-2024 I construct the portfolio and predict the next trading day returns using a ML model. Using these returns, I determine portfolio weights according to the following formula:
$$ w_i^{ml} = \frac{\tilde{p}_i}{\sum_{i=j}^{#assets} \tilde{p}_j} \text{ where } \tilde{p}_i = max(p_i,0)$$
Note that $p_i$ is the predicted return for stock $i$ (for next trading day) using the ML model. The rationale is that if the forecasted return is higher for some stocks, I want to allocate more budget to that stock, and prevent exposure to stocks from which the predicted return is negative. If all predictions are negative, the equal weighting method is used.

As ML-model to predict the next-day returns, XGBoost is chosen since it is relatively robust to overfitting. XGBoost combines a lot of weak learners (trees) into a strong learner (and the weak i.e. simple models tend to not overfit much). It iteratively gives more weight to observations misclassified by previous models. Hence the rationale is that each next weak model corrects the mistakes of previous model.

Right now, only the next-day return after portfolio construction is estimated and on the basis of this weights get computed. This functionality is currently restricted to only looking at predicted stock returns for the day after portfolio construction (2-1-2025), and determine the weights for 31-12-2024. Hence, the model could still be extended such that it dynamically adjusts portfolio weights over time using the one-day forecasts.

For further study, methods like hyperparameter optimization could be implemented to tune the parameters of XGBoost such that test performance improves. Currently only a basic XGBoost model is trained and general parameters are chosen. I also did not test the trained model due to time restrictions. Also other instruments apart from equity could be added to the model.