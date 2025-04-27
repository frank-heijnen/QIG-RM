# ASR-Portfolio-Tracker

This repository contains a command-line interface (CLI) application to track a simple investment portfolio of the magnificent seven. Note that I assume in our stylized economy it is only possible to invest in these stocks, but of course you could extend it to any universe of tickers. It is also assumed that the total budget is invested at 31-12-2024.

This repository consists of the following components:

- **Model.py**  
  The main engine: defines a `Stock` class and a `Portfolio` class with all the simulation & allocation logic.

- **Viewer.py**  
  A set of plotting functions (Matplotlib & Seaborn) to show current and historical price charts.

- **DataCollector.py**  
  Pulls in historical prices and other meta-/masterdata from Yahoo Finance API.

- **Controller.py**  
  The “control room” / CLI entry-point that ties together data fetching, modeling and viewing.

For the required packages and dependencies see `requirements.txt` (NumPy, pandas, yfinance, seaborn, etc.), run pip install -r requirements.txt in terminal.
