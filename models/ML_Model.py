## Helper functions to implement XGBoost to forecast 1-day returns

# Import relevant packages
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def prepare_training_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    r"""
    Given a multi-index dataframe called features,
    build a stacked X (features) and y (next-day 1-day return) dataset for all tickers.

    :params features: multi-index dataframe object containing the fetched features

    :returns: tuple of X feature matrix and y vector of estimated 1-day returns
    """

    tickers = features.columns.get_level_values(0).unique()
    X_list, y_list = [], []

    for ticker in tickers:
        df = features[ticker].copy()
        df['target'] = df['ret_1'].shift(-1) # Get last return
        df = df.dropna()

        X_list.append(df[['ret_1','ret_5','vol_20','mom_20']])
        y_list.append(df['target'])

    # Again join the lists of dataframes
    X = pd.concat(X_list, keys=tickers, names=['ticker','date'])
    y = pd.concat(y_list, keys=tickers, names=['ticker','date'])
    return X, y


def train_GBR(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 0) -> GradientBoostingRegressor:
    r"""
    Train a GradientBoostingRegressor to predict next-day returns.

    :params X: Dataframe object of the features
    :params y: target vector
    :params test_size: fraction of return data used as test set
    :params random_state:

    :returns: GradientBoostingRegressor object which is the trained XGBoost model
    """
    # chronological split
    split_i = int(len(X) * (1 - test_size))
    X_train, X_val = X.iloc[:split_i], X.iloc[split_i:]
    y_train, y_val = y.iloc[:split_i], y.iloc[split_i:]

    # Currently the model is not validated. Note that as test data returns after 31-12-2024 could be used

    # Create model
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=random_state)

    # Fit the model on the training data which is the first 80% of the data, adhering to the chronological format of time-series data
    model.fit(X_train, y_train)

    return model


def predict_next_returns(model, features: pd.DataFrame) -> pd.Series:
    r"""
    Given a fitted XGBoost model and the multi-index features dataframe, predict the next-day return for each ticker using its most recent row.

    :params model: fitted GradientBoostingRegressor object
    :params features: dataframe object containing the features

    :returns: vector of next-day return estimates, this is a pd.Series object
    """
    preds = {}
    # level 0 of columns holds tickers
    tickers = features.columns.get_level_values(0).unique()

    for ticker in tickers:
        # Get the feature info of a ticker
        df_t = features[ticker]
        # Select the last four features to make a precition
        x_last = df_t[['ret_1','ret_5','vol_20','mom_20']].iloc[-1]
        # predict (need 2D array)
        p = model.predict(x_last.values.reshape(1, -1))[0]
        preds[ticker] = p

    # Return a pd.Series object such that the returns are indexed by ticker
    return pd.Series(preds)
