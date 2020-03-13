from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd


class WrongData(Exception):
    pass


class WrongDataType(Exception):
    pass


def select_features(df, y, mode="regression", n_features=2):
    """
    Select Important Features from the input data using
    linear regression for mode = "regression" and
    logistic regression if mode is any other string.

    Arguments
    ---------
    df: pandas.core.frame.DataFrame
        pandas
    y: numpy.ndarray
        The y part of the data set
    n_features: int
        The number of features to return
    mode: str
        The mode for calculation (default = 'regression')

    Returns
    -------
    list
        A list of column names of length n_features
        selected via feature selection

    """

    assert isinstance(df, pd.DataFrame), "Input \
    Data is not dataframe"

    assert np.array([object()]).dtype not in list(df.dtypes), "String \
    Column Present in the data. Apply transform column function to fix"

    col = df.columns

    if mode == "regression":
        lr = LinearRegression()
        rfe = RFE(estimator=lr, n_features_to_select=n_features)
        rfe.fit(df, y)

    else:
        lg = LogisticRegression()
        rfe = RFE(estimator=lg, n_features_to_select=n_features)
        rfe.fit(df, y)

    li = []
    if len(np.argwhere(rfe.support_ > 0)) == 1:
        li.append(col[np.argwhere(rfe.support_ > 0).squeeze()])
    else:
        li = [col[i] for i in np.argwhere(rfe.support_ > 0).squeeze()]

    return li
