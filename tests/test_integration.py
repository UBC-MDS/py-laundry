# test suite for integration testing of the module

# importing required packages
from pylaundry.fill_missing import fill_missing
from pylaundry.categorize import categorize
from pylaundry.transform_columns import transform_columns
from pylaundry.select_features import select_features
import pandas as pd
import numpy as np

# dataset for testing integration
X_train = pd.DataFrame({
    'cat_column1': [1, 1, 1, 2, 2],
    'cat_column2': [3, 3, 2, 1, 1],
    'num_column1': [1.5, 2.5, 3.5, None, 4.5],
    'num_column2': [0.001, 0, 0.3, None, -0.8]
})

X_test = pd.DataFrame({
    'cat_column1': [1, 1, None],
    'cat_column2': [3, 3, 2],
    'num_column1': [1.5, 2.5, 3.5],
    'num_column2': [0.001, 0, 0.3]
})

y_train = np.array([3, 5, 7, 6, 9])


def run_pylaundry():
    """
    Runs all modules of pylaundry
    Arguments
    --------
    NA
    Returns
    ------
    features_selected = list of final features selected
    """
    col_dict = categorize(df=X_train)
    # second function - fill_missing
    clean_data = fill_missing(X_train, X_test,
                              col_dict,
                              num_imp="mean", cat_imp="mode")
    # third function - transform_columns
    transformed_data = transform_columns(clean_data['X_train'],
                                         clean_data['X_test'],
                                         col_dict)
    # fourth function - feature selection
    features_selected = select_features(transformed_data['X_train'],
                                        y_train, n_features=2)
    return features_selected


features_selected = run_pylaundry()
# test case for overall functioning


def test_pylaundry():
    """
    tests final output of the package after all steps
    Arguments
    ---------
    NA
    """
    assert len(features_selected) == 2, "number of output " \
                                        "features selected must be two"
    assert features_selected[0] == 'num_column1',\
        "Column 1 is most important feature to be selected"
    assert type(features_selected) == list, "Final lectures should be a list"
