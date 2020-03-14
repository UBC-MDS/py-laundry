from pylaundry.fill_missing import fill_missing
import pandas as pd
import numpy as np

# Define basic objects to be used in test suite function
X_train = pd.DataFrame({
    'cat1': [1, 2, None, 1, 1],
    'num1': [1.5, 2.5, 3.5, None, 4.5],
})

X_test = pd.DataFrame({
    'cat1': [1, None, 3, 1, 3],
    'num1': [1.5, 2.5, None, 2.0, 2.0]
})

col_dict = {'numeric': ['num1'],
            'categorical': ['cat1']}

"""
Test the output is the correct length, and that the
training and testing dataframes output are of the correct type
"""


def test_output_type():
    output = fill_missing(X_train,
                          X_test,
                          col_dict,
                          num_imp="mean",
                          cat_imp="mode")
    train_output = output['X_train']
    test_output = output['X_test']
    # Check length and type of output
    assert len(output) == 2, \
        "Output of fill_missing() should be two dataframes"
    assert isinstance(train_output, pd.DataFrame), \
        "Training df should be a Pandas DF"
    assert isinstance(test_output, pd.DataFrame), \
        "Test df should be a Pandas DF"


"""
Testing accuracy of mean imputation function
"""


def test_mean_num():
    output = fill_missing(X_train,
                          X_test,
                          col_dict,
                          num_imp="mean",
                          cat_imp="mode")
    train_output = output['X_train']
    test_output = output['X_test']

    # Mean column 1 is 3 for imputed value
    # Check for same imputation train and test
    assert train_output["num1"][3] == 3, \
        "Imputed mean value should be 3.0 in train set"
    assert test_output["num1"][2] == 3, \
        "Imputed mean value should be 3.0 in test set"


"""
Testing accuracy of median imputation function
"""


def test_median_num():
    output = fill_missing(X_train=pd.DataFrame({
        'cat1': [1, 2, None, 1, 1],
        'num1': [1.5, 2.5, 3.5, None, 4.5]
    }),
        X_test=pd.DataFrame({
            'cat1': [1, None, 3, 1, 3],
            'num1': [1.5, 2.5, None, 2.0, 2.0]
        }),
        column_dict={'numeric': ['num1'],
                     'categorical': ['cat1']},
        num_imp="median",
        cat_imp="mode")
    train_output = output['X_train']
    test_output = output['X_test']

    # Mean column 1 is 3 for imputed value
    # Check for same imputation train and test
    assert train_output["num1"][3] == 3, \
        "Imputed median value should be 3.0 in train set"
    assert test_output["num1"][2] == 3, \
        "Imputed median value should be 3.0 in test set"


"""
Testing accuracy of mode imputation function
"""


def test_mode_cat():
    output = fill_missing(X_train,
                          X_test,
                          col_dict,
                          num_imp="median",
                          cat_imp="mode")
    train_output = output['X_train']
    test_output = output['X_test']

    # Mean column 1 is 3 for imputed value
    # Check for same imputation train and test
    assert train_output["cat1"][2] == 1, \
        "Imputed mode value should be 1 in train set"
    assert test_output["cat1"][1] == 1, \
        "Imputed mode value should be 1 in test set"


"""
Testing for errors thrown with bad inputs
Non pandas dataframe X_train
"""


def test_bad_train_set():
    try:
        fill_missing(X_train=np.array(1),
                     X_test=pd.DataFrame({
                         'cat1': [1, None, 3, 1, 3],
                         'num1': [1.5, 2.5, None, 2.0, 2.0]
                     }),
                     column_dict={'numeric': ['num1'],
                                  'categorical': ['cat1']},
                     num_imp="median",
                     cat_imp="mode")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Non pandas dataframe X_test
"""


def test_bad_test_set():
    try:
        fill_missing(X_train=pd.DataFrame({
            'cat1': [1, None, 3, 1, 3],
            'num1': [1.5, 2.5, None, 2.0, 2.0]
        }),
            X_test=np.array(1),
            column_dict={'numeric': ['num1'],
                         'categorical': ['cat1']},
            num_imp="median",
            cat_imp="mode")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Non dictionary for column assignments
"""


def test_bad_col_dict():
    try:
        fill_missing(X_train,
                     X_test,
                     column_dict=["cat1", "num1"],
                     num_imp="median",
                     cat_imp="mode")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Not mean or median for numerical imputation
"""


def test_non_accepted_num():
    try:
        fill_missing(X_train=pd.DataFrame({
            'cat1': [1, 2, None, 1, 1],
            'num1': [1.5, 2.5, 3.5, None, 4.5]
        }),
            X_test=pd.DataFrame({
                'cat1': [1, None, 3, 1, 3],
                'num1': [1.5, 2.5, None, 2.0, 2.0]
            }),
            column_dict={'numeric': ['num1'],
                         'categorical': ['cat1']},
            num_imp="mode",
            cat_imp="mode")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Not mode for categorical imputation
"""


def test_non_accepted_cat():
    try:
        fill_missing(X_train,
                     X_test,
                     col_dict,
                     num_imp="median",
                     cat_imp="mean")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Different col names in X_train and X_test
"""


def test_diff_columns():
    try:
        fill_missing(X_train=pd.DataFrame({
            'cat1': [1, 2, None, 1, 1],
            'num2': [1.5, 2.5, 3.5, None, 4.5]
        }),
            X_test=pd.DataFrame({
                'cat1': [1, None, 3, 1, 3],
                'num1': [1.5, 2.5, None, 2.0, 2.0]
            }),
            column_dict={'numeric': ['num1'],
                         'categorical': ['cat1']},
            num_imp="median",
            cat_imp="mean")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Columns which were not named
"""


def test_no_name_columns():
    try:
        fill_missing(X_train=pd.DataFrame([1.5, 2.5, 3.5, None, 4.5]),
                     X_test=pd.DataFrame([1, None, 3, 1, 3]),
                     # insert column which is not in the df
                     column_dict={'numeric': ['num2'],
                                  'categorical': ['cat1']},
                     num_imp="median",
                     cat_imp="mean")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Column names which are not in the DF's
"""


def test_bad_columns():
    try:
        fill_missing(X_train=pd.DataFrame({
            'cat1': [1, 2, None, 1, 1],
            'num1': [1.5, 2.5, 3.5, None, 4.5]
        }),
            X_test=pd.DataFrame({
                'cat1': [1, None, 3, 1, 3],
                'num1': [1.5, 2.5, None, 2.0, 2.0]
            }),
            # insert column which is not in the df
            column_dict={'numeric': ['num2'],
                         'categorical': ['cat1']},
            num_imp="median",
            cat_imp="mean")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Column key which is not one of two specified names
"""


def test_bad_dict_keys():
    try:
        fill_missing(X_train,
                     X_test,
                     # insert wrong column key
                     column_dict={'numerical': ['num1'],
                                  'categorical': ['cat1']},
                     num_imp="median",
                     cat_imp="mean")

    except AssertionError:
        pass


"""
Testing for errors thrown with bad inputs
Columns which contain non-numeric values
"""


def test_non_numeric_columns():
    try:
        fill_missing(X_train=pd.DataFrame({
            'cat1': ['a', 'b', None, 'c', 'a'],
            'num1': [1.5, 2.5, 3.5, None, 4.5]
        }),
            X_test=X_test,
            column_dict={'numeric': ['num1'],
                         'categorical': ['cat1']},
            num_imp="median",
            cat_imp="mean")

    except AssertionError:
        pass
