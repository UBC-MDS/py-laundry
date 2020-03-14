# test suite for transform_columns function

import pandas as pd
import numpy as np
from pylaundry.transform_columns import transform_columns

# dataframes for testing

manager = ["M1", "M2", "M3", "M1", "M2",
           "M3", "M1", "M2", "M3", "M1"]
age = [23, 56, 34, 40, 34, 56, 45, 65, 54, 43]
sex = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
daily_wage = [100, 200, 100, 60, 80, 140, 320, 60, 90, 90]
X_train = {
    'manager': manager,
    'age': age,
    'sex': sex,
    'daily_wage': daily_wage}
X_train = pd.DataFrame(X_train)

manager = ["M1", "M2", "M3", "M1", "M2", "M3"]
age = [23, 56, 34, 40, 34, 56]
sex = ['M', 'F', 'M', 'F', 'M', 'F']
daily_wage = [80, 140, 320, 60, 90, 90]
X_test = {
    'manager': manager,
    'age': age,
    'sex': sex,
    'daily_wage': daily_wage}
X_test = pd.DataFrame(X_test)

column_dict = {'numeric': ['age', 'daily_wage'],
               'categorical': ['sex', 'manager']}


# test cases for bad inputs

# functon to test wrong X_train/X_test argument

def test_wrong_train_test_set():
    """
    Tests inputs for arguments X_train and X_test
    Arguments
    --------
    N/A
    """
    try:
        transform_columns(X_test=X_test, column_dict=column_dict, X_train=5)

    except AssertionError:
        pass

    try:
        transform_columns(column_dict=column_dict,
                          X_train=X_train, X_test=np.array([1, 2]))

    except AssertionError:
        pass

    try:
        transform_columns(column_dict=column_dict, X_train=X_train,
                          X_test=X_test[["manager"]])

    except AssertionError:
        pass

    try:
        transform_columns(column_dict=column_dict,
                          X_train=pd.DataFrame(([1, 2, 3, 4])),
                          X_test=pd.DataFrame(([1, 2, 3, 4])))
    except AssertionError:
        pass


def test_wrong_column_dict():
    """
    Tests inputs for argument column_dict
    Arguments
    --------
    N/A
    """

    try:
        transform_columns(X_train, X_test,
                          column_dict=['numeric',
                                       'categorical'])

    except AssertionError:
        pass

        # Wrong number of keys
    try:
        transform_columns(X_train,
                          X_test,
                          column_dict={'numeric': ['A', 'B'],
                                       'categorical': ['C', 'D'],
                                       'mix': ['E', 'F']})

    except AssertionError:
        pass

        # bad dictionary keys
    try:
        transform_columns(X_train, X_test,
                          column_dict={'numeric': ['A', 'B'],
                                       'mix_cat': ['C', 'D']})

    except AssertionError:
        pass

        # checking if dictionary has right column names for each key
    try:
        transform_columns(X_train, X_test,
                          column_dict={'numeric': ['A', 'B'],
                                       'categorical': ['C', 'D']})

    except AssertionError:
        pass

    # test cases for bad num_trans inputs


def test_wrong_num_trans():
    """
    Tests inputs for argument num_trans
    Arguments
    --------
    N/A
    """

    # bad input type
    try:
        transform_columns(X_train, X_test, column_dict, num_trans=5)

    except AssertionError:
        pass

        # bad argument
    try:
        transform_columns(X_train, X_test, column_dict,
                          num_trans='onehot_encoding')

    except AssertionError:
        pass

    try:
        transform_columns(X_train, X_test, column_dict,
                          num_trans='onehot_encoding',
                          cat_trans='minmax_scaling')

    except AssertionError:
        pass

    # test cases for bad cat_trans inputs


def test_wrong_cat_trans():
    """
    Tests inputs for argument cat_trans
    Arguments
    --------
    N/A
    """

    # bad input type
    try:
        transform_columns(X_train, X_test,
                          column_dict, cat_trans=5)

    except AssertionError:
        pass

        # bad argument
    try:
        transform_columns(X_train, X_test, column_dict,
                          cat_trans='minnmax_scaling')
    except AssertionError:
        pass

    # testing onehot_encoding


def test_onehot_encoding():
    """
    Tests if transform_columns does onehot_encoding correctly
    Arguments
    --------
    N/A
    """

    # number of columns
    assert len(transform_columns(X_train, X_test, column_dict)
               ['X_train'].columns) >= X_train.shape[1], \
        "Number of columns in transformed X_train " \
        "must be greater than or equal to input X_train after ohe"
    assert len(transform_columns(X_train, X_test, column_dict)
               ['X_test'].columns) >= X_test.shape[1], \
        "Number of columns in transformed X_test must" \
        " be greater than or equal to input X_train after ohe"

    assert np.array_equal((transform_columns(X_train,
                                             X_test, column_dict)
                           ['X_train'].columns),
                          (transform_columns(X_train,
                                             X_test,
                                             column_dict)
                           ['X_test'].columns)), \
        "Transformed X_train and X_test must hav" \
        "e same column names after ohe"


def test_label_encoding():
    """
    Tests if transform_columns does label_encoding correctly
    Arguments
    --------
    N/A
    """

    # number of columns
    assert len(transform_columns(X_train, X_test, column_dict,
                                 cat_trans="label_encoding")
               ['X_train'].columns) == X_train.shape[1], \
        "Number of columns in transformed X_train must be " \
        "equal to input X_train after label encoding"
    assert len(transform_columns(X_train, X_test,
                                 column_dict,
                                 cat_trans="label_encoding")
               ['X_test'].columns) == X_test.shape[1], \
        "Number of columns in " \
        "transformed X_test must be equal to input X_train"

    assert np.array_equal(
        (transform_columns(X_train, X_test,
                           column_dict, cat_trans="label_encoding")
         ['X_train'].columns),
        (transform_columns(X_train, X_test,
                           column_dict, cat_trans="label_encoding")
         ['X_test'].columns)), \
        "Transformed X_train and X_test must " \
        "have same column names after label encoding"


def test_standard_minmax_scaling():
    """
    Tests if transform_columns does
     minmax_scaling and label_encoding correctly
    Arguments
    --------
    N/A
    """

    assert abs(np.mean(transform_columns(X_train,
                                         X_test, column_dict)
                       ['X_train']
                       [column_dict['numeric'][0]])) < 10 ** (-3),\
        "X_train's numeric columns after"\
        " standard scaling must have mean close to 0"

    assert np.max(transform_columns(X_train, X_test,
                                    column_dict,
                                    num_trans='minmax_scaling')
                  ['X_train']
                  [column_dict['numeric'][0]]) <= 1,\
        "X_train's numeric columns after " \
        "minmax scaling must have maximum value of 1"

    assert np.min(transform_columns(X_train, X_test,
                                    column_dict,
                                    num_trans='minmax_scaling')
                  ['X_train'][
                      column_dict['numeric'][0]]) >= 0,\
        "X_train's numeric columns after minmax "\
        "scaling must have minimum value of 0"


def test_output():
    """
    Tests if transform_columns returns output dictionary correctly
    Arguments
    --------
    N/A
    """
    output_dict = transform_columns(X_train, X_test,
                                    column_dict, num_trans='minmax_scaling')

    # Check output length
    assert len(output_dict) == 2, \
        "Output of transform_columns() should " \
        "be a dictionary of length two"
    # output type
    assert isinstance(output_dict, dict), \
        "Output of transform_columns() should" \
        " be a dictionary of length two"
    # check elements of output_dict
    assert isinstance(output_dict['X_train'], pd.DataFrame), \
        "Dictionary elements must hold train" \
        " and test dataframes"
    assert isinstance(output_dict['X_test'], pd.DataFrame), \
        "Dictionary elements must hold train" \
        " and test dataframes"
    # correct dictionary keys
    for key in output_dict.keys():
        assert key in ['X_train', 'X_test'], \
            "output_dict keys can be only 'X_test' and 'X_train'"

    output_dict = transform_columns(X_train, X_test, column_dict)

    # Check output length
    assert len(output_dict) == 2, \
        "Output of transform_columns() should" \
        " be a dictionary of length two"
    # output type
    assert isinstance(output_dict, dict), \
        "Output of transform_columns() should " \
        "be a dictionary of length two"
    # check elements of output_dict
    assert isinstance(output_dict['X_train'], pd.DataFrame), \
        "Dictionary elements must hold train and test dataframes"
    assert isinstance(output_dict['X_test'], pd.DataFrame), \
        "Dictionary elements must hold train and test dataframes"
    # correct dictionary keys
    for key in output_dict.keys():
        assert key in ['X_train', 'X_test'], \
            "output_dict keys can be only 'X_test' and 'X_train'"

    output_dict = transform_columns(X_train, X_test, column_dict,
                                    num_trans='minmax_scaling',
                                    cat_trans='label_encoding')
    # Check output length
    assert len(output_dict) == 2, \
        "Output of transform_columns() should be a dictionary of length two"
    # output type
    assert isinstance(output_dict, dict), \
        "Output of transform_columns() should be a dictionary of length two"
    # check elements of output_dict
    assert isinstance(output_dict['X_train'], pd.DataFrame), \
        "Dictionary elements must hold train and test dataframes"
    assert isinstance(output_dict['X_test'], pd.DataFrame), \
        "Dictionary elements must hold train and test dataframes"
    # correct dictionary keys
    for key in output_dict.keys():
        assert key in ['X_train', 'X_test'], \
            "output_dict keys can be only 'X_test' and 'X_train'"

    output_dict = transform_columns(X_train, X_test,
                                    column_dict, cat_trans='label_encoding')
    # Check output length
    assert len(output_dict) == 2, \
        "Output of transform_columns() should be a dictionary of length two"
    # output type
    assert isinstance(output_dict, dict), \
        "Output of transform_columns() should be a dictionary of length two"
    # check elements of output_dict
    assert isinstance(output_dict['X_train'], pd.DataFrame), \
        "Dictionary elements must hold train and test dataframes"
    assert isinstance(output_dict['X_test'], pd.DataFrame), \
        "Dictionary elements must hold train and test dataframes"
    # correct dictionary keys
    for key in output_dict.keys():
        assert key in ['X_train', 'X_test'], \
            "output_dict keys can be only 'X_test' and 'X_train'"
