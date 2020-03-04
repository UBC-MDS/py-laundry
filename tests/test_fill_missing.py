import pytest
import pandas as pd
import numpy as np
from pylaundry.fill_missing import fill_missing

@pytest.fixture
def generate_train_set():
    df = pd.DataFrame({
        'cat1':[1,2,None,1,2,1,1,1], 
        'num1':[1,2,3,4,5,6,7,None],
        'num2':[10,10,10,10,10,10,10,None],
    })
    return df

@pytest.fixture
def generate_test_set():
    df = pd.DataFrame({
        'cat1':[1,None,3,1,3], 
        'num1':[1.5,2.5,None,2.0,2.0],
        'num2':[None,None,0.0,1.0,98.0]
    })
    return df

# I know not using the fixtures is gross but it was giving me assertion errors when I did...
def test_output_type():
    output = fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,2,1,1,1], 
                                    'num1':[1,2,3,4,5,6,7,None],
                                    'num2':[10,10,10,10,10,10,10,None],
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0],
                                'num2':[None,None,0.0,1.0,98.0]
                            }), 
                          column_dict = {'numeric': ['num1', 'num2'],
                                          'categorical': ['cat1']},
                          num_trans = "mean",
                          cat_trans = "mode")
    train_output = output[0]
    test_output = output[1]
    # Check length and type of output
    assert len(output) == 2, \
        "Output of fill_missing() should be two dataframes"
    assert isinstance(train_output, pd.DataFrame), \
        "Training df should be a Pandas DF"
    assert isinstance(test_output, pd.DataFrame), \
        "Test df should be a Pandas DF"


# Simple test of mean inputation function
def test_mean_num():
    output = fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "mean",
                          cat_trans = "mode")
    train_output = output[0]
    test_output = output[1]
    
    # Mean column 1 is 3 for inputed value 
    # Check for same inputation train and test
    assert train_output["num1"][3] == 3, \
        "Inputed mean value should be 3.0 in train set"
    assert test_output["num1"][2] == 3, \
        "Inputed mean value should be 3.0 in test set"

# Simple test of median inputation function
def test_median_num():
    output = fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mode")
    train_output = output[0]
    test_output = output[1]
    
    # Mean column 1 is 3 for inputed value 
    # Check for same inputation train and test
    assert train_output["num1"][3] == 3, \
        "Inputed median value should be 3.0 in train set"
    assert test_output["num1"][2] == 3, \
        "Inputed median value should be 3.0 in test set"

# Simple test of mode inputation function
def test_mode_cat():
    output = fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mode")
    train_output = output[0]
    test_output = output[1]
    
    # Mean column 1 is 3 for inputed value 
    # Check for same inputation train and test
    assert train_output["cat1"][2] == 1, \
        "Inputed mode value should be 1 in train set"
    assert test_output["cat1"][1] == 1, \
        "Inputed mode value should be 1 in test set"

# Test for bad inputs
# Non pandas train set
def test_bad_train_set():
    try: 
        fill_missing(train_df = np.array(1),
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mode")
        
    except AssertionError:
        pass 

# Non pandas DF test set
def test_bad_test_set():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }),
                          test_df = np.array(1), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mode")
        
    except AssertionError:
        pass 

# Non dictionary for columns
def test_bad_col_dict():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = ["cat1", "num1"],
                          num_trans = "median",
                          cat_trans = "mode")
        
    except AssertionError:
        pass 

# Not mean or median for num imputation
def test_non_accepted_num():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "mode",
                          cat_trans = "mode")
        
    except AssertionError:
        pass 

# Non mode cat imputation
def test_non_accepted_cat():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mean")
        
    except AssertionError:
        pass 

    
# Different column names in train and test set
def test_diff_columns():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num2':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          column_dict = {'numeric': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mean")
        
    except AssertionError:
        pass
    
# Columns in col_dict which aren't in df
def test_bad_columns():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          # insert column which is not in the df
                          column_dict = {'numeric': ['num2'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mean")
        
    except AssertionError:
        pass

# column key which is not one of two specified names
def test_bad_dict_keys():
    try: 
        fill_missing(train_df = pd.DataFrame({
                                    'cat1':[1,2,None,1,1], 
                                    'num1':[1.5,2.5,3.5,None,4.5]
                          }), 
                          test_df = pd.DataFrame({
                                'cat1':[1,None,3,1,3], 
                                'num1':[1.5,2.5,None,2.0,2.0]
                            }), 
                          # insert wrong column key
                          column_dict = {'numerical': ['num1'],
                                          'categorical': ['cat1']},
                          num_trans = "median",
                          cat_trans = "mean")
        
    except AssertionError:
        pass 