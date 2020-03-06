import numpy as np
import pandas as pd

## TODO: Categorical features which are strings? ["A", "B", "C"]....isnull does not seem to be working? 

def fill_missing(train_df, test_df, column_dict, num_trans, cat_trans):
    """
    Fill missing values in the dataframe based on user input.

    Arguments
    ---------     
    X_train: pandas.core.frame.DataFrame
        The training set, will be used for calculating and inputing values
    test_df: pandas.core.frame.DataFrame
        The test set, will be used for inputing values only.
    column_dict: dictionary
        A dictionary with keys = 'numeric','categorical',
        and values = a list of columns that fall into
        each respective category.
    num_trans -- string
        imputation method for numeric features, options are "mean", "median"
    cat_trans -- list
        imputation method for categorical features, options are "mode"

    Returns
    -------
    df_imputed-- pandas.core.frame.DataFrame  
        A pandas dataframe
    
    """
        
    # Check input types are as specified
    assert isinstance(train_df, pd.DataFrame), "train_df must be a Pandas DF"
    assert isinstance(test_df, pd.DataFrame), "test_df must be a Pandas DF"
    assert isinstance(column_dict, dict), "column_dict must be a dictionary"
    assert isinstance(num_trans, str), "num_trans should be a string"
    assert isinstance(cat_trans, str), "cat_trans should be a string"
    
    # Check train set and test set columns are the same
    assert np.array_equal(train_df.columns, test_df.columns), "train_df and test_df must have the same columns"
    
    # Check dictionary keys are numeric and categorical
    for key in column_dict.keys():
        assert key == 'numeric' or key == 'categorical', \
        "column_dict keys can be only 'numeric' and 'categorical'"
    
    # Check all the columns listed in dictionary are in the df
    for keys, values in column_dict.items():
        for column in values:
            assert column in train_df.columns, "columns in dictionary must be in dataframe"
            
    # Check that numerical imputation method is one of the two options 
    assert num_trans == "mean" or num_trans == "median", \
        "numerical imputation method can only be mean or median"
    
    # Check that categorical imputation method is the only option
    assert cat_trans == "mode"
    
    # Imputation methods for numerical transforms
    for column in column_dict['numeric']:
        # get column mean or median
        if num_trans == "mean":
            col_imp = train_df[column].mean()
        if num_trans == "median":
            col_imp = train_df[column].median()
            
        # Get index of NaN values in train columns
        # Todo: If these are empty (no Nan) is that fine
        index_train = train_df[column].index[train_df[column].apply(np.isnan)]
        index_test = test_df[column].index[test_df[column].apply(np.isnan)]
        
        # Use impute value on train set
        train_df.loc[index_train,column] = col_imp
        # Use same impute value on test set
        test_df.loc[index_test,column] = col_imp
            
    
    # Imputation methods for categorical transforms 
    for column in column_dict['categorical']:
       # Note:  If mode is a tie, pandas picks lower value pick the lower value!
        col_imp = train_df[column].mode()[0]
        
        # Get index of NaN values in train columns
        index_train = train_df[column].index[train_df[column].isnull()]
        index_test = test_df[column].index[test_df[column].isnull()]
        
        # Use impute value on train set
        train_df.loc[index_train,column] = col_imp
        # Use same impute value on test set
        test_df.loc[index_test,column] = col_imp
        
        
    return (train_df, test_df)
