def transform_columns(x_train, x_test,  column_dict, cat_trans = "onehot_encoding", num_trans = "standard_scaling"):
    """
    Transforms categorical and numerical features based on user input.

    Arguments
    ---------     
    x_train -- pandas.core.frame.DataFrame       
        A pandas dataframe for training set
    x_test -- pandas.core.frame.DataFrame       
        A pandas dataframe for training set
    column_dict: dictionary
        A dictionary with keys = 'numeric','categorical','text', 
	and values = a list of columns that fall into
	each respective category.
    cat_trans -- list
        transformation method for categorical features(default - 'ohe')
    num_trans -- list
        transformation method for numerical features(default - 'StandardScaler')
    
    
    Returns
    -------
    df_transformed -- pandas.core.frame.DataFrame  
        A pandas dataframe
    
    """
    