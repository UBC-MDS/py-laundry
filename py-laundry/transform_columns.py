def transform_columns(X, column_dict, cat_trans, num_trans):
    """
    Transforms categorical and numerical features based on user input.

    Arguments
    ---------     
    df -- pandas.core.frame.DataFrame       
        A pandas dataframe
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
    pass