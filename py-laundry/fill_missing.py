def fill_missing(df, column_dict, num_trans, cat_trans):
    """
    Fill missing values in the dataframe based on user input.

    Arguments
    ---------     
    df : pandas.core.frame.DataFrame
        A pandas dataframe
    column_dict: dictionary
        A dictionary with keys = 'numeric','categorical','text', 
	and values = a list of columns that fall into
	each respective category.
    num_trans -- string
        imputation method for numeric features
    cat_trans -- list
        imputation method for categorical features

    Returns
    -------
    df_imputed-- pandas.core.frame.DataFrame  
        A pandas dataframe
    
    """
    pass