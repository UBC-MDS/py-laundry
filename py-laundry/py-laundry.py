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
    
    """
    pass



def transform_columns(X, cat_list, num_list, cat_trans, num_trans):
    """
    Transforms categorical and numerical features based on user  inputs

    Arguments
    ---------     
    X -- numpy.ndarray        
        Features array
    cat_list -- list
        List of categorical features
    num_list -- list 
        List of numerical features
    cat_trans -- list
        transformation method for categorical features(default - 'ohe')
    num_trans -- list
        transformation method for numerical features(default - 'StandardScaler')
    
    
    Returns
    -------
    X_transformed -- numpy.ndarray
        array of transformed features
    
    """
    pass


def select_feature(df,y,mode="regression",n_features=2):

    """
    Select Important Features from the input data

    Arguments
    ---------     
    df -- dataframe        
        pandas
    y -- numpy.ndarray
        The y part of the data set
    n_features -- int
    
    Keyword arguments 
    -----------------
    mode -- str 
        The mode for calculation (default = 'regression') 
    
    Returns
    -------
    features -- list
    
    """

    pass

  
def categorize(df):
    """
    Categorizes each column in a dataframe as 'numeric', 
    'categorical' or 'text'.

    Arguments
    ----------
    df --  pandas.core.frame.DataFrame
        A pandas dataframe

    Returns
    -------
    column_categories -- dict
        A dictionary with keys = 'numeric','categorical','text', 
	and values = a list of columns that fall into
	each respective category
    
    Examples
    --------
    >>>> from py-laundry import py-laundry
    >>>> days = ['Monday','Tuesday','Wednesday']
    >>>> temps = [2.4,3.2,5.5]
    >>>> descriptions = ['Mostly sunny with passing clouds','Cloudy with scattered showers','Sunny but very windy']
    >>>> df = pd.DataFrame({'day':days, 'temp':temps, 'weather': descriptions})
    >>>> categorize(df)
    {'numeric':['temp'], 'categorical':['days'], 'text':['weather']}
    
    """
    pass


