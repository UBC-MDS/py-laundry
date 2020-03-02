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