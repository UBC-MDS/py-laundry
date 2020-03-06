import pandas as pd
import numpy as np

def categorize(df, max_cat = 10):
    """
    Identifies 'numeric' and 'categorical' columns
    in a dataframe, and returns a dictionary categorizing the columns with
    keys = 'numeric' and 'categorical', and values = lists of column names
    that fall into each respective category. If no numeric 
        or categorical columns exist in a dataframe, a dictionary with values =
        empty lists is returned.
    Categorical columns: Any columns with max_cat or fewer unique values,
        and any columns with dtype 'category'. A dtype of 'category' will 
        override the max_cat specification.
    Numeric columns: Any columns of dtype 'float64' or any 
        columns with dtype 'int' that have more than max_cat unique
        values. A dtype of 'float64' will override the max_cat specification.
    Other: Columns that do not meet any of the above criteria are not 
        categorized and excluded from the output dictionary. 

    Arguments
    ----------
    df --  pandas.core.frame.DataFrame
        A pandas dataframe
    
    max_cat -- int, default 10
        A positive integer denoting the maximum number of distinct values that
        a column can have for it to be marked 'categorical'.  The default
        value is 10. Any column with less than or equal to max_cat unique values
        will be marked 'categorical'

    Returns
    -------
    column_categories -- dict
        A dictionary with keys = 'numeric','categorical' 
	    and values = a list of columns that fall into
	    each respective category
    
    Examples
    --------
    >>>> from pylaundry import pylaundry
    >>>> days = ['Monday','Tuesday','Wednesday']
    >>>> temps = [2.4,3.2,5.5]
    >>>> descriptions = ['Mostly sunny with passing clouds','Cloudy with scattered showers','Sunny but very windy']
    >>>> df = pd.DataFrame({'day':days, 'temp':temps, 'weather': descriptions})
    >>>> categorize(df)
    {'numeric':['temp'], 'categorical':['days']}
    
    """
    # Check for valid parameters
    assert isinstance(df,pd.DataFrame), "df must be a Pandas DataFrame. Please see documentation."
    assert isinstance(max_cat, int), "max_cat must be an integer. Please see documentation."
    assert max_cat > 0, "max_cat must be a positive integer. Please see documentation."

    columns = set(df.columns)

    # Identify categorical columns
    # Remove columns with dtype 'float' from contention
    unique_values = df.select_dtypes(exclude='float64').nunique()
    # Add any columns with less than or equal to max_cat unique values
    # to the list of categorical columns
    categorical_set = set(unique_values[unique_values <= max_cat].index)
    # Add any columns explicitly specified as 'categorical' in DataFrame
    categorical_set.update(df.select_dtypes(include='category'))

    # Identify numerical columns
    # Mark all columns of dtype 'float64' as numeric
    all_numeric = set(df.select_dtypes(include = 'number').columns)
    # Include only integer columns that have not been categorized
    # as 'categorical'
    numeric_set = list(all_numeric - categorical_set)
    # Convert set to list
    categorical = list(categorical_set)
    numeric = list(numeric_set)

    return {'numeric':numeric, 'categorical':categorical}
