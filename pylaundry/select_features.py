from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd
class WrongData(Exception):
    pass

class WrongDataType(Exception):
    pass

def select_features(df,y,mode="regression",n_features=2):

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

    if not isinstance(df, pd.DataFrame):
        raise WrongDataType("Input Data is not dataframe")

    elif np.array([object()]).dtype in list(df.dtypes):
        raise WrongData("String Column Present in the data. Apply transform column function to fix")
        
    else:
        
        col = df.columns
        
        if mode == "regression":
            lr = LinearRegression()
            rfe = RFE(estimator = lr, n_features_to_select = n_features)
            rfe.fit(df, y)
            
        else:
            lg = LogisticRegression()
            rfe = RFE(estimator = lg, n_features_to_select = n_features)
            rfe.fit(df, y)
    


    li = []
    if len(np.argwhere(rfe.support_>0)) == 1:
        li.append(col[np.argwhere(rfe.support_>0).squeeze()])
    else:
        li = [col[i] for i in np.argwhere(rfe.support_>0).squeeze()]
        

    return li
