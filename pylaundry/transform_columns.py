import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import numpy as np 

def transform_columns(X_train, X_test,  column_dict, cat_trans = "onehot_encoding", num_trans = "standard_scaling"):
    """
    Transforms categorical and numerical features based on user input.

    Arguments
    ---------     
    X_train -- pandas.core.frame.DataFrame       
        A pandas dataframe for training set
    X_test -- pandas.core.frame.DataFrame       
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
    transformed_dict -- dictionary 
        A python dictionary with transformed training and test set with keys X_train and X_test respectively
    
    """
    
    ## code to retain name convention for function arguments
    X_train = X_train
    X_test = X_test
    numeric = column_dict['numeric']
    categorical = column_dict['categorical']
    
    if cat_trans == 'onehot_encoding':
        
        if num_trans == "standard_scaling":
            preprocessor = ColumnTransformer(transformers= [
                    ("stand_scaler", StandardScaler(), numeric),
                    ("ohe", OneHotEncoder(drop="first"), categorical)], sparse_threshold =0)

            
        elif num_trans == "minmax_scaling":
            preprocessor = ColumnTransformer(transformers= [
                    ("minmax_scaler", MinMaxScaler(), numeric),
                    ("ohe", OneHotEncoder(drop="first"), categorical)], sparse_threshold =0)
        
        # ## Applying transformations to training data set
        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), 
                                   index = X_train.index,
                                   columns = numeric +list(
                                           preprocessor.named_transformers_['ohe'].get_feature_names(numeric)))
            
        #applying transformations to test set
        X_test = pd.DataFrame(preprocessor.transform(X_test),
                              index = X_test.index,
                              columns= X_train.columns)
         
    elif cat_trans == "label_encoding":
        
        if num_trans == "standard_scaling":
            
            preprocessor = ColumnTransformer(transformers= [
                    ("stand_scaler", StandardScaler(), numeric),
                    ("ordinal", OrdinalEncoder(), categorical)], sparse_threshold =0)

            
        elif num_trans == "minmax_scaling":
            preprocessor = ColumnTransformer(transformers= [
                    ("minmax_scaler", MinMaxScaler(), numeric),
                    ("ordinal", OrdinalEncoder(), categorical)], sparse_threshold =0)
        
        # ## Applying transformations to training data set
        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), 
                                   index = X_train.index,
                                   columns = numeric +categorical)
            
        #applying transformations to test set
        X_test = pd.DataFrame(preprocessor.transform(X_test),
                              index = X_test.index,
                              columns= X_train.columns)
    
    transformed_dict = {'X_train' : X_train,
                        'X_test' : X_test}
    
    return transformed_dict


# test dataset
    
# dataframes for testing
employee_name = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
manager = ["M1", "M2", "M3", "M1", "M2", "M3", "M1", "M2", "M3", "M1"]
age = [23, 56,34,40, 34,56, 45, 65, 54,43]
sex = ['M', 'F','M', 'F','M', 'F','M', 'F', 'M', 'F']
daily_wage = [100,200, 100, 60, 80, 140, 320,60, 90, 90]
x_train = {'employee_name':employee_name,
           'manager': manager,
           'age': age,
           'sex':sex,
           'daily_wage' :daily_wage}
test_train = pd.DataFrame(x_train)


employee_name =["K", "L", "M", "N", "O", "P"]
manager = ["M1", "M2", "M3", "M1", "M2", "M3"]
age = [23, 56,34,40, 34,56]
sex = ['M', 'F','M', 'F','M', 'F']
daily_wage = [ 80, 140, 320,60, 90, 90]
x_test = {'employee_name':employee_name,
           'manager': manager,
           'age': age,
           'sex':sex,
           'daily_wage' :daily_wage}
test_test = pd.DataFrame(x_test)

column_dict = {'numeric': ['age', 'daily_wage'],
               'categorical': ['sex', 'manager']}

        
    
        
    
      
    
    

    