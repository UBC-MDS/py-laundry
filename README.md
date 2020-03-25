
## Summary
The `pylaundry` package performs many standard preprocessing techniques for Pandas dataframes, before use in statistical analysis and machine learning. The package functionality includes categorizing column types, handling missing data and imputation, transforming/standardizing columns and feature selection. The `pylaundry` package aims to remove much of the grunt work in the typical data science workflow, allowing the analyst maximum time and energy to devote to modelling!

![](https://github.com/UBC-MDS/pylaundry/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/pylaundry/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/pylaundry) ![Release](https://github.com/UBC-MDS/pylaundry/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/pylaundry/badge/?version=latest)](https://pylaundry.readthedocs.io/en/latest/?badge=latest)

## Installation:
```
pip install -i https://test.pypi.org/simple/ pylaundry
```

## Features
- `categorize`: This function will take in a Pandas dataframe, and output a dictionary with column types as keys (numerical or categorical), and a list of column names associated with each column type as values. Columns in the dataframe that do not fall into either category are omitted from the return (e.g. text or date columns)
    - Categorical criteria: Any column with fewer than a specified number of unique values OR with dtype = 'category'. A dtype of 'category' will override the specified unique value limit.
    - Numeric criteria: Any column with dtype = `float64', or any integer column that has more unique values than the specified limit for categorical columns.  

- `fill_missing`: This function takes in a training feature dataframe, a testing feature dataframe, and a dictionary denoting column type ('numeric' and 'categorical', like the output of `categorize`) and fills missing values in both dataframes depending on the column type. Missing values in 'numeric' columns can be filled by either the mean or median column value of the training dataframe, and values for 'categorical' columns are filled by the column mode of the training dataframe. Currently, this function supports only 'numeric' and 'categorical' columns. Two dataframes, the training and testing, with missing values filled are returned.

-  `transform_columns`: This function takes in a training dataframe, a testing feature dataframe, and a dictionary denoting column type ('numeric' and 'categorical', like the output of `categorize`) and applies appropriate pre-processing techniques to each column. Categorical columns may be transformed with One Hot Encoding or Ordinal Encoding based on the values in the training dataframe and numerical columns will be transformed with a Standard Scaler or MinMax Scaler based on the training dataframe. Two dataframes, the training and testing, with transformed columns are returned.

- `feature_selector`: This function takes in a feature dataframe, an array of targets, a mode (Regression or Classification), and a maximum number of features to select. The function returns the most important features to predict the target as a list.

### pylaundry in the Python ecosystem
- [pandas.DataFrame.dtypes] returns a Pandas series with the datatype of each column, similar to our `Categorize` function, which returns a dictionary. 
- [sklearn.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) offers similar functionality for the fill_missing and transform_columns functions, where similar functions can be wrapped in a Pipeline and carried out sequentially.
- There are many feature selection packages and functions, for instance [sklearn.feature_selection](https://scikit-learn.org/stable/modules/feature_selection.html), which carry out similar functionality to our `feature_selector` function

- The added advantage of `pyLaundry` is being the first package we are aware to abstract away the full dataframe pre-processing workflow with a unified and simple API.

## Dependencies

- Python 3.7.3 and Python packages:
  - pandas==0.24.2  
  - numpy==1.16.4  
  - scikit-learn==0.22   

## Documentation

View the [official documentation](https://pylaundry.readthedocs.io/en/latest/?badge=latest), hosted on `Read the Docs`. 
  
## Use Case and Examples

Use Case:

>You have a dataset that you want to pass in to a machine learning algorithm. You have already separated your target from features, and split the data into training and test sets. You know the dataset has NA values and features of varying types, and additionally, the dataset contains more features than you'd like to train on. 

#### categorize()

>Certain transformations are valid for only certain column types. (e.g. OHE is applicable to categorical features, and scaling is applicable to numerical features). `categorize()` allows you to pass your entire dataframe into a function and creates lists of categorical and numerical features that you can pass into a transformation pipeline. 

```
from pylaundry.categorize import categorize
import pandas as pd

df = pd.DataFrame({'a':[1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                       'b':[1.2, 3.4, 3.0, 4.9, 5.3, 6.1, 8.8, 9.4, 10.4, 3.5],
                       'c':['A','B','C','D','E','F','G','H','I','J']})

# categorize with default max_cat (10)
categorize(df)
>>> {'numeric':['b'], 
   'categorical':['a','c']}

# categorize with max_cat = 5 (column c is neither numeric or categorical 
# under this restriction)
categorize(df, max_cat = 5)
>>> {'numeric':['b'],
     'categorical':['a']}

# Explicitly setting dtype to category will override max_cat settings 
# for a column
df2 = df.astype({'b':'category'})
categorize(df2, max_cat = 5)
>>> {'numeric':[],
     'categorical':['a', 'b']}
```

#### fill_missing()

>The next step is to fill NA values in your dataframe. You can specify how numerical features are imputed (mean, median), and categorical NA values are imputed by the mode of the feature. The imputed values are calculated from the training set, and applied to NA values in both the training and test set. The training and test sets are passed in, and filled training and test sets are returned.


```
from pylaundry.fill_missing import fill_missing
import pandas as pd
import numpy as np

df_train = pd.DataFrame({'a':[1, 2, np.NaN, 4, 4],
                             'b':[1.2, 3.4, 3.0, 4.9, np.NaN]})

df_test = pd.DataFrame({'a':[6, np.NaN, 0],
                           'b':[0.5, 9.2, np.NaN]})

fill_missing(df_train, df_test, {'numeric':['b'], 'categorical':['a']}, 
             num_imp = 'median', cat_imp = 'mode')
>>>      
{'X_train':              a    b    
                    0    1  1.2    
                    1    2  3.4    
                    2    4  3.0    
                    3    4  4.9    
                    4    4  3.2   

'X_test':
                         a    b    
                    0    6  0.5    
                    1    4  9.2    
                    2    0  3.2  }
```

#### transform_columns()

>Now that you have parsed your dataframe into categorical and numeric features and filled the missing values, `transform_columns()` is a one-stop shop for applying common transformations to features by their column types. Categorical columns may be one-hot-encoded or ordinal encoded, and a standard scaler or MinMax scaler may be applied to numerical columns. Transformations will be fit on the training set, and applied to both the training and test sets. The training and test sets are passed in, and transformed training and test sets are returned.

```
from pylaundry.transform_columns import transform_columns
import pandas as pd

df_train = pd.DataFrame({'a':[1, 2, 3],
                             'b':[1.2, 3.4, 3.0],
                             'c':['A','B','C']})

df_test = pd.DataFrame({'a':[6, 2],
                            'b':[0.5, 9.2],
                            'c':['B', 'B']})

transform_columns(df_train, df_test, {'numeric':['a', 'b'], 'categorical':['c']})
>>> 
{'X_train':  
          a      b     c_B  c_C
    0  -1.22  -1.39      0    0
    1   0.0    0.91      1    0
    2   1.22   0.49      0    1

'X_test':
         a      b     c_B  c_C
    0   4.9   -2.13     1    0
    1   1.2    6.96     1    0 }
```

#### select_features()

>Finally, you have your filled and transformed training and testing datasets. Your test set contains more features than you'd like to use, since you know that some features are better predictors than others. You can use `select_features()` to obtain a specified number of the most important features, which you can then use to subset your feature datasets before passing them into a machine learning algorithm.

```
from pylaundry.select_features import select_features
import pandas as pd
import numpy as np

df = pd.DataFrame({'a':[1, 2, 3],
                       'b':[1.2, 2.2, 3.2],
                       'c':[8, 12, -5]})

target = np.array([1, 2, 3])

select_features(df, target, mode = 'regression', n_features = 2)
>>>  ['a', 'b']
```

#### Proposed end-to-end workflow using pylaundry

```
from pylaundry.categorize import categorize
from pylaundry.fill_missing import fill_missing
from pylaundry.transform_columns import transform_columns
from pylaundry.select_features import select_features
import pandas as pd
import numpy as np

X_train = pd.DataFrame({'a':[1, 2, np.NaN, 4, 5, 1, 2, 3, 4, 5],
                       'b':[1.2, 3.4, 3.0, 4.9, 5.3, 6.1, 8.8, 9.4, np.NaN, 1.2],
                       'c':[1, 1, 1, 3, 1, 1, 2, 3, 3, 2]})

X_test = pd.DataFrame({'a':[6, np.NaN, 0],
                       'b':[0.5, 9.2, np.NaN],
                       'c':[3, np.NaN, 2]})

y_train = np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5])

# Categorize columns
col_dict = categorize(X_train)

# Fill missing values
filled_dict = fill_missing(X_train, X_test, col_dict, 'mean', 'mode')
X_train_filled, X_test_filled = filled_dict['X_train'], filled_dict['X_test']

# Transform data
transformed_dict = transform_columns(X_train_filled, X_test_filled, col_dict)
X_train_transformed, X_test_transformed = transformed_dict['X_train'], transformed_dict['X_test']

# Select features
cols = select_features(X_train_transformed, y_train, mode = 'regression')

X_train = X_train_transformed[cols]
X_test = X_test_transformed[cols]

```

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).


