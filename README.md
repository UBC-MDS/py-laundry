
## Summary
The `pylaundry` package performs many standard preprocessing techniques for Pandas dataframes,  before use in statistical analysis and machine learning. The package functionality includes categorizing column types, handling missing data and imputation, transforming/standardizing columns and feature selection. The `pylaundry` package aims to remove much of the grunt work in the typical data science workflow, allowing the analyst maximum time and energy to devote to modelling!

![](https://github.com/UBC-MDS/py-laundry/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/py-laundry/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/py-laundry) ![Release](https://github.com/UBC-MDS/py-laundry/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/py-laundry/badge/?version=latest)](https://py-laundry.readthedocs.io/en/latest/?badge=latest)

## Installation:
```
pip install -i https://test.pypi.org/simple/ pylaundry
```

## Features
- `categorize`: This function will take in a Pandas dataframe, and output a dictionary with column types as keys (numerical or categorical), and a list of column names associated with each column type as values. Columns in the dataframe that do not fall into either category are omitted from the return (e.g. text or date columns)
    - Categorical criteria: Any column with fewer than a specified number of unique values OR with dtype = 'category'. A dtype of 'category' will override the specified unique value limit.
    - Numeric criteria: Any column with dtype = `float64', or any integer column that has more unique values than the specified limit for categorical columns.  

- `fill_missing`: This function takes in a trianing feature dataframe, a testing feature dataframe, and a dictionary denoting column type ('numeric' and 'categorical', like the output of `categorize`) and fills missing values in both dataframes depending on the column type. Missing values in 'numeric' columns can be filled by either the mean or median column value of the training dataframe (with default of mean), and values for 'categorical' columns are filled by the column mode of the training dataframe. Currently, this function supports only 'numeric' and 'categorical' columns. Two dataframes, the training and testing, with missing values filled are returned.

-  `transform_columns`: This function takes in a training dataframe, a testing feature dataframe, and a dictionary denoting column type ('numeric' and 'categorical', like the output of `categorize`) and applies appropriate pre-processing techniques to each column. Categorical columns may be transformed with One Hot Encoding or Ordinal Encoding based on the values in the training dataframe and numerical columns will be transformed with a Standard Scaler or MinMax Scaler based on the trianing dataframe. Two dataframes, the trianing and testing, with transformed columns are returned.

- `feature_selector`: This function takes in a feature dataframe, an array of targets, a mode (Regression or Classification), and a maximum number of features to select. The function returns the most important features to predict the target as a list.

### pylaundry in the Python ecosystem
- [sklearn.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) offers similar functionality for the fill_missing and transform_columns functions, where similar functions can be wrapped in a Pipeline and carried out sequentially.

- There are many feature selection packages and functions, for instance [sklearn.feature_selection](https://scikit-learn.org/stable/modules/feature_selection.html), which carry out similar functionality to our `feature_selector` function

- As far as we know, there are no similar packages for Categorizing Columns. `pyLaundry` is the first package we are aware of to abstract away the full dataframe pre-processing workflow with a unified and simple API.

## Dependencies

- Python 3.7.3 and Python packages:
  - pandas==0.24.2  
  - numpy==1.16.4  
  - sklearn==0.22    
  
## Usage and examples

#### categorize()

```
from pylaundry import categorize
import pandas

df = pandas.DataFrame({'a':[1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                       'b':[1.2, 3.4, 3.0, 4.9, 5.3, 6.1, 8.8, 9.4, 10.4, 1.2],
                       'c':['A','B','C','D','E','F','G','H','I','J']})

# categorize with default max_cat (10)
pylaundry.categorize(df)
>>> {'numeric':['b'], 
   'categorical':['a','b']}

# categorize with max_cat = 5 (column c is neither numeric or categorical 
# under this restriction)
pyalundry.categorize(df, max_cat = 5)
>>> {'numeric':['b'],
     'categorical':['a']}

# Explicitly setting dtype to override max_cat settings for a column
df = df.astype({'b':'category'})
pylaundry.categorize(df, max_cat = 5)
>>> {'numeric':[],
     'categorical':['a', 'b']}
```

#### fill_missing()

```
from pylaundry import fill_missing
import pandas

df_train = pandas.DataFrame({'a':[1, 2, None, 4, 4],
                             'b':[1.2, 3.4, 3.0, 4.9, None]})

df_test = pandas.DataFrame({'a':[6, NaN, 0],
                           'b':[0.5, 9.2, NaN]})

pylaundry.fill_missing(df_train, df_test, {'numeric':['b'], 'categorical':['a']}, 
             num_imp = 'median')
>>>      a    b    
    0    1  1.2    
    1    2  3.4    
    2    4  3.0    
    3    4  4.9    
    4    4  3.2   


         a    b    c
    0    6  0.5    B
    1    4  9.2    B
    2    0  3.2    C
```

#### transform_columns()

```
from pylaundry import transform_columns
import pandas

df_train = pandas.DataFrame({'a':[1, 2, 3],
                       'b':[1.2, 3.4, 3.0],
                       'c':['A','B','C']})

df_test = pandas.DataFrame({'a':[6, 2],
                       'b':[0.5, 9.2],
                       'c':['B', 'B']})

pyalundry.transform_columns(df_train, df_test, {'numeric':['a', 'b'], 'categorical':['c']})
>>>      a      b     A  B  C
    0  -1.2  -1.39    1  0  0
    1   0.0   0.91    0  1  0
    2   1.2   0.49    0  0  1

         a    b    A  B  C
    0  4.9  -2.1   0  1  0
    1  1.2  6.96   0  1  0
```

#### select_features()

```
from pylaundry import select_features
import pandas

df = pandas.DataFrame({'a':[1, 2, 3],
                       'b':[1.2, 2.2, 3.2],
                       'c':[8, 12, -5]})

target = np.array([1, 2, 3])




pyalundry.select_features(df, target, mode = 'regression', n_features = 2)
>>>  ['a', 'b']
```

#### Proposed workflow
```
from pylaundry import *
import pandas
import numpy

X_train = pandas.DataFrame({'a':[1, 2, NaN, 4, 5, 1, 2, 3, 4, 5],
                       'b':[1.2, 3.4, 3.0, 4.9, 5.3, 6.1, 8.8, 9.4, NaN, 1.2],
                       'c':['A','B','C','D','E','F','B','H','I','NaN']})

X_test = pandas.DataFrame({'a':[6, NaN, 0],
                       'b':[0.5, 9.2, NaN],
                       'c':[NaN, 'B', 'D']})

y_train = numpy.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5])

# Categorize columns
col_dict = pylaundry.categorize(X_train)

# Fill missing values
filled_dict = pylaundry.fill_missing(X_train, X_test, col_dict)
X_train_filled, X_test_filled = filled_dict['X_train'], filled_dict['X_test']

# Transform data
transformed_dict = pylaundry.transform_columns(X_train_filled, X_test_filled, col_dict)
X_train_transformed, X_test_transformed = transformed_dict['X_train'], transformed_dict['X_test']

# Select features
cols = pylaundry.select_features(X_train_transformed, y_train, mode = 'regression')

X_train = X_train_transformed[cols]
X_test = X_test_transformed[cols]

```

### Documentation
The official documentation is hosted on Read the Docs: <https://py-laundry.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).


