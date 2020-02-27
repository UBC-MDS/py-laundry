## Summary
The `pyLaundry` package performs many standard preprocessing techniques for Pandas dataframes,  before use in statistical analysis and machine learning. The package functionality includes categorizing column types, handling missing data and imputation, transforming/standardizing columns and feature selection. The `pyLaundry` package aims to remove much of the grunt work in the typical data science workflow, allowing the analyst maximum time and energy to devote to modelling!

![](https://github.com/UBC-MDS/py-laundry/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/py-laundry/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/py-laundry) ![Release](https://github.com/UBC-MDS/py-laundry/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/py-laundry/badge/?version=latest)](https://py-laundry.readthedocs.io/en/latest/?badge=latest)

### Installation:
```
pip install -i https://test.pypi.org/simple/ py-laundry
```

### Features
- `categorize`: This function will take in a Pandas dataframe, and output a dictionary with column types as keys (numerical, categorical, text), and a list of column names associated with each column type as values. 

- `fill_missing`: This function takes in a dataframe and depending on user input, will either remove all rows with missing values, or will fill missing values using `mean`, `median`, or `regression` imputation. 

-  `transform_columns`: This function will take in a dataframe and apply pre-processing techniques to each column. Categorical columns will be transformed with a One Hot Encoding, numerical columns will be transformed with a Standard Scaler, and text columns will be transformed with a Count Vectorizer. 

- `feature_selector`: This function takes in a dataframe which has X and y columns specified, a target task (Regression or Classification), and a maximum number of features to select. The function returns the most important features for the target task. 

### pyLaundry in the Python ecosystem
- [sklearn.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) offers similar functionality for the fill_missing and transform_columns functions, where similar functions can be wrapped in a Pipeline and carried out sequentially.

- There are many feature selection packages and functions, for instance [sklearn.feature_selection](https://scikit-learn.org/stable/modules/feature_selection.html), which carry out similar functionality to our `feature_selector` function

- As far as we know, there are no similar packages for Categorizing Columns. `pyLaundry` is the first package we are aware of to abstract away the full dataframe pre-processing workflow with a unified and simple API.

### Dependencies

- Python 3.7.3 and Python packages:
  - pandas==0.24.2  
  - numpy==1.16.4  
  - sklearn==0.22    
  
### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://py-laundry.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
