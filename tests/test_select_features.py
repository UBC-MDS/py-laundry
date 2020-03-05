
#from pylaundry.select_features import select_features
import pytest
import pandas as pd
import numpy as np
import random
import unittest
import string
from pylaundry import select_features



@pytest.fixture
def generate_data_regression():
   x1 = np.random.randint(low = 0, high = 200, size = 1000)
   x2 = np.random.randint(low = 100, high = 600, size = 1000)
   x3 = np.random.randint(low = -100, high = 300, size = 1000)
   y = 2*x1
   df = pd.DataFrame({"x1":x1, "x2":x2, "x3":x3, "y": y})
   
   return df

@pytest.fixture
def generate_data_regression_one():
   x1 = np.random.randint(low = 0, high = 200, size = 1000)
   x2 = np.random.randint(low = 100, high = 600, size = 1000)
   x3 = np.random.randint(low = -100, high = 300, size = 1000)
   y = 2*x1 + 3*x2
   df = pd.DataFrame({"x1":x1, "x2":x2, "x3":x3, "y": y})
   
   return df



@pytest.fixture
def generate_data_classification():
   x1 = np.random.randint(low = 0, high = 200, size = 1000)
   x2 = np.random.randint(low = 100, high = 600, size = 1000)
   x3 = np.random.randint(low = -100, high = 300, size = 1000)
   df = pd.DataFrame({"x1":x1, "x2":x2, "x3":x3})
   df['y'] = df['x1'].apply(lambda x: 1 if x > 150 else 0)
   
   return df

@pytest.fixture
def generate_wrong_data():
    x1 = np.random.randint(low = 0, high = 200, size = 1000)
    x2 = np.random.randint(low = 100, high = 600, size = 1000)
    x3 = [random.choice(string.ascii_lowercase) for i in range(1000)]
    y = 2*x1
    df = pd.DataFrame({"x1":x1, "x2":x2, "x3":x3, "y": y})

    return df

@pytest.fixture
def generate_wrong_data_one():
    x1 = np.random.randint(low = 0, high = 200, size = 1000)
    y = 2*x1
    

    return x1, y
    
    
def test_regression(generate_data_regression):
    df = generate_data_regression
    y = df['y'].values
    df = df[['x1','x2','x3']]
    assert select_features.select_features(df,y,n_features=1) == ["x1"]

def test_regression_one(generate_data_regression_one):
    df = generate_data_regression_one
    y = df['y'].values
    df = df[['x1','x2','x3']]
    assert select_features.select_features(df,y,n_features=2) == ["x1", "x2"]


def test_classification(generate_data_classification):
    df = generate_data_classification
    y = df['y'].values
    df = df[['x1','x2','x3']]
    assert select_features.select_features(df,y,mode = "classification",n_features=1) == ["x1"]

def test_dataframe(generate_wrong_data):
    df = generate_wrong_data
    y = df['y'].values
    df = df[['x1','x2','x3']]

    class exceptiontest(unittest.TestCase):
        def testmyexception(self):
            with self.assertRaises(Exception) as context:
                select_features.select_features(df,y,n_features=1)
            
            self.assertTrue('String Column Present in the data. Apply transform_column function to fix' in str(context.exception))
            #print ("success!")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)


def test_input(generate_wrong_data_one):
    df, y = generate_wrong_data_one
    

    class exceptiontest(unittest.TestCase):
        def testmyexception(self):
            with self.assertRaises(Exception) as context:
                select_features.select_features(df,y,n_features=1)
            
            self.assertTrue('Input Data is not dataframe' in str(context.exception))
            #print ("success!")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)






