import pytest
import pandas as pd
from pylaundry.fill_missing import fill_missing

# Sample input
x = [1, 1]
df = pd.DataFrame([[1, 2, None, 3], [3, 2, 4, 5], [4, 1,  5, 6], [2, None, 4, 3]])
threshold = 0.9

def test_mean_imputation():
    assert (fill_missing(df, dict(), "mean", "mode") == 5)