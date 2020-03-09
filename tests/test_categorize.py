from pylaundry.categorize import categorize
import pytest
import datetime
import numpy as np
import pandas as pd


@pytest.fixture
def generate_data():
    base_date = datetime.date(year=2020, month=1, day=1)
    input = pd.DataFrame({
        'cat1': [1, 2, 3, 1, 4, 3, 3, 2, 1, 1,
                 2, 3, 2, 3, 1, 3, 2, 4, 1, 1,
                 3, 2, 4, 2, 2, 2, 3, 1, 2, 3],
        'cat2': ['A', 'B', 'C', 'A', 'B', 'C',
                 'A', 'B', 'C', 'A', 'B', 'C',
                 'A', 'B', 'C', 'A', 'B', 'C',
                 'A', 'B', 'C', 'A', 'B', 'C',
                 'A', 'B', 'C', 'A', 'B', 'C'],
        'num1': np.random.uniform(0.0, 5.0, 30),
        'num2': np.random.randint(-10, 10, 30),
        'date1': [base_date + datetime.timedelta(days=x) for x in range(30)],
        'text1': [str(x+0.5)+'this is some text' for x in range(30)],
        'num3': np.random.uniform(-100, 100, 30),
        'cat3': np.repeat(('Monday', 'Tuesday', 'Wednesday',
                          'Thursday', 'Friday'), 6),
        'text2': ['Text instance #'+str(x) for x in range(30)]
    })
    return input


@pytest.fixture
def generate_data_no_cat():
    base_date = datetime.date(year=2020, month=1, day=1)
    input = pd.DataFrame({
        'date1': [base_date + datetime.timedelta(days=x) for x in range(30)],
        'text1': [str(x+0.5)+'this is some text' for x in range(30)],
        'text2': ['Text instance #'+str(x) for x in range(30)]
    })
    return input


def test_output_type(generate_data):
    output = categorize(generate_data)
    assert isinstance(output, dict), \
        "Output of categorize() should be a dictionary"
    assert isinstance(output['numeric'], list), \
        "Dictionary value should be list"
    assert isinstance(output['categorical'], list), \
        "Dictionary value should be list"


def test_categorize_categorical(generate_data):
    output = categorize(generate_data)
    assert set(output['categorical']) == set(['cat1', 'cat2', 'cat3']), "cat1,\
         cat2 and cat3 of generate_data() should be categorized as categorical"


def test_categorize_cat_dtype(generate_data):
    df = pd.DataFrame({'col': [1.1, 2.2, 3.3, 4.4]})
    df['col'] = df['col'].astype('category')
    output = categorize(df, max_cat=2)
    assert output['categorical'] == ['col'], "A column with \
        dtype = category should override the max_cat specification"


def test_categorize_float_dtype(generate_data):
    df = pd.DataFrame({'col': [1.1, 1.1, 2.2, 2.2]})
    output = categorize(df)
    assert output['numeric'] == ['col'], "A column with dtype \
        = float64 should override the max_cat specification"


def test_categorize_numeric(generate_data):
    output = categorize(generate_data)
    assert set(output['numeric']) == set(['num1', 'num2', 'num3']), "num1, \
        num2 and num3 of generate_data() should be categorized as numeric"


def test_categorize_max_cat(generate_data):
    output = categorize(generate_data, max_cat=3)
    assert output['categorical'] == ['cat2'], "Only cat2 of generate_data()\
        should be categorical with max_cat = 3"
    assert set(output['numeric']) == set(['num1', 'num2', 'num3', 'cat1']),\
        "Cat 1 should be marked numeric with max_cat = 3"


def test_categorize_bad_input(generate_data):
    try:
        categorize('hello!')
    except AssertionError:
        pass
    try:
        categorize(np.ones((3, 3)))
    except AssertionError:
        pass
    try:
        categorize(generate_data, max_cat=1.3)
    except AssertionError:
        pass
    try:
        categorize(generate_data, max_cat=-1)
    except AssertionError:
        pass


def test_categorize_no_cat(generate_data_no_cat):
    assert categorize(pd.DataFrame()) == {'numeric': [],
                                          'categorical': []}, "Empty\
                                           dataframe should return dictionary \
                                           with keys but empty list values"
    assert categorize(generate_data_no_cat) == {'numeric': [],
                                                'categorical': []}, "Dataframe\
                                                s with no categorizable colum\
                                                ns should return dict with ke\
                                                ys and empty list values"
