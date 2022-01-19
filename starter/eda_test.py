import pandas as pd
import pytest
import starter.eda


@pytest.fixture
def data_read():
    dataframe = pd.read_csv('data/raw/census.csv', skipinitialspace=True)
    processed_dataframe = starter.eda.clean(dataframe)
    return processed_dataframe


def test_nulls(data_read):
    assert data_read.shape == data_read.dropna().shape


def test_question_marks(data_read):
    assert '?' not in data_read.values
