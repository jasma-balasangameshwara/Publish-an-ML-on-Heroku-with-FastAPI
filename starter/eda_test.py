import pandas as pd
import pytest
import starter.eda


@pytest.fixture
def data_read():
    dataframe = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    processed_dataframe = starter.eda.clean(dataframe)
    return processed_dataframe


def test_nulls():
    dataframe = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    processed_dataframe = starter.eda.clean(dataframe)
    assert processed_dataframe.shape == processed_dataframe.dropna().shape


def test_question_marks(data_read):
    assert '?' not in data_read.values
