import pandas as pd
import pytest
import starter.eda


@pytest.fixture
def data_read():
    dataframe = starter.eda.clean("data/raw/census.csv", "data/processed/cleaned_census.csv")
    return dataframe


def test_nulls(data_read):
    assert data_read.shape == data_read.dropna().shape


def test_question_marks(data_read):
    assert '?' not in data_read.values
