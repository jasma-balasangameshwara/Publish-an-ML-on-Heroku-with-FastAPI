import pandas as pd
import pytest
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split


def test_data_split():
    data = 'data/raw/census.csv'
    assert len(data) != 0


@pytest.fixture(scope='session')
def test_model_train_1():
    data = pd.read_csv("data/processed/cleaned_census.csv")
    train, test = train_test_split(data, test_size=0.20)
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    train = train.drop(["salary"], axis=1)
    x_categorical = train[categorical_features].values
    x_continuous = train.drop(*[categorical_features], axis=1)
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    x_categorical = encoder.fit_transform(x_categorical)
    assert len(x_continuous) != 0
    assert len(x_categorical) != 0


@pytest.fixture(scope='session')
def test_model_train_2():
    data = pd.read_csv("data/processed/cleaned_census.csv")
    train, test = train_test_split(data, test_size=0.20)
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country", ]
    y_train = train["salary"]
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train.values).ravel()
    assert len(y_train) != 0
