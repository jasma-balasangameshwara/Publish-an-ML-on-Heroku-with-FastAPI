import pandas as pd
import numpy as np
import pytest
import train_model
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def test_data_split():
    data = 'data/raw/census.csv'
    assert len(data) != 0


def test_model_train_1():
    _, train, _, _ = train_model.data_split()
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


def test_model_train_2():
    _, train, test, _ = train_model.data_split()
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
