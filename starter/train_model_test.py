import pandas as pd
import numpy as np
import pytest
import train_model
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split


def clean(input_path, output_path):
    census_df = pd.read_csv(input_path)
    census_df.replace({'?': None}, inplace=True)
    census_df = census_df.dropna()
    census_df.columns = census_df.columns.str.strip()
    census_df.drop("fnlgt", axis="columns", inplace=True)
    census_df.drop("education-num", axis="columns", inplace=True)
    census_df.to_csv(output_path, index=False)
    return census_df


def test_data_split():
    data = 'data/raw/census.csv'
    assert len(data) != 0


def test_model_train_1():
    data = clean(
        "data/raw/census.csv",
        "data/processed/cleaned_census.csv")
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


def test_model_train_2():
    data = clean(
        "data/raw/census.csv",
        "data/processed/cleaned_census.csv")
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
