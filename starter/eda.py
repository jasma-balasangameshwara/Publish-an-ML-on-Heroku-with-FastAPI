#Data Cleaning
import pandas as pd


def clean(dataframe):
    dataframe.replace({'?': None}, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe.drop("fnlgt", axis="columns", inplace=True)
    dataframe.drop("education-num", axis="columns", inplace=True)
    return dataframe


if __name__ == '__main__':
    raw_dataframe = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    processed_dataframe = clean(raw_dataframe)
    processed_dataframe.to_csv("data/processed/cleaned_census.csv", index=False)

