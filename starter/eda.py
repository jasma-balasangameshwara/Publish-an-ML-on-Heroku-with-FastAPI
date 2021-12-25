import pandas as pd


def clean():
    census_df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    census_df = census_df.dropna()
    census_df.replace({'?': None})
    census_df.to_csv("data/processed/cleaned_census.csv", index=False)
    return census_df
