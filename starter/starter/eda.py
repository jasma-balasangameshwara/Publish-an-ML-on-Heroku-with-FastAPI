import pandas as pd

path = "../data/raw/census.csv"
census_df = pd.read_csv(path, skipinitialspace=True)
census_df = census_df.dropna(inplace=True)

