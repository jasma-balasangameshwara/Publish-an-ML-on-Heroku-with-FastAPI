import pandas as pd


def clean(input_path, output_path):
    census_df = pd.read_csv(input_path, skipinitialspace=True)
    census_df = census_df.dropna()
    census_df.replace({'?': None})
    census_df.to_csv(output_path, index=False)
    return census_df


if __name__ == '__main__':
    dataframe = clean("data/raw/census.csv", "data/processed/cleaned_census.csv")
