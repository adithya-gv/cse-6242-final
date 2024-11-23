import pandas as pd

def clean_game_data():

    # Load your CSV file
    df = pd.read_csv("game_data.csv")

    # Drop columns with any missing values
    df_cleaned = df.dropna(axis=1, how='any')

    # Save the cleaned DataFrame back to a CSV file, if needed
    df_cleaned.to_csv("cleaned_game_data.csv", index=False)

clean_game_data()