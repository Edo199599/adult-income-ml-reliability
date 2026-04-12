import pandas as pd
from src.config import RAW_DATA_PATH, TARGET_COL, PROCESSED_DATA_PATH


def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def standardize_column_names(df):
    df = df.rename(
        columns={
            "education-num": "education_num",
            "marital-status": "marital_status",
            "capital-gain": "capital_gain",
            "capital-loss": "capital_loss",
            "hours-per-week": "hours_per_week",
            "native-country": "native_country",
        }
    )
    return df

def clean_raw_dataframe(df):
    df = df.replace("?", pd.NA)
    df = df.dropna()
    return df

def encode_target(df):
    df[TARGET_COL] = df[TARGET_COL].str.replace(".", "", regex=False)
    df[TARGET_COL] = df[TARGET_COL].map({
        "<=50K": 0,
        ">50K": 1,
    }).astype(int)
    return df

def save_processed_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

def main():
    df = load_raw_data()
    df = standardize_column_names(df)
    df = clean_raw_dataframe(df)
    #print(df[TARGET_COL].unique())
    df = encode_target(df)
    save_processed_data(df)

    #print(df.shape)
    #print(df.head(2))


if __name__ == "__main__":
    main()