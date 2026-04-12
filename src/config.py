from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "adult_clean.csv"

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET_COL = "income"

RANDOM_STATE = 42
TEST_SIZE = 0.2
