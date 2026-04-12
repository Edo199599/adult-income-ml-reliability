import pandas as pd
from src.config import PROCESSED_DATA_PATH, TARGET_COL, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    return X_train, X_test, y_train, y_test



def build_training_pipeline():

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
            ]
        )
    
    pipe = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )

    return pipe



def main():
    X_train, X_test, y_train, _ = load_data()
    print(X_train.shape)
    print(X_test.shape)
    pipe = build_training_pipeline()
    pipe.fit(X_train, y_train)
    return pipe



if __name__ == "__main__":
    main()