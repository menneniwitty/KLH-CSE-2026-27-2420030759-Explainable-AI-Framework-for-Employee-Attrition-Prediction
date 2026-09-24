import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")


def load_data():
    """Load the IBM HR Analytics Employee Attrition dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":

    df = load_data()

    print("\n========== DATASET INFORMATION ==========")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\n========== COLUMN NAMES ==========")
    for column in df.columns:
        print(column)

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())

    print("\n========== ATTRITION DISTRIBUTION ==========")
    print(df["Attrition"].value_counts())

    print("\n========== FIRST 5 RECORDS ==========")
    print(df.head())