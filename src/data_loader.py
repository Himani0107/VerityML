
from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bank.csv"


def load_dataset(file_path=DATA_PATH, target_column="deposit"):
    """Load a CSV dataset and separate its features from its target."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist in the dataset."
        )

    if df[target_column].isnull().any():
        raise ValueError("The target column contains missing values.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def get_feature_types(X):
    """Identify numerical and categorical feature columns."""

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


if __name__ == "__main__":
    X, y = load_dataset()
    numerical, categorical = get_feature_types(X)

    print("Dataset loaded successfully!")
    print("Feature rows and columns:", X.shape)
    print("Target column: deposit")
    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nNumerical features:", numerical)
    print("\nCategorical features:", categorical)

