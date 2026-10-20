
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import get_feature_types


def create_model_pipeline(X):
    """Create a preprocessing and classification pipeline."""

    numerical_columns, categorical_columns = get_feature_types(X)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    return model


def prepare_target(y):
    """Convert the bank dataset's yes/no labels into 1/0."""

    target_mapping = {"no": 0, "yes": 1}

    unexpected_labels = set(y.dropna().unique()) - set(target_mapping)

    if unexpected_labels:
        raise ValueError(
            f"Unexpected target labels: {unexpected_labels}. "
            "Expected only 'no' and 'yes'."
        )

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    return y.map(target_mapping).astype(int)
