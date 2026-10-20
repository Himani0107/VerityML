
import pandas as pd
import pytest

from data_loader import load_dataset, get_feature_types


def test_load_dataset_separates_features_and_target(tmp_path):
    # Create a tiny sample dataset for testing.
    df = pd.DataFrame({
        "age": [22, 35, 41],
        "job": ["student", "admin", "technician"],
        "deposit": ["yes", "no", "yes"],
    })

    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    X, y = load_dataset(file_path)

    assert X.shape == (3, 2)
    assert list(X.columns) == ["age", "job"]
    assert list(y) == ["yes", "no", "yes"]


def test_missing_target_column_is_rejected(tmp_path):
    df = pd.DataFrame({
        "age": [22, 35],
        "job": ["student", "admin"],
    })

    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="does not exist"):
        load_dataset(file_path)


def test_empty_dataset_is_rejected(tmp_path):
    df = pd.DataFrame(columns=["age", "deposit"])

    file_path = tmp_path / "empty.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_dataset(file_path)


def test_feature_types_are_identified():
    X = pd.DataFrame({
        "age": [20, 25, 30],
        "balance": [100, 200, 300],
        "job": ["student", "admin", "teacher"],
    })

    numerical, categorical = get_feature_types(X)

    assert numerical == ["age", "balance"]
    assert categorical == ["job"]

