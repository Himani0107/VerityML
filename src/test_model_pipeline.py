
import pandas as pd
import pytest

from model_pipeline import create_model_pipeline, prepare_target


def test_target_is_converted_to_binary():
    y = pd.Series(["yes", "no", "yes", "no"])

    result = prepare_target(y)

    assert result.tolist() == [1, 0, 1, 0]


def test_unexpected_target_labels_are_rejected():
    y = pd.Series(["yes", "maybe", "no"])

    with pytest.raises(ValueError, match="Unexpected target labels"):
        prepare_target(y)


def test_pipeline_trains_and_predicts_probabilities():
    X = pd.DataFrame({
        "age": [22, 35, 41, 28, 50, 31, 45, 24],
        "job": [
            "student", "admin", "technician", "student",
            "admin", "technician", "admin", "student",
        ],
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])

    model = create_model_pipeline(X)
    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    assert len(probabilities) == len(y)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


if __name__ == "__main__":
    print("Run tests using: python -m pytest src/test_model_pipeline.py -v")
