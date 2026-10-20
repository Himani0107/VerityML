
import numpy as np

from evaluation import calculate_ece, evaluate_probabilities


def test_ece_for_perfect_calibration():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    assert calculate_ece(y_true, probabilities) == 0.0


def test_evaluation_returns_all_metrics():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])

    metrics = evaluate_probabilities(y_true, probabilities)

    assert set(metrics.keys()) == {
        "roc_auc",
        "brier_score",
        "log_loss",
        "ece",
    }
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["ece"] >= 0.0


def test_invalid_probabilities_are_rejected():
    y_true = np.array([0, 1])
    probabilities = np.array([0.2, 1.5])

    try:
        evaluate_probabilities(y_true, probabilities)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid probabilities to fail.")


if __name__ == "__main__":
    test_ece_for_perfect_calibration()
    test_evaluation_returns_all_metrics()
    test_invalid_probabilities_are_rejected()
    print("All evaluation tests passed!")
