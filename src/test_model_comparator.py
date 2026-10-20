
import pytest

from model_comparator import compare_models


def test_improved_candidate_metrics():
    baseline = {
        "roc_auc": 0.80,
        "brier_score": 0.20,
        "log_loss": 0.60,
        "ece": 0.08,
    }

    candidate = {
        "roc_auc": 0.85,
        "brier_score": 0.15,
        "log_loss": 0.45,
        "ece": 0.03,
    }

    changes = compare_models(baseline, candidate)

    assert changes["roc_auc"] > 0
    assert changes["brier_score"] < 0
    assert changes["log_loss"] < 0
    assert changes["ece"] < 0


def test_missing_metric_is_rejected():
    baseline = {
        "roc_auc": 0.80,
        "brier_score": 0.20,
        "log_loss": 0.60,
    }

    candidate = {
        "roc_auc": 0.85,
        "brier_score": 0.15,
        "log_loss": 0.45,
        "ece": 0.03,
    }

    with pytest.raises(ValueError, match="missing metrics"):
        compare_models(baseline, candidate)


if __name__ == "__main__":
    print("Run using: python -m pytest src/test_model_comparator.py -v")
