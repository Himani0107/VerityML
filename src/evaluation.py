
import numpy as np

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calculate_ece(y_true, probabilities, n_bins=10):
    """Calculate ECE using equal-width probability bins."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite numbers.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    ece = 0.0

    for bin_number in range(n_bins):
        mask = bin_indices == bin_number

        if not mask.any():
            continue

        mean_confidence = probabilities[mask].mean()
        actual_rate = y_true[mask].mean()
        weight = mask.sum() / len(y_true)

        ece += weight * abs(mean_confidence - actual_rate)

    return float(ece)


def evaluate_probabilities(y_true, probabilities, n_bins=10):
    """Return reusable classification probability metrics."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite numbers.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC-AUC requires both classes in y_true.")

    metrics = {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "ece": calculate_ece(
            y_true,
            probabilities,
            n_bins=n_bins,
        ),
    }

    return metrics
