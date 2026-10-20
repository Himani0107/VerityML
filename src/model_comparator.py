

def compare_models(baseline, candidate, tolerance=0.001):
    """Compare two models evaluated on the same dataset."""

    metric_directions = {
        "roc_auc": "higher",
        "brier_score": "lower",
        "log_loss": "lower",
        "ece": "lower",
    }

    required = set(metric_directions)

    for name, metrics in [
        ("baseline", baseline),
        ("candidate", candidate),
    ]:
        missing = required - set(metrics)
        if missing:
            raise ValueError(
                f"{name} is missing metrics: {sorted(missing)}"
            )

    changes = {}

    print("VERITYML MODEL COMPARISON")
    print("=" * 45)

    for metric, direction in metric_directions.items():
        old = baseline[metric]
        new = candidate[metric]
        change = new - old
        changes[metric] = change

        if abs(change) <= tolerance:
            status = "approximately unchanged"
        elif (direction == "higher" and change > 0) or (
            direction == "lower" and change < 0
        ):
            status = "improved"
        else:
            status = "worsened"

        print(
            f"{metric}: {old:.4f} -> {new:.4f} "
            f"({change:+.4f}) — {status}"
        )

    print(
        "\nReminder: Small differences may reflect sampling "
        "variation. Evaluate both models on the same examples."
    )

    return changes


if __name__ == "__main__":
    baseline_metrics = {
        "roc_auc": 0.9026,
        "brier_score": 0.1269,
        "log_loss": 0.4101,
        "ece": 0.0542,
    }

    isotonic_metrics = {
        "roc_auc": 0.9022,
        "brier_score": 0.1233,
        "log_loss": 0.3902,
        "ece": 0.0114,
    }

    compare_models(baseline_metrics, isotonic_metrics)
