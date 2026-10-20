
def interpret_metrics(metrics):
    """
    Convert classification reliability metrics into
    cautious, plain-language explanations.
    """

    explanations = []

    roc_auc = metrics["roc_auc"]
    brier = metrics["brier_score"]
    logloss = metrics["log_loss"]
    ece = metrics["ece"]

    # Ranking interpretation
    if roc_auc >= 0.90:
        explanations.append(
            "Ranking: The model separates positive and negative "
            "examples well on this evaluation dataset."
        )
    elif roc_auc >= 0.80:
        explanations.append(
            "Ranking: The model shows good ability to rank "
            "positive examples above negative examples."
        )
    elif roc_auc >= 0.70:
        explanations.append(
            "Ranking: The model shows moderate ranking ability."
        )
    else:
        explanations.append(
            "Ranking: The model's ranking ability is limited "
            "on this evaluation dataset."
        )

    # Calibration interpretation
    if ece < 0.02:
        explanations.append(
            "Calibration: The average confidence gap across the "
            "chosen bins is small. Individual bins may still be "
            "miscalibrated."
        )
    elif ece < 0.05:
        explanations.append(
            "Calibration: The average confidence gap is modest, "
            "but inspect the reliability diagram for problematic bins."
        )
    else:
        explanations.append(
            "Calibration: The average confidence gap is noticeable. "
            "Predicted probabilities may not consistently match "
            "observed outcome frequencies."
        )

    # Probability quality
    explanations.append(
        f"Brier score: {brier:.4f}. Lower is better when comparing "
        "models evaluated on the same outcomes."
    )
    explanations.append(
        f"Log loss: {logloss:.4f}. Lower is better; confident "
        "incorrect predictions are penalized strongly."
    )

    # Limitations
    explanations.append(
        "Limitation: These results describe this evaluation split. "
        "They do not guarantee reliability on future or different data."
    )

    return explanations


if __name__ == "__main__":
    example_metrics = {
        "roc_auc": 0.9022,
        "brier_score": 0.1233,
        "log_loss": 0.3902,
        "ece": 0.0114,
    }

    print("VERITYML RELIABILITY INTERPRETER")
    print("=" * 40)

    for explanation in interpret_metrics(example_metrics):
        print(f"\n• {explanation}")
