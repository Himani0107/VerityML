
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from model_pipeline import create_model_pipeline, prepare_target
from evaluation import evaluate_probabilities


def main():
    X, y = load_dataset()
    y = prepare_target(y)

    # Keep the same split as run_experiment.py
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=y_temp
    )

    models = {
        "Baseline": create_model_pipeline(X_train),
        "Sigmoid": CalibratedClassifierCV(
            estimator=create_model_pipeline(X_train),
            method="sigmoid",
            cv=5,
        ),
        "Isotonic": CalibratedClassifierCV(
            estimator=create_model_pipeline(X_train),
            method="isotonic",
            cv=5,
        ),
    }

    # Fit using training data; compare using validation data
    validation_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_validation)[:, 1]
        metrics = evaluate_probabilities(y_validation, probabilities)
        validation_results[name] = metrics

    print("VALIDATION COMPARISON")
    print("-" * 65)
    print(f"{'Model':<12} {'ROC-AUC':>10} {'Brier':>10} "
          f"{'Log loss':>10} {'ECE':>10}")

    for name, metrics in validation_results.items():
        print(
            f"{name:<12} "
            f"{metrics['roc_auc']:>10.4f} "
            f"{metrics['brier_score']:>10.4f} "
            f"{metrics['log_loss']:>10.4f} "
            f"{metrics['ece']:>10.4f}"
        )

    # Choose using validation Brier score, not test performance
    best_name = min(
        validation_results,
        key=lambda name: validation_results[name]["brier_score"],
    )

    print(f"\nSelected using validation Brier score: {best_name}")

    # Evaluate the selected model on the test set
    best_model = models[best_name]
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_probabilities)

    print("\nFINAL TEST RESULTS FOR SELECTED MODEL")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
