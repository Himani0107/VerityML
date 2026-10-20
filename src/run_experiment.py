
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from model_pipeline import create_model_pipeline, prepare_target
from evaluation import evaluate_probabilities


def main():
    # 1. Load the dataset
    X, y_raw = load_dataset()
    y = prepare_target(y_raw)

    # 2. Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print("Dataset split:")
    print(f"Training:   {len(X_train)} rows")
    print(f"Validation: {len(X_validation)} rows")
    print(f"Test:       {len(X_test)} rows")

    # 3. Train the model using only training data
    model = create_model_pipeline(X_train)
    model.fit(X_train, y_train)

    # 4. Use validation data for an initial evaluation
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    validation_metrics = evaluate_probabilities(
        y_validation, validation_probabilities
    )

    print("\nValidation metrics:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 5. Final evaluation on the held-out test set
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_probabilities)

    print("\nFinal test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
