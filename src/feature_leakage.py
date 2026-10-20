
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calculate_ece(y_true, probabilities, n_bins=10):
    """Calculate Expected Calibration Error using equal-width bins."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(
        probabilities, bin_edges[1:-1], right=False
    )

    ece = 0.0
    total = len(y_true)

    for bin_number in range(n_bins):
        mask = bin_indices == bin_number

        if not np.any(mask):
            continue

        mean_confidence = probabilities[mask].mean()
        actual_rate = y_true[mask].mean()
        bin_weight = mask.sum() / total

        ece += bin_weight * abs(mean_confidence - actual_rate)

    return ece



# 1. Load the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "bank.csv")

X = df.drop(columns=["deposit"])
y = df["deposit"].map({"no": 0, "yes": 1})

if y.isna().any():
    raise ValueError("Unexpected target values found in deposit column.")


# 2. Recreate the same 60/20/20 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp,
)

print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")

# We deliberately do not evaluate or use X_test in this experiment.


# 3. Build a pipeline based on the available features
def create_model_pipeline(features):
    numeric_columns = features.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = features.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


# 4. Compare validation probabilities
def evaluate_model(name, model, X_validation, y_validation):
    probabilities = model.predict_proba(X_validation)[:, 1]

    brier = brier_score_loss(y_validation, probabilities)
    loss = log_loss(y_validation, probabilities)
    auc = roc_auc_score(y_validation, probabilities)

    print(f"\n--- {name} ---")
    print(f"Brier score: {brier:.4f}")
    print(f"Log loss: {loss:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    return probabilities


# Model A: includes duration
model_with_duration = create_model_pipeline(X_train)
model_with_duration.fit(X_train, y_train)

probabilities_with_duration = evaluate_model(
    "Model A: WITH duration",
    model_with_duration,
    X_val,
    y_val,
)


# Model B: excludes duration
X_train_without_duration = X_train.drop(columns=["duration"])
X_val_without_duration = X_val.drop(columns=["duration"])

model_without_duration = create_model_pipeline(
    X_train_without_duration
)
model_without_duration.fit(
    X_train_without_duration,
    y_train,
)

probabilities_without_duration = evaluate_model(
    "Model B: WITHOUT duration",
    model_without_duration,
    X_val_without_duration,
    y_val,
)


# 5. Summarize the comparison
print("\n=== VALIDATION COMPARISON ===")
print(
    "Brier score difference (without - with):",
    round(
        brier_score_loss(y_val, probabilities_without_duration)
        - brier_score_loss(y_val, probabilities_with_duration),
        4,
    ),
)
print(
    "Log loss difference (without - with):",
    round(
        log_loss(y_val, probabilities_without_duration)
        - log_loss(y_val, probabilities_with_duration),
        4,
    ),
)
print(
    "ROC-AUC difference (without - with):",
    round(
        roc_auc_score(y_val, probabilities_without_duration)
        - roc_auc_score(y_val, probabilities_with_duration),
        4,
    ),
)


# 6. Calibration experiment without duration
# Use training data for fitting and cross-validation.
# Use validation data only for evaluation.

print("\n\n=== CALIBRATION WITHOUT DURATION ===")

# Create an isotonic-calibrated pipeline.
# Preprocessing is fitted independently inside each CV fold.
calibrated_model_without_duration = CalibratedClassifierCV(
    estimator=create_model_pipeline(X_train_without_duration),
    method="isotonic",
    cv=5,
)

calibrated_model_without_duration.fit(
    X_train_without_duration,
    y_train,
)

# Predict probabilities on the validation set.
calibrated_probabilities = (
    calibrated_model_without_duration.predict_proba(
        X_val_without_duration
    )[:, 1]
)

# Compare the uncalibrated and calibrated models.
uncalibrated_brier = brier_score_loss(
    y_val,
    probabilities_without_duration,
)
calibrated_brier = brier_score_loss(
    y_val,
    calibrated_probabilities,
)

uncalibrated_log_loss = log_loss(
    y_val,
    probabilities_without_duration,
)
calibrated_log_loss = log_loss(
    y_val,
    calibrated_probabilities,
)

uncalibrated_auc = roc_auc_score(
    y_val,
    probabilities_without_duration,
)
calibrated_auc = roc_auc_score(
    y_val,
    calibrated_probabilities,
)

print("\n--- WITHOUT duration: Uncalibrated ---")
print(f"Brier score: {uncalibrated_brier:.4f}")
print(f"Log loss: {uncalibrated_log_loss:.4f}")
print(f"ROC-AUC: {uncalibrated_auc:.4f}")

print("\n--- WITHOUT duration: Isotonic calibrated ---")
print(f"Brier score: {calibrated_brier:.4f}")
print(f"Log loss: {calibrated_log_loss:.4f}")
print(f"ROC-AUC: {calibrated_auc:.4f}")

print("\n=== CHANGE AFTER CALIBRATION ===")
print(
    "Brier score change:",
    round(calibrated_brier - uncalibrated_brier, 4),
)
print(
    "Log loss change:",
    round(calibrated_log_loss - uncalibrated_log_loss, 4),
)
print(
    "ROC-AUC change:",
    round(calibrated_auc - uncalibrated_auc, 4),
)


# 7. Compare calibration error without duration

uncalibrated_ece = calculate_ece(
    y_val,
    probabilities_without_duration,
)

calibrated_ece = calculate_ece(
    y_val,
    calibrated_probabilities,
)

print("\n=== ECE COMPARISON WITHOUT DURATION ===")
print(f"Uncalibrated ECE: {uncalibrated_ece:.4f}")
print(f"Isotonic ECE:     {calibrated_ece:.4f}")
print(
    "ECE change (calibrated - uncalibrated):",
    round(calibrated_ece - uncalibrated_ece, 4),
)



