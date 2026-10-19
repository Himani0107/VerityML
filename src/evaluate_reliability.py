

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# 1. Load the dataset
df = pd.read_csv("data/bank.csv")

X = df.drop(columns=["deposit"])
y = df["deposit"].map({"no": 0, "yes": 1})

# 2. Split into training, validation, and test sets
# 60% training, 20% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# 3. Identify numerical and categorical features
numerical_columns = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Build preprocessing pipeline
def create_model_pipeline():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


# 5. Evaluate probability metrics
def evaluate_probabilities(name, y_true, probabilities):
    print(f"\n--- {name} ---")
    print(
        "Brier score:",
        round(brier_score_loss(y_true, probabilities), 4),
    )
    print(
        "Log loss:",
        round(log_loss(y_true, probabilities), 4),
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_true, probabilities), 4),
    )


# 6. Calculate Expected Calibration Error (ECE)
def calibration_report(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_ids = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    total = len(y_true)
    ece = 0.0
    rows = []

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        if count == 0:
            continue

        mean_confidence = probabilities[mask].mean()
        actual_fraction = y_true[mask].mean()
        gap = abs(actual_fraction - mean_confidence)

        ece += (count / total) * gap

        rows.append({
            "bin": (
                f"{bin_edges[bin_id]:.1f}-"
                f"{bin_edges[bin_id + 1]:.1f}"
            ),
            "count": count,
            "mean_confidence": round(float(mean_confidence), 4),
            "actual_positive_rate": round(float(actual_fraction), 4),
            "absolute_gap": round(float(gap), 4),
        })

    report = pd.DataFrame(rows)

    print(f"\nECE ({n_bins} equal-width bins): {ece:.4f}")
    print(report.to_string(index=False))

    return ece, report


# 7. Train baseline model
baseline_model = create_model_pipeline()
baseline_model.fit(X_train, y_train)

# 8. Train sigmoid-calibrated model
# The full preprocessing + classifier pipeline is cross-validated.
calibrated_model = CalibratedClassifierCV(
    estimator=create_model_pipeline(),
    method="sigmoid",
    cv=5,
)

calibrated_model.fit(X_train, y_train)
# 9. Train an isotonic-calibrated model
isotonic_model = CalibratedClassifierCV(
    estimator=create_model_pipeline(),
    method="isotonic",
    cv=5,
)

isotonic_model.fit(X_train, y_train)


# 9. Compare probabilities on validation data
baseline_probs = baseline_model.predict_proba(X_val)[:, 1]
calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]
isotonic_probs = isotonic_model.predict_proba(X_val)[:, 1]

evaluate_probabilities(
    "Baseline Logistic Regression",
    y_val,
    baseline_probs,
)

evaluate_probabilities(
    "Sigmoid-Calibrated Logistic Regression",
    y_val,
    calibrated_probs,
)

evaluate_probabilities(
    "Isotonic-Calibrated Logistic Regression",
    y_val,
    isotonic_probs,
)

print("\n=== BASELINE CALIBRATION REPORT ===")
baseline_ece, baseline_report = calibration_report(
    y_val, baseline_probs
)

print("\n=== SIGMOID CALIBRATION REPORT ===")
calibrated_ece, calibrated_report = calibration_report(
    y_val, calibrated_probs
)

print("\n=== ISOTONIC CALIBRATION REPORT ===")
isotonic_ece, isotonic_report = calibration_report(
    y_val, isotonic_probs
)


# 10. Plot reliability diagrams
plt.figure(figsize=(8, 6))

for name, probabilities in [
    ("Baseline", baseline_probs),
    ("Sigmoid calibrated", calibrated_probs),
    ("Isotonic calibrated", isotonic_probs),
]:
    actual_fraction, mean_probability = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    plt.plot(
        mean_probability,
        actual_fraction,
        marker="o",
        label=name,
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Actual positive fraction")
plt.title("VerityML: Probability Reliability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = Path("reports/reliability_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(
    "reports/reliability_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
print("Reliability plot saved successfully.")

plt.close()

# Final evaluation on the untouched test set
# Model selection was based on validation results.

baseline_test_probs = baseline_model.predict_proba(X_test)[:, 1]
isotonic_test_probs = isotonic_model.predict_proba(X_test)[:, 1]

evaluate_probabilities(
    "FINAL TEST: Baseline Logistic Regression",
    y_test,
    baseline_test_probs,
)

evaluate_probabilities(
    "FINAL TEST: Isotonic-Calibrated Logistic Regression",
    y_test,
    isotonic_test_probs,
)

print("\n=== FINAL TEST CALIBRATION REPORT ===")

test_ece, test_report = calibration_report(
    y_test,
    isotonic_test_probs,
)

print("\nFinal test evaluation completed.")