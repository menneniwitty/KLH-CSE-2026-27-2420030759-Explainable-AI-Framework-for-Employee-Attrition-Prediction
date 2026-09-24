import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "results/models/xgboost_model.pkl"
TEST_PATH = "data/processed/test_processed.csv"

RESULTS_DIR = "results/evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

print("\nLoading test data...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop("Attrition", axis=1)
y_test = test_df["Attrition"]

print("Testing data shape:", X_test.shape)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n========================================")
print("XGBOOST PERFORMANCE EVALUATION")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Attrition", "Attrition"],
    zero_division=0
)

print(report)


with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(report)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "performance_metrics.csv"
    ),
    index=False
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Attrition",
        "Attrition"
    ]
)

disp.plot()

plt.title("XGBoost Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("XGBoost ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# METRIC BAR CHART
# --------------------------------------------------

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
]

plt.figure()

plt.bar(
    metric_names,
    metric_values
)

plt.ylim(0, 1)

plt.ylabel("Score")

plt.title("XGBoost Performance Metrics")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "performance_metrics.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# COMPLETION MESSAGE
# --------------------------------------------------

print("\n========================================")
print("PERFORMANCE EVALUATION COMPLETED")
print("========================================")

print("\nGenerated files:")

print(
    "1.",
    os.path.join(
        RESULTS_DIR,
        "performance_metrics.csv"
    )
)

print(
    "2.",
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    )
)

print(
    "3.",
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )
)

print(
    "4.",
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    )
)

print(
    "5.",
    os.path.join(
        RESULTS_DIR,
        "performance_metrics.png"
    )
)