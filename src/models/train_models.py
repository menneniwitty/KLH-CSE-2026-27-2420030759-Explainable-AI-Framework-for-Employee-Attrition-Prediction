import pandas as pd
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

RESULTS_DIR = Path("results/metrics")
MODELS_DIR = Path("results/models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("\n========== LOADING PROCESSED DATA ==========")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training data shape:", train_df.shape)
print("Testing data shape :", test_df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df.drop("Attrition", axis=1)
y_train = train_df["Attrition"]

X_test = test_df.drop("Attrition", axis=1)
y_test = test_df["Attrition"]

print("\nNumber of training features:", X_train.shape[1])
print("Number of testing features :", X_test.shape[1])


# ============================================================
# 4. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n========== MODEL TRAINING ==========")

for name, model in models.items():

    print(f"\nTraining: {name}")

    # Train model
    model.fit(X_train, y_train)

    # Store trained model
    trained_models[name] = model

    # Predictions
    predictions = model.predict(X_test)

    # Probability of Attrition = 1
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Store results
    results.append({

        "Model": name,

        "Accuracy": round(accuracy, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1_Score": round(f1, 4),

        "ROC_AUC": round(roc_auc, 4)

    })

    # Display results
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)


# ============================================================
# 7. SAVE MODEL COMPARISON
# ============================================================

comparison_file = RESULTS_DIR / "model_comparison.csv"

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 8. IDENTIFY BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]


# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

best_model_file = MODELS_DIR / "best_model.pkl"

joblib.dump(
    best_model,
    best_model_file
)


# ============================================================
# 10. SAVE XGBOOST MODEL
# ============================================================

xgb_model_file = MODELS_DIR / "xgboost_model.pkl"

joblib.dump(
    trained_models["XGBoost"],
    xgb_model_file
)


# ============================================================
# 11. DISPLAY FINAL RESULTS
# ============================================================

print("\n========== MODEL COMPARISON ==========\n")

print(
    results_df.to_string(index=False)
)

print("\n=======================================")

print(
    f"\nBest model based on ROC-AUC: "
    f"{best_model_name}"
)

print(
    f"Best ROC-AUC: "
    f"{results_df.iloc[0]['ROC_AUC']}"
)

print(
    f"\nSaved comparison to:"
    f"\n{comparison_file}"
)

print(
    f"\nSaved best model to:"
    f"\n{best_model_file}"
)

print(
    f"\nSaved XGBoost model to:"
    f"\n{xgb_model_file}"
)

print("\n========== MODEL TRAINING COMPLETED ==========")
