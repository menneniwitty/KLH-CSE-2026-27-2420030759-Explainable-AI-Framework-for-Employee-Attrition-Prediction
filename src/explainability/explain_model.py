import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "train_processed.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "test_processed.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "results", "models", "xgboost_model.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "explanations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load data and trained XGBoost model
# --------------------------------------------------

print("Loading processed data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

model = joblib.load(MODEL_PATH)

print("Training data shape:", train_df.shape)
print("Testing data shape :", test_df.shape)


# Separate features and target

X_train = train_df.drop(columns=["Attrition"])
X_test = test_df.drop(columns=["Attrition"])

y_test = test_df["Attrition"]


# --------------------------------------------------
# 3. SHAP Explainability
# --------------------------------------------------

print("\nRunning SHAP explainability...")

explainer = shap.TreeExplainer(model)

# Use a sample of test data for the global explanation
X_sample = X_test.iloc[:100]

shap_values = explainer.shap_values(X_sample)

# Handle different SHAP versions
if isinstance(shap_values, list):
    shap_values = shap_values[1]

elif hasattr(shap_values, "values"):
    shap_values = shap_values.values

if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]


# --------------------------------------------------
# 4. SHAP Summary Plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

shap_summary_path = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)

plt.savefig(
    shap_summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", shap_summary_path)


# --------------------------------------------------
# 5. SHAP Feature Importance Bar Plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

shap_bar_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.png"
)

plt.savefig(
    shap_bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", shap_bar_path)


# --------------------------------------------------
# 6. Individual Employee SHAP Explanation
# --------------------------------------------------

employee_index = 0

employee_data = X_test.iloc[[employee_index]]

employee_shap_values = shap_values[employee_index]

# Create a table containing feature contributions

shap_explanation = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP_Value": employee_shap_values,
    "Feature_Value": X_test.iloc[employee_index].values
})

# Sort by absolute SHAP value

shap_explanation["Absolute_SHAP"] = (
    shap_explanation["SHAP_Value"].abs()
)

shap_explanation = shap_explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
)

shap_csv_path = os.path.join(
    OUTPUT_DIR,
    "individual_shap_explanation.csv"
)

shap_explanation.to_csv(
    shap_csv_path,
    index=False
)

print("Saved:", shap_csv_path)


# --------------------------------------------------
# 7. LIME Explainability
# --------------------------------------------------

print("\nRunning LIME explainability...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Attrition", "Attrition"],
    mode="classification",
    random_state=42
)

lime_explanation = lime_explainer.explain_instance(
    X_test.iloc[employee_index].values,
    model.predict_proba,
    num_features=10
)


# Save LIME explanation as HTML

lime_html_path = os.path.join(
    OUTPUT_DIR,
    "lime_employee_explanation.html"
)

lime_explanation.save_to_file(
    lime_html_path
)

print("Saved:", lime_html_path)


# --------------------------------------------------
# 8. Display prediction for explained employee
# --------------------------------------------------

prediction = model.predict(employee_data)[0]

probability = model.predict_proba(employee_data)[0][1]

print("\nIndividual Employee Explanation")
print("--------------------------------")
print("Predicted Class:",
      "Attrition" if prediction == 1 else "No Attrition")

print("Attrition Probability:",
      round(probability * 100, 2), "%")


# --------------------------------------------------
# 9. Finish
# --------------------------------------------------

print("\n======================================")
print("SHAP + LIME EXPLAINABILITY COMPLETED")
print("======================================")

print("\nGenerated files:")
print("1. shap_summary.png")
print("2. shap_feature_importance.png")
print("3. individual_shap_explanation.csv")
print("4. lime_employee_explanation.html")
