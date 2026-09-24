from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import pandas as pd
import shap

app = Flask(__name__)
CORS(app)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "xgboost_model.pkl"
)

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train_processed.csv"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)

print("XGBoost model loaded successfully.")


# =========================================================
# LOAD RAW DATA
# =========================================================

raw_df = pd.read_csv(RAW_DATA_PATH)

print("Raw dataset loaded:", raw_df.shape)


# =========================================================
# LOAD PROCESSED TRAINING DATA
# =========================================================

train_df = pd.read_csv(TRAIN_PATH)

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column != "Attrition"
]

print("Number of model features:", len(FEATURE_COLUMNS))


# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(model)

print("SHAP explainer initialized.")


# =========================================================
# REMOVE UNUSED COLUMNS
# =========================================================

COLUMNS_TO_REMOVE = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]

working_df = raw_df.drop(
    columns=COLUMNS_TO_REMOVE
)


# =========================================================
# CREATE DEFAULT EMPLOYEE VALUES
# =========================================================

default_values = {}

for column in working_df.columns:

    # Skip target
    if column == "Attrition":
        continue

    # IMPORTANT:
    # Numeric columns -> median
    # Categorical columns -> mode

    if pd.api.types.is_numeric_dtype(
        working_df[column]
    ):

        default_values[column] = (
            working_df[column].median()
        )

    else:

        default_values[column] = (
            working_df[column].mode()[0]
        )


# =========================================================
# CONVERT SATISFACTION VALUES
# =========================================================

def convert_satisfaction(value):

    if isinstance(value, str):

        value = value.strip()

        if value.startswith("1"):
            return 1

        if value.startswith("2"):
            return 2

        if value.startswith("3"):
            return 3

        if value.startswith("4"):
            return 4

    return int(value)


# =========================================================
# CREATE MODEL INPUT
# =========================================================

def create_model_input(user_data):

    # Start with dataset-based defaults
    employee = default_values.copy()

    # -----------------------------------------------------
    # Employee Information
    # -----------------------------------------------------

    if "Age" in user_data:
        employee["Age"] = int(
            user_data["Age"]
        )

    if "Gender" in user_data:
        employee["Gender"] = (
            user_data["Gender"]
        )

    if "MaritalStatus" in user_data:
        employee["MaritalStatus"] = (
            user_data["MaritalStatus"]
        )

    if "DistanceFromHome" in user_data:
        employee["DistanceFromHome"] = int(
            user_data["DistanceFromHome"]
        )

    # -----------------------------------------------------
    # Job Information
    # -----------------------------------------------------

    if "Department" in user_data:
        employee["Department"] = (
            user_data["Department"]
        )

    if "JobRole" in user_data:
        employee["JobRole"] = (
            user_data["JobRole"]
        )

    if "MonthlyIncome" in user_data:
        employee["MonthlyIncome"] = float(
            user_data["MonthlyIncome"]
        )

    if "YearsAtCompany" in user_data:
        employee["YearsAtCompany"] = int(
            user_data["YearsAtCompany"]
        )

    # -----------------------------------------------------
    # Work Environment
    # -----------------------------------------------------

    if "JobSatisfaction" in user_data:

        employee["JobSatisfaction"] = (
            convert_satisfaction(
                user_data["JobSatisfaction"]
            )
        )

    if "EnvironmentSatisfaction" in user_data:

        employee["EnvironmentSatisfaction"] = (
            convert_satisfaction(
                user_data["EnvironmentSatisfaction"]
            )
        )

    if "OverTime" in user_data:

        employee["OverTime"] = (
            user_data["OverTime"]
        )

    if "WorkLifeBalance" in user_data:

        employee["WorkLifeBalance"] = (
            convert_satisfaction(
                user_data["WorkLifeBalance"]
            )
        )

    # -----------------------------------------------------
    # Convert to DataFrame
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [employee]
    )

    # -----------------------------------------------------
    # Find categorical columns
    # -----------------------------------------------------

    categorical_columns = (
        input_df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
    )

    # -----------------------------------------------------
    # SAME ENCODING AS PREPROCESSING
    # -----------------------------------------------------

    input_df = pd.get_dummies(
        input_df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # -----------------------------------------------------
    # MATCH EXACT 44 MODEL FEATURES
    # -----------------------------------------------------

    input_df = input_df.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    # Make sure everything is numeric
    input_df = input_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    return input_df


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message":
            "Employee Attrition Prediction API is running",

        "status":
            "success"
    })


# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # RECEIVE DATA
        # -------------------------------------------------

        user_data = request.get_json()

        if not user_data:

            return jsonify({
                "success": False,
                "error":
                    "No input data received"
            }), 400

        print("\nReceived employee data:")
        print(user_data)

        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

        input_df = create_model_input(
            user_data
        )

        print(
            "Model input shape:",
            input_df.shape
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = int(
            model.predict(input_df)[0]
        )

        probability = float(
            model.predict_proba(
                input_df
            )[0][1]
        )

        # -------------------------------------------------
        # PREDICTION LABEL
        # -------------------------------------------------

        if prediction == 1:

            prediction_text = "Attrition"

        else:

            prediction_text = "No Attrition"

        # -------------------------------------------------
        # SHAP VALUES
        # -------------------------------------------------

        shap_values = (
            explainer.shap_values(
                input_df
            )
        )

        if isinstance(
            shap_values,
            list
        ):

            shap_values = shap_values[1]

        elif hasattr(
            shap_values,
            "values"
        ):

            shap_values = (
                shap_values.values
            )

        if len(
            shap_values.shape
        ) == 3:

            shap_values = (
                shap_values[:, :, 1]
            )

        shap_values = shap_values[0]

        # -------------------------------------------------
        # SHAP DATAFRAME
        # -------------------------------------------------

        explanation_df = pd.DataFrame({

            "feature":
                FEATURE_COLUMNS,

            "shap_value":
                shap_values,

            "value":
                input_df.iloc[0].values

        })

        explanation_df[
            "absolute_shap"
        ] = explanation_df[
            "shap_value"
        ].abs()

        explanation_df = (
            explanation_df.sort_values(
                "absolute_shap",
                ascending=False
            )
        )

        # -------------------------------------------------
        # TOP 5 EXPLANATIONS
        # -------------------------------------------------

        explanations = []

        for _, row in (
            explanation_df.head(5).iterrows()
        ):

            shap_value = float(
                row["shap_value"]
            )

            if shap_value > 0:

                contribution = (
                    "Higher Attrition Risk"
                )

            else:

                contribution = (
                    "Lower Attrition Risk"
                )

            explanations.append({

                "feature":
                    str(row["feature"]),

                "value":
                    float(row["value"]),

                "shap_value":
                    round(
                        shap_value,
                        4
                    ),

                "contribution":
                    contribution

            })

        # -------------------------------------------------
        # PRINT RESULT IN TERMINAL
        # -------------------------------------------------

        print("\nPrediction:")
        print(prediction_text)

        print(
            "Attrition Probability:",
            round(
                probability * 100,
                2
            ),
            "%"
        )

        print("\nTop SHAP explanations:")

        print(explanations)

        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return jsonify({

            "success":
                True,

            "prediction":
                prediction_text,

            "prediction_value":
                prediction,

            "attrition_probability":
                round(
                    probability * 100,
                    2
                ),

            "explanation":
                explanations

        })

    except Exception as e:

        print(
            "\nPrediction error:",
            str(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
