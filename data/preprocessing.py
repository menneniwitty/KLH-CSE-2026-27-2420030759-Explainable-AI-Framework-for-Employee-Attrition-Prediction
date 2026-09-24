import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_PATH = Path(
    "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# REMOVE NON-PREDICTIVE COLUMNS
# --------------------------------------------------

columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]

df = df.drop(columns=columns_to_remove)

print("After removing unnecessary columns:", df.shape)


# --------------------------------------------------
# ENCODE TARGET VARIABLE
# --------------------------------------------------

df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# --------------------------------------------------
# IDENTIFY CATEGORICAL FEATURES
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)


# --------------------------------------------------
# ONE-HOT ENCODING
# --------------------------------------------------

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


print("\nEncoded feature shape:", X.shape)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nBefore SMOTE:")
print("Training set:", X_train.shape)
print("Testing set :", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())


# --------------------------------------------------
# HANDLE CLASS IMBALANCE
# SMOTE IS APPLIED ONLY TO TRAINING DATA
# --------------------------------------------------

smote = SMOTE(
    random_state=42
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)


print("\nAfter SMOTE:")
print(
    y_train_resampled.value_counts()
)


# --------------------------------------------------
# SAVE PROCESSED TRAINING DATA
# --------------------------------------------------

train_data = X_train_resampled.copy()
train_data["Attrition"] = y_train_resampled.values

train_data.to_csv(
    PROCESSED_DIR / "train_processed.csv",
    index=False
)


# --------------------------------------------------
# SAVE PROCESSED TEST DATA
# --------------------------------------------------

test_data = X_test.copy()
test_data["Attrition"] = y_test.values

test_data.to_csv(
    PROCESSED_DIR / "test_processed.csv",
    index=False
)


print("\n========== PREPROCESSING COMPLETED ==========")

print(
    "\nSaved:",
    PROCESSED_DIR / "train_processed.csv"
)

print(
    "Saved:",
    PROCESSED_DIR / "test_processed.csv"
)