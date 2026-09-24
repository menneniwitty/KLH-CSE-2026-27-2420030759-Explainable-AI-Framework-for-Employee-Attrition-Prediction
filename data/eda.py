import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset path
DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Output directory
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 1. ATTRITION DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(7, 5))

df["Attrition"].value_counts().plot(kind="bar")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "attrition_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. OVERTIME VS ATTRITION
# --------------------------------------------------

overtime_attrition = pd.crosstab(
    df["OverTime"],
    df["Attrition"]
)

overtime_attrition.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Overtime vs Employee Attrition")
plt.xlabel("OverTime")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.legend(title="Attrition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overtime_vs_attrition.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. JOB SATISFACTION VS ATTRITION
# --------------------------------------------------

job_satisfaction = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"]
)

job_satisfaction.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Job Satisfaction vs Employee Attrition")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.legend(title="Attrition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "job_satisfaction_vs_attrition.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 4. AGE VS ATTRITION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

df[df["Attrition"] == "No"]["Age"].plot(
    kind="hist",
    bins=15,
    alpha=0.6,
    label="No"
)

df[df["Attrition"] == "Yes"]["Age"].plot(
    kind="hist",
    bins=15,
    alpha=0.6,
    label="Yes"
)

plt.title("Age Distribution by Employee Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.legend(title="Attrition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_vs_attrition.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 5. MONTHLY INCOME VS ATTRITION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

df.boxplot(
    column="MonthlyIncome",
    by="Attrition"
)

plt.title("Monthly Income vs Employee Attrition")
plt.suptitle("")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_income_vs_attrition.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 6. YEARS AT COMPANY VS ATTRITION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

df.boxplot(
    column="YearsAtCompany",
    by="Attrition"
)

plt.title("Years at Company vs Employee Attrition")
plt.suptitle("")
plt.xlabel("Attrition")
plt.ylabel("Years at Company")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "years_at_company_vs_attrition.png",
    dpi=300
)

plt.close()


print("\n========== EDA COMPLETED ==========")

print("\nGenerated files:")

for file in OUTPUT_DIR.glob("*.png"):
    print(file)