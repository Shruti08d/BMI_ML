import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import make_scorer, f1_score

np.random.seed(42)


#LOAD THE DATASET

print("Loading the dataset")
df = pd.read_csv("final_data.csv")

print(f"\n  Total rows    : {len(df)}")
print(f"  Total columns : {len(df.columns)}")

drop_these = [
    "diabetes_risk_score",
    "hypertension_risk_score",
    "heart_disease_risk_score",
    "obesity_risk_score",
    "cholesterol_risk_score",
    "family_history_diabetes",
]

df = df.drop(columns=drop_these)



#  CREATE THE TARGET VARIABLE
# HbA1c >= 6.5 means the person has diabetes risk
# 1 = at risk, 0 = not at risk
print("Building the target variable")


df["target"] = (df["hba1c"] >= 6.5).astype(int)

at_risk     = df["target"].sum()
not_at_risk = len(df) - at_risk

print(f"\n  Target: HbA1c >= 6.5 (WHO diabetes threshold)")
print(f"\n  At Risk     (1) : {at_risk} people ({at_risk/len(df)*100:.1f}%)")
print(f"  Not At Risk (0) : {not_at_risk} people ({not_at_risk/len(df)*100:.1f}%)")



#CLEAN AND ENCODE

print("Cleaning and encoding data")
before = len(df)
df     = df.dropna()


text_columns = [
    "gender", "ethnicity", "education_level",
    "income_level", "employment_status", "smoking_status"
]

for col in text_columns:
    enc     = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    

# SET UP FEATURES AND TARGET
# we predict the same target with both and compare the results
# the difference between them is our finding


# Model A - just BMI
features_bmi = ["bmi"]

# Model B - lifestyle habits only
# no blood tests, no BMI, no genetic history
features_full = [
    "Age",
    "gender",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "smoking_status",
    "alcohol_consumption_per_week",
    "hypertension_history",
    "cardiovascular_history",
    "waist_to_hip_ratio",
]

y = df["target"].values

print(f"\n  Model A (BMI only)  : {features_bmi}")
print(f"  Model B (lifestyle) : {len(features_full)} features")

#  SCALE THE FEATURES
# bring everything to the same range
print("Scaling features")

# two separate scalers so each model gets the right scale
scaler_bmi  = StandardScaler()
scaler_full = StandardScaler()

X_bmi  = scaler_bmi.fit_transform(df[features_bmi])
X_full = scaler_full.fit_transform(df[features_full])


#  THREE MODELS
# we test three different algorithms and compare them
# Logistic Regression 
# Random Forest   
# K-Nearest Neighbors
print("Defining three models")
models = {
    "Logistic Regression" : LogisticRegression(
                                max_iter     = 1000,
                                random_state = 42
                            ),
    "Random Forest"       : RandomForestClassifier(
                                n_estimators     = 100,
                                max_depth        = 5,
                                min_samples_leaf = 10,
                                random_state     = 42
                            ),
    "K-Nearest Neighbors" : KNeighborsClassifier(
                                n_neighbors = 9
                            ),
}

print("go")



# STRATIFIED 5-FOLD CROSS VALIDATION
# AUC is our main metric
# AUC of 0.5 = model is just guessing randomly
# AUC of 1.0 = perfect predictions
# the gap between Model A and Model B is our main finding

print(" 5-Fold Cross Validation")
print("BMI only  vs  Full Lifestyle")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy" : "accuracy",
    "f1"       : make_scorer(f1_score, zero_division=0),
    "auc"      : "roc_auc",
}

all_results = []

for name, model in models.items():

    print(f"\n  {name}")
    print(f"  {'-' * 45}")

    # train and test on BMI only across 5 folds
    res_bmi  = cross_validate(model, X_bmi,  y, cv=cv, scoring=scoring)

    # train and test on full lifestyle features across 5 folds
    res_full = cross_validate(model, X_full, y, cv=cv, scoring=scoring)

    acc_bmi  = res_bmi["test_accuracy"].mean()
    auc_bmi  = res_bmi["test_auc"].mean()
    acc_full = res_full["test_accuracy"].mean()
    auc_full = res_full["test_auc"].mean()
    auc_std  = res_full["test_auc"].std()

    print(f"  {'':12} {'BMI Only':>10} {'Full Model':>12} {'Gap':>8}")
    print(f"  {'Accuracy':<12} {acc_bmi:>10.3f} {acc_full:>12.3f} {acc_full-acc_bmi:>+8.3f}")
    print(f"  {'AUC':<12} {auc_bmi:>10.3f} {auc_full:>12.3f} {auc_full-auc_bmi:>+8.3f}")
    print(f"  AUC std dev: +/- {auc_std:.3f}")

    all_results.append({
        "Model"    : name,
        "AUC BMI"  : round(auc_bmi,  3),
        "AUC Full" : round(auc_full, 3),
        "AUC Gap"  : round(auc_full - auc_bmi, 3),
        "Acc BMI"  : round(acc_bmi,  3),
        "Acc Full" : round(acc_full, 3),
    })

results_df = pd.DataFrame(all_results)

print(f"\n\n  RESULTS SUMMARY")
print(f"  {results_df[['Model','AUC BMI','AUC Full','AUC Gap']].to_string(index=False)}")

results_df.to_csv("results.csv", index=False)
print(f"\n  Saved ")



#  BMI vs LIFESTYLE SIDE BY SIDE
print("PART 8: Chart 1 - BMI vs Lifestyle AUC")

BLUE   = "#3B8BD4"
ORANGE = "#E87D4C"
GREEN  = "#4CAF82"

model_labels = ["Logistic\nRegression", "Random\nForest", "KNN"]
x     = np.arange(len(results_df))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars_bmi  = ax.bar(x - width/2, results_df["AUC BMI"],
                   width, label="BMI Only",
                   color=ORANGE, alpha=0.88)

bars_full = ax.bar(x + width/2, results_df["AUC Full"],
                   width, label="Full Lifestyle Model",
                   color=BLUE, alpha=0.88)

# show the score on top of every bar
for bar in list(bars_bmi) + list(bars_full):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.004,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold"
    )

# dotted line shows where random guessing sits
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
ax.text(2.45, 0.503, "random guess (0.5)",
        fontsize=8, color="gray", ha="right")

ax.set_title(
    "AUC Score: BMI Only vs Full Lifestyle Model\n"
    "Higher = better   |   5-Fold Cross Validation",
    fontsize=11, fontweight="bold", pad=12
)
ax.set_ylabel("AUC Score", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylim(0, 0.80)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("chart1_bmi_vs_lifestyle.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved")


# charts


print("PART 9: Chart 2 - Improvement over BMI")


fig, ax = plt.subplots(figsize=(7, 5))

bars_gap = ax.bar(x, results_df["AUC Gap"],
                  width=0.45,
                  color=GREEN, alpha=0.88)

# show the improvement value on top of every bar
for bar in bars_gap:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"+{bar.get_height():.3f}",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#2a7a50"
    )

ax.set_title(
    "How Much Did Lifestyle Features Improve Over BMI?\n"
    "AUC Gained by Using Full Lifestyle Model",
    fontsize=11, fontweight="bold", pad=12
)
ax.set_ylabel("AUC Gained", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylim(0, 0.10)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("chart2_improvement.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved")
print("\n  yaay!")

