import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import seaborn as sns
#import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#from sklearn.inspection import permutation_importance
#from scipy import stats
#from scipy.stats import kruskal, f_oneway

#try:
#    import xgboost as xgb
#    XGB_AVAILABLE = True
#except ImportError:
#    XGB_AVAILABLE = False
#    print("xgboost not installed — skipping XGBoost model")

# np.random.seed(42)


# upload data
# ignore some data points
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

print(f"... Loading ... {len(df)} rows \n" )

# account for the 2 different numbers in blood pressure
# blood pressure -> systolic / diastolic
def parse_blood_pressure(bp_str):
    try:
        s, d = bp_str.strip().split("/")
        return float(s), float(d)
    except Exception:
        return np.nan, np.nan
df[["Systolic", "Diastolic"]] = df["Blood Pressure"].apply(
    lambda x: pd.Series(parse_blood_pressure(x))
)

# take only the data parts that we will need 
DATA_COLS = [
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Systolic",
    "Diastolic",
    "Heart Rate",
    "Daily Steps",
]

df = df.dropna(subset=DATA_COLS).copy()
X_raw = df[DATA_COLS].copy()

# ---------------- HEALTH SCORE ------------------------------------------ ##
# using a health score to determine an individual's healthiness
" scores are made on a 0-100 scale"
" it will be weighted"
" 0 ( on death) to 100 ( healthy as a horse )"

# ensure that calculated value is not out of range or max and min
def clamp(value, low, high):
    return max(low, min(high, value))

def score_sleep_duration(hours):
    # doctor says that optimal sleep is between 7 to 9 hours
    # apparently 10 + hours is likely to mean depression, diabetes, obesity, etc
    if 7 <= hours <= 9:
        return 100
    # less sleep
    elif hours < 7:
        return clamp(100 - (7 - hours) * 25, 0, 100)
    # too much sleep 
    else:
        return clamp(100 - (hours - 9) * 25, 0, 100)
    

def score_sleep_quality(quality):
    # turn 1 - 10 scale into 100
    return clamp( (quality - 1) / 9 * 100, 0, 100)

# physical activity - minutes per day
def score_activity(mins):
    # recommended is 2.5 hours per week
    # say that 60 mins is ideal
    return clamp(mins / 60 * 100, 0, 100)

# stress levels - scale of 0 to 10
def score_stress(sl):
    # 0 being good = low stress
    # 10 being bad = high stress levels
    # 10 - stress level to compute the health score (high score = low stress)
    return clamp( (10 - sl) / 9 * 100, 0, 100)

# blood pressure ______________________
def score_systolic(sbp):
    "# GOOD SBP : less than 120 --> 100"
    "# BAD SBP : more than 140 --> 0"
    "# bad sbp means hypertension"
    if sbp < 120:
        return 100
    elif sbp < 130:
        return clamp( 100 - (sbp - 120) * 5, 0, 100)
    elif sbp < 140:
        return clamp( 50 - (sbp - 130) * 5, 0, 100)
    else:
        return 0
def score_diastolic(dbp):
    "# GOOD DBP: less than 80 ---> 100"
    "# BAD DBP:  MORE THAN 90 ---> 0"
    if dbp < 80:
        return 100
    elif dbp < 90:
        return clamp( 100 - (dbp - 80) * 10, 0, 100)
    else:
        return 0
    

# heart rate 
def score_heart_rate(hr):
    if 60 <= hr <= 80:
        return 100
    elif 50 <= hr < 60:
        return clamp(100 - (60 - hr) * 5, 0, 100)
    elif 80 <= hr < 100:
        return clamp(100 - (hr - 80) * 5, 0, 100)
    else:
        return 0
    
# daily steps per day
def score_steps(steps):
    " they say you need 10,000 steps a day "
    " 10,000 --> score of 100"
    return clamp( steps / 10000 * 100, 0, 100)

WEIGHTS = {
    "sleep_dur":  0.15,
    "sleep_qual": 0.15,
    "activity":   0.15,
    "stress":     0.20,   
    "systolic":   0.12,
    "diastolic":  0.08,
    "hr":         0.05,
    "steps":      0.10,
}
 
def compute_health_score(row):
    s = {
        "sleep_dur":  score_sleep_duration(row["Sleep Duration"]),
        "sleep_qual": score_sleep_quality(row["Quality of Sleep"]),
        "activity":   score_activity(row["Physical Activity Level"]),
        "stress":     score_stress(row["Stress Level"]),
        "systolic":   score_systolic(row["Systolic"]),
        "diastolic":  score_diastolic(row["Diastolic"]),
        "hr":         score_heart_rate(row["Heart Rate"]),
        "steps":      score_steps(row["Daily Steps"]),
    }
    return sum(WEIGHTS[k] * v for k, v in s.items())
 
df["Health Score"] = df.apply(compute_health_score, axis=1).round(2)

# ---------------------- Train ML Model --------------------

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)
 
y = df["Health Score"].values
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
 
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
 

print("-----MODEL PERFORMANCE-----")
print(f"  Mean Absolute Error : {mae:.2f} pts")
print(f"  R2 Score            : {r2:.4f}")
print("─" * 45)

# ------ simplify the health scores into simple terms (tiers) ------

def gauge_tier(score):
    if score >= 80:   return "Excellent"
    elif score >= 60: return "Good"
    elif score >= 40: return "Fair"
    else:             return "Poor"
 
df["Predicted Score"] = model.predict(X_scaled).round(2)
df["Gauge Tier"]      = df["Predicted Score"].apply(gauge_tier)
 
DISPLAY_COLS = [
    "Person ID", "Age", "Occupation",
    "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level",
    "Systolic", "Diastolic", "Heart Rate", "Daily Steps",
    "BMI Category", "Health Score", "Predicted Score", "Gauge Tier"
]
 
result = df[DISPLAY_COLS].sort_values("Predicted Score", ascending=False)
 
# print("\nHEALTH SCORES (top 10)\n")
# print(result.head(10).to_string(index=False))

# ----- save results to csv
result.to_csv("health_scores_output.csv", index=False)
print("\nSaved → health_scores_output.csv")

# ----- results : compare to health score vs their given bmi category

# combine the normal and normal weight together
df["BMI Category"] = df["BMI Category"].str.strip().replace("Normal Weight", "Normal")
# turn the string into int for easy comparison
BMI_RANK = {
    "Normal":        3,
    "Overweight":    2,
    "Obese":         1,
    "Underweight":   0,
}
df["BMI Rank"] = df["BMI Category"].str.strip().map(BMI_RANK)
 
print("\nHEALTH SCORE vs BMI CATEGORY — COMPARISON")
print("─" * 55)

# separate the results according to their given category

# Average health score per BMI category
bmi_summary = (
    df.groupby("BMI Category")["Predicted Score"]
    .agg(Count="count", Mean="mean", Min="min", Max="max")
    .round(2)
    .sort_values("Mean", ascending=False)
)
print("\nAverage Predicted Health Score by BMI Category:\n")
print(bmi_summary.to_string())

# Correlation between BMI rank and health score
valid_mask = df["BMI Rank"].notna()
if valid_mask.sum() > 1:
    corr = df.loc[valid_mask, ["BMI Rank", "Predicted Score"]].corr().iloc[0, 1]
    print(f"\nCorrelation (BMI rank vs Predicted Score): {corr:.3f}")
    if corr >= 0.5:
        note = "Strong positive — higher BMI rank aligns well with higher health score."
    elif corr >= 0.2:
        note = "Moderate positive — BMI and health score partially agree."
    elif corr >= -0.2:
        note = "Weak / no correlation — health score captures factors BMI alone misses."
    else:
        note = "Negative — some high-BMI individuals score well on other metrics."
    print(f"  Interpretation: {note}")


# count inconsistencies - good score but poor BMI, or vice versa
df["BMI_Health_Flag"] = "Consistent"
 
mask_high_score_poor_bmi = (
    (df["Predicted Score"] >= 60) &
    (df["BMI Category"].str.strip().isin(["Obese", "Overweight"]))
)
df.loc[mask_high_score_poor_bmi, "BMI_Health_Flag"] = "High Score / Poor BMI"
 
mask_low_score_good_bmi = (
    (df["Predicted Score"] < 60) &
    (df["BMI Category"].str.strip().isin(["Normal", "Normal Weight"]))
)
df.loc[mask_low_score_good_bmi, "BMI_Health_Flag"] = "Low Score / Normal BMI"
 
flag_counts = df["BMI_Health_Flag"].value_counts()
print("\nAgreement / Disagreements :\n")
print(flag_counts.to_string())



# Export enriched comparison CSV
compare_cols = [
    "Person ID", "Age", "Occupation", "Gender",
    "BMI Category", "Predicted Score", "Gauge Tier", "BMI_Health_Flag"
]
df[compare_cols].to_csv("health_vs_bmi_comparison.csv", index=False)
print("\nSaved → health_vs_bmi_comparison.csv")


# predict for new person
" Make a new person "
#new_person = pd.DataFrame([{
#    "Sleep Duration":          7.5,
#    "Quality of Sleep":        8,
#    "Physical Activity Level": 45,
#    "Stress Level":            4,
#    "Systolic":                118,
#    "Diastolic":               76,
#    "Heart Rate":              68,
#    "Daily Steps":             8500,
#}])

new_person = pd.DataFrame([{
    "Sleep Duration":          np.round(np.random.uniform(4.5, 9.0), 1),   # hours
    "Quality of Sleep":        np.random.randint(1, 11),                    # 1–10
    "Physical Activity Level": np.random.randint(10, 100),                  # arbitrary scale
    "Stress Level":            np.random.randint(1, 11),                    # 1–10
    "Systolic":                np.random.randint(90, 141),                  # mmHg
    "Diastolic":               np.random.randint(60, 91),                   # mmHg
    "Heart Rate":              np.random.randint(50, 101),                  # bpm
    "Daily Steps":             np.random.randint(1000, 15001),              # steps
}])

new_scaled = scaler.transform(new_person[DATA_COLS])
new_score  = model.predict(new_scaled)[0]
 
print("\n") 
print("─" * 45)
print("PREDICT FOR A NEW PERSON")
print(new_person.to_string(index=False))
print(f"\n  Predicted Health Score : {new_score:.1f} / 100")
print(f"  Gauge Tier             : {gauge_tier(new_score)}")

