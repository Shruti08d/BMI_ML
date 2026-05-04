import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import kruskal, f_oneway

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("xgboost not installed — skipping XGBoost model")

np.random.seed(42)


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