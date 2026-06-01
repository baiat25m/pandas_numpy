"""
================================================================================
MLOPS 01 — Feature Engineering with Pandas & NumPy
================================================================================

This is where theory meets practice. Feature engineering is the most
impactful skill in ML — a good feature beats a fancy model.

Topics covered:
  1. Encoding categorical variables (Label, One-hot, Target encoding)
  2. Scaling numeric features (Min-Max, Standard, Robust)
  3. Creating new features from existing ones
  4. Binning / discretization
  5. Date/time features
  6. Text features (basic)
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Realistic MLOps feature engineering scenario: customer churn prediction
raw = pd.DataFrame({
    "customer_id":  range(1, n+1),
    "age":          np.random.randint(18, 75, n),
    "tenure_months":np.random.randint(1, 120, n),
    "monthly_charge":np.random.uniform(20, 120, n).round(2),
    "total_charges": None,  # will derive
    "n_support_calls":np.random.poisson(2, n),
    "plan":         np.random.choice(["basic", "standard", "premium"], n, p=[0.4, 0.35, 0.25]),
    "contract":     np.random.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20]),
    "last_login":   pd.date_range("2023-01-01", periods=n, freq="8h"),
    "churn":        np.random.choice([0, 1], n, p=[0.73, 0.27]),
})
raw["total_charges"] = (raw["monthly_charge"] * raw["tenure_months"]).round(2)
raw = raw.set_index("customer_id")

print(f"Raw features:\n{raw.head(3)}\n")

print("=" * 60)
print("1. ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

df = raw.copy()

# --- Label Encoding (ordinal — when order matters) ---
plan_order = {"basic": 0, "standard": 1, "premium": 2}
df["plan_label"] = df["plan"].map(plan_order)
print(f"Label encoding (plan): {df[['plan','plan_label']].drop_duplicates().to_dict('records')}")

# --- One-Hot Encoding (nominal — when no order) ---
contract_dummies = pd.get_dummies(df["contract"], prefix="contract", drop_first=True)
# drop_first=True avoids multicollinearity (dummy trap)
print(f"\nOne-hot (contract):\n{contract_dummies.head(4)}")
df = pd.concat([df, contract_dummies], axis=1)

# --- Target Encoding (encode with mean of target per category) ---
# Useful for high-cardinality categoricals
target_enc = df.groupby("plan")["churn"].mean()
df["plan_target_enc"] = df["plan"].map(target_enc)
print(f"\nTarget encoding (plan → mean churn rate):\n{target_enc.round(3)}")

print()
print("=" * 60)
print("2. SCALING NUMERIC FEATURES")
print("=" * 60)

numeric_cols = ["age", "tenure_months", "monthly_charge", "total_charges", "n_support_calls"]

# --- Min-Max Scaling [0, 1] ---
def minmax_scale(series):
    return (series - series.min()) / (series.max() - series.min())

# --- Standard (Z-score) Scaling — mean=0, std=1 ---
def standard_scale(series):
    return (series - series.mean()) / series.std()

# --- Robust Scaling — uses median and IQR (outlier-resistant) ---
def robust_scale(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series - series.median()) / iqr

scaled = pd.DataFrame(index=df.index)
for col in numeric_cols:
    scaled[f"{col}_minmax"]   = minmax_scale(df[col])
    scaled[f"{col}_standard"] = standard_scale(df[col])
    scaled[f"{col}_robust"]   = robust_scale(df[col])

print(f"Scaling comparison for 'monthly_charge':")
print(f"  original: mean={df['monthly_charge'].mean():.1f}  std={df['monthly_charge'].std():.1f}")
print(f"  minmax:   mean={scaled['monthly_charge_minmax'].mean():.3f}  std={scaled['monthly_charge_minmax'].std():.3f}")
print(f"  standard: mean={scaled['monthly_charge_standard'].mean():.3f}  std={scaled['monthly_charge_standard'].std():.3f}")
print(f"  robust:   median={scaled['monthly_charge_robust'].median():.3f}  iqr-range≈1")

print()
print("=" * 60)
print("3. CREATING DERIVED FEATURES")
print("=" * 60)

# Ratio features
df["charge_per_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
df["support_per_month"] = df["n_support_calls"] / df["tenure_months"].clip(lower=1)

# Polynomial / interaction features
df["age_x_tenure"] = df["age"] * df["tenure_months"]
df["monthly_charge_sq"] = df["monthly_charge"] ** 2

# Log transform — useful for right-skewed distributions
df["log_total_charges"] = np.log1p(df["total_charges"])   # log1p = log(1+x) handles 0

# Binary flags
df["high_value"]    = (df["monthly_charge"] > df["monthly_charge"].quantile(0.75)).astype(int)
df["veteran"]       = (df["tenure_months"] > 24).astype(int)
df["at_risk"]       = (df["n_support_calls"] > 5).astype(int)

print(f"Derived features:\n{df[['charge_per_month','log_total_charges','high_value','veteran','at_risk']].head(5)}\n")

print("=" * 60)
print("4. BINNING / DISCRETIZATION")
print("=" * 60)

# Equal-width bins
df["age_bin"] = pd.cut(df["age"], bins=[18, 30, 45, 60, 75], labels=["18-30", "30-45", "45-60", "60-75"])

# Quantile-based bins (equal frequency) — better for skewed data
df["charge_quartile"] = pd.qcut(df["monthly_charge"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

print(f"Age bins:\n{df['age_bin'].value_counts().sort_index()}\n")
print(f"Charge quartiles:\n{df['charge_quartile'].value_counts().sort_index()}\n")

# Churn rate by bin
print(f"Churn rate by age bin:")
print(df.groupby("age_bin", observed=True)["churn"].mean().round(3))

print()
print("=" * 60)
print("5. DATE/TIME FEATURES")
print("=" * 60)

reference_date = pd.Timestamp("2024-01-01")
df["days_since_login"] = (reference_date - df["last_login"]).dt.days
df["login_hour"]       = df["last_login"].dt.hour
df["login_dayofweek"]  = df["last_login"].dt.dayofweek
df["login_is_weekend"] = (df["login_dayofweek"] >= 5).astype(int)

print(f"Datetime features:\n{df[['days_since_login','login_hour','login_is_weekend']].describe().round(1)}\n")

print("=" * 60)
print("6. FINAL FEATURE MATRIX")
print("=" * 60)

feature_cols = [
    # Encoded categoricals
    "plan_label", "plan_target_enc",
    "contract_one-year", "contract_two-year",
    # Scaled numerics (use standard scaling for most models)
    "age", "tenure_months", "monthly_charge",
    # Derived features
    "charge_per_month", "support_per_month", "log_total_charges",
    # Binary flags
    "high_value", "veteran", "at_risk",
    # Datetime
    "days_since_login", "login_is_weekend",
]

X = df[feature_cols].copy()
y = df["churn"].copy()

# Final check before training
print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"\nFeature summary:\n{X.describe().round(2).T[['mean','std','min','max']].head(8)}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Add a feature 'customer_lifetime_value' = total_charges / churn_risk
   where churn_risk = plan_target_enc. Handle division by zero safely.

2. Create a 'recency_score' feature:
   - Assign score 3 if days_since_login <= 7
   - Score 2 if 7 < days_since_login <= 30
   - Score 1 if 30 < days_since_login <= 90
   - Score 0 if > 90

3. Apply log transformation to 'n_support_calls' (watch out for zeros!)
   and compare the skewness before and after.

4. Create a 5-bin quantile column for 'tenure_months' and compute
   the mean churn rate for each bin. Which tenure group churns most?
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
df["customer_lifetime_value"] = df["total_charges"] / df["plan_target_enc"].replace(0, np.nan)
df["customer_lifetime_value"] = df["customer_lifetime_value"].fillna(0)
print(f"1. CLV stats:\n{df['customer_lifetime_value'].describe().round(0)}\n")

# 2.
conditions = [
    df["days_since_login"] <= 7,
    df["days_since_login"].between(8, 30),
    df["days_since_login"].between(31, 90),
]
choices = [3, 2, 1]
df["recency_score"] = np.select(conditions, choices, default=0)
print(f"2. Recency score distribution:\n{df['recency_score'].value_counts().sort_index()}\n")

# 3.
skew_before = df["n_support_calls"].skew()
skew_after  = np.log1p(df["n_support_calls"]).skew()
print(f"3. Skewness before: {skew_before:.3f}  after: {skew_after:.3f}\n")

# 4.
df["tenure_bin"] = pd.qcut(df["tenure_months"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
churn_by_tenure = df.groupby("tenure_bin", observed=True)["churn"].mean().round(3)
print(f"4. Churn rate by tenure quintile:\n{churn_by_tenure}")
print(f"   Highest churn: {churn_by_tenure.idxmax()} ({churn_by_tenure.max():.1%})")
