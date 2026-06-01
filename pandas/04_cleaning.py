"""
================================================================================
PANDAS 04 — Data Cleaning: Missing Values, Duplicates, Types
================================================================================

THEORY:
  Garbage in → garbage model. Data cleaning is 60-80% of real MLOps work.

  Key problems to fix before training:
    1. Missing values (NaN)    — drop, fill, interpolate, flag
    2. Duplicates              — exact rows, near-duplicates
    3. Wrong dtypes            — object instead of float, string dates
    4. Outliers                — values outside expected range
    5. Inconsistent strings    — "NYC", "nyc", "New York" → same thing

  Rule of thumb:
    - <5% missing → fill with mean/median/mode or constant
    - >30% missing → consider dropping the column
    - Categorical missing → fill with "UNKNOWN" or mode
    - Sequential data → forward-fill or interpolate
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Create a messy dataset (realistic)
n = 100
df = pd.DataFrame({
    "id":         range(1, n+1),
    "name":       [f"user_{i}" for i in range(1, n+1)],
    "age":        np.where(np.random.rand(n) < 0.1, np.nan, np.random.randint(18, 70, n).astype(float)),
    "income":     np.where(np.random.rand(n) < 0.15, np.nan, np.random.normal(50000, 15000, n)),
    "score":      np.where(np.random.rand(n) < 0.05, np.nan, np.random.uniform(0, 100, n).round(1)),
    "city":       np.random.choice(["New York", "NYC", "nyc", "LA", "Los Angeles", None, "Chicago"], n),
    "category":   np.random.choice(["A", "B", "C", None], n, p=[0.4, 0.3, 0.2, 0.1]),
    "joined_date":pd.date_range("2020-01-01", periods=n, freq="3D").astype(str),
    "salary_str": [f"${np.random.randint(30000, 100000):,}" for _ in range(n)],
})
# Introduce duplicates
df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # 5 exact duplicates
print(f"Messy dataset: {df.shape}\n{df.head(3)}")

print()
print("=" * 60)
print("1. EXPLORING MISSING VALUES")
print("=" * 60)

# How many missing per column
missing = df.isnull().sum()
missing_pct = df.isnull().mean() * 100
missing_report = pd.DataFrame({"count": missing, "pct": missing_pct.round(1)})
print(f"Missing value report:\n{missing_report[missing_report['count'] > 0]}\n")

# Which rows have ANY missing value
rows_with_na = df[df.isnull().any(axis=1)]
print(f"Rows with at least one NaN: {len(rows_with_na)}")

# Any/all — check if all values in a column are null
print(f"Any null in 'age': {df['age'].isnull().any()}")
print(f"All null in 'age': {df['age'].isnull().all()}")

print()
print("=" * 60)
print("2. HANDLING MISSING VALUES")
print("=" * 60)

df2 = df.copy()

# Drop rows where ALL values are NaN (rare but worth checking)
before = len(df2)
df2 = df2.dropna(how="all")
print(f"dropna(how='all'): removed {before - len(df2)} rows")

# Drop columns with too much missing
df2 = df2.loc[:, df2.isnull().mean() < 0.5]   # keep cols with <50% missing
print(f"After dropping high-missing cols: {df2.columns.tolist()}")

# Fill numeric columns
df2["age"]    = df2["age"].fillna(df2["age"].median())         # median (robust to outliers)
df2["income"] = df2["income"].fillna(df2["income"].mean())     # mean
df2["score"]  = df2["score"].fillna(df2["score"].median())

# Fill categorical columns
df2["category"] = df2["category"].fillna("UNKNOWN")
df2["city"]     = df2["city"].fillna("UNKNOWN")

# Forward fill — for time-ordered data
# df2["sensor"] = df2["sensor"].ffill()   # carry last known value forward
# df2["sensor"] = df2["sensor"].bfill()   # use next known value backward

print(f"\nAfter filling:")
print(f"  Remaining NaNs: {df2.isnull().sum().sum()}")

print()
print("=" * 60)
print("3. DUPLICATES")
print("=" * 60)

df3 = df2.copy()

print(f"Total rows:      {len(df3)}")
print(f"Duplicated rows: {df3.duplicated().sum()}")
print(f"Duplicate rows:\n{df3[df3.duplicated()].head(3)}\n")

# Keep first occurrence, remove rest
df3 = df3.drop_duplicates(keep="first")
print(f"After dedup: {len(df3)} rows")

# Check duplicates only on subset of columns (e.g., same user, different timestamp)
# df3.drop_duplicates(subset=["id", "name"], keep="last")

print()
print("=" * 60)
print("4. FIXING DATA TYPES")
print("=" * 60)

df4 = df3.copy()

# String → datetime
df4["joined_date"] = pd.to_datetime(df4["joined_date"])
print(f"joined_date dtype: {df4['joined_date'].dtype}")
print(f"  year: {df4['joined_date'].dt.year.unique()[:3]}")
print(f"  month range: {df4['joined_date'].dt.month.min()} to {df4['joined_date'].dt.month.max()}")

# String with dollar signs → float
df4["salary"] = df4["salary_str"].str.replace("[$,]", "", regex=True).astype(float)
print(f"\nsalary_str → salary: {df4[['salary_str','salary']].head(3).to_string()}")

# Object → category (memory efficient for low-cardinality strings)
print(f"\ncategory before: {df4['category'].dtype}  memory: {df4['category'].memory_usage(deep=True)} B")
df4["category"] = df4["category"].astype("category")
print(f"category after:  {df4['category'].dtype}  memory: {df4['category'].memory_usage(deep=True)} B")

# Numeric downcast
df4["age"] = df4["age"].astype("int8")   # 18-70 fits in int8 (-128..127)
print(f"\nage dtype: {df4['age'].dtype}")

print()
print("=" * 60)
print("5. STRING STANDARDIZATION")
print("=" * 60)

df5 = df4.copy()

# Inspect unique city values
print(f"City values: {df5['city'].unique()}")

# Normalize
df5["city"] = df5["city"].str.strip().str.lower()

# Map known aliases
city_map = {
    "new york": "new_york",
    "nyc":      "new_york",
    "la":       "los_angeles",
    "los angeles": "los_angeles",
}
df5["city"] = df5["city"].map(city_map).fillna(df5["city"])
print(f"After normalization: {df5['city'].value_counts().head()}\n")

print()
print("=" * 60)
print("6. OUTLIER DETECTION AND HANDLING")
print("=" * 60)

# IQR-based outlier removal for 'income'
Q1 = df5["income"].quantile(0.25)
Q3 = df5["income"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier_mask = (df5["income"] < lower) | (df5["income"] > upper)
print(f"Income outliers: {outlier_mask.sum()} rows")
print(f"  Range kept: [{lower:,.0f}, {upper:,.0f}]")

# Option 1: Remove outliers
df_no_outliers = df5[~outlier_mask].copy()

# Option 2: Clip outliers to boundary values (less data loss)
df5["income"] = df5["income"].clip(lower=lower, upper=upper)
print(f"After clipping: min={df5['income'].min():,.0f}  max={df5['income'].max():,.0f}")

print()
print("=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)

final_cols = ["id", "age", "income", "score", "category", "city", "salary", "joined_date"]
df_clean = df5[final_cols].copy()
print(f"Clean dataset: {df_clean.shape}")
print(f"Dtypes:\n{df_clean.dtypes}")
print(f"\nMissing values: {df_clean.isnull().sum().sum()}")
print(f"\nSample:\n{df_clean.head(3)}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Load data/loans.csv, find columns with any missing values,
   and fill them: numerics with median, strings with 'UNKNOWN'.

2. After filling, verify there are zero NaN values remaining.

3. The 'income' column has some unrealistic values (negative or > 200000).
   Count them and clip them to [0, 200000].

4. Convert 'approved' to bool dtype and 'category' to 'category' dtype.
   Compare memory_usage(deep=True) before and after.
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)
import os
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loans.csv")

df_loan = pd.read_csv(csv_path)

# 1+2.
print(f"1. Missing before:\n{df_loan.isnull().sum()[df_loan.isnull().sum() > 0]}")
for col in df_loan.select_dtypes(include="number").columns:
    df_loan[col] = df_loan[col].fillna(df_loan[col].median())
for col in df_loan.select_dtypes(include="object").columns:
    df_loan[col] = df_loan[col].fillna("UNKNOWN")
print(f"2. Missing after: {df_loan.isnull().sum().sum()}\n")

# 3.
bad = ((df_loan["income"] < 0) | (df_loan["income"] > 200000)).sum()
print(f"3. Unrealistic income values: {bad}")
df_loan["income"] = df_loan["income"].clip(0, 200000)
print(f"   After clipping: min={df_loan['income'].min():.0f}  max={df_loan['income'].max():.0f}\n")

# 4.
before = df_loan.memory_usage(deep=True).sum()
df_loan["approved"] = df_loan["approved"].astype(bool)
df_loan["category"] = df_loan["category"].astype("category")
after = df_loan.memory_usage(deep=True).sum()
print(f"4. Memory: {before:,} B → {after:,} B  ({100*(1-after/before):.1f}% reduction)")
