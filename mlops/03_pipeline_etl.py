"""
================================================================================
MLOPS 03 — End-to-End ETL Pipeline
================================================================================

THEORY:
  ETL = Extract, Transform, Load.
  In MLOps this means: raw data → clean features → feature store / model input.

  A production ETL pipeline:
    1. Extract  — read from source (CSV, DB, API, Parquet lake)
    2. Validate — schema + value checks (fail fast)
    3. Clean    — handle nulls, types, duplicates
    4. Transform — feature engineering
    5. Load     — write to feature store (Parquet, database)
    6. Log      — record what happened (lineage, stats, errors)

  This file builds a complete, repeatable pipeline for the loan dataset.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# STEP 0: Generate raw source data (simulates upstream source)
# ============================================================

def generate_raw_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Simulate messy raw data as it would come from a source system."""
    rng = np.random.default_rng(seed)
    data = {
        "loan_id":       [f"L{i:05d}" for i in range(1, n+1)],
        "applicant_age": rng.integers(18, 80, n).astype(float),
        "annual_income": rng.normal(55000, 20000, n),
        "credit_score":  rng.integers(300, 900, n).astype(float),
        "loan_amount":   rng.uniform(1000, 50000, n),
        "term_months":   rng.choice([12, 24, 36, 48, 60], n),
        "purpose":       rng.choice(["home", "auto", "personal", "business", None], n, p=[0.25, 0.2, 0.3, 0.2, 0.05]),
        "employment":    rng.choice(["employed", "self-employed", "unemployed", "EMPLOYED", " employed "], n,
                                    p=[0.5, 0.2, 0.1, 0.1, 0.1]),
        "approved":      rng.choice([0, 1], n, p=[0.35, 0.65]),
        "application_ts":pd.date_range("2023-01-01", periods=n, freq="6h").astype(str),
    }
    df = pd.DataFrame(data)

    # Inject realistic issues
    rng2 = np.random.default_rng(seed + 1)
    null_mask = rng2.random(n) < 0.08
    df.loc[null_mask, "applicant_age"] = np.nan
    df.loc[rng2.random(n) < 0.05, "credit_score"] = np.nan
    df.loc[rng2.random(n) < 0.03, "annual_income"] = -9999   # sentinel for missing
    df.loc[rng2.random(n) < 0.01, "credit_score"]  = 9999    # data entry error
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)     # 10 duplicate rows

    return df


# ============================================================
# STEP 1: EXTRACT
# ============================================================

class PipelineLogger:
    """Simple logger that tracks what happened in each pipeline step."""
    def __init__(self, pipeline_name: str):
        self.name = pipeline_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats: dict = {"run_id": self.run_id, "steps": []}

    def log_step(self, step: str, input_rows: int, output_rows: int,
                 notes: str = ""):
        record = {"step": step, "input": input_rows, "output": output_rows,
                  "dropped": input_rows - output_rows, "notes": notes}
        self.stats["steps"].append(record)
        print(f"  [{step}] {input_rows} → {output_rows} rows  {notes}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"  Pipeline log saved to {path}")


def extract(source_path: str) -> pd.DataFrame:
    """Extract raw data from source."""
    df = pd.read_csv(source_path)
    return df


# ============================================================
# STEP 2: VALIDATE
# ============================================================

def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove rows with critical issues; return clean df and list of warnings."""
    warnings = []
    n0 = len(df)

    # Critical: remove rows where loan_id is null (can't trace record)
    df = df[df["loan_id"].notna()].copy()

    # Critical: remove rows where approved label is missing
    df = df[df["approved"].isin([0, 1])].copy()

    n1 = len(df)
    if n0 != n1:
        warnings.append(f"Removed {n0-n1} rows with critical missing data")

    # Non-critical warnings
    null_pct = df.isnull().mean()
    for col in null_pct[null_pct > 0.15].index:
        warnings.append(f"High null rate in '{col}': {null_pct[col]:.1%}")

    return df, warnings


# ============================================================
# STEP 3: CLEAN
# ============================================================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset=["loan_id"], keep="first")

    # Fix income sentinel (-9999 means missing)
    df["annual_income"] = df["annual_income"].replace(-9999, np.nan)

    # Fix credit_score outlier (9999 = data entry error)
    df.loc[df["credit_score"] > 850, "credit_score"] = np.nan

    # Clip to valid ranges
    df["applicant_age"] = df["applicant_age"].clip(18, 100)
    df["credit_score"]  = df["credit_score"].clip(300, 850)
    df["annual_income"] = df["annual_income"].clip(lower=0)
    df["loan_amount"]   = df["loan_amount"].clip(lower=100)

    # Fill missing values
    df["applicant_age"] = df["applicant_age"].fillna(df["applicant_age"].median())
    df["credit_score"]  = df["credit_score"].fillna(df["credit_score"].median())
    df["annual_income"] = df["annual_income"].fillna(df["annual_income"].median())
    df["purpose"]       = df["purpose"].fillna("unknown")

    # Standardize strings
    df["employment"] = df["employment"].str.strip().str.lower()
    df["purpose"]    = df["purpose"].str.strip().str.lower()

    # Parse datetime
    df["application_ts"] = pd.to_datetime(df["application_ts"])

    # Correct dtypes
    df["applicant_age"] = df["applicant_age"].astype(int)
    df["credit_score"]  = df["credit_score"].astype(int)
    df["approved"]      = df["approved"].astype(int)

    return df


# ============================================================
# STEP 4: TRANSFORM (feature engineering)
# ============================================================

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio features
    df["debt_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)
    df["monthly_payment"] = df["loan_amount"] / df["term_months"]
    df["income_per_age"]  = df["annual_income"] / df["applicant_age"]

    # Log transforms for right-skewed features
    df["log_income"]      = np.log1p(df["annual_income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])

    # Buckets
    df["age_group"] = pd.cut(
        df["applicant_age"],
        bins=[18, 25, 35, 50, 65, 100],
        labels=["18-25", "25-35", "35-50", "50-65", "65+"]
    )
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[300, 580, 670, 740, 800, 850],
        labels=["poor", "fair", "good", "very_good", "exceptional"]
    )

    # Encode categoricals
    employment_map = {"employed": 2, "self-employed": 1, "unemployed": 0}
    df["employment_enc"] = df["employment"].map(employment_map).fillna(1)

    purpose_dummies = pd.get_dummies(df["purpose"], prefix="purpose", drop_first=True)
    df = pd.concat([df, purpose_dummies], axis=1)

    # Datetime features
    df["app_hour"]    = df["application_ts"].dt.hour
    df["app_month"]   = df["application_ts"].dt.month
    df["app_weekday"] = df["application_ts"].dt.dayofweek
    df["is_weekend"]  = (df["app_weekday"] >= 5).astype(int)

    return df


# ============================================================
# STEP 5: LOAD
# ============================================================

def load(df: pd.DataFrame, output_path: str):
    """Write processed features to Parquet (the feature store)."""
    df.to_csv(output_path, index=False)


# ============================================================
# RUN THE PIPELINE
# ============================================================

def run_pipeline():
    logger = PipelineLogger("loan_feature_pipeline")
    print(f"\n{'='*60}")
    print(f"PIPELINE: {logger.name}  run_id={logger.run_id}")
    print(f"{'='*60}")

    # Generate raw data and save as source
    raw_path    = os.path.join(DATA_DIR, "raw_loans.csv")
    output_path = os.path.join(DATA_DIR, "features_loans.csv")
    log_path    = os.path.join(DATA_DIR, "pipeline_log.json")

    raw = generate_raw_data(500)
    raw.to_csv(raw_path, index=False)
    print(f"Source: {raw_path}  ({len(raw)} rows)")

    # EXTRACT
    print("\nStep 1: EXTRACT")
    df = extract(raw_path)
    logger.log_step("extract", len(raw), len(df))

    # VALIDATE
    print("\nStep 2: VALIDATE")
    df, warnings = validate(df)
    for w in warnings:
        print(f"  ⚠  {w}")
    logger.log_step("validate", len(raw), len(df), notes=f"{len(warnings)} warnings")

    # CLEAN
    print("\nStep 3: CLEAN")
    n_before = len(df)
    df = clean(df)
    logger.log_step("clean", n_before, len(df), notes="dedup + impute + normalize")

    # TRANSFORM
    print("\nStep 4: TRANSFORM")
    n_before = len(df)
    df = transform(df)
    logger.log_step("transform", n_before, len(df),
                    notes=f"→ {df.shape[1]} features")

    # LOAD
    print("\nStep 5: LOAD")
    load(df, output_path)
    print(f"  Written {len(df)} rows × {df.shape[1]} cols to {output_path}")
    logger.log_step("load", len(df), len(df), notes=output_path)

    # Save log
    logger.save(log_path)

    return df


df_features = run_pipeline()

print(f"\n{'='*60}")
print("FINAL FEATURE SUMMARY")
print(f"{'='*60}")

feature_cols = [
    "applicant_age", "annual_income", "credit_score", "loan_amount",
    "debt_to_income", "monthly_payment", "log_income",
    "employment_enc", "app_hour", "is_weekend",
]
print(f"\nShape: {df_features.shape}")
print(f"Feature stats:\n{df_features[feature_cols].describe().round(2).T[['mean','std','min','max']]}")
print(f"\nTarget distribution:\n{df_features['approved'].value_counts(normalize=True).round(3)}")
print(f"\nMissing values: {df_features.isnull().sum().sum()}")
