"""
================================================================================
MLOPS 02 — Data Validation & Schema Checks
================================================================================

THEORY:
  In production MLOps, bad data is the #1 cause of silent model failures.
  Unlike code bugs (which crash loudly), data bugs produce wrong outputs
  that look valid.

  What to validate:
    1. Schema — right columns, right types
    2. Value ranges — no impossible ages, no negative prices
    3. Distributions — has the data distribution shifted? (data drift)
    4. Referential integrity — no orphan foreign keys
    5. Freshness — data isn't stale

  The goal is to FAIL FAST — catch problems at ingestion, not at serving time.
================================================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

np.random.seed(0)

print("=" * 60)
print("SCHEMA VALIDATION")
print("=" * 60)

# Define expected schema as a dict of {column: expected_dtype}
EXPECTED_SCHEMA = {
    "user_id":      "int64",
    "age":          "float64",
    "income":       "float64",
    "credit_score": "int64",
    "approved":     "int64",
    "category":     "object",
}

REQUIRED_COLUMNS = set(EXPECTED_SCHEMA.keys())

def validate_schema(df: pd.DataFrame, expected: dict) -> list[str]:
    """Return list of schema violations."""
    errors = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    for col, expected_dtype in expected.items():
        if col in df.columns and str(df[col].dtype) != expected_dtype:
            errors.append(f"Column '{col}': expected {expected_dtype}, got {df[col].dtype}")

    return errors


# Good data
good = pd.DataFrame({
    "user_id":      [1, 2, 3],
    "age":          [25.0, 30.0, 45.0],
    "income":       [50000.0, 60000.0, 75000.0],
    "credit_score": [700, 720, 680],
    "approved":     [1, 1, 0],
    "category":     ["A", "B", "A"],
})

# Bad data — wrong dtype, missing column
bad = pd.DataFrame({
    "user_id":      ["u1", "u2"],    # should be int
    "age":          [25.0, 30.0],
    "income":       [50000.0, 60000.0],
    # credit_score missing!
    "approved":     [1, 1],
    "category":     ["A", "B"],
})

good_errors = validate_schema(good, EXPECTED_SCHEMA)
bad_errors  = validate_schema(bad,  EXPECTED_SCHEMA)
print(f"Good data errors: {good_errors}")
print(f"Bad data errors:  {bad_errors}")

print()
print("=" * 60)
print("VALUE RANGE VALIDATION")
print("=" * 60)

# Define constraints
@dataclass
class ColumnConstraint:
    min_val:    Optional[float] = None
    max_val:    Optional[float] = None
    allowed:    Optional[list]  = None
    nullable:   bool = True
    max_null_pct: float = 0.1    # max 10% nulls

CONSTRAINTS = {
    "age":          ColumnConstraint(min_val=0, max_val=120, nullable=False),
    "income":       ColumnConstraint(min_val=0, max_val=1_000_000),
    "credit_score": ColumnConstraint(min_val=300, max_val=850, nullable=False),
    "approved":     ColumnConstraint(allowed=[0, 1], nullable=False),
    "category":     ColumnConstraint(allowed=["A", "B", "C", "UNKNOWN"]),
}

def validate_values(df: pd.DataFrame, constraints: dict) -> list[str]:
    errors = []
    for col, c in constraints.items():
        if col not in df.columns:
            continue
        s = df[col]

        # Null check
        null_pct = s.isnull().mean()
        if not c.nullable and null_pct > 0:
            errors.append(f"'{col}': not nullable but has {null_pct:.1%} nulls")
        elif null_pct > c.max_null_pct:
            errors.append(f"'{col}': {null_pct:.1%} nulls exceeds threshold {c.max_null_pct:.1%}")

        s_notnull = s.dropna()

        # Range checks
        if c.min_val is not None:
            n_below = (s_notnull < c.min_val).sum()
            if n_below > 0:
                errors.append(f"'{col}': {n_below} values below min {c.min_val}")
        if c.max_val is not None:
            n_above = (s_notnull > c.max_val).sum()
            if n_above > 0:
                errors.append(f"'{col}': {n_above} values above max {c.max_val}")

        # Allowed values
        if c.allowed is not None:
            invalid = set(s_notnull.unique()) - set(c.allowed)
            if invalid:
                errors.append(f"'{col}': unexpected values {invalid}")

    return errors

# Generate data with intentional errors
n = 200
df_test = pd.DataFrame({
    "user_id":      range(n),
    "age":          np.where(np.random.rand(n) < 0.02, -5, np.random.randint(18, 70, n)).astype(float),
    "income":       np.random.normal(50000, 20000, n),
    "credit_score": np.where(np.random.rand(n) < 0.03, 1000, np.random.randint(300, 850, n)),
    "approved":     np.random.choice([0, 1, 2], n, p=[0.4, 0.55, 0.05]),  # '2' is invalid!
    "category":     np.random.choice(["A", "B", "C", "D", None], n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
})

errors = validate_values(df_test, CONSTRAINTS)
print("Value validation errors:")
for e in errors:
    print(f"  ✗ {e}")

print()
print("=" * 60)
print("DISTRIBUTION VALIDATION (DATA DRIFT DETECTION)")
print("=" * 60)

# Population Stability Index (PSI) — industry standard for drift detection
def compute_psi(reference: np.ndarray, current: np.ndarray,
                bins: int = 10, epsilon: float = 1e-6) -> float:
    """
    PSI < 0.1:  no significant change
    PSI 0.1-0.2: moderate change, investigate
    PSI > 0.2:  significant drift, retrain model
    """
    # Create bins based on reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current)   + epsilon

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# Kolmogorov-Smirnov test statistic (simpler)
def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Max absolute difference between empirical CDFs. Range [0,1]. Higher = more drift."""
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    ref_cdf = np.arange(1, len(ref_sorted)+1) / len(ref_sorted)
    cur_cdf = np.interp(ref_sorted, cur_sorted, np.arange(1, len(cur_sorted)+1) / len(cur_sorted),
                        left=0, right=1)
    return np.max(np.abs(ref_cdf - cur_cdf))

np.random.seed(1)
# Reference (training) distribution
ref_income = np.random.normal(50000, 10000, 5000)

# Scenario 1: stable production data
stable_income = np.random.normal(50000, 10500, 1000)

# Scenario 2: drifted production data (different population)
drifted_income = np.random.normal(58000, 12000, 1000)

for name, prod in [("stable", stable_income), ("drifted", drifted_income)]:
    psi = compute_psi(ref_income, prod)
    ks  = ks_statistic(ref_income, prod)
    status = "OK" if psi < 0.1 else ("WARNING" if psi < 0.2 else "DRIFT DETECTED")
    print(f"{name:10s}:  PSI={psi:.4f}  KS={ks:.4f}  [{status}]")

print()
print("=" * 60)
print("COMPLETENESS & FRESHNESS CHECKS")
print("=" * 60)

# Simulate a pipeline that should produce daily data
pipeline_log = pd.DataFrame({
    "date":          pd.date_range("2024-01-01", periods=30, freq="D"),
    "rows_produced": np.random.randint(900, 1100, 30),
    "null_rate":     np.random.uniform(0.01, 0.05, 30),
})

# Inject anomalies
pipeline_log.loc[5, "rows_produced"] = 50    # data gap!
pipeline_log.loc[12, "null_rate"]    = 0.35  # quality issue!
pipeline_log.loc[20, "rows_produced"]= 0     # complete gap!

def check_pipeline_health(log: pd.DataFrame) -> list[str]:
    alerts = []
    expected_rows = log["rows_produced"].median()

    for _, row in log.iterrows():
        if row["rows_produced"] < expected_rows * 0.5:
            alerts.append(f"{row['date'].date()}: LOW VOLUME ({row['rows_produced']} rows, expected ~{expected_rows:.0f})")
        if row["null_rate"] > 0.10:
            alerts.append(f"{row['date'].date()}: HIGH NULL RATE ({row['null_rate']:.1%})")

    # Freshness check
    latest = log["date"].max()
    staleness_days = (pd.Timestamp("2024-01-31") - latest).days
    if staleness_days > 1:
        alerts.append(f"STALE DATA: latest date is {latest.date()}, {staleness_days} days ago")

    return alerts

alerts = check_pipeline_health(pipeline_log)
print(f"Pipeline health alerts ({len(alerts)} found):")
for a in alerts:
    print(f"  ⚠  {a}")

print()
print("=" * 60)
print("PUTTING IT ALL TOGETHER: Validation Pipeline")
print("=" * 60)

def run_validation(df: pd.DataFrame, name: str = "dataset") -> bool:
    print(f"\n{'='*40}")
    print(f"Validating: {name} ({df.shape})")
    print(f"{'='*40}")
    all_errors = []

    schema_errors = validate_schema(df, EXPECTED_SCHEMA)
    if schema_errors:
        print(f"SCHEMA ({len(schema_errors)} errors):")
        for e in schema_errors: print(f"  ✗ {e}")
        all_errors.extend(schema_errors)

    # Only run value checks if schema is OK
    val_cols = {k: v for k, v in CONSTRAINTS.items() if k in df.columns}
    value_errors = validate_values(df, val_cols)
    if value_errors:
        print(f"VALUES ({len(value_errors)} errors):")
        for e in value_errors: print(f"  ✗ {e}")
        all_errors.extend(value_errors)

    if not all_errors:
        print("  ✓ All checks passed")
    return len(all_errors) == 0

run_validation(good, "good_data")
run_validation(bad, "bad_data")
