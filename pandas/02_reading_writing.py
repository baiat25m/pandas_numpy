"""
================================================================================
PANDAS 02 — Reading & Writing Data
================================================================================

THEORY:
  Real MLOps workflows read data from many formats. The most common:
    CSV     — human-readable, slow for large files, no type info
    JSON    — nested structures, APIs, event logs
    Parquet — columnar, compressed, fast, preserves types → preferred for ML
    Pickle  — Python-native, fast but not cross-language safe

  Parquet is the gold standard for ML feature stores and data lakes.
  Always prefer Parquet over CSV when you own the pipeline.

  Key parameters to know:
    sep, header, names, usecols, dtype, nrows, skiprows, na_values (CSV)
    orient, lines (JSON)
    columns, engine, compression (Parquet)
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("GENERATING SAMPLE DATA")
print("=" * 60)

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "user_id":      np.arange(1, n+1),
    "age":          np.random.randint(18, 70, n),
    "income":       np.random.normal(55000, 20000, n).round(2),
    "credit_score": np.random.randint(300, 850, n),
    "loan_amount":  np.random.uniform(5000, 50000, n).round(2),
    "approved":     np.random.choice([0, 1], n, p=[0.35, 0.65]),
    "category":     np.random.choice(["A", "B", "C", None], n, p=[0.4, 0.3, 0.2, 0.1]),
})
print(f"Sample data: {df.shape}\n{df.head(3)}")

print()
print("=" * 60)
print("CSV — read & write")
print("=" * 60)

csv_path = os.path.join(DATA_DIR, "loans.csv")
df.to_csv(csv_path, index=False)      # index=False → don't write row numbers
print(f"Written to {csv_path}")

# Read it back
df_csv = pd.read_csv(csv_path)
print(f"Read back: {df_csv.shape}")

# Useful read_csv parameters
df_partial = pd.read_csv(
    csv_path,
    usecols=["user_id", "age", "approved"],   # only load these columns
    nrows=10,                                   # first 10 rows only
    dtype={"approved": bool},                  # override dtype
)
print(f"Partial load (usecols, nrows=10):\n{df_partial.head(3)}")

# Handling bad separators / custom na values
custom_na = pd.read_csv(csv_path, na_values=["NA", "N/A", "null", ""])
print(f"Missing values after custom na_values: {custom_na.isnull().sum().sum()}")

print()
print("=" * 60)
print("JSON — read & write")
print("=" * 60)

json_path = os.path.join(DATA_DIR, "loans.json")

# orient='records' → each row is a JSON object in a list (most common)
df.head(5).to_json(json_path, orient="records", indent=2)
print(f"Written JSON ({os.path.getsize(json_path)} bytes)")

df_json = pd.read_json(json_path, orient="records")
print(f"Read back: {df_json.shape}\n{df_json.head(2)}")

# Line-delimited JSON (JSONL) — standard for event logs and streaming
jsonl_path = os.path.join(DATA_DIR, "events.jsonl")
df.head(5).to_json(jsonl_path, orient="records", lines=True)

df_jsonl = pd.read_json(jsonl_path, orient="records", lines=True)
print(f"\nJSONL read back: {df_jsonl.shape}")

print()
print("=" * 60)
print("PARQUET — the ML engineer's best friend")
print("=" * 60)

parquet_path = os.path.join(DATA_DIR, "loans.parquet")
df.to_parquet(parquet_path, index=False)

sizes = {
    "CSV":     os.path.getsize(csv_path),
    "JSON":    os.path.getsize(json_path),
    "Parquet": os.path.getsize(parquet_path),
}
print("File size comparison:")
for fmt, size in sizes.items():
    print(f"  {fmt:8s}: {size:,} bytes")

# Read parquet — optionally load only specific columns (efficient!)
df_parq = pd.read_parquet(parquet_path, columns=["user_id", "age", "approved"])
print(f"\nParquet partial load: {df_parq.shape}")
print(f"Types preserved: {df_parq.dtypes.to_dict()}")

print()
print("=" * 60)
print("CHUNKED READING — for files that don't fit in RAM")
print("=" * 60)

# Write a larger CSV to demo chunking
big_csv = os.path.join(DATA_DIR, "big_loans.csv")
df_big = pd.concat([df] * 50, ignore_index=True)   # 10,000 rows
df_big.to_csv(big_csv, index=False)

chunk_size = 500
total_approved = 0
total_rows = 0

for chunk in pd.read_csv(big_csv, chunksize=chunk_size):
    total_approved += chunk["approved"].sum()
    total_rows += len(chunk)

print(f"Chunked processing {total_rows:,} rows → approved total: {total_approved:,}")
print(f"Approval rate: {total_approved/total_rows:.1%}")

print()
print("=" * 60)
print("DTYPE OPTIMIZATION — reducing memory footprint")
print("=" * 60)

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to smallest type that fits."""
    df = df.copy()
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique()
        if n_unique / len(df) < 0.5:   # less than 50% unique → use category
            df[col] = df[col].astype("category")
    return df

before_mem = df_big.memory_usage(deep=True).sum() / 1024
df_opt = optimize_dtypes(df_big)
after_mem = df_opt.memory_usage(deep=True).sum() / 1024

print(f"Memory before: {before_mem:.1f} KB")
print(f"Memory after:  {after_mem:.1f} KB  ({100*(1-after_mem/before_mem):.0f}% reduction)")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Using the loans.csv in the data/ folder:
   - Read only the 'age', 'income', 'approved' columns
   - Print mean income by approved/not-approved

2. Save a subset (approved==1) to a Parquet file called 'approved_loans.parquet'.
   Then read it back and verify the shape.

3. Write a function that reads any CSV in chunks of 1000 rows and returns
   the overall mean of a given numeric column.

4. Read loans.csv and print which columns have null values and how many.
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
df1 = pd.read_csv(csv_path, usecols=["age", "income", "approved"])
print(f"1. Mean income by approved:\n{df1.groupby('approved')['income'].mean().round(2)}\n")

# 2.
approved_df = pd.read_csv(csv_path)
approved_df[approved_df["approved"] == 1].to_parquet(
    os.path.join(DATA_DIR, "approved_loans.parquet"), index=False
)
check = pd.read_parquet(os.path.join(DATA_DIR, "approved_loans.parquet"))
print(f"2. approved loans parquet: {check.shape}\n")

# 3.
def mean_chunked(path, col, chunk_size=1000):
    total, count = 0.0, 0
    for chunk in pd.read_csv(path, usecols=[col], chunksize=chunk_size):
        total += chunk[col].sum()
        count += chunk[col].count()
    return total / count

mean_age = mean_chunked(csv_path, "age")
print(f"3. Mean age (chunked): {mean_age:.2f}\n")

# 4.
df4 = pd.read_csv(csv_path)
nulls = df4.isnull().sum()
print(f"4. Null counts:\n{nulls[nulls > 0]}")
