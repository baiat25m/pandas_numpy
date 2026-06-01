"""
================================================================================
PANDAS 01 — Series & DataFrame Basics
================================================================================

THEORY:
  Pandas is built on top of NumPy. It adds:
    - LABELS on rows (index) and columns (names)
    - Mixed data types across columns (unlike NumPy)
    - Rich I/O, groupby, merge, time-series support

  Two core types:
    Series    — 1D labeled array. Think: a single spreadsheet column.
    DataFrame — 2D labeled table. Think: a spreadsheet or SQL table.

  In MLOps, DataFrames are how you load, inspect, clean, and transform
  data before feeding it to a model or writing it back to a data store.
================================================================================
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("SERIES — 1D labeled array")
print("=" * 60)

# From a list — auto-generates integer index
s1 = pd.Series([10, 20, 30, 40, 50])
print(f"Series from list:\n{s1}\n")
print(f"  dtype: {s1.dtype}")
print(f"  index: {s1.index.tolist()}")
print(f"  values: {s1.values}  (NumPy array!)")
print(f"  s1[2]: {s1[2]}")

# From a dict — keys become the index
s2 = pd.Series({"accuracy": 0.94, "precision": 0.91, "recall": 0.89, "f1": 0.90})
print(f"\nSeries from dict:\n{s2}\n")
print(f"  s2['accuracy']: {s2['accuracy']}")
print(f"  s2[s2 > 0.9]:   {s2[s2 > 0.90].to_dict()}")

# Useful Series attributes
print(f"\n  name: {s2.name}")
s2.name = "metrics"
print(f"  after naming: {s2.name}")
print(f"  shape: {s2.shape}")
print(f"  size:  {s2.size}")

print()
print("=" * 60)
print("DATAFRAME — 2D labeled table")
print("=" * 60)

# From a dict of lists — most common way
df = pd.DataFrame({
    "model_id":    ["model_a", "model_b", "model_c", "model_d"],
    "accuracy":    [0.94, 0.87, 0.91, 0.96],
    "latency_ms":  [120, 45, 85, 230],
    "deployed":    [True, False, True, True],
    "version":     [3, 1, 2, 4],
})
print(f"DataFrame:\n{df}\n")

# Core attributes
print(f"shape:    {df.shape}   (rows, cols)")
print(f"columns:  {df.columns.tolist()}")
print(f"index:    {df.index.tolist()}")
print(f"dtypes:\n{df.dtypes}\n")

# Quick overview — your first command on any new dataset
print(f"info():")
df.info()
print()
print(f"describe() — stats for numeric columns:\n{df.describe()}\n")

print()
print("=" * 60)
print("ACCESSING COLUMNS")
print("=" * 60)

# Single column → Series
accuracy_col = df["accuracy"]
print(f"df['accuracy'] type: {type(accuracy_col)}")
print(f"{accuracy_col}\n")

# Multiple columns → DataFrame
subset = df[["model_id", "accuracy", "latency_ms"]]
print(f"df[['model_id','accuracy','latency_ms']]:\n{subset}\n")

# Dot notation (only for simple column names, avoid if col has spaces)
print(f"df.version:\n{df.version}")

print()
print("=" * 60)
print("ADDING & MODIFYING COLUMNS")
print("=" * 60)

# New column — vectorized operation
df["score"] = df["accuracy"] * 0.7 + (1 / df["latency_ms"]) * 0.3
df["is_fast"] = df["latency_ms"] < 100
df["model_id_upper"] = df["model_id"].str.upper()

print(f"After adding columns:\n{df}\n")

# Modify in-place
df["latency_ms"] = df["latency_ms"] * 1.05   # simulate 5% slowdown
print(f"Latency after 5% slowdown: {df['latency_ms'].tolist()}")

print()
print("=" * 60)
print("BASIC ROW OPERATIONS")
print("=" * 60)

# Head / tail — always do this on a new dataset
print(f"head(2):\n{df.head(2)}\n")
print(f"tail(2):\n{df.tail(2)}\n")

# Sampling — useful for quick inspection of large datasets
print(f"sample(2):\n{df.sample(2, random_state=42)}\n")

# Sort
sorted_df = df.sort_values("accuracy", ascending=False)
print(f"Sorted by accuracy:\n{sorted_df[['model_id','accuracy']]}\n")

print()
print("=" * 60)
print("INDEX — the row labels")
print("=" * 60)

# Set a meaningful index (good practice for model registries, etc.)
df_indexed = df.set_index("model_id")
print(f"With model_id as index:\n{df_indexed[['accuracy','latency_ms']]}\n")

# Reset back to integer index
df_reset = df_indexed.reset_index()
print(f"Reset index — model_id back as column: {df_reset.columns.tolist()}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Create a DataFrame with 5 experiments:
   - columns: run_id (1-5), lr (learning rates: 0.1, 0.01, 0.001, 0.0001, 0.00001),
     train_loss, val_loss (make up values)
   - Print its shape, dtypes, and describe().

2. Add a column "overfit" that is True when val_loss > train_loss * 1.1.

3. Sort by val_loss ascending and print only the top 3 runs.

4. Set run_id as the index. Access the row for run_id=3 using .loc[3].
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
exp = pd.DataFrame({
    "run_id":     [1, 2, 3, 4, 5],
    "lr":         [0.1, 0.01, 0.001, 0.0001, 0.00001],
    "train_loss": [0.80, 0.55, 0.35, 0.40, 0.45],
    "val_loss":   [0.85, 0.60, 0.33, 0.55, 0.60],
})
print(f"1. shape={exp.shape}")
print(f"   dtypes:\n{exp.dtypes}\n")
print(f"   describe:\n{exp.describe()}\n")

# 2.
exp["overfit"] = exp["val_loss"] > exp["train_loss"] * 1.1
print(f"2. overfit column:\n{exp[['run_id','train_loss','val_loss','overfit']]}\n")

# 3.
top3 = exp.sort_values("val_loss").head(3)
print(f"3. top 3 by val_loss:\n{top3[['run_id','val_loss']]}\n")

# 4.
exp_idx = exp.set_index("run_id")
print(f"4. run_id=3:\n{exp_idx.loc[3]}")
