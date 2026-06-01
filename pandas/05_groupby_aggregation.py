"""
================================================================================
PANDAS 05 — GroupBy & Aggregation
================================================================================

THEORY:
  GroupBy follows the Split-Apply-Combine pattern:
    1. SPLIT   — divide DataFrame into groups by one or more columns
    2. APPLY   — run a function on each group independently
    3. COMBINE — collect results back into a single DataFrame

  In MLOps you use this for:
    - Computing per-class metrics (accuracy per label, loss per batch)
    - Feature statistics per segment (avg income per city)
    - Temporal aggregations (daily/weekly model performance)
    - Detecting data imbalance in label distributions

  Key aggregation functions: mean, sum, count, min, max, std, median,
  nunique, first, last, and custom functions via .agg() or .apply()
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(0)
n = 500

df = pd.DataFrame({
    "model":       np.random.choice(["modelA", "modelB", "modelC"], n),
    "env":         np.random.choice(["prod", "staging", "dev"], n, p=[0.5, 0.3, 0.2]),
    "region":      np.random.choice(["us-east", "us-west", "eu"], n),
    "latency_ms":  np.random.gamma(2, 50, n).round(1),
    "accuracy":    np.random.uniform(0.70, 0.99, n).round(4),
    "error_count": np.random.poisson(2, n),
    "requests":    np.random.randint(100, 10000, n),
    "date":        pd.date_range("2024-01-01", periods=n, freq="12h"),
})
print(f"Dataset: {df.shape}\n{df.head(3)}\n")

print("=" * 60)
print("BASIC GROUPBY")
print("=" * 60)

# Single column groupby
g = df.groupby("model")

print(f"Groups: {list(g.groups.keys())}")
print(f"Group sizes:\n{g.size()}\n")

# Single aggregation
print(f"Mean latency by model:\n{g['latency_ms'].mean().round(1)}\n")

# Multiple aggregations at once with .agg()
summary = g.agg(
    mean_latency  = ("latency_ms", "mean"),
    p95_latency   = ("latency_ms", lambda x: x.quantile(0.95)),
    mean_accuracy = ("accuracy", "mean"),
    total_errors  = ("error_count", "sum"),
    total_requests= ("requests", "sum"),
)
summary = summary.round(2)
print(f"Model summary:\n{summary}\n")

print("=" * 60)
print("MULTI-COLUMN GROUPBY")
print("=" * 60)

multi = df.groupby(["model", "env"]).agg(
    mean_latency = ("latency_ms", "mean"),
    count        = ("latency_ms", "count"),
).round(2)

print(f"Model × Environment:\n{multi}\n")

# Unstack — pivot inner group level to columns (like a cross-tab)
print(f"Unstacked (model rows, env columns):\n{multi['mean_latency'].unstack()}\n")

print("=" * 60)
print("TRANSFORM — keep original DataFrame shape")
print("=" * 60)

# transform() returns same-length result aligned with original rows
# Useful for: z-scoring within a group, computing group-mean features

# Add group mean as a feature (relative performance within model)
df["model_mean_latency"] = df.groupby("model")["latency_ms"].transform("mean")
df["latency_vs_model"]   = df["latency_ms"] - df["model_mean_latency"]

# Z-score within group
df["latency_zscore"] = df.groupby("model")["latency_ms"].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(f"Added transform columns:\n{df[['model','latency_ms','model_mean_latency','latency_zscore']].head(5)}\n")

print("=" * 60)
print("APPLY — run any function on each group")
print("=" * 60)

# More flexible than agg, but slower for large datasets
def model_report(grp):
    return pd.Series({
        "n":           len(grp),
        "error_rate":  grp["error_count"].sum() / grp["requests"].sum(),
        "sla_breach":  (grp["latency_ms"] > 150).mean(),   # % above SLA
        "top_region":  grp.groupby("region").size().idxmax(),
    })

report = df.groupby("model").apply(model_report, include_groups=False)
print(f"Custom model report:\n{report}\n")

print("=" * 60)
print("PIVOT TABLE — the Excel-style summary")
print("=" * 60)

pivot = df.pivot_table(
    values="latency_ms",
    index="model",
    columns="env",
    aggfunc="mean",
    margins=True,        # adds row/column totals
    margins_name="All",
).round(1)

print(f"Pivot: mean latency (model × env):\n{pivot}\n")

# With multiple values
pivot2 = df.pivot_table(
    values=["latency_ms", "accuracy"],
    index="model",
    columns="env",
    aggfunc="mean",
).round(3)

print(f"Multi-value pivot:\n{pivot2}\n")

print("=" * 60)
print("CROSSTAB — count frequency of combinations")
print("=" * 60)

ct = pd.crosstab(df["model"], df["env"], normalize="index") * 100
print(f"Model × env distribution (%):\n{ct.round(1)}\n")

print("=" * 60)
print("RESAMPLE — time-based groupby (very common in monitoring)")
print("=" * 60)

df_ts = df.set_index("date").sort_index()

# Daily averages
daily = df_ts["latency_ms"].resample("D").agg(["mean", "max", "count"]).round(1)
print(f"Daily latency stats (first 7 days):\n{daily.head(7)}\n")

# Weekly accuracy per model
weekly_acc = df_ts.groupby("model")["accuracy"].resample("W").mean().round(4)
print(f"Weekly accuracy per model (first few):\n{weekly_acc.head(9)}\n")

print("=" * 60)
print("PRACTICAL: COMPUTING CLASSIFICATION METRICS PER CLASS")
print("=" * 60)

# Simulate model predictions
np.random.seed(5)
predictions = pd.DataFrame({
    "true_label": np.random.choice(["cat", "dog", "bird"], 1000, p=[0.5, 0.3, 0.2]),
    "pred_label": np.random.choice(["cat", "dog", "bird"], 1000, p=[0.5, 0.3, 0.2]),
    "confidence": np.random.uniform(0.5, 1.0, 1000),
})

# Per-class accuracy
def class_accuracy(grp):
    return (grp["pred_label"] == grp["true_label"]).mean()

per_class = predictions.groupby("true_label").apply(class_accuracy, include_groups=False)
print(f"Per-class accuracy:\n{per_class.round(3)}\n")

# Confusion matrix via crosstab
confusion = pd.crosstab(
    predictions["true_label"],
    predictions["pred_label"],
    rownames=["Actual"],
    colnames=["Predicted"],
)
print(f"Confusion matrix:\n{confusion}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Using the monitoring df, compute per-region stats:
   - total requests
   - mean accuracy
   - error rate (error_count / requests)
   Sort by total requests descending.

2. Add a column 'high_error' = True where error_count > per-model median.
   Use groupby + transform.

3. Create a pivot table showing the COUNT (not mean) of rows per model × region.
   Which model-region combination has the most observations?

4. Resample the data daily and compute the 95th percentile of latency.
   Is there a day with unusually high p95?
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
reg = df.groupby("region").agg(
    total_requests = ("requests", "sum"),
    mean_accuracy  = ("accuracy", "mean"),
    error_rate     = ("error_count", lambda x: x.sum() / df.loc[x.index, "requests"].sum()),
).sort_values("total_requests", ascending=False).round(4)
print(f"1.\n{reg}\n")

# 2.
df["model_median_err"] = df.groupby("model")["error_count"].transform("median")
df["high_error"] = df["error_count"] > df["model_median_err"]
print(f"2. high_error counts:\n{df.groupby('model')['high_error'].sum()}\n")

# 3.
piv3 = pd.crosstab(df["model"], df["region"])
idx = piv3.stack().idxmax()
print(f"3. Most observations: {idx}  ({piv3.stack().max()} rows)\n")

# 4.
p95_daily = df.set_index("date").sort_index()["latency_ms"].resample("D").quantile(0.95).round(1)
max_day = p95_daily.idxmax()
print(f"4. Highest p95: {p95_daily.max()} ms on {max_day.date()}")
