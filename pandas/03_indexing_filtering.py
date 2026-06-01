"""
================================================================================
PANDAS 03 — Indexing & Filtering: loc, iloc, boolean filters
================================================================================

THEORY:
  Pandas has two primary indexers:
    .loc[row_label, col_label]   — label-based (use index names / column names)
    .iloc[row_position, col_pos] — position-based (pure integers, like NumPy)

  When to use which:
    .loc  → when you care about what the label IS (e.g., user_id="u123")
    .iloc → when you care about where it is (e.g., first 100 rows)

  CHAINED INDEXING PITFALL:
    df["col"][mask] = value  ← may or may not modify original (undefined!)
    ALWAYS use:  df.loc[mask, "col"] = value
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(0)
n = 20

df = pd.DataFrame({
    "user_id":    [f"u{i:03d}" for i in range(1, n+1)],
    "age":        np.random.randint(20, 60, n),
    "score":      np.random.uniform(0, 100, n).round(1),
    "tier":       np.random.choice(["bronze", "silver", "gold"], n),
    "active":     np.random.choice([True, False], n, p=[0.7, 0.3]),
    "purchases":  np.random.randint(0, 50, n),
})
df = df.set_index("user_id")
print(f"DataFrame (indexed by user_id):\n{df.head(8)}\n")

print("=" * 60)
print("LOC — label-based indexing")
print("=" * 60)

# Single row by label
print(f"df.loc['u003']:\n{df.loc['u003']}\n")

# Multiple rows
print(f"df.loc[['u001', 'u005', 'u010']]:\n{df.loc[['u001', 'u005', 'u010']]}\n")

# Row + column
print(f"df.loc['u003', 'score']:  {df.loc['u003', 'score']}")
print(f"df.loc['u003', ['age', 'score']]:\n{df.loc['u003', ['age', 'score']]}\n")

# Slice of rows (label-based, INCLUSIVE of both ends unlike Python)
print(f"df.loc['u003':'u006', 'age':'score']:\n{df.loc['u003':'u006', 'age':'score']}\n")

print("=" * 60)
print("ILOC — position-based indexing")
print("=" * 60)

print(f"df.iloc[0]:            first row\n{df.iloc[0]}\n")
print(f"df.iloc[-1]:           last row\n{df.iloc[-1]}\n")
print(f"df.iloc[0:3]:          first 3 rows\n{df.iloc[0:3]}\n")
print(f"df.iloc[[0, 2, 4], 0:2]:\n{df.iloc[[0, 2, 4], 0:2]}\n")

print("=" * 60)
print("BOOLEAN FILTERING — the workhorse of data selection")
print("=" * 60)

# Simple filter
gold_users = df[df["tier"] == "gold"]
print(f"Gold users ({len(gold_users)}):\n{gold_users.head(3)}\n")

# Compound filters — always use & | ~ (not and/or/not)
active_gold = df[(df["tier"] == "gold") & (df["active"] == True)]
print(f"Active gold users: {len(active_gold)}\n")

high_value = df[(df["score"] > 70) | (df["purchases"] > 40)]
print(f"Score>70 or purchases>40: {len(high_value)}\n")

young_active = df[~(df["age"] > 40) & df["active"]]
print(f"Active and not >40yo: {len(young_active)}\n")

# isin — check membership in a list
bronze_silver = df[df["tier"].isin(["bronze", "silver"])]
print(f"Bronze or silver: {len(bronze_silver)}\n")

# between — range check (inclusive)
mid_score = df[df["score"].between(40, 70)]
print(f"Score between 40-70: {len(mid_score)}\n")

# query() — SQL-like string syntax (readable for complex conditions)
result = df.query("age > 30 and score > 60 and tier == 'gold'")
print(f"query syntax result: {len(result)} rows\n")

print("=" * 60)
print("SAFE ASSIGNMENT WITH LOC")
print("=" * 60)

df2 = df.copy()

# WRONG — SettingWithCopyWarning risk
# df[df["score"] > 80]["score"] = 100   # never do this

# CORRECT — always use .loc for assignment
df2.loc[df2["score"] > 80, "score"] = 100.0
print(f"Scores set to 100 where >80:")
print(f"  Count of 100.0: {(df2['score'] == 100.0).sum()}")

# Add a new computed column conditionally
df2["segment"] = "low"
df2.loc[df2["score"] >= 40, "segment"] = "mid"
df2.loc[df2["score"] >= 70, "segment"] = "high"
print(f"\nSegment distribution:\n{df2['segment'].value_counts()}\n")

print("=" * 60)
print("AT / IAT — fast single-value access")
print("=" * 60)

# .at and .iat are faster than .loc/.iloc for single cells
print(f"df.at['u001', 'score']:  {df.at['u001', 'score']}")    # label
print(f"df.iat[0, 1]:            {df.iat[0, 1]}")               # position

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Using df (with user_id index), select all users where:
   - age is between 25 and 45 (inclusive)
   - score is above 50
   - tier is NOT bronze
   Print how many users match.

2. Use .loc to update the 'tier' of all users with purchases > 45 to 'platinum'.
   Then verify by printing the count of 'platinum' users.

3. Using .iloc, extract a "corner" of the DataFrame:
   - last 5 rows
   - first 3 columns
   Print the result.

4. Use the .query() method to find users where:
   (score > 80 OR purchases > 30) AND active == True
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
mask = df["age"].between(25, 45) & (df["score"] > 50) & (df["tier"] != "bronze")
print(f"1. Users matching: {mask.sum()}\n")

# 2.
df3 = df.copy()
df3.loc[df3["purchases"] > 45, "tier"] = "platinum"
print(f"2. Platinum users: {(df3['tier'] == 'platinum').sum()}\n")

# 3.
corner = df.iloc[-5:, :3]
print(f"3. Corner (last 5 rows, first 3 cols):\n{corner}\n")

# 4.
result4 = df.query("(score > 80 or purchases > 30) and active == True")
print(f"4. query result: {len(result4)} users")
