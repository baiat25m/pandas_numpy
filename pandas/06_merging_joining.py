"""
================================================================================
PANDAS 06 — Merging, Joining & Concatenating
================================================================================

THEORY:
  Combining DataFrames is fundamental in MLOps:
    - Join features from multiple tables (user data + events + metadata)
    - Concat train batches from multiple files
    - Merge model predictions with ground truth labels

  CONCAT  — stacks DataFrames vertically (more rows) or horizontally (more cols)
  MERGE   — SQL-style JOIN on key column(s). Types:
              inner  → only rows with match in BOTH tables
              left   → all rows from left + matches from right (NaN if no match)
              right  → all rows from right + matches from left
              outer  → all rows from both (NaN where no match)
  JOIN    — shorthand for merge on index

  CONCAT is for stacking. MERGE is for enriching (adding info from another table).
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

print("=" * 60)
print("SETUP — three related tables")
print("=" * 60)

# Table 1: users
users = pd.DataFrame({
    "user_id":  [1, 2, 3, 4, 5],
    "name":     ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "city":     ["NYC", "LA", "NYC", "Chicago", "LA"],
    "tier":     ["gold", "silver", "bronze", "gold", "silver"],
})

# Table 2: model predictions for users (not all users have predictions)
predictions = pd.DataFrame({
    "user_id":   [1, 2, 3, 6, 7],    # user 4,5 missing; 6,7 unknown
    "pred_score": [0.9, 0.4, 0.7, 0.6, 0.8],
    "model_version": ["v2", "v2", "v1", "v2", "v2"],
})

# Table 3: user events log
events = pd.DataFrame({
    "user_id":   [1, 1, 2, 2, 3, 4, 4, 4],
    "event":     ["click", "purchase", "click", "click", "purchase", "view", "purchase", "click"],
    "amount":    [0, 50, 0, 0, 120, 0, 75, 0],
})

print(f"Users:\n{users}\n")
print(f"Predictions:\n{predictions}\n")
print(f"Events:\n{events}\n")

print("=" * 60)
print("MERGE — SQL-style joins")
print("=" * 60)

# Inner join — only rows that exist in BOTH
inner = pd.merge(users, predictions, on="user_id", how="inner")
print(f"INNER join (user_id in both): {len(inner)} rows\n{inner}\n")

# Left join — all users, predictions where available
left = pd.merge(users, predictions, on="user_id", how="left")
print(f"LEFT join (all users): {len(left)} rows\n{left}\n")
print(f"  Users without predictions (NaN): {left['pred_score'].isnull().sum()}")

# Right join — all predictions, users where available
right = pd.merge(users, predictions, on="user_id", how="right")
print(f"\nRIGHT join (all predictions): {len(right)} rows\n{right}\n")

# Outer join — everything
outer = pd.merge(users, predictions, on="user_id", how="outer")
print(f"OUTER join (all rows): {len(outer)} rows\n{outer}\n")

print("=" * 60)
print("MERGE WITH DIFFERENT COLUMN NAMES")
print("=" * 60)

# Sometimes the key has a different name in each table
users2     = users.rename(columns={"user_id": "uid"})
preds2     = predictions.rename(columns={"user_id": "customer_id"})

merged = pd.merge(users2, preds2, left_on="uid", right_on="customer_id", how="inner")
print(f"Different key names:\n{merged}\n")

# Merge on multiple keys (composite key)
df_a = pd.DataFrame({"model": ["A","A","B"], "env": ["prod","dev","prod"], "v": [1,2,3]})
df_b = pd.DataFrame({"model": ["A","B","B"], "env": ["prod","prod","dev"], "score": [0.9,0.8,0.7]})
multi_merge = pd.merge(df_a, df_b, on=["model", "env"], how="outer")
print(f"Multi-key merge:\n{multi_merge}\n")

print("=" * 60)
print("HANDLING DUPLICATE COLUMN NAMES AFTER MERGE")
print("=" * 60)

# When both tables have same non-key column
df_left  = pd.DataFrame({"id": [1,2], "score": [0.9, 0.8], "ts": ["2024-01", "2024-02"]})
df_right = pd.DataFrame({"id": [1,2], "score": [0.85, 0.75]})

merged_dup = pd.merge(df_left, df_right, on="id", suffixes=("_train", "_test"))
print(f"Duplicate col resolved:\n{merged_dup}\n")

print("=" * 60)
print("CONCAT — stacking DataFrames")
print("=" * 60)

# Vertical concat (more rows) — e.g., combining batches
batch1 = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
batch2 = pd.DataFrame({"x": [3, 4], "y": [30, 40]})
batch3 = pd.DataFrame({"x": [5, 6], "y": [50, 60]})

combined = pd.concat([batch1, batch2, batch3], ignore_index=True)
print(f"Vertical concat:\n{combined}\n")

# With keys — adds hierarchical index to track source
combined_keyed = pd.concat([batch1, batch2, batch3], keys=["b1", "b2", "b3"])
print(f"Concat with keys:\n{combined_keyed}\n")

# Horizontal concat (more columns) — axis=1
df1 = pd.DataFrame({"A": [1, 2, 3]})
df2 = pd.DataFrame({"B": [4, 5, 6]})
df3 = pd.DataFrame({"C": [7, 8, 9]})
horiz = pd.concat([df1, df2, df3], axis=1)
print(f"Horizontal concat:\n{horiz}\n")

# CAUTION: concat with different columns → NaN filled for missing cols
df_miss = pd.DataFrame({"A": [7, 8], "C": [70, 80]})   # no column B
with_missing = pd.concat([batch1, df_miss], ignore_index=True)
print(f"Concat with missing cols:\n{with_missing}\n")

print("=" * 60)
print("PRACTICAL: BUILDING A FEATURE TABLE")
print("=" * 60)

# Aggregate events per user → feature table
event_features = events.groupby("user_id").agg(
    n_events     = ("event", "count"),
    n_purchases  = ("event", lambda x: (x == "purchase").sum()),
    total_spend  = ("amount", "sum"),
).reset_index()

# Merge with user data
user_features = pd.merge(users, event_features, on="user_id", how="left")
user_features[["n_events", "n_purchases", "total_spend"]] = \
    user_features[["n_events", "n_purchases", "total_spend"]].fillna(0)

# Merge with latest prediction
latest_pred = predictions.groupby("user_id")["pred_score"].last().reset_index()
user_features = pd.merge(user_features, latest_pred, on="user_id", how="left")

print(f"Final feature table:\n{user_features}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Using users and predictions, do a left join. Then add a column
   'has_prediction' = True/False. What % of users have predictions?

2. Create two DataFrames representing train and test sets with columns
   [id, feature1, feature2, label]. Concatenate them vertically,
   adding a 'split' column ('train' or 'test') to distinguish them.

3. You have model_a_preds and model_b_preds DataFrames each with
   (user_id, pred_score). Merge them to compare. Add a column
   'winner' = 'a' if model_a wins, 'b' if model_b wins.

4. Given a merge that produces unexpected rows, how would you diagnose
   whether it's a 1:1, 1:many, or many:many join?
   Hint: use .value_counts() and check for indicator=True.
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
left_join = pd.merge(users, predictions, on="user_id", how="left")
left_join["has_prediction"] = left_join["pred_score"].notna()
pct = left_join["has_prediction"].mean() * 100
print(f"1. {pct:.0f}% of users have predictions\n")

# 2.
np.random.seed(0)
train_df = pd.DataFrame({"id": range(100), "feature1": np.random.rand(100),
                         "feature2": np.random.rand(100), "label": np.random.randint(0,2,100)})
test_df  = pd.DataFrame({"id": range(100, 120), "feature1": np.random.rand(20),
                         "feature2": np.random.rand(20), "label": np.random.randint(0,2,20)})
all_data = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")],
                     ignore_index=True)
print(f"2. Combined shape: {all_data.shape}, splits: {all_data['split'].value_counts().to_dict()}\n")

# 3.
model_a = pd.DataFrame({"user_id": [1,2,3,4,5], "pred_score_a": [0.9,0.4,0.7,0.6,0.5]})
model_b = pd.DataFrame({"user_id": [1,2,3,4,5], "pred_score_b": [0.8,0.6,0.5,0.7,0.55]})
compared = pd.merge(model_a, model_b, on="user_id")
compared["winner"] = np.where(compared["pred_score_a"] >= compared["pred_score_b"], "a", "b")
print(f"3.\n{compared}\n")

# 4.
print("4. Diagnose join cardinality:")
print("   users['user_id'].value_counts()     → should be all 1s for 1:1")
print("   events['user_id'].value_counts()    → many → 1:many join expected")
print("   Use pd.merge(..., indicator=True) to see _merge column: left_only/right_only/both")
indicator_example = pd.merge(users, predictions, on="user_id", how="outer", indicator=True)
print(f"   indicator result:\n{indicator_example['_merge'].value_counts()}")
