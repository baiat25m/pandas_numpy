"""
================================================================================
NUMPY 02 — Indexing, Slicing, Boolean Masking
================================================================================

THEORY:
  Indexing in NumPy is zero-based like Python lists, but much more powerful.

  Three types:
  1. Basic indexing   → arr[0], arr[1:3]  — returns a VIEW (no copy)
  2. Boolean masking  → arr[arr > 5]      — returns a COPY
  3. Fancy indexing   → arr[[0, 2, 4]]    — returns a COPY

  VIEW vs COPY matters in MLOps:
    - A view shares memory with the original → mutations affect the source
    - Use .copy() explicitly when you need independence
================================================================================
"""

import numpy as np

print("=" * 60)
print("1D INDEXING")
print("=" * 60)

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

print(f"array:         {arr}")
print(f"arr[0]:        {arr[0]}")       # first element
print(f"arr[-1]:       {arr[-1]}")      # last element
print(f"arr[2:5]:      {arr[2:5]}")     # elements at index 2,3,4
print(f"arr[:4]:       {arr[:4]}")      # first 4
print(f"arr[6:]:       {arr[6:]}")      # from index 6 to end
print(f"arr[::2]:      {arr[::2]}")     # every 2nd element (step=2)
print(f"arr[::-1]:     {arr[::-1]}")    # reversed

print()
print("=" * 60)
print("2D INDEXING — row, col syntax")
print("=" * 60)

m = np.array([[1,  2,  3,  4],
              [5,  6,  7,  8],
              [9,  10, 11, 12]])

print(f"matrix:\n{m}\n")
print(f"m[0, 0]:       {m[0, 0]}")        # row 0, col 0 → 1
print(f"m[1, 2]:       {m[1, 2]}")        # row 1, col 2 → 7
print(f"m[-1, -1]:     {m[-1, -1]}")      # last row, last col → 12

print()
print(f"m[0]:          {m[0]}")            # entire row 0
print(f"m[:, 1]:       {m[:, 1]}")         # entire column 1
print(f"m[0:2, 1:3]:\n{m[0:2, 1:3]}")     # sub-matrix rows 0-1, cols 1-2

print()
print("=" * 60)
print("BOOLEAN MASKING — the most important pattern in data work")
print("=" * 60)

scores = np.array([45, 82, 91, 34, 76, 55, 88, 63])

# The condition creates a boolean array
mask = scores > 70
print(f"scores:          {scores}")
print(f"scores > 70:     {mask}")
print(f"filtered:        {scores[mask]}")     # or: scores[scores > 70]

# Compound conditions — use & (and), | (or), ~ (not)
# Note: must use & | ~, NOT Python's `and` `or` `not`
high_scores = scores[(scores > 60) & (scores < 90)]
print(f"between 60-90:   {high_scores}")

# Where — returns indices of True values
indices = np.where(scores > 70)
print(f"indices > 70:    {indices[0]}")

# np.where(condition, x, y) — ternary: x if True, y if False
graded = np.where(scores >= 60, "pass", "fail")
print(f"pass/fail:       {graded}")

print()
print("=" * 60)
print("FANCY INDEXING — select specific rows/cols by index list")
print("=" * 60)

data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90],
                 [11, 22, 33]])

# Select rows 0 and 2
print(f"rows [0,2]:\n{data[[0, 2]]}\n")

# Select specific elements (row-col pairs)
rows = [0, 1, 3]
cols = [2, 0, 1]
print(f"elements at (0,2),(1,0),(3,1): {data[rows, cols]}")

print()
print("=" * 60)
print("VIEW vs COPY — critical for avoiding silent bugs")
print("=" * 60)

original = np.array([1, 2, 3, 4, 5])

# Slicing → VIEW (shares memory)
view = original[1:4]
view[0] = 999
print(f"After modifying view[0]=999:")
print(f"  view:     {view}")
print(f"  original: {original}  ← CHANGED! view shares memory")

# Explicit copy → independent
original2 = np.array([1, 2, 3, 4, 5])
copy = original2[1:4].copy()
copy[0] = 999
print(f"\nAfter modifying copy[0]=999:")
print(f"  copy:      {copy}")
print(f"  original2: {original2}  ← unchanged")

print()
print("MLOps context: when splitting train/val/test sets, use .copy()")
print("to avoid accidental mutations bleeding between sets.")

print()
print("=" * 60)
print("MODIFYING VALUES with boolean masks")
print("=" * 60)

data = np.array([1.0, -0.5, 3.2, -1.1, 0.8, -2.0])
print(f"Original:   {data}")

# Set all negative values to 0 (ReLU activation!)
relu = data.copy()
relu[relu < 0] = 0
print(f"After ReLU: {relu}")

# Clip values to a range
clipped = np.clip(data, -1.0, 1.0)
print(f"Clipped [-1,1]: {clipped}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Given arr = np.arange(20), extract every 3rd element starting from index 1.

2. Given a 4x4 matrix (np.arange(16).reshape(4,4)), extract the 2x2
   bottom-right sub-matrix.

3. Given scores = np.array([55, 72, 89, 43, 95, 61]), count how many
   scores are above 70 using boolean masking (no loops!).

4. Given data = np.array([3.5, -1.2, 0.0, 7.8, -0.5, 4.1]),
   replace all values below 0 with the mean of the positive values.

5. What happens if you do: a = np.array([1,2,3]); b = a; b[0] = 99
   What is a[0]? Why? How do you fix it?
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
arr = np.arange(20)
print(f"1. {arr[1::3]}")

# 2.
m = np.arange(16).reshape(4, 4)
print(f"2.\n{m[2:, 2:]}")

# 3.
scores = np.array([55, 72, 89, 43, 95, 61])
print(f"3. {np.sum(scores > 70)} scores above 70")

# 4.
data = np.array([3.5, -1.2, 0.0, 7.8, -0.5, 4.1])
mean_pos = data[data > 0].mean()
result = data.copy()
result[result < 0] = mean_pos
print(f"4. {result}")

# 5.
a = np.array([1, 2, 3])
b = a            # b is a reference, NOT a copy
b[0] = 99
print(f"5. a[0] = {a[0]}  ← b = a makes b point to same object")
print("   Fix: b = a.copy()")
