"""
================================================================================
NUMPY 03 — Math Operations & Broadcasting
================================================================================

THEORY:
  NumPy operates element-wise by default — no loops needed.

  Broadcasting: when arrays have different shapes, NumPy automatically
  "stretches" the smaller one to match — if the rules allow it.

  Broadcasting rules (apply right-to-left on shape tuples):
    1. Dimensions with size 1 can be stretched to match the other.
    2. Missing dimensions (on the left) are treated as size 1.

  Example: (3, 4) + (4,) → (3, 4) [the (4,) is broadcast across rows]

  This is essential for:
    - Normalizing features: (X - mean) / std
    - Adding bias terms in neural networks
    - Computing pairwise distances
================================================================================
"""

import numpy as np

print("=" * 60)
print("ELEMENT-WISE ARITHMETIC")
print("=" * 60)

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a:       {a}")
print(f"b:       {b}")
print(f"a + b:   {a + b}")
print(f"a - b:   {a - b}")
print(f"a * b:   {a * b}")
print(f"b / a:   {b / a}")
print(f"a ** 2:  {a ** 2}")
print(f"b % 3:   {b % 3}")   # modulo

# Scalar operations (scalar is broadcast to all elements)
print(f"\na + 100: {a + 100}")
print(f"a * 2.5: {a * 2.5}")

print()
print("=" * 60)
print("UNIVERSAL FUNCTIONS (ufuncs) — vectorized math")
print("=" * 60)

x = np.array([0, 0.5, 1.0, 1.5, 2.0])

print(f"x:            {x}")
print(f"np.sqrt(x):   {np.sqrt(x)}")
print(f"np.exp(x):    {np.exp(x).round(3)}")
print(f"np.log(x+1):  {np.log(x + 1).round(3)}")  # log(0) = -inf, add 1
print(f"np.abs(x-1):  {np.abs(x - 1)}")
print(f"np.sin(x):    {np.sin(x).round(3)}")
print(f"np.ceil(x):   {np.ceil(x)}")
print(f"np.floor(x):  {np.floor(x)}")
print(f"np.round(x):  {np.round(x)}")

print()
print("=" * 60)
print("BROADCASTING — the superpower")
print("=" * 60)

# Example 1: add a 1D array to each row of a 2D matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
bias = np.array([10, 20, 30])   # shape (3,)

result = matrix + bias           # bias broadcast across each row
print(f"matrix:\n{matrix}")
print(f"bias: {bias}  shape={bias.shape}")
print(f"matrix + bias:\n{result}\n")

# Example 2: column vector broadcast across columns
col_scale = np.array([[100],     # shape (3, 1)
                       [200],
                       [300]])
result2 = matrix * col_scale
print(f"col_scale:\n{col_scale}  shape={col_scale.shape}")
print(f"matrix * col_scale:\n{result2}\n")

# MLOps: Z-score normalization (standardization)
features = np.array([[2.0, 100.0, 0.5],
                     [3.0, 150.0, 0.7],
                     [1.0,  80.0, 0.3],
                     [4.0, 200.0, 0.9]])

mean = features.mean(axis=0)    # mean of each COLUMN (feature)
std  = features.std(axis=0)     # std of each COLUMN

normalized = (features - mean) / std   # broadcasting handles shape mismatch!

print(f"Raw features:\n{features}")
print(f"mean: {mean}")
print(f"std:  {std.round(3)}")
print(f"Normalized (z-score):\n{normalized.round(3)}")
print(f"Verify mean≈0: {normalized.mean(axis=0).round(10)}")
print(f"Verify std≈1:  {normalized.std(axis=0).round(10)}")

print()
print("=" * 60)
print("AGGREGATION FUNCTIONS — axis matters!")
print("=" * 60)

m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"matrix:\n{m}\n")
print(f"np.sum(m):          {np.sum(m)}         ← all elements")
print(f"np.sum(m, axis=0):  {np.sum(m, axis=0)}  ← sum each COLUMN (collapse rows)")
print(f"np.sum(m, axis=1):  {np.sum(m, axis=1)}  ← sum each ROW (collapse cols)")
print()
print(f"np.mean(m, axis=0): {np.mean(m, axis=0)}")
print(f"np.max(m, axis=1):  {np.max(m, axis=1)}")
print(f"np.argmax(m, axis=1): {np.argmax(m, axis=1)}  ← index of max per row")
print(f"np.cumsum(m[0]):    {np.cumsum(m[0])}")

print()
print("Memory aid: axis=0 collapses ROWS → result has shape of a row")
print("            axis=1 collapses COLS → result has shape of a column")

print()
print("=" * 60)
print("DOT PRODUCT & MATRIX MULTIPLICATION")
print("=" * 60)

# 1D dot product (inner product)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"v1 · v2 = {np.dot(v1, v2)}")   # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication — (m,k) @ (k,n) → (m,n)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A @ B:\n{A @ B}")               # preferred syntax (Python 3.5+)
print(f"np.dot(A,B):\n{np.dot(A, B)}")  # same result

# MLOps: linear layer output = X @ W + b
batch  = np.random.randn(32, 10)   # 32 samples, 10 features
W      = np.random.randn(10, 5)    # weight matrix
b      = np.zeros(5)               # bias
output = batch @ W + b
print(f"\nLinear layer: {batch.shape} @ {W.shape} + {b.shape} → {output.shape}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Compute the softmax of [2.0, 1.0, 0.1] using only numpy operations.
   softmax(x) = exp(x) / sum(exp(x))

2. Given a feature matrix of shape (100, 5), normalize each feature
   to be in range [0, 1] using min-max scaling:
   x_scaled = (x - x_min) / (x_max - x_min)
   Apply column-wise without any loops.

3. Given predictions = np.array([0.9, 0.2, 0.8, 0.3]) and
   labels = np.array([1, 0, 1, 1]), compute Binary Cross-Entropy:
   BCE = -mean(y * log(p) + (1-y) * log(1-p))

4. Matrix A is (4, 3) and matrix B is (3, 2). What is the shape of A @ B?
   Compute it with random arrays and verify.
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
x = np.array([2.0, 1.0, 0.1])
softmax = np.exp(x) / np.sum(np.exp(x))
print(f"1. softmax: {softmax.round(4)}  (sums to {softmax.sum():.1f})")

# 2.
np.random.seed(42)
features = np.random.randn(100, 5) * 10
x_min = features.min(axis=0)
x_max = features.max(axis=0)
scaled = (features - x_min) / (x_max - x_min)
print(f"2. min after scaling: {scaled.min(axis=0).round(3)}")
print(f"   max after scaling: {scaled.max(axis=0).round(3)}")

# 3.
preds  = np.array([0.9, 0.2, 0.8, 0.3])
labels = np.array([1, 0, 1, 1])
eps = 1e-10   # avoid log(0)
bce = -np.mean(labels * np.log(preds + eps) + (1 - labels) * np.log(1 - preds + eps))
print(f"3. BCE loss: {bce:.4f}")

# 4.
A = np.random.randn(4, 3)
B = np.random.randn(3, 2)
C = A @ B
print(f"4. (4,3) @ (3,2) → {C.shape}")
