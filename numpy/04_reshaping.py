"""
================================================================================
NUMPY 04 — Reshaping, Stacking, Splitting
================================================================================

THEORY:
  In MLOps you constantly need to restructure arrays:
    - Flatten images for a dense layer: (batch, H, W, C) → (batch, H*W*C)
    - Add batch dimension:              (H, W) → (1, H, W)
    - Stack multiple arrays into a batch
    - Split a dataset into train/val/test

  KEY RULE: reshape never copies data (returns a view) as long as the
  total number of elements stays the same.
================================================================================
"""

import numpy as np

print("=" * 60)
print("RESHAPE")
print("=" * 60)

arr = np.arange(12)
print(f"Original: {arr}  shape={arr.shape}")

r1 = arr.reshape(3, 4)
r2 = arr.reshape(4, 3)
r3 = arr.reshape(2, 2, 3)

print(f"\nreshape(3,4):\n{r1}")
print(f"\nreshape(4,3):\n{r2}")
print(f"\nreshape(2,2,3):\n{r3}")

# Use -1 to let NumPy infer one dimension automatically
r4 = arr.reshape(3, -1)   # -1 → 12/3 = 4
r5 = arr.reshape(-1, 4)   # -1 → 12/4 = 3
print(f"\nreshape(3,-1): {r4.shape}  (12/3=4 inferred)")
print(f"reshape(-1,4): {r5.shape}  (12/4=3 inferred)")

# Flatten — collapse to 1D
flat = r1.flatten()      # returns a COPY
ravel = r1.ravel()       # returns a VIEW if possible (preferred)
print(f"\nflatten:  {flat}  (copy)")
print(f"ravel:    {ravel}  (view)")

print()
print("=" * 60)
print("ADDING/REMOVING DIMENSIONS — very common in ML pipelines")
print("=" * 60)

img = np.zeros((28, 28))        # single grayscale image
print(f"img shape:          {img.shape}")

# Add batch dimension at front
batch = img[np.newaxis, :]      # or: np.expand_dims(img, axis=0)
print(f"add batch dim:      {batch.shape}")

# Add channel dimension at end (for convolutional layers)
img_with_chan = img[:, :, np.newaxis]   # or: np.expand_dims(img, axis=-1)
print(f"add channel dim:    {img_with_chan.shape}")

# Remove dimensions of size 1 (squeeze)
squeezed = batch.squeeze()
print(f"squeeze batch dim:  {squeezed.shape}")

# Practical: convert (H, W) grayscale to (1, H, W, 1) for Keras
keras_ready = img[np.newaxis, :, :, np.newaxis]
print(f"Keras batch input:  {keras_ready.shape}")

print()
print("=" * 60)
print("TRANSPOSE")
print("=" * 60)

m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"original shape: {m.shape}")
print(f"transposed shape: {m.T.shape}")
print(f"m.T:\n{m.T}")

# 3D transpose (e.g., converting NHWC ↔ NCHW image formats)
tensor = np.zeros((32, 224, 224, 3))   # batch, H, W, channels (TensorFlow/Keras)
nchw = tensor.transpose(0, 3, 1, 2)   # → batch, channels, H, W (PyTorch)
print(f"\nNHWC {tensor.shape} → NCHW {nchw.shape}")

print()
print("=" * 60)
print("STACKING ARRAYS")
print("=" * 60)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

vstack = np.vstack([a, b, c])     # vertical stack → (3, 3) matrix
hstack = np.hstack([a, b, c])     # horizontal stack → (9,) 1D
stack0 = np.stack([a, b, c], axis=0)   # new axis at position 0 → (3, 3)
stack1 = np.stack([a, b, c], axis=1)   # new axis at position 1 → (3, 3)

print(f"vstack:\n{vstack}  shape={vstack.shape}")
print(f"hstack:  {hstack}  shape={hstack.shape}")
print(f"stack(axis=0):\n{stack0}  shape={stack0.shape}")
print(f"stack(axis=1):\n{stack1}  shape={stack1.shape}")

# Batching images: stack individual images into a batch
images = [np.random.rand(28, 28) for _ in range(4)]   # 4 separate images
batch = np.stack(images, axis=0)   # → (4, 28, 28)
print(f"\n4 images (28,28) stacked → batch {batch.shape}")

print()
print("=" * 60)
print("SPLITTING ARRAYS")
print("=" * 60)

data = np.arange(24).reshape(6, 4)
print(f"data:\n{data}\n")

# Split into 3 equal pieces along axis=0 (rows)
parts = np.split(data, 3, axis=0)
for i, p in enumerate(parts):
    print(f"part {i}: shape={p.shape}\n{p}")

print()
# array_split allows unequal splits
pieces = np.array_split(np.arange(10), 3)
print(f"array_split [0..9] into 3: {[p.tolist() for p in pieces]}")

# Train/val/test split (practical MLOps pattern)
np.random.seed(42)
N = 1000
X = np.random.randn(N, 10)
y = np.random.randint(0, 2, N)

indices = np.random.permutation(N)    # shuffle indices
train_end = int(0.7 * N)
val_end   = int(0.85 * N)

X_train = X[indices[:train_end]]
X_val   = X[indices[train_end:val_end]]
X_test  = X[indices[val_end:]]
y_train = y[indices[:train_end]]
y_val   = y[indices[train_end:val_end]]
y_test  = y[indices[val_end:]]

print(f"Train/Val/Test split (70/15/15):")
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. You have 500 grayscale images each of size 32x32 stored as a list.
   Stack them into a numpy array and reshape to (500, 1024).

2. Convert a (10, 3) feature matrix to (10, 3, 1) — add a trailing dimension.
   Then squeeze it back to (10, 3).

3. Given arr = np.arange(100), split it into train (60%), val (20%), test (20%)
   without shuffling.

4. You have two matrices A shape (50, 5) and B shape (50, 3). Stack them
   horizontally to form a (50, 8) feature matrix.
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
imgs = [np.random.rand(32, 32) for _ in range(500)]
batch = np.stack(imgs, axis=0)
flat  = batch.reshape(500, -1)
print(f"1. stacked: {batch.shape} → reshaped: {flat.shape}")

# 2.
mat = np.random.randn(10, 3)
expanded = mat[:, :, np.newaxis]
back = expanded.squeeze(-1)
print(f"2. (10,3) → {expanded.shape} → squeezed: {back.shape}")

# 3.
arr = np.arange(100)
train, val, test = arr[:60], arr[60:80], arr[80:]
print(f"3. train: {train.shape}, val: {val.shape}, test: {test.shape}")

# 4.
A = np.random.randn(50, 5)
B = np.random.randn(50, 3)
combined = np.hstack([A, B])
print(f"4. hstack (50,5)+(50,3) → {combined.shape}")
