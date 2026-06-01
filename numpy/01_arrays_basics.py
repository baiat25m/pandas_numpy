"""
================================================================================
NUMPY 01 — Arrays, dtypes, shapes
================================================================================

WHY NUMPY?
  - Python lists are slow: each element is a Python object with overhead.
  - NumPy arrays store data in contiguous memory as raw C types → 10-100x faster.
  - Every ML/data library (pandas, scikit-learn, PyTorch) speaks NumPy internally.

CORE CONCEPT: The ndarray
  - n-dimensional array with a fixed dtype (float32, int64, bool, etc.)
  - Shape describes dimensions: (3,) = 1D with 3 elements, (3,4) = 2D matrix
  - All elements must be the same type (unlike Python lists)
================================================================================
"""

import numpy as np

print("=" * 60)
print("CREATING ARRAYS")
print("=" * 60)

# From Python list
a = np.array([1, 2, 3, 4, 5])
print(f"From list:      {a}")
print(f"  type:         {type(a)}")
print(f"  dtype:        {a.dtype}")  # int64 on most systems
print(f"  shape:        {a.shape}")  # (5,) — 1D, 5 elements
print(f"  ndim:         {a.ndim}")  # 1
print(f"  size:         {a.size}")  # total elements = 5
print(f"  itemsize:     {a.itemsize}")  # bytes per element

print()

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D array:\n{matrix}")
print(f"  shape: {matrix.shape}")  # (2, 3) → 2 rows, 3 cols
print(f"  ndim:  {matrix.ndim}")  # 2

print()
print("=" * 60)
print("FACTORY FUNCTIONS — building arrays without listing values")
print("=" * 60)

# zeros / ones — most common for initializing
zeros = np.zeros((3, 4))  # 3x4 matrix of 0.0
ones = np.ones((2, 3))  # 2x3 matrix of 1.0
full = np.full((2, 2), 7.5)  # 2x2 matrix of 7.5
eye = np.eye(3)  # 3x3 identity matrix (1s on diagonal)

print(f"zeros (3,4):\n{zeros}\n")
print(f"ones (2,3):\n{ones}\n")
print(f"full (2,2) with 7.5:\n{full}\n")
print(f"identity (3,3):\n{eye}\n")

# Ranges — like Python range() but returns an array
arange = np.arange(0, 10, 2)  # start=0, stop=10, step=2
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced points from 0 to 1

print(f"arange(0,10,2):    {arange}")
print(f"linspace(0,1,5):   {linspace}")

print()
print("=" * 60)
print("DTYPES — data types matter for memory and performance")
print("=" * 60)

# Explicit dtype
float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 4 bytes each
float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 8 bytes each (default)
int8 = np.array([1, 2, 3], dtype=np.int8)  # 1 byte, range -128..127
boolean = np.array([True, False, True], dtype=bool)

print(f"float32: {float32}  dtype={float32.dtype}  itemsize={float32.itemsize}B")
print(f"float64: {float64}  dtype={float64.dtype}  itemsize={float64.itemsize}B")
print(f"int8:    {int8}      dtype={int8.dtype}     itemsize={int8.itemsize}B")
print(f"bool:    {boolean}  dtype={boolean.dtype}")

# MLOps context: model weights are often float32 to save memory/VRAM
# logs/IDs are int32 or int64, feature flags are bool

# Type conversion (casting)
x = np.array([1.7, 2.3, 3.9])
x_int = x.astype(int)  # truncates (doesn't round!)
print(f"\nCasting float→int: {x} → {x_int}  (truncates, not rounds!)")

print()
print("=" * 60)
print("SHAPE INSPECTION — what you'll check constantly")
print("=" * 60)

data = np.zeros((100, 28, 28, 3))  # batch of 100 RGB 28x28 images
print(f"Image batch shape:  {data.shape}")
print(f"  batch_size:       {data.shape[0]}")
print(f"  height:           {data.shape[1]}")
print(f"  width:            {data.shape[2]}")
print(f"  channels:         {data.shape[3]}")
print(f"  total elements:   {data.size:,}")
print(f"  memory (MB):      {data.nbytes / 1e6:.1f}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Try these yourself (scroll to ANSWERS below when done):

1. Create a 1D array of odd numbers from 1 to 19.

2. Create a 5x5 matrix filled with zeros, then set the diagonal to 1
   using np.eye() — but also try doing it by assignment.

3. Create an array [1, 2, 3, 4, 5] as float32, then compute its memory
   usage vs float64. How many bytes does each use?

4. What is the shape, ndim, and size of np.zeros((4, 3, 2))?

5. Create a linearly spaced array from -1 to 1 with 100 points
   (useful for creating synthetic features in tests).
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
ans1 = np.arange(1, 20, 2)
print(f"1. {ans1}")

# 2.
ans2 = np.eye(5)
print(f"2.\n{ans2}")

# 3.
arr_f32 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
arr_f64 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(f"3. float32: {arr_f32.nbytes}B   float64: {arr_f64.nbytes}B")

# 4.
arr4 = np.zeros((4, 3, 2))
print(f"4. shape={arr4.shape}  ndim={arr4.ndim}  size={arr4.size}")

# 5.
ans5 = np.linspace(-1, 1, 100)
print(f"5. {ans5[:5]} ... {ans5[-5:]}  (len={len(ans5)})")
