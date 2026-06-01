"""
================================================================================
NUMPY 05 — Statistics, Random Numbers & Distributions
================================================================================

THEORY:
  Statistics in NumPy form the backbone of:
    - Data profiling (understanding your dataset before training)
    - Feature engineering (mean, std, percentiles)
    - Reproducible experiments (seeding random number generators)
    - Synthetic data generation for testing pipelines

  Always set np.random.seed() at the top of experiments for reproducibility.
================================================================================
"""

import numpy as np

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

# Simulate model inference latencies in milliseconds
np.random.seed(42)
latencies = np.random.gamma(shape=2.0, scale=50, size=1000)  # right-skewed

print(f"n samples:    {len(latencies)}")
print(f"mean:         {np.mean(latencies):.2f} ms")
print(f"median:       {np.median(latencies):.2f} ms")
print(f"std:          {np.std(latencies):.2f} ms")
print(f"variance:     {np.var(latencies):.2f}")
print(f"min:          {np.min(latencies):.2f} ms")
print(f"max:          {np.max(latencies):.2f} ms")
print(f"range:        {np.ptp(latencies):.2f} ms")   # peak-to-peak = max - min

# Percentiles / quantiles
p25, p50, p75, p95, p99 = np.percentile(latencies, [25, 50, 75, 95, 99])
print(f"\nPercentiles:")
print(f"  p25:  {p25:.1f} ms")
print(f"  p50:  {p50:.1f} ms  (median)")
print(f"  p75:  {p75:.1f} ms")
print(f"  p95:  {p95:.1f} ms  ← SLA target in MLOps")
print(f"  p99:  {p99:.1f} ms  ← tail latency")

# IQR (Interquartile Range) — used for outlier detection
iqr = p75 - p25
lower_fence = p25 - 1.5 * iqr
upper_fence = p75 + 1.5 * iqr
outliers = latencies[(latencies < lower_fence) | (latencies > upper_fence)]
print(f"\nIQR: {iqr:.1f}  fences: [{lower_fence:.1f}, {upper_fence:.1f}]")
print(f"Outliers: {len(outliers)} ({100*len(outliers)/len(latencies):.1f}%)")

print()
print("=" * 60)
print("RANDOM NUMBER GENERATION")
print("=" * 60)

rng = np.random.default_rng(seed=42)  # modern API (preferred over np.random.seed)

# Uniform: values between [low, high)
uniform = rng.uniform(low=0.0, high=1.0, size=5)
print(f"uniform [0,1):   {uniform.round(3)}")

# Normal (Gaussian): mean=0, std=1
normal = rng.normal(loc=0.0, scale=1.0, size=5)
print(f"normal(0,1):     {normal.round(3)}")

# Integers
integers = rng.integers(low=0, high=10, size=8)
print(f"integers [0,10): {integers}")

# Bernoulli via binomial
bernoulli = rng.binomial(n=1, p=0.3, size=10)
print(f"bernoulli(p=0.3):{bernoulli}")

# Choice with/without replacement
population = np.arange(100)
sample     = rng.choice(population, size=10, replace=False)
print(f"sample 10 from 100: {sorted(sample)}")

# Shuffle in-place
arr = np.arange(10)
rng.shuffle(arr)
print(f"shuffled: {arr}")

print()
print("=" * 60)
print("COMMON DISTRIBUTIONS IN ML")
print("=" * 60)

# Standard Normal — initial weights in neural networks
weights_init = rng.normal(0, 0.01, size=(64, 64))
print(f"Xavier-like init:  mean={weights_init.mean():.4f}  std={weights_init.std():.4f}")

# Kaiming / He initialization for ReLU networks
fan_in = 64
he_init = rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, 32))
print(f"He init:           mean={he_init.mean():.4f}  std={he_init.std():.4f}")

# Dropout mask
keep_prob = 0.8
mask = rng.uniform(0, 1, size=(32, 128)) < keep_prob
dropped = mask.astype(float) / keep_prob   # inverted dropout
print(f"Dropout mask:      {mask.sum()/(32*128)*100:.1f}% kept (target {keep_prob*100:.0f}%)")

print()
print("=" * 60)
print("CORRELATION & COVARIANCE")
print("=" * 60)

# Generate correlated features
n = 500
x1 = rng.normal(0, 1, n)
x2 = 0.8 * x1 + 0.2 * rng.normal(0, 1, n)   # x2 is highly correlated with x1
x3 = rng.normal(0, 1, n)                       # x3 is independent

features = np.column_stack([x1, x2, x3])

corr = np.corrcoef(features.T)   # expects shape (n_features, n_samples)
print(f"Correlation matrix:")
print(corr.round(2))
print(f"\ncorr(x1,x2) = {corr[0,1]:.2f}  ← expected ~0.8")
print(f"corr(x1,x3) = {corr[0,2]:.2f}  ← expected ~0.0")

print()
print("=" * 60)
print("HISTOGRAM / DISTRIBUTION ANALYSIS (without plotting)")
print("=" * 60)

data = rng.normal(5, 2, size=1000)
counts, bin_edges = np.histogram(data, bins=10)

print("Histogram bins:")
for i in range(len(counts)):
    bar = "#" * (counts[i] // 10)
    print(f"  [{bin_edges[i]:5.1f}, {bin_edges[i+1]:5.1f}): {counts[i]:4d} {bar}")

print()
print("=" * 60)
print("PRACTICAL: DETECTING DATA DRIFT")
print("=" * 60)

# Production data drift scenario
rng2 = np.random.default_rng(0)

# Reference distribution (training data)
train_feature = rng2.normal(loc=0.5, scale=0.1, size=10000)

# Production data — mean has shifted (drift!)
prod_feature = rng2.normal(loc=0.65, scale=0.12, size=1000)

train_mean, train_std = train_feature.mean(), train_feature.std()
prod_mean,  prod_std  = prod_feature.mean(),  prod_feature.std()

mean_shift = abs(prod_mean - train_mean) / train_std  # in units of std

print(f"Train:      mean={train_mean:.3f}  std={train_std:.3f}")
print(f"Production: mean={prod_mean:.3f}  std={prod_std:.3f}")
print(f"Mean shift: {mean_shift:.2f} standard deviations")
print(f"Drift alert: {'YES ⚠' if mean_shift > 0.5 else 'No'}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Generate 10,000 values from a normal distribution (mean=100, std=15)
   representing IQ scores. Compute the % of values above 130.

2. Using the IQR method, count outliers in:
   np.array([2,3,3,4,5,5,6,7,8,8,9,100,-50])

3. Generate a reproducible train/test split of 1000 samples:
   - Set seed=99
   - Shuffle indices
   - Split 80/20
   - Verify: re-run with same seed and confirm splits are identical.

4. Compute the rolling mean of a 1D array using only numpy operations
   (no pandas). Use a window of 3.
   arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
iq = np.random.default_rng(0).normal(100, 15, 10000)
pct_above_130 = (iq > 130).mean() * 100
print(f"1. {pct_above_130:.2f}% above 130 (expect ~2.28%)")

# 2.
outlier_arr = np.array([2,3,3,4,5,5,6,7,8,8,9,100,-50])
q1, q3 = np.percentile(outlier_arr, [25, 75])
iqr2 = q3 - q1
low, high = q1 - 1.5*iqr2, q3 + 1.5*iqr2
print(f"2. outliers: {outlier_arr[(outlier_arr < low) | (outlier_arr > high)]}")

# 3.
def make_split(seed=99, n=1000):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:800], idx[800:]

tr1, te1 = make_split()
tr2, te2 = make_split()
print(f"3. Reproducible: {np.array_equal(tr1, tr2)}")  # True

# 4.
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
w = 3
rolling = np.array([arr[i:i+w].mean() for i in range(len(arr)-w+1)])
print(f"4. rolling mean (window=3): {rolling}")
