"""
================================================================================
PANDAS 07 — Time Series: Datetime, Resample, Rolling
================================================================================

THEORY:
  MLOps is full of time-series data:
    - Model prediction logs (timestamps, latency, accuracy)
    - Training runs (loss over epochs, over time)
    - Infrastructure metrics (CPU, memory, request rate)
    - Feature drift over time

  Key operations:
    pd.to_datetime()    — parse strings/ints to datetime
    .dt accessor        — extract year, month, day, hour, weekday...
    .resample()         — group by time period (like groupby for time)
    .rolling()          — sliding window aggregations
    .shift()            — lag/lead for time-difference features
    .diff()             — difference between consecutive rows
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

print("=" * 60)
print("CREATING DATETIME SERIES")
print("=" * 60)

# Parse from strings
dates_str = pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-05 14:30:00"], format="mixed")
print(f"Parsed from strings:\n{dates_str}\n")

# Date range — very useful for generating test data
daily = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")
hourly = pd.date_range(start="2024-01-01", periods=24, freq="h")
print(f"Daily range: {len(daily)} days, from {daily[0].date()} to {daily[-1].date()}")
print(f"Hourly range (24h): {hourly[:3].tolist()}\n")

# Frequency aliases: D=day, h=hour, min=minute, W=week, ME=month-end, QE=quarter-end, YE=year-end

print("=" * 60)
print("DATETIME ACCESSOR (.dt)")
print("=" * 60)

# Build a monitoring log
n = 500
log = pd.DataFrame({
    "timestamp":  pd.date_range("2024-01-01", periods=n, freq="2h"),
    "latency_ms": np.random.gamma(2, 50, n).round(1),
    "errors":     np.random.poisson(1, n),
    "model":      np.random.choice(["v1", "v2"], n),
})

# Extract components
log["year"]    = log["timestamp"].dt.year
log["month"]   = log["timestamp"].dt.month
log["day"]     = log["timestamp"].dt.day
log["hour"]    = log["timestamp"].dt.hour
log["weekday"] = log["timestamp"].dt.day_name()   # Monday, Tuesday...
log["is_weekend"] = log["timestamp"].dt.dayofweek >= 5

print(f"Log with datetime features:\n{log[['timestamp','hour','weekday','is_weekend']].head(6)}\n")

# Set timestamp as index for time-series operations
log = log.set_index("timestamp").sort_index()

print("=" * 60)
print("RESAMPLE — time-based aggregation")
print("=" * 60)

# Daily summary
daily_stats = log["latency_ms"].resample("D").agg(["mean", "max", "count"]).round(1)
print(f"Daily latency stats:\n{daily_stats.head(7)}\n")

# Weekly error rate
weekly_errors = log.resample("W").agg(
    total_errors   = ("errors", "sum"),
    total_requests = ("errors", "count"),
    mean_latency   = ("latency_ms", "mean"),
)
weekly_errors["error_rate"] = weekly_errors["total_errors"] / weekly_errors["total_requests"]
print(f"Weekly error metrics:\n{weekly_errors.head(5).round(3)}\n")

# Per-group resample (model v1 vs v2)
model_daily = log.groupby("model")["latency_ms"].resample("D").mean().round(1)
print(f"Daily latency per model (first 6):\n{model_daily.head(6)}\n")

print("=" * 60)
print("ROLLING WINDOWS — sliding aggregations")
print("=" * 60)

daily_ts = log["latency_ms"].resample("D").mean()

# Rolling mean (smoothing out noise)
rolling_7  = daily_ts.rolling(window=7, min_periods=1).mean()
rolling_30 = daily_ts.rolling(window=30, min_periods=1).mean()

# Rolling std (detect volatility)
rolling_std = daily_ts.rolling(window=7, min_periods=1).std()

comparison = pd.DataFrame({
    "raw":        daily_ts,
    "7d_ma":      rolling_7,
    "30d_ma":     rolling_30,
    "7d_std":     rolling_std,
}).round(2)

print(f"Raw vs rolling means:\n{comparison.head(10)}\n")

# Expanding window — cumulative stats (grows from start)
cumulative_mean = daily_ts.expanding().mean()
print(f"Cumulative mean latency (expanding):\n{cumulative_mean.head(5).round(2)}\n")

print("=" * 60)
print("SHIFT & DIFF — lag features")
print("=" * 60)

daily_ts_df = daily_ts.to_frame("latency")

# Lag features (previous N periods)
daily_ts_df["lag_1d"] = daily_ts_df["latency"].shift(1)   # yesterday's latency
daily_ts_df["lag_7d"] = daily_ts_df["latency"].shift(7)   # last week

# Difference (rate of change)
daily_ts_df["delta_1d"]  = daily_ts_df["latency"].diff(1)   # change vs yesterday
daily_ts_df["pct_change"] = daily_ts_df["latency"].pct_change(1) * 100

print(f"Lag/diff features:\n{daily_ts_df.head(10).round(2)}\n")

print("=" * 60)
print("TIME FILTERING — slicing by date")
print("=" * 60)

# Slice by date string — in pandas 3.x must use .loc[] for partial string indexing
jan = log.loc["2024-01"]                         # all of January
feb_week1 = log.loc["2024-02-01":"2024-02-07"]   # first week of Feb

print(f"January records:     {len(jan)}")
print(f"Feb week 1 records:  {len(feb_week1)}")

# Filter using datetime comparison
recent = log[log.index >= "2024-03-01"]
print(f"March onwards:       {len(recent)}\n")

print("=" * 60)
print("PRACTICAL: MODEL DRIFT DETECTION OVER TIME")
print("=" * 60)

# Simulate accuracy over time — gradually degrading
n_days = 90
dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
# Accuracy starts high, drifts down after day 60
accuracy = np.concatenate([
    np.random.normal(0.92, 0.02, 60),
    np.random.normal(0.85, 0.03, 30),   # model starts to degrade
])

perf = pd.DataFrame({"accuracy": accuracy}, index=dates)
perf["7d_mean"] = perf["accuracy"].rolling(7).mean()
perf["7d_std"]  = perf["accuracy"].rolling(7).std()

# Detect drift: 7-day rolling mean drops below baseline - 2*std
baseline_mean = perf["accuracy"][:30].mean()
baseline_std  = perf["accuracy"][:30].std()
threshold = baseline_mean - 2 * baseline_std

perf["drift_alert"] = perf["7d_mean"] < threshold

print(f"Baseline accuracy: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"Drift threshold:   {threshold:.3f}")
print(f"Drift alerts triggered on {perf['drift_alert'].sum()} days")
drift_days = perf[perf["drift_alert"]].index
if len(drift_days):
    print(f"First drift alert:  {drift_days[0].date()}")
    print(f"Last drift alert:   {drift_days[-1].date()}")

print()
print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. Create a DataFrame with hourly timestamps for all of January 2024,
   a 'load' column (random 0-100), and extract: hour, day_of_week, is_weekend.
   Compute mean load for weekdays vs weekends.

2. Using the log DataFrame from this file, compute a 24-hour rolling
   max of latency. Which day had the worst rolling max?

3. Build a simple anomaly detector:
   - Compute 7-day rolling mean and std of daily latency
   - Flag days where latency > mean + 2*std as anomalies

4. Create a 'time since last error' feature:
   - Start with the daily error count
   - For each day, compute how many days since the last day with errors > 2
""")

print("=" * 60)
print("ANSWERS")
print("=" * 60)

# 1.
jan_hourly = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", "2024-01-31 23:00", freq="h"),
    "load":      np.random.randint(0, 100, 31*24),
})
jan_hourly["hour"]       = jan_hourly["timestamp"].dt.hour
jan_hourly["day_of_week"]= jan_hourly["timestamp"].dt.day_name()
jan_hourly["is_weekend"] = jan_hourly["timestamp"].dt.dayofweek >= 5
wd_we = jan_hourly.groupby("is_weekend")["load"].mean().round(1)
print(f"1. Mean load — Weekday: {wd_we[False]}  Weekend: {wd_we[True]}\n")

# 2.
rolling_24h_max = log["latency_ms"].rolling("24h").max()
worst_day = rolling_24h_max.resample("D").max().idxmax()
print(f"2. Worst 24h rolling max day: {worst_day.date()}\n")

# 3.
daily_lat = log["latency_ms"].resample("D").mean()
r_mean = daily_lat.rolling(7, min_periods=3).mean()
r_std  = daily_lat.rolling(7, min_periods=3).std()
anomalies = daily_lat[daily_lat > r_mean + 2*r_std]
print(f"3. Anomaly days: {len(anomalies)}")
if len(anomalies):
    print(f"   First anomaly: {anomalies.index[0].date()}\n")

# 4.
daily_errors = log["errors"].resample("D").sum()
has_high_err = (daily_errors > 2).astype(int)
# Days since last high-error day
days_since = has_high_err.copy().astype(float)
counter = 0
for i, val in enumerate(has_high_err):
    if val:
        counter = 0
    else:
        counter += 1
    days_since.iloc[i] = counter
print(f"4. Days since last high-error (last 7 days):\n{days_since.tail(7).astype(int)}")
