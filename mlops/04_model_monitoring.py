"""
================================================================================
MLOPS 04 — Model Monitoring & Performance Tracking
================================================================================

THEORY:
  After deploying a model, you need to watch it continuously. Models degrade.

  What to monitor:
    1. Prediction distribution — is the model still outputting similar scores?
    2. Feature drift          — have input features changed? (PSI, KS test)
    3. Model performance      — accuracy, precision, recall, AUC (when labels arrive)
    4. Operational metrics    — latency p50/p95/p99, error rate, throughput
    5. Concept drift          — same inputs, different correct outputs (hardest)

  This file builds a monitoring dashboard using only pandas + numpy.
================================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ============================================================
# SIMULATE PRODUCTION DATA (3 months of daily model logs)
# ============================================================

def simulate_production_logs(n_days: int = 90) -> pd.DataFrame:
    """Simulate daily model prediction logs over 3 months."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    logs = []
    for i, date in enumerate(dates):
        n = rng.integers(800, 1200)  # variable request count per day

        # Simulate gradual model degradation after day 60
        accuracy_base = 0.92 if i < 60 else 0.92 - (i - 60) * 0.003
        drift_offset  = 0.0  if i < 60 else (i - 60) * 0.005  # input distribution shift

        day_logs = pd.DataFrame({
            "date":        date,
            "pred_score":  np.clip(rng.normal(0.6 + drift_offset, 0.2, n), 0, 1),
            "pred_label":  (rng.random(n) < (0.65 + drift_offset)).astype(int),
            "true_label":  (rng.random(n) < 0.65).astype(int),
            "latency_ms":  rng.gamma(2, 50, n),
            "feature_age": rng.normal(35 + drift_offset * 20, 10, n),   # drifting feature
            "feature_income": rng.normal(55000 + drift_offset * 5000, 15000, n),
        })
        # Inject accuracy based on simulated truth
        correct = (rng.random(n) < accuracy_base)
        day_logs["true_label"] = np.where(correct, day_logs["pred_label"],
                                          1 - day_logs["pred_label"])
        logs.append(day_logs)

    return pd.concat(logs, ignore_index=True)

logs = simulate_production_logs(90)
logs["date"] = pd.to_datetime(logs["date"])
print(f"Production logs: {logs.shape}")
print(f"Date range: {logs['date'].min().date()} to {logs['date'].max().date()}\n")

# ============================================================
# PART 1: DAILY METRICS COMPUTATION
# ============================================================

print("=" * 60)
print("1. DAILY PERFORMANCE METRICS")
print("=" * 60)

def compute_metrics(grp: pd.DataFrame) -> pd.Series:
    """Compute classification metrics for a group."""
    y_true = grp["true_label"]
    y_pred = grp["pred_label"]

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({
        "n_predictions":    len(grp),
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "mean_pred_score":  grp["pred_score"].mean(),
        "positive_rate":    y_pred.mean(),
        "p50_latency":      grp["latency_ms"].quantile(0.50),
        "p95_latency":      grp["latency_ms"].quantile(0.95),
        "p99_latency":      grp["latency_ms"].quantile(0.99),
    })

daily = logs.groupby("date").apply(compute_metrics, include_groups=False).round(4)
print(f"Daily metrics (first 5 days):\n{daily.head()}\n")

# ============================================================
# PART 2: ROLLING PERFORMANCE WITH ALERTING
# ============================================================

print("=" * 60)
print("2. ROLLING METRICS & ALERTS")
print("=" * 60)

# 7-day rolling metrics
daily["accuracy_7d"]   = daily["accuracy"].rolling(7, min_periods=3).mean()
daily["f1_7d"]         = daily["f1"].rolling(7, min_periods=3).mean()
daily["p95_latency_7d"]= daily["p95_latency"].rolling(7).mean()

# Establish baseline from first 30 days
baseline = daily.iloc[:30]
baseline_acc_mean = baseline["accuracy"].mean()
baseline_acc_std  = baseline["accuracy"].std()
baseline_p95_mean = baseline["p95_latency"].mean()

# Alert thresholds
ACC_THRESHOLD     = baseline_acc_mean - 2 * baseline_acc_std
LATENCY_THRESHOLD = baseline_p95_mean * 1.5   # 50% increase

daily["acc_alert"]     = daily["accuracy_7d"] < ACC_THRESHOLD
daily["latency_alert"] = daily["p95_latency_7d"] > LATENCY_THRESHOLD

print(f"Baseline (first 30 days):")
print(f"  Accuracy:  {baseline_acc_mean:.3f} ± {baseline_acc_std:.3f}")
print(f"  p95 latency: {baseline_p95_mean:.1f} ms")
print(f"\nAlert thresholds:")
print(f"  Accuracy < {ACC_THRESHOLD:.3f}")
print(f"  p95 Latency > {LATENCY_THRESHOLD:.1f} ms")

acc_alerts     = daily[daily["acc_alert"]]
latency_alerts = daily[daily["latency_alert"]]
print(f"\nAccuracy alerts fired: {len(acc_alerts)} days")
print(f"Latency alerts fired:  {len(latency_alerts)} days")

if len(acc_alerts):
    print(f"First accuracy alert:  {acc_alerts.index[0].date()}")
    print(f"Accuracy on that day:  {acc_alerts.iloc[0]['accuracy_7d']:.3f}")

# ============================================================
# PART 3: FEATURE DRIFT DETECTION
# ============================================================

print()
print("=" * 60)
print("3. FEATURE DRIFT DETECTION")
print("=" * 60)

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    ref_pct = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    cur_pct = np.histogram(current,   bins=breakpoints)[0] / len(current)   + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

# Reference = first 30 days
ref_logs = logs[logs["date"] <= "2024-01-30"]
features_to_monitor = ["feature_age", "feature_income", "pred_score"]

print(f"{'Feature':<20} {'PSI':>8} {'Status':>15}")
print("-" * 45)

weekly_data = logs.groupby(pd.Grouper(key="date", freq="W"))
for week_start, week_grp in weekly_data:
    if len(week_grp) < 100:
        continue
    # Only show last few weeks
    if week_start < pd.Timestamp("2024-02-01"):
        continue

    psivs = {}
    for feat in features_to_monitor:
        psi = compute_psi(ref_logs[feat].values, week_grp[feat].values)
        psivs[feat] = psi

    for feat, psi in psivs.items():
        status = "OK" if psi < 0.1 else ("WARNING" if psi < 0.2 else "DRIFT!")
        print(f"{week_start.date()} {feat:<18} {psi:>8.4f} {status:>12}")
    print()

# ============================================================
# PART 4: DATA PROFILING REPORT
# ============================================================

print("=" * 60)
print("4. DATA PROFILING REPORT")
print("=" * 60)

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a profile report for all columns."""
    report = pd.DataFrame()
    report["dtype"]      = df.dtypes
    report["n_null"]     = df.isnull().sum()
    report["null_pct"]   = (df.isnull().mean() * 100).round(2)
    report["n_unique"]   = df.nunique()
    report["unique_pct"] = (df.nunique() / len(df) * 100).round(2)

    for col in df.select_dtypes(include="number").columns:
        report.loc[col, "mean"]    = df[col].mean()
        report.loc[col, "std"]     = df[col].std()
        report.loc[col, "min"]     = df[col].min()
        report.loc[col, "p25"]     = df[col].quantile(0.25)
        report.loc[col, "median"]  = df[col].median()
        report.loc[col, "p75"]     = df[col].quantile(0.75)
        report.loc[col, "max"]     = df[col].max()
        report.loc[col, "skew"]    = df[col].skew()

    return report.round(3)

profile = profile_dataframe(logs[features_to_monitor + ["true_label", "latency_ms"]])
print(f"\nData profile:\n{profile[['dtype','n_null','null_pct','mean','std','min','max','skew']]}")

# ============================================================
# PART 5: MONITORING DASHBOARD (text-based)
# ============================================================

print()
print("=" * 60)
print("5. MONITORING DASHBOARD")
print("=" * 60)

last_7 = daily.tail(7)
last_30 = daily.tail(30)

def bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    filled = int(value / max_val * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.3f}"

print(f"\nLast 7-day average metrics:")
print(f"  Accuracy:   {bar(last_7['accuracy'].mean())}")
print(f"  Precision:  {bar(last_7['precision'].mean())}")
print(f"  Recall:     {bar(last_7['recall'].mean())}")
print(f"  F1 Score:   {bar(last_7['f1'].mean())}")
print(f"\nLatency (ms):  p50={last_7['p50_latency'].mean():.1f}  p95={last_7['p95_latency'].mean():.1f}  p99={last_7['p99_latency'].mean():.1f}")

print(f"\nAccuracy trend (last 30 days):")
for i, (date, row) in enumerate(last_30[["accuracy"]].iterrows()):
    marker = " ←" if i == 29 else ""
    bar_len = int(row["accuracy"] * 30)
    print(f"  {date.strftime('%m-%d')} {'█' * bar_len}{marker}")

print(f"\nActive alerts: ", end="")
active_alerts = []
if daily.iloc[-1]["acc_alert"]:
    active_alerts.append("ACCURACY_DEGRADATION")
if daily.iloc[-1]["latency_alert"]:
    active_alerts.append("HIGH_LATENCY")
print(", ".join(active_alerts) if active_alerts else "None")

# ============================================================
# PART 6: RETRAINING SIGNAL
# ============================================================

print()
print("=" * 60)
print("6. RETRAINING SIGNAL COMPUTATION")
print("=" * 60)

# Score from 0-100. High score → trigger retraining.
def compute_retraining_score(daily: pd.DataFrame, ref_data: pd.DataFrame) -> dict:
    recent = daily.tail(7)
    score = 0
    reasons = []

    # Accuracy drop
    acc_drop = baseline_acc_mean - recent["accuracy"].mean()
    if acc_drop > 0.03:
        score += 40
        reasons.append(f"accuracy drop: {acc_drop:.3f}")

    # Increasing latency
    lat_increase = recent["p95_latency"].mean() / baseline_p95_mean - 1
    if lat_increase > 0.2:
        score += 20
        reasons.append(f"latency increase: {lat_increase:.1%}")

    # Positive rate shift
    pos_rate_shift = abs(recent["positive_rate"].mean() - 0.65)
    if pos_rate_shift > 0.1:
        score += 30
        reasons.append(f"positive rate shift: {pos_rate_shift:.3f}")

    # Volume anomaly
    expected_n = daily["n_predictions"].mean()
    recent_n   = recent["n_predictions"].mean()
    if abs(recent_n - expected_n) / expected_n > 0.3:
        score += 10
        reasons.append(f"volume anomaly: {recent_n:.0f} vs expected {expected_n:.0f}")

    return {"score": min(score, 100), "reasons": reasons,
            "recommendation": "RETRAIN" if score >= 50 else "MONITOR" if score >= 25 else "OK"}

retrain_signal = compute_retraining_score(daily, ref_logs)
print(f"Retraining score: {retrain_signal['score']}/100")
print(f"Recommendation:   {retrain_signal['recommendation']}")
print(f"Reasons:")
for r in retrain_signal["reasons"]:
    print(f"  • {r}")
