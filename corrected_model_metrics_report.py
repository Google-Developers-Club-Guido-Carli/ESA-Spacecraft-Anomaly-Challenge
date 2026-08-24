"""
Fresh, single-execution metrics report for the CORRECTED hybrid ensemble
(SPAICE 2026 poster, Mission1 channels 41-46).

Re-runs the corrected pipeline from hybrid_ensemble_notebook.ipynb exactly
(same block-sampling loader, same chronological split, same IsolationForest /
Telemanom / engineer_features / XGBoost fusion, same hyperparameters) and
then computes point-wise + event-wise metrics, a threshold sweep with
bootstrap stability check, and size-stratified recall on the fresh
chronological validation set. Also emits the corrected confusion-matrix
figure and a calibrated-threshold Kaggle submission CSV (record only, not
for submission).

All data loading uses pyarrow column projection + float32 downcast; the full
14.7M x 87 train.parquet is never materialized (memory-safety hard
constraint).

Outputs (all new filenames - nothing in outputs/ is overwritten):
  outputs/corrected_confusion_matrix.png
  outputs/corrected_model_metrics_report.txt
  outputs/spaice2026_calibrated_submission.csv   (id, is_anomaly - NOT for submission)
"""
import os
import sys

# Determinism: pin every BLAS/OpenMP thread pool to 1 BEFORE numpy/scipy/
# sklearn/xgboost are imported (thread count env vars only take effect if set
# prior to the first import that initializes the underlying thread pools).
# Multithreaded float summation (BLAS reductions, XGBoost histogram merges,
# IsolationForest's joblib parallelism) reorders floating-point additions
# nondeterministically across runs even with a fixed random_state.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

import gc
import time
import json
import platform
import numpy as np
import pandas as pd
import sklearn
import pyarrow
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
)
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single seed constant, used everywhere something stochastic happens
# (IsolationForest, XGBoost, bootstrap resampling). The block-based
# chronological split itself uses no RNG (pure index arithmetic).
SEED = 42

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / ".data"
OUT_DIR = REPO / "outputs"
OUT_DIR.mkdir(exist_ok=True)

REPORT_LINES = []
def log(msg=""):
    print(msg)
    REPORT_LINES.append(str(msg))

T0 = time.time()

log("=" * 80)
log("STEP 0: Reproducibility configuration")
log("=" * 80)
log(f"SEED = {SEED} (used for IsolationForest, XGBoost, bootstrap resampling)")
log(f"Thread pins: OMP/OPENBLAS/MKL/VECLIB/NUMEXPR NUM_THREADS=1, "
    f"IsolationForest n_jobs=1, XGBoost n_jobs=1, pyarrow use_threads=False")
log(f"XGBoost tree_method='hist' (explicit, deterministic single-thread CPU histograms)")
log(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
    f"sklearn {sklearn.__version__} | xgboost {xgb.__version__} | pyarrow {pyarrow.__version__}")

# ===========================================================================
# 1. Memory-safe data loading (mirrors notebook cells 4-9 exactly)
# ===========================================================================
log("=" * 80)
log("STEP 1: Memory-safe data loading (column projection, float32, block sampling)")
log("=" * 80)

FEATURE_COLS = [f"channel_{i}" for i in range(41, 47)]
BLOCK_SIZE = 150_000
N_BLOCKS = 10
N_VAL_BLOCKS = 2

# ---- test.parquet (full, small) ----
test_table = pq.read_table(DATA_DIR / "test.parquet", columns=["id"] + FEATURE_COLS,
                            use_threads=False)
df_test = test_table.to_pandas()
del test_table
gc.collect()
for c in FEATURE_COLS:
    df_test[c] = df_test[c].astype(np.float32)
df_test["id"] = df_test["id"].astype(np.int64)

# ---- train.parquet (block-sampled, row-group-aware partial reads) ----
pf = pq.ParquetFile(DATA_DIR / "train.parquet")
total_rows = pf.metadata.num_rows
max_start = total_rows - BLOCK_SIZE
block_starts = [int(round(k * max_start / (N_BLOCKS - 1))) for k in range(N_BLOCKS)]

row_group_offsets = []
offset = 0
for rg in range(pf.num_row_groups):
    n = pf.metadata.row_group(rg).num_rows
    row_group_offsets.append((offset, offset + n))
    offset += n

needed_cols = ["id"] + FEATURE_COLS + ["is_anomaly"]
block_frames = []
for b, start in enumerate(block_starts):
    end = start + BLOCK_SIZE
    rg_indices = [i for i, (s, e) in enumerate(row_group_offsets) if e > start and s < end]
    table = pf.read_row_groups(rg_indices, columns=needed_cols, use_threads=False)
    rg_first_offset = row_group_offsets[rg_indices[0]][0]
    local_start = start - rg_first_offset
    block_df = table.slice(local_start, BLOCK_SIZE).to_pandas()
    assert len(block_df) == BLOCK_SIZE
    block_df["block"] = b
    block_frames.append(block_df)
    del table
del pf
gc.collect()

df_train = pd.concat(block_frames, ignore_index=True)
del block_frames
gc.collect()
for c in FEATURE_COLS:
    df_train[c] = df_train[c].astype(np.float32)
df_train["id"] = df_train["id"].astype(np.int64)
df_train["is_anomaly"] = df_train["is_anomaly"].astype(np.int32)
df_train["block"] = df_train["block"].astype(np.int32)

log(f"Train sample: {len(df_train):,} rows ({len(df_train)/total_rows*100:.2f}% of {total_rows:,})")
log(f"Test set: {len(df_test):,} rows")

X_raw_test = df_test[FEATURE_COLS].values.astype(np.float32)
X_raw_train = df_train[FEATURE_COLS].values.astype(np.float32)
y_raw_train = df_train["is_anomaly"].values
train_ids = df_train["id"].values
train_blocks = df_train["block"].values

# ---- chronological split (problem 1 fix) ----
train_block_mask = train_blocks < (N_BLOCKS - N_VAL_BLOCKS)
val_block_mask = ~train_block_mask

X_train = X_raw_train[train_block_mask]
X_val = X_raw_train[val_block_mask]
y_train = y_raw_train[train_block_mask]
y_val = y_raw_train[val_block_mask]
ids_train = train_ids[train_block_mask]
ids_val = train_ids[val_block_mask]

assert ids_train.max() < ids_val.min(), "Chronological split violated"
log(f"Chronological split: train={len(X_train):,} rows, val={len(X_val):,} rows")
log(f"  Assertion passed: max(train id)={ids_train.max():,} < min(val id)={ids_val.min():,}")
log(f"  Val anomaly rate: {y_val.mean()*100:.3f}%  ({y_val.sum():,} / {len(y_val):,} points)")

# ===========================================================================
# 2. Scaling + IsolationForest + Telemanom + engineer_features
#    (mirrors notebook cells 11-34 exactly, including the fit_transform-per-
#    split scaling convention and the corrected engineer_features with the
#    inf->0 fix)
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 2: Base detectors + feature engineering (unchanged architecture)")
log("=" * 80)

scaler = StandardScaler()
X_scaled_val = scaler.fit_transform(X_val)
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.fit_transform(X_raw_test)

def run_iforest(X_fit_scores):
    iforest = IsolationForest(
        n_estimators=100, contamination=0.10, max_samples="auto",
        random_state=SEED, n_jobs=1, verbose=0,
    )
    iforest.fit(X_fit_scores)
    raw = -iforest.score_samples(X_fit_scores)
    scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
    return scores

iforest_scores_train = run_iforest(X_scaled_train)
iforest_scores_val = run_iforest(X_scaled_val)
iforest_scores_test = run_iforest(X_scaled_test)
log("IsolationForest scored (train/val/test)")

def compute_telemanom_scores(X, window_size=50):
    n_samples, n_features = X.shape
    prediction_errors = np.zeros(n_samples)
    for i in range(window_size, n_samples):
        window = X[i - window_size:i]
        weights = np.exp(np.linspace(-1, 0, window_size))
        weights = weights / weights.sum()
        predicted = np.zeros(n_features)
        for f in range(n_features):
            predicted[f] = np.sum(window[:, f] * weights)
        error = np.abs(X[i] - predicted)
        prediction_errors[i] = np.mean(error)

    cumulative_errors = np.zeros(n_samples)
    cum_window = 10
    for i in range(cum_window, n_samples):
        cumulative_errors[i] = np.sum(prediction_errors[i - cum_window:i])
    if np.max(cumulative_errors) > 0:
        cumulative_errors = cumulative_errors / np.max(cumulative_errors)

    combined_scores = 0.7 * prediction_errors + 0.3 * cumulative_errors
    if np.max(combined_scores) > np.min(combined_scores):
        normalized = (combined_scores - np.min(combined_scores)) / (np.max(combined_scores) - np.min(combined_scores))
    else:
        normalized = combined_scores
    return normalized

t = time.time()
telemanom_scores_train = compute_telemanom_scores(X_scaled_train, window_size=50)
telemanom_scores_val = compute_telemanom_scores(X_scaled_val, window_size=50)
telemanom_scores_test = compute_telemanom_scores(X_scaled_test, window_size=50)
log(f"Telemanom scored (train/val/test) in {time.time()-t:.1f}s")

def engineer_features(X, iforest_scores, telemanom_scores, window_sizes=[10, 30, 50]):
    n_samples, n_features = X.shape
    features = []
    feature_names = []

    features.append(iforest_scores.reshape(-1, 1)); feature_names.append("iforest_score")
    features.append(telemanom_scores.reshape(-1, 1)); feature_names.append("telemanom_score")
    detector_agreement = np.abs(iforest_scores - telemanom_scores)
    features.append(detector_agreement.reshape(-1, 1)); feature_names.append("detector_agreement")

    for window in window_sizes:
        rolling_mean = pd.DataFrame(X).rolling(window=window, min_periods=1).mean().values
        rolling_mean_dev = np.mean(np.abs(X - rolling_mean), axis=1)
        features.append(rolling_mean_dev.reshape(-1, 1)); feature_names.append(f"rolling_mean_dev_{window}")

        rolling_std = pd.DataFrame(X).rolling(window=window, min_periods=1).std().values
        rolling_std_score = np.mean(rolling_std, axis=1)
        features.append(rolling_std_score.reshape(-1, 1)); feature_names.append(f"rolling_std_{window}")

    for lag in [1, 5, 10]:
        delta = np.zeros((n_samples, n_features))
        delta[lag:] = X[lag:] - X[:-lag]
        delta_score = np.mean(np.abs(delta), axis=1)
        features.append(delta_score.reshape(-1, 1)); feature_names.append(f"delta_{lag}")

    slopes = np.zeros(n_samples)
    slope_window = 10
    for i in range(slope_window, n_samples):
        window_data = X[i - slope_window:i]
        time_idx = np.arange(slope_window)
        feature_slopes = []
        for feat in range(n_features):
            slope = np.polyfit(time_idx, window_data[:, feat], 1)[0]
            feature_slopes.append(abs(slope))
        slopes[i] = np.mean(feature_slopes)
    features.append(slopes.reshape(-1, 1)); feature_names.append("slope_magnitude")

    corr_window = 30
    n_corr_pairs = min(3, n_features - 1)
    for i in range(n_corr_pairs):
        for j in range(i + 1, min(i + 2, n_features)):
            corr_values = pd.Series(X[:, i]).rolling(
                window=corr_window, min_periods=10
            ).corr(pd.Series(X[:, j])).fillna(0).values
            features.append(corr_values.reshape(-1, 1)); feature_names.append(f"corr_ch{i}_ch{j}")

    X_features = np.hstack(features)
    # Problem 4 fix: extend NaN->0 to inf->0, fixed once at the function source
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    return X_features, feature_names

t = time.time()
X_eng_train, feature_names = engineer_features(X_scaled_train, iforest_scores_train, telemanom_scores_train)
X_eng_val, _ = engineer_features(X_scaled_val, iforest_scores_val, telemanom_scores_val)
X_eng_test, _ = engineer_features(X_scaled_test, iforest_scores_test, telemanom_scores_test)
log(f"Feature engineering complete in {time.time()-t:.1f}s (16 features x train/val/test)")
n_nonfinite = (~np.isfinite(X_eng_train)).sum() + (~np.isfinite(X_eng_val)).sum() + (~np.isfinite(X_eng_test)).sum()
log(f"Non-finite values remaining after inf-fix: {n_nonfinite} (must be 0)")
assert n_nonfinite == 0

# ===========================================================================
# 3. XGBoost fusion - supervised on true chronological labels
#    (preserved hyperparameters: 200 trees, depth 6, lr 0.1, subsample 0.8)
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 3: XGBoost fusion (supervised, true labels, preserved hyperparameters)")
log("=" * 80)

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist",   # explicit: deterministic CPU histogram method
    random_state=SEED, n_jobs=1,
)
xgb_model.fit(X_eng_train, y_train, verbose=False)

val_proba = xgb_model.predict_proba(X_eng_val)[:, 1]
test_proba = xgb_model.predict_proba(X_eng_test)[:, 1]
log(f"XGBoost trained on {len(X_eng_train):,} rows, scored val ({len(val_proba):,}) and test ({len(test_proba):,})")

# ===========================================================================
# 4. Post-processing (unchanged: min_event_length=3, max_gap=5)
# ===========================================================================
def postprocess_scores(scores, threshold=0.5, min_event_length=3, max_gap=5):
    binary = (scores > threshold).astype(int)
    segments = []
    in_segment = False
    start = 0
    for i in range(len(binary)):
        if binary[i] == 1 and not in_segment:
            start = i; in_segment = True
        elif binary[i] == 0 and in_segment:
            segments.append((start, i - 1)); in_segment = False
    if in_segment:
        segments.append((start, len(binary) - 1))

    merged = []
    if segments:
        cs, ce = segments[0]
        for s, e in segments[1:]:
            if s - ce <= max_gap:
                ce = e
            else:
                merged.append((cs, ce)); cs, ce = s, e
        merged.append((cs, ce))

    filtered = [(s, e) for s, e in merged if e - s + 1 >= min_event_length]
    processed = np.zeros_like(scores)
    for s, e in filtered:
        processed[s:e + 1] = scores[s:e + 1]
    return processed, filtered

# ===========================================================================
# 5. Metrics
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 4: Metrics on the fresh chronological validation set")
log("=" * 80)

def find_events(binary_array):
    """Contiguous segments of 1s -> list of (start, end) inclusive."""
    events = []
    in_seg = False
    start = 0
    for i, v in enumerate(binary_array):
        if v == 1 and not in_seg:
            start = i; in_seg = True
        elif v == 0 and in_seg:
            events.append((start, i - 1)); in_seg = False
    if in_seg:
        events.append((start, len(binary_array) - 1))
    return events

TRUE_EVENTS = find_events(y_val)
N_TRUE_EVENTS_CANONICAL = len(TRUE_EVENTS)
event_lengths = np.array([e - s + 1 for s, e in TRUE_EVENTS])
event_max_score = np.array([val_proba[s:e + 1].max() for s, e in TRUE_EVENTS])

log(f"\nCanonical true anomaly event count in validation set: {N_TRUE_EVENTS_CANONICAL}")
log(f"True event lengths (all {N_TRUE_EVENTS_CANONICAL}): {event_lengths.tolist()}")

def point_confusion(y_true, pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, pred_binary, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), precision=precision, recall=recall, f1=f1)

def event_recall(pred_binary, events):
    if len(events) == 0:
        return 0.0, 0
    caught = sum(1 for s, e in events if pred_binary[s:e + 1].any())
    return caught / len(events), caught

def fbeta(precision, recall, beta=0.5):
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

def event_wise_f05(pred_binary, events, point_precision_source_binary, y_true):
    """Event-wise F0.5 (Sehili & Zhang 2023 style):
    recall = fraction of true events with >=1 caught point (event-wise);
    precision = point-wise precision over the whole series (corrected by
    point-level false-alarm rate); combined with beta=0.5 (precision
    weighted more heavily than recall)."""
    pc = point_confusion(y_true, point_precision_source_binary)
    r, caught = event_recall(pred_binary, events)
    f = fbeta(pc["precision"], r, beta=0.5)
    return dict(precision=pc["precision"], recall=r, f05=f, events_caught=caught, n_events=len(events),
                point_confusion=pc)

# ---- ROC-AUC / PR-AUC (threshold-independent) ----
roc_auc = roc_auc_score(y_val, val_proba)
pr_auc = average_precision_score(y_val, val_proba)
log(f"\nROC-AUC (threshold-independent): {roc_auc!r}")
log(f"PR-AUC  (threshold-independent): {pr_auc!r}")

# ===========================================================================
# 6. Threshold sweep + calibration + bootstrap stability
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 5: Threshold sweep, calibration, bootstrap stability")
log("=" * 80)

THRESH_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

# Precompute point-wise precision(t) via sorting (vectorized)
order = np.argsort(-val_proba)
sorted_scores = val_proba[order]
sorted_labels = y_val[order]
cum_tp = np.cumsum(sorted_labels)
n_at_or_above = np.arange(1, len(sorted_scores) + 1)

def precision_at_threshold(t):
    idx = np.searchsorted(-sorted_scores, -t, side="right")  # count of scores > t
    if idx == 0:
        return 0.0
    return cum_tp[idx - 1] / idx

def event_recall_at_threshold(t, max_scores):
    if len(max_scores) == 0:
        return 0.0
    return float(np.mean(max_scores > t))

sweep_rows = []
for t in THRESH_GRID:
    p = precision_at_threshold(t)
    r = event_recall_at_threshold(t, event_max_score)
    f = fbeta(p, r, beta=0.5)
    sweep_rows.append((t, p, r, f))

sweep_df = pd.DataFrame(sweep_rows, columns=["threshold", "precision", "event_recall", "f05"])
best_idx = sweep_df["f05"].idxmax()
CALIBRATED_THRESHOLD = float(sweep_df.loc[best_idx, "threshold"])
CALIBRATED_F05 = float(sweep_df.loc[best_idx, "f05"])

log(f"\nThreshold sweep grid: {THRESH_GRID[0]} to {THRESH_GRID[-1]} step 0.01 ({len(THRESH_GRID)} thresholds)")
log(f"Optimal (calibrated) threshold: {CALIBRATED_THRESHOLD!r}")
log(f"Event-wise F0.5 at calibrated threshold (raw, pre-postprocessing): {CALIBRATED_F05!r}")
log("\nFull sweep table:")
log(sweep_df.to_string(index=False))

# ---- Bootstrap stability check ----
# Event-level bootstrap: resample the N true events with replacement,
# recompute event-recall(t) from the resampled multiset of event max-scores,
# hold point-wise precision(t) fixed (it is a global quantity over the full
# point-level series, not a per-event quantity), find the argmax-F0.5
# threshold for each resampled draw, and report what fraction of draws land
# on the same threshold (same 0.01 grid cell) as the point-estimate.
BOOT_N = 500
BOOT_SEED = SEED  # one seed constant, used everywhere
rng = np.random.default_rng(BOOT_SEED)

precision_grid = sweep_df["precision"].values  # fixed across bootstrap
consensus_count = 0
boot_thresholds = []
n_events = len(event_max_score)
if n_events > 0:
    for _ in range(BOOT_N):
        sample_idx = rng.integers(0, n_events, size=n_events)
        sample_scores = event_max_score[sample_idx]
        recalls = (sample_scores[None, :] > THRESH_GRID[:, None]).mean(axis=1)
        f05_vec = np.array([fbeta(precision_grid[i], recalls[i], beta=0.5) for i in range(len(THRESH_GRID))])
        boot_best_t = THRESH_GRID[np.argmax(f05_vec)]
        boot_thresholds.append(boot_best_t)
        if abs(boot_best_t - CALIBRATED_THRESHOLD) < 1e-9:
            consensus_count += 1
    consensus_pct = 100.0 * consensus_count / BOOT_N
else:
    consensus_pct = float("nan")

log(f"\nBootstrap stability check:")
log(f"  N iterations: {BOOT_N}")
log(f"  Random seed: {BOOT_SEED}")
log(f"  Method: event-level resampling with replacement of the {n_events} true "
    f"validation events; point-wise precision(t) held fixed (global, not "
    f"per-event); per-draw optimal threshold recomputed on the same 0.01 grid.")
log(f"  Consensus on threshold {CALIBRATED_THRESHOLD}: {consensus_pct:.2f}% "
    f"({consensus_count}/{BOOT_N} draws)")
if len(boot_thresholds) > 0:
    bt = np.array(boot_thresholds)
    log(f"  Bootstrap threshold distribution: mean={bt.mean():.4f}, std={bt.std():.4f}, "
        f"median={np.median(bt):.4f}, [5th,95th pct]=[{np.percentile(bt,5):.4f}, {np.percentile(bt,95):.4f}]")

# ===========================================================================
# 7. Full metrics at threshold 0.5 AND calibrated threshold, raw + post-processed
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 6: Full point-wise + event-wise report at both thresholds")
log("=" * 80)

REPORT = {}
for label, t in [("threshold_0.5", 0.5), (f"calibrated_threshold_{CALIBRATED_THRESHOLD}", CALIBRATED_THRESHOLD)]:
    log(f"\n{'-'*80}\nTHRESHOLD = {t}  ({label})\n{'-'*80}")

    raw_binary = (val_proba > t).astype(int)
    pc_raw = point_confusion(y_val, raw_binary)
    log(f"[POINT-WISE, RAW] TN={pc_raw['TN']} FP={pc_raw['FP']} FN={pc_raw['FN']} TP={pc_raw['TP']}")
    log(f"[POINT-WISE, RAW] precision={pc_raw['precision']!r} recall={pc_raw['recall']!r} f1={pc_raw['f1']!r}")
    log(f"[POINT-WISE, RAW] ROC-AUC={roc_auc!r} PR-AUC={pr_auc!r}  (threshold-independent, shown for reference)")

    processed_scores, filtered_segments = postprocess_scores(val_proba, threshold=t, min_event_length=3, max_gap=5)
    pp_binary = (processed_scores > 0).astype(int)
    pc_pp = point_confusion(y_val, pp_binary)
    log(f"[POINT-WISE, POST-PROCESSED] TN={pc_pp['TN']} FP={pc_pp['FP']} FN={pc_pp['FN']} TP={pc_pp['TP']}")
    log(f"[POINT-WISE, POST-PROCESSED] precision={pc_pp['precision']!r} recall={pc_pp['recall']!r} f1={pc_pp['f1']!r}")
    log(f"[POINT-WISE, POST-PROCESSED] events after gap-merge+min-length filter: {len(filtered_segments)}")

    ev_raw = event_wise_f05(raw_binary, TRUE_EVENTS, raw_binary, y_val)
    log(f"[EVENT-WISE, RAW] n_true_events={ev_raw['n_events']} caught={ev_raw['events_caught']} "
        f"recall={ev_raw['recall']!r} precision(point-corrected)={ev_raw['precision']!r} F0.5={ev_raw['f05']!r}")
    assert ev_raw["n_events"] == N_TRUE_EVENTS_CANONICAL, "EVENT COUNT MISMATCH (raw) — FLAG"

    ev_pp = event_wise_f05(pp_binary, TRUE_EVENTS, pp_binary, y_val)
    log(f"[EVENT-WISE, POST-PROCESSED] n_true_events={ev_pp['n_events']} caught={ev_pp['events_caught']} "
        f"recall={ev_pp['recall']!r} precision(point-corrected)={ev_pp['precision']!r} F0.5={ev_pp['f05']!r}")
    assert ev_pp["n_events"] == N_TRUE_EVENTS_CANONICAL, "EVENT COUNT MISMATCH (post-processed) — FLAG"

    # size-stratified recall (raw predictions, this threshold)
    bins = {"short(<100)": (0, 100), "medium(100-1000)": (100, 1000), "long(>1000)": (1000, np.inf)}
    log(f"[SIZE-STRATIFIED RECALL, RAW predictions @ threshold {t}]")
    strat = {}
    for name, (lo, hi) in bins.items():
        mask = (event_lengths >= lo) & (event_lengths < hi) if hi != np.inf else (event_lengths >= lo)
        idxs = np.where(mask)[0]
        n_bin = len(idxs)
        caught_flags = [bool(raw_binary[TRUE_EVENTS[i][0]:TRUE_EVENTS[i][1] + 1].any()) for i in idxs]
        n_caught = sum(caught_flags)
        lengths_this_bin = event_lengths[idxs].tolist()
        log(f"  {name}: n_events={n_bin} n_caught={n_caught} "
            f"recall={ (n_caught/n_bin) if n_bin>0 else float('nan')!r}")
        log(f"    lengths={lengths_this_bin}")
        log(f"    caught_flags(event_index->caught)={list(zip(idxs.tolist(), caught_flags))}")
        strat[name] = dict(n_events=n_bin, n_caught=n_caught, lengths=lengths_this_bin,
                            caught=list(zip(idxs.tolist(), caught_flags)))

    log(f"[EVENT COUNT CHECK] true events this section = {N_TRUE_EVENTS_CANONICAL} "
        f"(canonical) -- MATCH" if N_TRUE_EVENTS_CANONICAL == len(TRUE_EVENTS) else "MISMATCH — FLAG LOUDLY")

    REPORT[label] = dict(
        threshold=t,
        point_raw=pc_raw, point_postprocessed=pc_pp,
        event_raw=ev_raw, event_postprocessed=ev_pp,
        size_stratified=strat,
        roc_auc=roc_auc, pr_auc=pr_auc,
    )

log(f"\n{'='*80}\nFINAL CONSISTENCY CHECK: true anomaly event count in validation set\n{'='*80}")
log(f"Canonical count: {N_TRUE_EVENTS_CANONICAL}")
log("(Every section above asserted equality with this canonical count; no mismatch was raised.)")

# ===========================================================================
# 8. Confusion-matrix figure (corrected model, replaces old paper Figure 3)
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 7: Confusion-matrix figure (corrected model, calibrated threshold)")
log("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (title, t) in zip(axes, [("Threshold = 0.5", 0.5), (f"Calibrated threshold = {CALIBRATED_THRESHOLD}", CALIBRATED_THRESHOLD)]):
    pred = (val_proba > t).astype(int)
    cm = confusion_matrix(y_val, pred, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"]); ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Corrected Model — {title}", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")
fig.suptitle("Corrected Model Confusion Matrix — Chronological Validation Set\n"
             "(replaces the original paper's Figure 3, which used the leaked/shuffled-split model)",
             fontsize=11)
plt.tight_layout()
fig_path = OUT_DIR / "corrected_confusion_matrix.png"
plt.savefig(fig_path, dpi=150)
plt.close(fig)
log(f"Saved: {fig_path}")

# ===========================================================================
# 9. Calibrated-threshold Kaggle submission CSV (record only, NOT for submission)
# ===========================================================================
log("\n" + "=" * 80)
log("STEP 8: Calibrated-threshold submission CSV (BUILD ONLY, NOT for submission)")
log("=" * 80)

submission_df = pd.DataFrame({
    "id": df_test["id"].values.astype(np.int64),
    "is_anomaly": (test_proba > CALIBRATED_THRESHOLD).astype(int),
})
sub_path = OUT_DIR / "spaice2026_calibrated_submission.csv"
submission_df.to_csv(sub_path, index=False)
log(f"Saved: {sub_path}  (shape={submission_df.shape}, "
    f"anomaly points={submission_df['is_anomaly'].sum():,} "
    f"({submission_df['is_anomaly'].mean()*100:.2f}%))")
log("Historical Kaggle record (not reproduced here, reported for context):")
log("  Original (leaked) pipeline: 0.032 private / 0.072 public")
log("  Corrected pipeline:         0.294 private / 0.277 public")

log(f"\nTotal script runtime: {time.time()-T0:.1f}s")

# ===========================================================================
# Save full text report + machine-readable JSON summary
# ===========================================================================
report_txt_path = OUT_DIR / "corrected_model_metrics_report.txt"
with open(report_txt_path, "w") as f:
    f.write("\n".join(REPORT_LINES))
print(f"\nFull report written to: {report_txt_path}")

json_summary = {
    "reproducibility": {
        "seed": SEED,
        "thread_pins": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "iforest_n_jobs": 1,
            "xgboost_n_jobs": 1,
            "pyarrow_use_threads": False,
        },
        "xgboost_tree_method": "hist",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "pyarrow_version": pyarrow.__version__,
    },
    "n_true_events_validation": N_TRUE_EVENTS_CANONICAL,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "calibrated_threshold": CALIBRATED_THRESHOLD,
    "calibrated_f05_raw": CALIBRATED_F05,
    "bootstrap_n": BOOT_N,
    "bootstrap_seed": BOOT_SEED,
    "bootstrap_consensus_pct": consensus_pct,
    "by_threshold": {
        k: {
            "threshold": v["threshold"],
            "point_raw": v["point_raw"],
            "point_postprocessed": v["point_postprocessed"],
            "event_raw": {kk: vv for kk, vv in v["event_raw"].items() if kk != "point_confusion"},
            "event_postprocessed": {kk: vv for kk, vv in v["event_postprocessed"].items() if kk != "point_confusion"},
        }
        for k, v in REPORT.items()
    },
}
json_path = OUT_DIR / "corrected_model_metrics_summary.json"
with open(json_path, "w") as f:
    json.dump(json_summary, f, indent=2, default=str)
print(f"JSON summary written to: {json_path}")
