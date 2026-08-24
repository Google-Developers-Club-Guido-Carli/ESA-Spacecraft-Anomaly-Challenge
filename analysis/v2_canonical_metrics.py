#!/usr/bin/env python3
"""
Canonical, single-source-of-truth metrics for the paper's Results section.

THE ONE MODEL: the fully corrected v2 pipeline as it lives in
hybrid_ensemble_notebook.ipynb (commit 199a294 and later) --
  - chronological block selection: last 2 of 10 timeline blocks (8, 9) = validation,
    blocks 0-7 = training. No random selection anywhere.
  - no "id" feature: feature set is exactly channel_41-46 (6 channels).
  - engineer_features() with its built-in inf -> 0 sanitization (the corrected
    version, not the pre-fix one).
  - StandardScaler + IsolationForest fit on training blocks only.
  - XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, objective='binary:logistic', eval_metric='auc',
    random_state=42, n_jobs=-1).

Real Kaggle result for this exact model (independently confirmed, per the user):
private 0.294 / public 0.277, submitted as outputs/v2_corrected_submission.csv
(threshold=0.95, built from this same corrected pipeline).

HOW TO RUN THIS YOURSELF:
    cd /Users/yaseminates/projects/ESA-Spacecraft-Anomaly-Challenge
    python3 analysis/v2_canonical_metrics.py

Requires: numpy, pandas, pyarrow, scikit-learn, xgboost, psutil (all already
used elsewhere in this project). Reads only .data/train.parquet (via chunked,
column-projected pyarrow reads -- never the full 89-column file). No files
outside analysis/ are written; no submission file is built; no protected file
in outputs/ is read or touched.

Everything below is computed fresh in this one execution -- nothing is loaded
from a prior run or from memory of an earlier session. Outputs are written to:
  analysis/v2_canonical_metrics_results.json   (machine-readable, exact values)
  analysis/v2_canonical_metrics_report.md      (human-readable tables)
"""
import gc
import io
import json
import time
import contextlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix)
import xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / '.data'
ANALYSIS_DIR = REPO / 'analysis'
CHANNEL_COLS = [f'channel_{i}' for i in range(41, 47)]

MEM_FLOOR_GB = 0.40
t0 = time.time()
def elapsed():
    return f"{time.time()-t0:6.1f}s"

def check_memory(stage):
    avail = psutil.virtual_memory().available / 1e9
    print(f"  [mem @ {stage}] available: {avail:.3f} GB", flush=True)
    if avail < MEM_FLOOR_GB:
        raise MemoryError(f"Available RAM {avail:.3f} GB below floor at '{stage}'.")
    return avail


# ============================================================
# Verbatim pipeline functions, reproduced exactly as they currently exist in
# hybrid_ensemble_notebook.ipynb (cells 18 / 28 / 39 -- the corrected versions)
# ============================================================

def compute_telemanom_scores(X, window_size=50, use_lstm=False):
    n_samples, n_features = X.shape
    prediction_errors = np.zeros(n_samples)
    for i in range(window_size, n_samples):
        window = X[i-window_size:i]
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
        cumulative_errors[i] = np.sum(prediction_errors[i-cum_window:i])
    if np.max(cumulative_errors) > 0:
        cumulative_errors = cumulative_errors / np.max(cumulative_errors)
    combined_scores = 0.7 * prediction_errors + 0.3 * cumulative_errors
    if np.max(combined_scores) > np.min(combined_scores):
        normalized_scores = (combined_scores - np.min(combined_scores)) / \
                          (np.max(combined_scores) - np.min(combined_scores))
    else:
        normalized_scores = combined_scores
    return normalized_scores, prediction_errors, cumulative_errors


def engineer_features(X, iforest_scores, telemanom_scores, window_sizes=[10, 30, 50]):
    """Current (post-leakage-fix) 16-feature version, including the inf->0 fix."""
    n_samples, n_features = X.shape
    features = []
    feature_names = []
    features.append(iforest_scores.reshape(-1, 1)); feature_names.append('iforest_score')
    features.append(telemanom_scores.reshape(-1, 1)); feature_names.append('telemanom_score')
    detector_agreement = np.abs(iforest_scores - telemanom_scores)
    features.append(detector_agreement.reshape(-1, 1)); feature_names.append('detector_agreement')
    for window in window_sizes:
        rolling_mean = pd.DataFrame(X).rolling(window=window, min_periods=1).mean().values
        rolling_mean_dev = np.mean(np.abs(X - rolling_mean), axis=1)
        features.append(rolling_mean_dev.reshape(-1, 1)); feature_names.append(f'rolling_mean_dev_{window}')
        rolling_std = pd.DataFrame(X).rolling(window=window, min_periods=1).std().values
        rolling_std_score = np.mean(rolling_std, axis=1)
        features.append(rolling_std_score.reshape(-1, 1)); feature_names.append(f'rolling_std_{window}')
    for lag in [1, 5, 10]:
        delta = np.zeros((n_samples, n_features))
        delta[lag:] = X[lag:] - X[:-lag]
        delta_score = np.mean(np.abs(delta), axis=1)
        features.append(delta_score.reshape(-1, 1)); feature_names.append(f'delta_{lag}')
    slopes = np.zeros(n_samples)
    slope_window = 10
    for i in range(slope_window, n_samples):
        window_data = X[i-slope_window:i]
        time_idx = np.arange(slope_window)
        feature_slopes = []
        for feat in range(n_features):
            slope = np.polyfit(time_idx, window_data[:, feat], 1)[0]
            feature_slopes.append(abs(slope))
        slopes[i] = np.mean(feature_slopes)
    features.append(slopes.reshape(-1, 1)); feature_names.append('slope_magnitude')
    corr_window = 30
    n_corr_pairs = min(3, n_features - 1)
    for i in range(n_corr_pairs):
        for j in range(i + 1, min(i + 2, n_features)):
            corr_values = pd.Series(X[:, i]).rolling(
                window=corr_window, min_periods=10
            ).corr(pd.Series(X[:, j])).fillna(0).values
            features.append(corr_values.reshape(-1, 1)); feature_names.append(f'corr_ch{i}_ch{j}')
    X_features = np.hstack(features)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    return X_features, feature_names


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
    merged_segments = []
    if segments:
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if start - current_end <= max_gap:
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end))
    filtered_segments = [(s, e) for s, e in merged_segments if e - s + 1 >= min_event_length]
    processed_scores = np.zeros_like(scores)
    for start, end in filtered_segments:
        processed_scores[start:end + 1] = scores[start:end + 1]
    return processed_scores, filtered_segments


def find_events(binary_labels):
    events = []
    in_event = False
    start = 0
    for i in range(len(binary_labels)):
        if binary_labels[i] == 1 and not in_event:
            start = i; in_event = True
        elif binary_labels[i] == 0 and in_event:
            events.append((start, i - 1)); in_event = False
    if in_event:
        events.append((start, len(binary_labels) - 1))
    return events


def catch_events(events, binary_preds):
    caught, missed = [], []
    for start, end in events:
        if np.any(binary_preds[start:end + 1] == 1):
            caught.append((start, end))
        else:
            missed.append((start, end))
    return caught, missed


def overlaps_any(interval, events):
    a_start, a_end = interval
    for b_start, b_end in events:
        if a_start <= b_end and a_end >= b_start:
            return True
    return False


def fbeta(p, r, beta_val=0.5):
    denom = beta_val ** 2 * p + r
    return ((1 + beta_val ** 2) * p * r) / denom if denom > 0 else 0.0


def bin_label(length, edges=(100, 1000)):
    if length < edges[0]: return "short (<100)"
    elif length < edges[1]: return "medium (100-1000)"
    else: return "long (>1000)"


def event_metrics_per_block(pred_binary_full, block_lengths, true_events_per_block, nominal_mask_full, n_nominal_points):
    """Exact event-wise TPe/FNe/FPe + TNR-corrected precision + F0.5, aggregated per block."""
    TPe = FNe = FPe = 0
    offset = 0
    for length, block_true_events in zip(block_lengths, true_events_per_block):
        block_pred = pred_binary_full[offset:offset + length]
        caught, missed = catch_events(block_true_events, block_pred)
        TPe += len(caught); FNe += len(missed)
        block_pred_events = find_events(block_pred)
        FPe += sum(1 for pe in block_pred_events if not overlaps_any(pe, block_true_events))
        offset += length
    n_true = TPe + FNe
    ev_recall = TPe / n_true if n_true > 0 else 0.0
    ev_prec_uncorr = TPe / (TPe + FPe) if (TPe + FPe) > 0 else 0.0
    fp_points_mask = nominal_mask_full & (pred_binary_full == 1)
    n_fp_points = int(fp_points_mask.sum())
    tnr = (1 - n_fp_points / n_nominal_points) if n_nominal_points > 0 else 0.0
    ev_prec = ev_prec_uncorr * tnr if ev_prec_uncorr > 0 else 0.0
    ev_f05 = fbeta(ev_prec, ev_recall, 0.5)
    return dict(TPe=TPe, FNe=FNe, FPe=FPe, event_recall=ev_recall,
                event_precision_raw=ev_prec_uncorr, tnr=tnr,
                event_precision_corrected=ev_prec, f05=ev_f05)


def postprocess_full(val_proba, block_lengths, threshold, min_event_length=3, max_gap=5):
    out = []
    offset = 0
    for length in block_lengths:
        block = val_proba[offset:offset + length]
        with contextlib.redirect_stdout(io.StringIO()):
            processed, _ = postprocess_scores(block, threshold=threshold,
                                               min_event_length=min_event_length, max_gap=max_gap)
        out.append((processed > 0).astype(int))
        offset += length
    return np.concatenate(out)


# ============================================================
# STEP 0: reconstruct the ONE corrected v2 model + its own chronological val set
# ============================================================
print(f"[{elapsed()}] STEP 0: reconstructing the corrected v2 pipeline", flush=True)
check_memory("start")

pf = pq.ParquetFile(DATA_DIR / 'train.parquet')
n_full = pf.metadata.num_rows
block_size = 150000
n_blocks_target = 10
block_starts = np.linspace(0, n_full - block_size, n_blocks_target).astype(int)
block_ends = block_starts + block_size
val_block_idx = [8, 9]          # chronological: last 2 of 10 blocks
train_block_idx = [0, 1, 2, 3, 4, 5, 6, 7]
print(f"  train_block_idx={train_block_idx}  val_block_idx={val_block_idx} (chronological, no randomness)")

needed_cols = CHANNEL_COLS + ['is_anomaly']
all_ranges = sorted((int(block_starts[b]), int(block_ends[b]), b) for b in range(n_blocks_target))
buffers = {b: [] for b in range(n_blocks_target)}
row_offset = 0
n_rg = pf.metadata.num_row_groups
for rg_idx in range(n_rg):
    rg_nrows = pf.metadata.row_group(rg_idx).num_rows
    rg_start, rg_end = row_offset, row_offset + rg_nrows
    overlaps = [(s, e, b) for (s, e, b) in all_ranges if e > rg_start and s < rg_end]
    if overlaps:
        table = pf.read_row_group(rg_idx, columns=needed_cols)
        arr = {c: table.column(c).to_numpy(zero_copy_only=False) for c in needed_cols}
        for s, e, b in overlaps:
            lo, hi = max(s, rg_start) - rg_start, min(e, rg_end) - rg_start
            piece_X = np.stack([arr[c][lo:hi] for c in CHANNEL_COLS], axis=1).astype(np.float32)
            piece_y = arr['is_anomaly'][lo:hi].astype(np.uint8)
            buffers[b].append((piece_X, piece_y))
        del table, arr
    row_offset = rg_end
    gc.collect()
raw_blocks_X, raw_blocks_y = {}, {}
for b in range(n_blocks_target):
    parts = buffers[b]
    raw_blocks_X[b] = np.concatenate([p[0] for p in parts], axis=0)
    raw_blocks_y[b] = np.concatenate([p[1] for p in parts], axis=0)
buffers.clear(); gc.collect()
check_memory("after block extraction")

raw_train_X_v2 = np.concatenate([raw_blocks_X[b] for b in train_block_idx], axis=0)
scaler_v2 = StandardScaler()
scaler_v2.fit(raw_train_X_v2)
scaled_blocks_X = {b: scaler_v2.transform(raw_blocks_X[b]) for b in range(n_blocks_target)}
del raw_blocks_X, raw_train_X_v2; gc.collect()

scaled_train_X_v2 = np.concatenate([scaled_blocks_X[b] for b in train_block_idx], axis=0)
iforest_v2 = IsolationForest(n_estimators=100, contamination=0.10, max_samples="auto",
                              random_state=42, n_jobs=-1, verbose=0)
iforest_v2.fit(scaled_train_X_v2)
iforest_raw_blocks = {b: -iforest_v2.score_samples(scaled_blocks_X[b]) for b in range(n_blocks_target)}
train_raw_concat = np.concatenate([iforest_raw_blocks[b] for b in train_block_idx])
_gmin, _gmax = train_raw_concat.min(), train_raw_concat.max()
iforest_scores_blocks = {b: (iforest_raw_blocks[b] - _gmin) / (_gmax - _gmin + 1e-10) for b in range(n_blocks_target)}
del scaled_train_X_v2, iforest_raw_blocks, train_raw_concat; gc.collect()
print(f"[{elapsed()}] scaler_v2/iforest_v2 fit. Stage 2 (Telemanom + feature engineering, all 10 blocks)...")

X_eng_blocks = {}
for b in range(n_blocks_target):
    tele, _, _ = compute_telemanom_scores(scaled_blocks_X[b], window_size=50)
    X_eng, feature_names = engineer_features(scaled_blocks_X[b], iforest_scores_blocks[b], tele, window_sizes=[10, 30, 50])
    X_eng_blocks[b] = X_eng
    del tele
del scaled_blocks_X; gc.collect()
check_memory("after Stage 2")

X_eng_train = np.concatenate([X_eng_blocks[b] for b in train_block_idx], axis=0)
y_train = np.concatenate([raw_blocks_y[b] for b in train_block_idx], axis=0)
X_eng_val = np.concatenate([X_eng_blocks[b] for b in val_block_idx], axis=0)
y_val = np.concatenate([raw_blocks_y[b] for b in val_block_idx], axis=0)
val_block_lengths = [len(raw_blocks_y[b]) for b in val_block_idx]
del X_eng_blocks; gc.collect()

xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               objective="binary:logistic", eval_metric="auc",
                               random_state=42, n_jobs=-1)
xgb_model.fit(X_eng_train, y_train, verbose=False)
val_proba = xgb_model.predict_proba(X_eng_val)[:, 1]
del X_eng_train, X_eng_val; gc.collect()
print(f"[{elapsed()}] Model trained. Validation set: {len(y_val):,} rows, "
      f"anomaly rate {y_val.mean()*100:.4f}%")
check_memory("after training")

true_events_per_block = []
offset = 0
for length in val_block_lengths:
    block_y = y_val[offset:offset + length]
    true_events_per_block.append(find_events(block_y))
    offset += length
n_true_events = sum(len(e) for e in true_events_per_block)
all_true_event_lengths = sorted(e - s + 1 for block in true_events_per_block for s, e in block)
nominal_mask = (y_val == 0)
n_nominal_points = int(nominal_mask.sum())

print(f"\n{'='*90}")
print(f" TRUE ANOMALY EVENTS IN THIS VALIDATION SET: {n_true_events}")
print(f" Event lengths (points): {all_true_event_lengths}")
print(f"{'='*90}")

results = {
    "model_description": (
        "Corrected v2 pipeline: chronological block selection (last 2 of 10 = val), "
        "no id feature (6 channels only), engineer_features() with inf->0 fix, "
        "trained on blocks 0-7 (1,200,000 rows), evaluated on blocks 8-9 (300,000 rows)."
    ),
    "real_kaggle_score": {
        "private": 0.294, "public": 0.277,
        "submission_file": "outputs/v2_corrected_submission.csv",
        "submission_threshold": 0.95,
        "note": "Independently confirmed by actual Kaggle submission of this exact corrected model."
    },
    "validation_set": {
        "n_rows": int(len(y_val)),
        "anomaly_rate_pct": float(y_val.mean() * 100),
        "val_block_idx": val_block_idx,
        "n_true_events": n_true_events,
        "true_event_lengths": all_true_event_lengths,
    },
}

# ============================================================
# STEPS 1-2: point-wise metrics at threshold=0.5 and threshold=0.95
# ============================================================
print(f"\n[{elapsed()}] STEPS 1-2: point-wise metrics at threshold=0.5 and threshold=0.95", flush=True)

roc_auc = roc_auc_score(y_val, val_proba)   # threshold-independent
pr_auc = average_precision_score(y_val, val_proba)  # threshold-independent

point_wise = {}
for thr in [0.5, 0.95]:
    pred = (val_proba > thr).astype(int)
    cm = confusion_matrix(y_val, pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    prec = precision_score(y_val, pred, pos_label=1, zero_division=0)
    rec = recall_score(y_val, pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_val, pred, pos_label=1, zero_division=0)
    point_wise[str(thr)] = {
        "threshold": thr, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
    }
    print(f"\n  --- threshold={thr} ---")
    print(f"  Confusion matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"  Precision={prec:.6f}  Recall={rec:.6f}  F1={f1:.6f}")
    print(f"  ROC-AUC={roc_auc:.6f}  PR-AUC={pr_auc:.6f}  (threshold-independent, same at both thresholds)")

results["point_wise_metrics"] = point_wise

# ============================================================
# STEPS 3-4: event-wise metrics at threshold=0.5 and threshold=0.95, raw + post-processed
# ============================================================
print(f"\n[{elapsed()}] STEPS 3-4: event-wise metrics (raw + post-processed) at both thresholds", flush=True)

event_wise = {}
size_stratified = {}
for thr in [0.5, 0.95]:
    pred_raw = (val_proba > thr).astype(int)
    res_raw = event_metrics_per_block(pred_raw, val_block_lengths, true_events_per_block, nominal_mask, n_nominal_points)

    pred_pp_full = postprocess_full(val_proba, val_block_lengths, thr, min_event_length=3, max_gap=5)
    res_pp = event_metrics_per_block(pred_pp_full, val_block_lengths, true_events_per_block, nominal_mask, n_nominal_points)

    event_wise[str(thr)] = {"raw": res_raw, "post_processed": res_pp}

    print(f"\n  --- threshold={thr} ---")
    print(f"  RAW:            TPe={res_raw['TPe']} FNe={res_raw['FNe']} FPe={res_raw['FPe']}  "
          f"recall={res_raw['event_recall']:.6f}  prec_corrected={res_raw['event_precision_corrected']:.6f}  "
          f"F0.5={res_raw['f05']:.6f}")
    print(f"  POST-PROCESSED: TPe={res_pp['TPe']} FNe={res_pp['FNe']} FPe={res_pp['FPe']}  "
          f"recall={res_pp['event_recall']:.6f}  prec_corrected={res_pp['event_precision_corrected']:.6f}  "
          f"F0.5={res_pp['f05']:.6f}")

    # size-stratified recall (post-processed), with exact per-event caught/missed detail
    offset_map = []
    offset = 0
    for length, block_true in zip(val_block_lengths, true_events_per_block):
        for (s, e) in block_true:
            offset_map.append((s + offset, e + offset))
        offset += length
    event_rows = []
    for (s, e) in offset_map:
        length = e - s + 1
        caught = bool(np.any(pred_pp_full[s:e + 1] == 1))
        event_rows.append({"length": int(length), "bin": bin_label(length), "caught": caught})
    event_rows.sort(key=lambda r: r["length"])

    bins_summary = {}
    for b in ["short (<100)", "medium (100-1000)", "long (>1000)"]:
        sub = [r for r in event_rows if r["bin"] == b]
        n_total = len(sub)
        n_caught = sum(1 for r in sub if r["caught"])
        bins_summary[b] = {
            "n_events": n_total, "n_caught": n_caught,
            "recall": (n_caught / n_total) if n_total > 0 else None,
            "event_lengths_caught": [r["length"] for r in sub if r["caught"]],
            "event_lengths_missed": [r["length"] for r in sub if not r["caught"]],
        }
    size_stratified[str(thr)] = {"events": event_rows, "bins": bins_summary}

    print(f"\n  SIZE-STRATIFIED RECALL @ threshold={thr} (post-processed):")
    for r in event_rows:
        print(f"    length={r['length']:5d}  bin={r['bin']:<20}  caught={r['caught']}")
    for b, s in bins_summary.items():
        rstr = f"{s['recall']:.3f}" if s['recall'] is not None else "n/a"
        print(f"    {b:<20} events={s['n_events']}  caught={s['n_caught']}  recall={rstr}")

results["event_wise_metrics"] = event_wise
results["size_stratified_recall"] = size_stratified

# consistency check on true event count
event_counts_seen = {n_true_events}
for thr_key, sd in size_stratified.items():
    event_counts_seen.add(len(sd["events"]))
for thr_key, ew in event_wise.items():
    event_counts_seen.add(ew["raw"]["TPe"] + ew["raw"]["FNe"])
    event_counts_seen.add(ew["post_processed"]["TPe"] + ew["post_processed"]["FNe"])
print(f"\n[{elapsed()}] CONSISTENCY CHECK: true-event counts seen across all steps: {sorted(event_counts_seen)}")
if len(event_counts_seen) == 1:
    print(f"  OK -- single consistent value ({n_true_events}) used throughout.")
else:
    print(f"  *** WARNING: inconsistent event counts found across steps: {sorted(event_counts_seen)} ***")
results["event_count_consistency_check"] = {
    "values_seen": sorted(event_counts_seen),
    "consistent": len(event_counts_seen) == 1,
}

# ============================================================
# STEP 6: threshold sweep + bootstrap validation (rerun fresh, not recalled)
# ============================================================
print(f"\n[{elapsed()}] STEP 6: threshold sweep (0.05-0.99) + bootstrap validation (500 iterations)", flush=True)
thresholds = np.round(np.arange(0.05, 1.00, 0.02), 2)

precomputed = {}
sweep_rows = []
for t in thresholds:
    pred_by_block = []
    offset = 0
    for length in val_block_lengths:
        block_proba = val_proba[offset:offset + length]
        with contextlib.redirect_stdout(io.StringIO()):
            processed, _ = postprocess_scores(block_proba, threshold=t, min_event_length=3, max_gap=5)
        pred_by_block.append((processed > 0).astype(int))
        offset += length
    pred_events_by_block = [find_events(p) for p in pred_by_block]
    pred_full = np.concatenate(pred_by_block)
    fp_points_mask = nominal_mask & (pred_full == 1)
    n_fp_points = int(fp_points_mask.sum())
    tnr = (1 - n_fp_points / n_nominal_points) if n_nominal_points > 0 else 0.0
    precomputed[t] = dict(pred_by_block=pred_by_block, pred_events_by_block=pred_events_by_block, tnr=tnr)

    res = event_metrics_per_block(pred_full, val_block_lengths, true_events_per_block, nominal_mask, n_nominal_points)
    sweep_rows.append({"threshold": float(t), **res})

sweep_df = pd.DataFrame(sweep_rows)
print(f"\n  {'threshold':>10}{'TPe':>6}{'FNe':>6}{'FPe':>7}{'recall':>9}{'prec_corr':>11}{'F0.5':>9}")
for _, r in sweep_df.iterrows():
    print(f"  {r['threshold']:>10.2f}{int(r['TPe']):>6d}{int(r['FNe']):>6d}{int(r['FPe']):>7d}"
          f"{r['event_recall']:>9.3f}{r['event_precision_corrected']:>11.4f}{r['f05']:>9.4f}")

best_row = sweep_df.loc[sweep_df['f05'].idxmax()]
best_threshold = float(best_row['threshold'])
print(f"\n  >>> OPTIMAL THRESHOLD (post-processed event-wise F0.5): {best_threshold:.2f}, "
      f"F0.5={best_row['f05']:.6f}, TPe={int(best_row['TPe'])}, FPe={int(best_row['FPe'])} <<<")

# bootstrap
event_pool = []
for block_pos, events in enumerate(true_events_per_block):
    for ev in events:
        event_pool.append((block_pos, ev))

def lightweight_f05(boot_true_events_per_block, t):
    pre = precomputed[t]
    TPe = FNe = FPe = 0
    for block_pos in range(len(val_block_lengths)):
        block_true = boot_true_events_per_block[block_pos]
        block_pred = pre['pred_by_block'][block_pos]
        caught, missed = catch_events(block_true, block_pred)
        TPe += len(caught); FNe += len(missed)
        FPe += sum(1 for pe in pre['pred_events_by_block'][block_pos] if not overlaps_any(pe, block_true))
    n_true = TPe + FNe
    recall = TPe / n_true if n_true > 0 else 0.0
    prec_uncorr = TPe / (TPe + FPe) if (TPe + FPe) > 0 else 0.0
    prec_corr = prec_uncorr * pre['tnr'] if prec_uncorr > 0 else 0.0
    return fbeta(prec_corr, recall, 0.5)

N_ITER = 500
SEED = 12345
rng_boot = np.random.RandomState(SEED)
TOL = 1e-9
iters_where_t_is_best = {t: 0 for t in thresholds}
best_f05_per_iter = []

for it in range(N_ITER):
    draw_idx = rng_boot.randint(0, len(event_pool), size=len(event_pool))
    resampled = [event_pool[i] for i in draw_idx]
    boot_true_events_per_block = [[] for _ in val_block_lengths]
    for block_pos, ev in resampled:
        boot_true_events_per_block[block_pos].append(ev)
    f05_by_t = {t: lightweight_f05(boot_true_events_per_block, t) for t in thresholds}
    max_f05 = max(f05_by_t.values())
    tied = set(t for t, f in f05_by_t.items() if abs(f - max_f05) <= TOL)
    for t in tied:
        iters_where_t_is_best[round(t, 2)] += 1
    best_f05_per_iter.append(max_f05)

best_f05_per_iter = np.array(best_f05_per_iter)
n_best_thr = iters_where_t_is_best.get(round(best_threshold, 2), 0)
bootstrap_consensus_pct = n_best_thr / N_ITER * 100
print(f"\n  Bootstrap: N_ITER={N_ITER}, seed={SEED}, event pool size={len(event_pool)}")
print(f"  >>> Threshold {best_threshold:.2f} was among tied-best in {n_best_thr}/{N_ITER} "
      f"iterations ({bootstrap_consensus_pct:.1f}% consensus) <<<")
top5 = sorted(iters_where_t_is_best.items(), key=lambda kv: -kv[1])[:5]
print(f"  Top 5 thresholds by consensus: {[(t, f'{v}/{N_ITER}') for t, v in top5]}")

results["threshold_sweep"] = {
    "thresholds_tested": [float(t) for t in thresholds],
    "sweep_table": sweep_rows,
    "optimal_threshold": best_threshold,
    "optimal_f05": float(best_row['f05']),
}
results["bootstrap_validation"] = {
    "n_iterations": N_ITER,
    "seed": SEED,
    "event_pool_size": len(event_pool),
    "threshold_tested": best_threshold,
    "consensus_count": n_best_thr,
    "consensus_pct": bootstrap_consensus_pct,
    "top5_thresholds_by_consensus": [{"threshold": t, "count": v} for t, v in top5],
    "best_f05_per_iteration_mean": float(best_f05_per_iter.mean()),
    "best_f05_per_iteration_median": float(np.median(best_f05_per_iter)),
    "best_f05_per_iteration_std": float(best_f05_per_iter.std()),
}

# ============================================================
# Write outputs
# ============================================================
json_path = ANALYSIS_DIR / 'v2_canonical_metrics_results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[{elapsed()}] Wrote {json_path}")

print(f"[{elapsed()}] DONE. All numbers above were computed fresh in this single execution.")
