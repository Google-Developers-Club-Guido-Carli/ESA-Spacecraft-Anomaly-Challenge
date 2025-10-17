# Ensemble Implementation Summary

## What Was Implemented

Implemented the ensemble anomaly detection algorithm pseudocode into the `ensemble_test/` folder using LocalSTD, GlobalSTD and iForest.

### Files Created

1. **ensemble_utils.py** (289 lines)

   - `z_to_score()` - Transform z-scores to [0,1] using exponential mapping
   - `percentile_normalize()` - Robust normalization using percentile clipping
   - `extract_events()` - Extract contiguous anomaly segments
   - `prune_short_events()` - Remove events shorter than L_min
   - `merge_nearby_events()` - Merge events separated by small gaps
   - `postprocess_events()` - Complete postprocessing pipeline
   - `eventwise_precision_recall_f05()` - Event-wise metrics computation
   - `pointwise_metrics()` - Point-wise metrics for debugging
2. **ensemble.py** (343 lines)

   - `EnsembleDetector` class with:
     - `fit()` - Fit detectors on training data
     - `_detect_globalstd()` - Global mean/std outlier detection
     - `_detect_localstd()` - Rolling window local z-score detection
     - `_detect_iforest()` - Isolation Forest multivariate detection
     - `score()` - Compute ensemble scores with consensus bonus
     - `predict()` - Binary predictions with temporal postprocessing
     - `calibrate_threshold()` - Optimize threshold on validation set
3. **run_ensemble.py** (177 lines)

   - Complete demo script that:
     - Loads train/test from `.data/` (parquet files)
     - Splits train into train/val (85%/15%)
     - Fits ensemble on training data
     - Calibrates threshold on validation (maximizes event-wise F0.5)
     - Evaluates on validation with event-wise and point-wise metrics
     - Predicts on test set
     - Saves submission and detailed scores to `outputs/`
4. **README.md**

   - Documentation of algorithm, usage, hyperparameters, and outputs
5. **test_load.py**

   - Diagnostic script to verify data loading

## Algorithm Details

### Detector Combination

The ensemble combines three detectors:

1. **GlobalSTD** (weight=0.35)

   - Compares each point to global mean ± N×std
   - Fast, catches gross outliers
   - z_max(t) = max_c |X(t,c) - μ_c| / σ_c
   - score = 1 - exp(-z_max/3.0)
2. **LocalSTD** (weight=0.35)

   - Rolling window (size=51) local statistics
   - Adaptive to changing baselines
   - z_local_max(t) = max_c |X(t,c) - μ_local_c(t)| / σ_local_c(t)
   - score = 1 - exp(-z_local_max/3.0)
3. **IsolationForest** (weight=0.30)

   - Multivariate structural anomaly detection
   - Catches correlated anomalies across channels
   - Scores normalized via percentile clipping (p5, p95)

### Aggregation

```
S_raw(t) = 0.35×s_global + 0.35×s_local + 0.30×s_iforest

votes(t) = count of detectors with score ≥ threshold (0.5)
bonus = 0.05 if votes ≥ 2

S(t) = clip(S_raw + bonus, 0, 1)
```

### Temporal Postprocessing

After thresholding S(t) ≥ T_final:

1. Remove events < 3 samples long
2. Merge events separated by ≤ 2 samples
3. Output compact event predictions

### Calibration

Threshold T_final is calibrated by:

- Sweeping 200 values in [0,1]
- Computing event-wise F0.5 on validation
- Selecting threshold that maximizes F0.5

## Dataset

- **Train**: 14,728,321 samples (channels 41-46 + is_anomaly)

  - Train subset: 12,519,072 samples (85%)
  - Val subset: 2,209,249 samples (15%)
  - Anomaly rate: 10.48%
- **Test**: 521,280 samples (channels 41-46, no labels)

## Expected Outputs

Once complete, the script will generate:

1. **outputs/ensemble_submission.csv**

   - Columns: id, is_anomaly
   - Binary predictions for test set
2. **outputs/ensemble_scores.csv**

   - Columns: id, score, s_global, s_local, s_iforest
   - Detailed per-detector scores for test set
3. **Console output**

   - Validation metrics (event-wise and point-wise)
   - Best threshold found
   - Number of detected events
   - Test prediction statistics

## Metrics

The algorithm optimizes for **event-wise F0.5**:

- Precision favored over recall (β=0.5)
- Each contiguous anomaly segment = 1 event
- Predicted event is TP if it overlaps any true event

Both event-wise and point-wise metrics are reported for validation.

## Next Steps:

1. Review validation F0.5 score
2. Examine detected vs true events
3. Consider hyperparameter tuning:
   - Detector weights
   - Window size for LocalSTD
   - z0 parameters
   - Postprocessing params (L_min, gap_merge)
4. Add Telemanom LSTM detector (weight=0.4) (?)
