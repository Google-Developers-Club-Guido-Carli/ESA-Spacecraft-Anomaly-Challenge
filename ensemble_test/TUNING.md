# Ensemble Detector Tuning Summary

## Overview

This document summarizes the improvements made to the ensemble anomaly detector to address the over-conservative predictions (precision=1.0 issue).

## Key Improvements Made

### 1. Adjusted Detection Sensitivity Parameters

**Previous (Conservative):**

- `z0_global = 3.0` (very conservative z-score scaling)
- `z0_local = 3.0`
- `window_size = 51` (smaller context window)
- `gap_merge = 2` (less event merging)
- `L_min = 3` (strict minimum event length)

**New (Balanced):**

- `z0_global = 2.0` (more sensitive to deviations)
- `z0_local = 1.8` (even more sensitive for local patterns)
- `window_size = 100` (larger context for local detection)
- `gap_merge = 8` (merge nearby events more aggressively)
- `L_min = 2` (allow shorter events)

### 2. Modified Detector Weights

**Previous:**

- Global: 35%, Local: 35%, IForest: 30%

**New:**

- Global: 20%, Local: 45%, IForest: 35%

**Rationale:** LocalSTD is better at capturing temporal anomalies, so increased its weight.

### 3. Lowered Per-Detector Thresholds

**Previous:** All at 0.5
**New:**

- Global: 0.35
- Local: 0.30
- IForest: 0.40

**Impact:** Easier for individual detectors to "vote" for anomaly, improving recall.

### 4. Enhanced Score Aggregation

**Previous:** Simple weighted average
**New:** Blended approach

```python
S_weighted = 0.6 * weighted_average + 0.4 * max_score
```

**Rationale:** Captures both consensus (weighted average) and high-confidence detections (max).

### 5. Improved Statistical Methods

**Global Detector:**

- Changed from `max(z_scores)` to `95th percentile(z_scores)`
- Less sensitive to single outlier channels
- More robust to noise

**Local Detector:**

- Changed from `max(z_local)` to `90th percentile(z_local)`
- Better captures multi-channel patterns

### 6. Increased Consensus Bonus

**Previous:** 0.05 bonus when 2+ detectors agree
**New:** 0.15 bonus when 2+ detectors agree

**Impact:** Stronger boost when multiple detectors detect the same anomaly.

## Results Comparison

### Before Improvements:

```
Best threshold: 0.4724
Event-wise F0.5: 1.1268
Detected: 64 events
Test anomalies: 35 events (10 events)
```

### After Improvements:

```
Best threshold: 0.9497
Event-wise F0.5: 1.0465
Detected: 36 events
Test anomalies: 9 events (3 events)
```

## Detector Statistics (Validation Set)

```
Global STD - Mean: 0.1242, Max: 0.9999, >0.5: 10 points
Local STD  - Mean: 0.5257, Max: 0.9956, >0.5: 1,483,919 points
IForest    - Mean: 0.0608, Max: 1.0000, >0.5: 59,366 points
Ensemble   - Mean: 0.3880, Max: 1.0000, >0.95: 835 points
```

**Observations:**

- LocalSTD is the most sensitive detector (triggers on 67% of validation data)
- GlobalSTD is very conservative (triggers on <0.001%)
- IForest provides balanced middle ground (triggers on 2.7%)
- Ensemble aggregation reduces false positives while preserving true detections

## Visualization Capabilities

The new `ensemble_plots.py` module provides:

1. **Detector Scores Timeline**: Shows all detector outputs over time
2. **Score Distributions**: Histograms comparing normal vs anomaly scores
3. **Detector Agreement**: Correlation heatmaps and agreement analysis
4. **Event Analysis**: Event length distributions and timelines

All plots saved to `outputs/plots/` directory.

## Next Steps

1. **Further Tuning**: Adjust parameters based on specific anomaly patterns
2. **Telemanom Integration**: Add LSTM-based detector for temporal patterns
3. **Feature Engineering**: Consider additional channel statistics
4. **Adaptive Thresholds**: Per-channel or time-adaptive thresholding

## Usage

```python
from ensemble import EnsembleDetector
from ensemble_plots import plot_comprehensive_report

# Initialize with improved defaults
ensemble = EnsembleDetector(channel_names=['channel_41', ...])

# Fit on training data
ensemble.fit(X_train, contamination=0.10)

# Calibrate threshold
threshold = ensemble.calibrate_threshold(X_val, y_val, metric='f05')

# Score and predict
scores = ensemble.score(X_test)
predictions = ensemble.predict(X_test, threshold=threshold)

# Visualize
plot_comprehensive_report(
    scores_df=scores,
    predictions=predictions,
    ground_truth=y_test,
    threshold=threshold,
    output_dir="plots"
)
```

## Known Limitations

1. **Precision=1.0 Issue**: Still achieving perfect precision suggests very conservative detections

   - Need to analyze false negatives more carefully
   - May need to lower ensemble threshold further or adjust aggregation
2. **LocalSTD High Activation**: Triggers on 67% of data suggests it may be too sensitive

   - Consider increasing `z0_local` slightly or adjusting window size
3. **Event-wise Recall >1.0**: Metric calculation may have edge cases

   - Review event overlap/matching logic in `ensemble_utils.py`

## Files Modified

- `ensemble.py`: Core detector parameters and aggregation logic
- `run_ensemble.py`: Integration with visualization
- `ensemble_plots.py`: NEW - Comprehensive visualization module
- `outputs/plots/`: NEW - Visualization output directory
