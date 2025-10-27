

# Adapter & XGBoost – Implementation

This folder contains the implementation of an adapter-based preprocessing layer and an XGBoost classifier for ESA spacecraft telemetry anomaly detection (channels 41–46).

## Files

- **adapter&xgboost.ipynb** — Main notebook combining feature adaptation, preprocessing, training, and anomaly scoring
- (Optional) **adapter.py** — Lightweight transformation block (if extracted later)
- (Optional) **xgb_utils.py** — Helper functions for scoring, threshold tuning, and event postprocessing

## Method Overview

This pipeline wraps raw telemetry features through a normalization adapter and then trains an **XGBoost** model to classify anomalies at the point level. Compared to unsupervised detectors, this method learns discriminative patterns directly from labeled anomalies.

### Adapter Layer

The adapter standardizes each feature dimension:

- Scaling to zero-mean, unit-variance
- Robust clipping to remove extreme values
- Optional exponential smoothing for temporal drift

This improves model stability and reduces bias from volatile channels.

### XGBoost Classifier

The model uses gradient-boosted decision trees:

- Handles non-linear relationships
- Captures cross-channel interactions
- Resistant to noise and scaling issues
- Naturally encodes feature importance

### Scoring Pipeline

For each timepoint *t*:

1. Pass features through the adapter
2. Compute class probability `p(t) = P(anomaly | x(t))`
3. Apply percentile calibration on validation
4. Threshold probabilities into binary predictions

### Event Postprocessing

After thresholding:

1. Remove events shorter than **3 samples**
2. Merge events separated by **≤2 samples**
3. Output compact start/end timestamps

These rules suppress flickering predictions and tighten event boundaries.

## Usage

Open the notebook and run all cells:

```bash
jupyter notebook adapter&xgboost.ipynb
```

The notebook will:

1. Load preprocessed train/test `.data` series
2. Apply adapter normalization
3. Train XGBoost using labeled anomalies
4. Tune threshold to maximize event-wise **F1**
5. Predict anomaly events on test data
6. Save submissions to `outputs/`

## Requirements

```
numpy
pandas
scikit-learn
xgboost
matplotlib
```

Install via:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

## Hyperparameters

Default configuration (validated on sample dataset):

- **max_depth**: 5
- **n_estimators**: 300
- **learning_rate**: 0.05
- **subsample**: 0.9
- **colsample_bytree**: 0.8
- **eval_metric**: logloss

Adapter parameters:

- **clip_percentiles**: (1%, 99%)
- **temporal_smoothing**: α = 0.15

Event rules:

- **L_min**: 3
- **gap_merge**: 2

## Outputs

- `outputs/xgb_submission.csv` — Binary anomaly predictions
- `outputs/xgb_scores.csv` — Per-point anomaly probabilities
- `outputs/feature_importance.png` — Ranked feature contributions

## Metrics

Validation compares:

- **Point-wise** accuracy, precision, recall, F1
- **Event-wise** precision, recall, F1 (primary metric)

Event-wise scoring rewards temporal coherence and minimizes overprediction.

---

## Comparison vs Other Approaches

### Telemanom (LSTM Forecasting)
✅ Models temporal evolution directly
✅ Excellent recall on gradual drift
❌ Training cost is high
❌ Can overpredict on noisy channels

Telemanom is ideal when *sequence dynamics* dominate anomaly behavior.

---

### Ensemble (GlobalSTD, LocalSTD, IsolationForest)
✅ Very fast, unsupervised
✅ Good precision due to consensus voting
❌ Limited non-linear interaction modeling
❌ Sensitive to window sizes

Ensembles shine when labels are scarce and runtime matters.

---

### XGBoost (This Implementation)
✅ Learns discriminative shape patterns
✅ Handles multi-channel feature interactions
✅ Built-in feature importance
❌ Requires labeled anomalies
❌ Slightly weaker recall on rare-long events

XGBoost performs best when you have *annotated anomalies* and need robust precision.

---

## When to Choose Which

| Method          | Best For | Weak On |
|----------------|----------|---------|
| Telemanom      | Temporal drifts & slow anomalies | Fast spikes |
| Ensemble       | Label scarcity, fast inference   | Complex interactions |
| XGBoost        | Multi-feature discriminative classification | Long rare anomalies |

You can also **hybridize**:
- Telemanom scores → meta features
- XGBoost → final classifier

---

## Design Details

This implementation follows internal design discussions:

- Adapter-based normalization for fair feature comparison
- Smoothed probabilities to prevent flicker anomalies
- Percentile-based threshold calibration
- Event merging aligned with ESA scoring rules

Further tuning can improve recall on extended events (see notebook notes).

