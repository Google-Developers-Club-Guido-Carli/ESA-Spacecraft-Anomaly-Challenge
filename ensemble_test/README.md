# Ensemble Anomaly Detector - Implementation

This folder contains the implementation of the ensemble anomaly detection algorithm for ESA spacecraft telemetry (channels 41-46).

## Files

- **ensemble.py** - Core `EnsembleDetector` class combining GlobalSTD, LocalSTD, and IsolationForest
- **ensemble_utils.py** - Helper functions for score transforms, event postprocessing, and metrics
- **run_ensemble.py** - Demo script to run ensemble on `.data` files

## Algorithm Overview

The ensemble combines three complementary detectors:

1. **GlobalSTD** - Global statistical outlier detection (mean ± N×std)
2. **LocalSTD** - Rolling window local z-score detection (adaptive)
3. **IsolationForest** - Multivariate structural anomaly detection

### Scoring Pipeline

For each timepoint t:

1. Compute per-detector scores s_d(t) ∈ [0,1]
2. Weighted aggregation: S_raw = Σ w_d × s_d(t)
3. Consensus bonus: +0.05 if ≥2 detectors agree
4. Final score: S(t) = clip(S_raw + bonus, 0, 1)

### Temporal Postprocessing

After thresholding:

1. Remove events shorter than L_min=3 samples
2. Merge events separated by ≤2 samples
3. Output compact event predictions

## Usage

```powershell
cd ensemble_test
python run_ensemble.py
```

This will:

1. Load train/test data from `../.data/`
2. Split train into train/val (85%/15%)
3. Fit ensemble on training data
4. Calibrate threshold on validation (maximize event-wise F0.5)
5. Predict on test set
6. Save results to `outputs/ensemble_submission.csv`

## Requirements

```
pandas
numpy
scikit-learn
```

Install via:

```powershell
pip install pandas numpy scikit-learn
```

## Hyperparameters

Default settings (tuned for F0.5 optimization):

- **Weights**: Global=0.35, Local=0.35, IForest=0.30
- **Window size** (LocalSTD): 51 samples
- **z0 parameters**: 3.0 (for z→score transform)
- **Event pruning**: L_min=3, gap_merge=2
- **Consensus**: bonus=0.05, min_votes=2

## Outputs

- `outputs/ensemble_submission.csv` - Binary predictions (id, is_anomaly)
- `outputs/ensemble_scores.csv` - Detailed scores per detector

## Metrics

The algorithm optimizes for **event-wise F0.5** (precision favored):

- Precision: fraction of predicted events that overlap true events
- Recall: fraction of true events that are detected
- F0.5: weighted harmonic mean favoring precision

Both event-wise and point-wise metrics are reported for validation.

## Design Details

See the pseudocode and algorithm design in the whatsapp groupchat (ChatGpt output, not using Telemanom for now). This implementation follows the specified formulas for:

- z-score to anomaly score transform: s = 1 - exp(-z/z0)
- Percentile normalization for IsolationForest
- Event extraction and postprocessing
- Consensus voting mechanism
