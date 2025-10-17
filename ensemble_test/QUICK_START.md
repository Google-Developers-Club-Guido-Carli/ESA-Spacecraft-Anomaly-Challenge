# Quick Start Guide - Ensemble Anomaly Detector

## What's Running Now

The ensemble detector is currently running on the full ESA spacecraft dataset:

- **Training**: 12.5M samples from channels 41-46
- **Validation**: 2.2M samples for threshold calibration
- **Test**: 521K samples for final predictions

Current status: **Calibrating thresholds** (optimizing event-wise F0.5 on validation)

## After Completion

### Check Results

1. **Validation Metrics** (console output)

   ```
   Event-wise Precision: X.XXXX
   Event-wise Recall: X.XXXX
   Event-wise F0.5: X.XXXX
   ```
2. **Output Files** (in `ensemble_test/outputs/`)

   - `ensemble_submission.csv` - Binary predictions for test set
   - `ensemble_scores.csv` - Detailed scores per detector

### Interpretation

**Good Performance**:

- Event-wise F0.5 > 0.4 (decent)
- Event-wise F0.5 > 0.6 (good)
- Event-wise Precision > 0.7 (high precision, as intended)

**Expected Behavior**:

- Precision should be higher than recall (F0.5 favors precision)
- Point-wise metrics will be lower than event-wise (normal)
- Fewer detected events than ground truth (conservative detector)

## Quick Modifications to test in person

### 1. Use Smaller Sample for Faster Testing

Edit `run_ensemble.py`:

```python
# After loading data, add:
train_df = train_df.sample(n=100000, random_state=42)  # Use 100K samples
```

### 2. Adjust Detector Weights

Edit `run_ensemble.py`, line ~102:

```python
ensemble = EnsembleDetector(
    channel_names=channel_names,
    weights={'global': 0.4, 'local': 0.3, 'iforest': 0.3},  # Change these
    ...
)
```

### 3. Change Postprocessing

Edit `run_ensemble.py`, line ~105:

```python
L_min=5,        # Minimum event length (currently 3)
gap_merge=5,    # Merge events within 5 samples (currently 2)
```

### 4. Skip Threshold Calibration

Edit `run_ensemble.py`, comment out calibration block and use fixed threshold:

```python
# best_threshold = ensemble.calibrate_threshold(...)
best_threshold = 0.5  # Fixed threshold
```

## Run Again (After This Completes)

```powershell
cd C:\Users\macca\Desktop\ESA\ESA-Spacecraft-Anomaly-Challenge\ensemble_test
..\.venv\Scripts\python.exe run_ensemble.py
```

Or from workspace root:

```powershell
.venv\Scripts\python.exe ensemble_test\run_ensemble.py
```

## Once we have our results...

1. **Review submission file**

   ```powershell
   Get-Content outputs\ensemble_submission.csv -Head 20
   ```
2. **Check detected anomaly rate**

   ```powershell
   Import-Csv outputs\ensemble_submission.csv | Measure-Object -Property is_anomaly -Sum
   ```
3. **Analyze per-detector scores**
   Open `outputs/ensemble_scores.csv` in Excel/Python to see which detector contributed most
4. **Tune hyperparameters** based on validation results
5. **Add Telemanom detector** (requires LSTM model) (?)

## Understanding the Algorithm

### What Each Detector Does

- **GlobalSTD**: Catches values far from global mean (simple baseline)
- **LocalSTD**: Catches local deviations using rolling window (adaptive)
- **IsolationForest**: Catches multivariate anomalies (correlated changes)

### How Ensemble Combines Them

```
Final Score = 0.35×GlobalSTD + 0.35×LocalSTD + 0.30×IForest + Consensus Bonus
```

- Consensus bonus (+0.05) when ≥2 detectors agree
- Final threshold calibrated to maximize validation F0.5

### Why Event-Wise F0.5?

- ESA challenge uses event-wise scoring (each anomaly segment = 1 event)
- F0.5 favors precision (minimize false alarms)
- Critical for spacecraft operations (false alarms are costly)

## Troubleshooting

### "Out of Memory"

- Reduce train sample size (add `.sample(n=1000000)`)
- Increase IsolationForest max_samples parameter

### "Taking Too Long"

- Reduce `n_thresholds` in `calibrate_threshold()` from 200 to 50
- Skip threshold calibration, use fixed threshold=0.5

### "Low F0.5 Score"

- Try adjusting detector weights
- Increase L_min (more conservative, higher precision)
- Tune z0 parameters (lower = more sensitive)

## Files Structure

```
ensemble_test/
├── ensemble.py              # Core detector class
├── ensemble_utils.py        # Helper functions
├── run_ensemble.py          # Main demo script
├── README.md               # Algorithm documentation
├── IMPLEMENTATION_STATUS.md # Implementation summary
├── QUICK_START.md          # This file
└── outputs/                # Results (created at runtime)
    ├── ensemble_submission.csv
    └── ensemble_scores.csv
```
