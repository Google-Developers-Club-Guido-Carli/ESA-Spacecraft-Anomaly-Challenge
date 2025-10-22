# Telemanom-ESA Baseline (Channels 41–46, Lite Data)
## Small changes/improvements can be made: 
*Note on “non-pruned” vs “pruned”: In this repo both .ts files currently contain the same series (the Telemanom output is already pruned + buffered inside telemanom/errors.py).*


This folder contains our implementation of the **Telemanom-ESA anomaly detection baseline** for the ESA Spacecraft Anomaly Challenge.  
Our task was to **adapt and run the baseline on the lighter sample of the original train and test datasets**.
This implementation is adapted from the official [Telemanom-ESA baseline](https://github.com/kplabs-pl/ESA-ADB/tree/main/TimeEval-algorithms/telemanom_esa).  
We downloaded the original folder from that repository and then added my own files and modifications (JSON configs, Colab notebook, partial training run, etc.) on top of it.

## Our Contributions

- Added **JSON config files** (`args_train.json`, `args_test.json`) for running training/testing.  
- Wrote a **converter script (`converter.py`)** to prepare the light datasets into the expected TimeEval format.  
- Ran **partial training** for **16 epochs out of 35** (partially trained model due to runtime limits on GoogleCollab GPU).  
- Integrated everything inside the original ESA baseline folder structure.  
- Prepared **testing configuration** for generating errors and anomaly scores. 
- Post-export helper: make_ts_from_algorithm_output.py to write Kaggle/ESA-style .ts binaries.
- Prepred Plots to show results 

## Environment

- Platform: **Google Colab** (originally done in there) 
- Hardware: **GPU (limited quota) + CPU fallback**  
- Python version: 3.10 (Colab default)  
- TensorFlow: 2.15  
- Runtime limits: Training stopped after ~16 epochs due to GPU usage cap.  


## File & Folder Overview

### Core Code
- `algorithm.py` → **Main entry point**. Runs either training or testing depending on the JSON config.  
- `telemanom/` → Package with helper modules:  
  - `modeling.py` → LSTM model definition.  
  - `detector.py` → Prediction, thresholding, pruning.   
  - `channel.py` → Channel-specific preprocessing.  
  - `helpers.py` → Utility functions.  

### Data
- `data/train_lite.csv` → Original lighter training dataset.  
- `data/test_lite.csv` → Original lighter test dataset.  
- `data/train_te.csv` → **Converted training dataset** (from `converter.py`, TimeEval format).  
- `data/test_te.csv` → **Converted test dataset** (from `converter.py`, TimeEval format).  

### Configs (that we added)
- `json_files/args_train.json` → Training configuration (used for 16-epoch training).  
- `json_files/args_test.json` → Testing configuration.  

### Utilities
- `converter.py` → Script we wrote/modified to convert the raw CSVs (`*_lite.csv`) into TimeEval format (`*_te.csv`).  
- `manifest.json` → Metadata describing the algorithm (used for TimeEval/Docker runs).  
- `requirements.txt` → Dependencies (numpy, pandas, tensorflow, etc.).  
- `LICENSE` → License of the original repo.  
- `README.md` → This documentation file.  

### Outputs
- `runs/` → Folder for model checkpoints and output files.
- `internal-run-id.log` → Log file automatically generated during runs.  

## Steps We Executed

1. **Cloned the official ESA Telemanom-ESA baseline repo.**

2. **Converted the datasets** into TimeEval format:  
```sh
python converter.py
```
This produced: train_te.csv and test_te.csv in data folder.

3. **Trained the model using the lighter dataset and our training JSON**
```sh
python algorithm.py "$(tr -d '\n' < json_files/args_train.json)"
```
Training was stopped at 16 epochs out of 35 due to runtime/GPU time limits. Model is saved in runs/model.keras.

4. **Testing the model using the lighter dataset and our testing JSON (with thresholding on)** 
```sh
python algorithm.py "$(tr -d '\n' < json_files/args_test.json)"
```
We lowered "min_error_value" so the baseline actually flags anomalies on this scale.
It produced: 
- errors.csv - (per-channel matrix from testing (0/1 with thresholding on) )
- errors-no-threshold.csv - raw smoothed errors (padded)

6. **Export Kaggle/ESA-style binaries & generated plot**
```sh
python make_ts_from_algorithm_output.py
```
This writes:
- runs/anomaly_scores_nonPruned.ts
- runs/anomaly_scores_Pruned.ts

Then the plotting notebook reads those files and saves figures under runs/figures/.



## Next Steps (To-Do)
1. Finish full training (35 epochs) and compare metrics to the partial model.

2. Expose and export a true non-pruned sequence (pre-prune_anoms) to make nonPruned.ts differ from Pruned.ts. 

3. (Optional) Containerize with Docker/TimeEval for reproducible benchmarking.

## EXTRA:  Modifications to `algorithm.py`
We made small changes in `algorithm.py` to make the baseline work correctly with the **lite test dataset** (which does not contain anomaly labels).
### Key Edits
1. **Only use anomaly columns that actually exist**
   - Original code assumed `is_anomaly_channel_*` columns were always present.  
   - We added a check (`existing_anom`) so the code skips them if they are missing.
2. **Keep data columns aligned with channels only**
   - Ensures the model input indices map only to real telemetry channels, not anomalies.
3. **Safe slicing when no anomaly columns are present**
   - Prevented arrays with zero features (fixes `Matrix size-incompatible: In[0]: [70,0]` error during LSTM inference).
4. **More robust CSV parsing**
   - Explicit `parse_dates=["timestamp"]` and flexible handling whether anomaly columns exist or not.

### Effect
- Works on **training data** (with labels) and **test data** (without labels).  
- Prevents KeyErrors and zero-feature tensors.  
- Does **not** change the model architecture or training procedure — only makes input loading more robust.

We also made a small change in `modelling.py`
Added `restore_best_weights=True` to **EarlyStopping** and improved **ModelCheckpoint** (monitoring `val_loss`, saving best model only).  
These changes ensure the best model is always saved and reloaded, improving training stability under Colab GPU limits.


## Quick Start (Colab Reproduction)
These are the exact steps we followed in Google Colab to set up, train, and prepare testing.

1. **Mount Google Drive** (to save converted datasets and trained model):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ``` 
2. **Get the Telemanom-ESA baseline code** (clone the official repo and copy the folder):
```sh
!git clone https://github.com/kplabs-pl/ESA-ADB.git
cp -r ESA-ADB/TimeEval-algorithms/telemanom_esa /content/telemanom_esa
cd /content/telemanom_esa
```
3. **Install dependencies** (from my requirements.txt):
```sh
!pip install -r requirements.txt
```
4. **Convert the lighter datasets into TimeEval format** (produces train_te.csv and test_te.csv):
```sh
!python converter.py
```
5. **Run training using my JSON config** (partial run, stopped after 16/35 epochs due to GPU limit):
```sh
!python algorithm.py "$(tr -d '\n' < json_files/args_train.json)"
```
it saves model to runs/model.keras.

6. **Run testing**:
```sh
!python algorithm.py "$(tr -d '\n' < json_files/args_test.json)"
```
7. **Export .ts binaries & plot:**
```sh
python make_ts_from_algorithm_output.py
```

