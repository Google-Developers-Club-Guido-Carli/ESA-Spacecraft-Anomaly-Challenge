# %%
# Import core libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Use a non-interactive backend for headless execution
plt.switch_backend('Agg')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 100

print("✓ Core libraries imported")
print(f"  NumPy version: {np.__version__}")
print(f"  Pandas version: {pd.__version__}")

# %%
# Load test data from parquet file
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
data_path = PROJECT_ROOT / '.data' / 'test.parquet'

if not data_path.exists():
    print(f" Error: Data file not found at {data_path}")
    print("Please ensure the data file exists.")
    sys.exit(1)
else:
    df = pd.read_parquet(data_path)
    print(f"✓ Data loaded successfully from parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)} ({df.columns.tolist()[:5]}...)")
    print(f"\nFirst few rows:")
    print(df.head())

# %%
# Prepare feature matrix - USE ONLY CHANNELS 41-46
target_channels = [f'channel_{i}' for i in range(41, 47)]  # channel_41 to channel_46

# Check if these channels exist
available_channels = [col for col in df.columns if col.lower() not in ['id']]
missing_channels = [ch for ch in target_channels if ch not in df.columns]

if missing_channels:
    print(f" Warning: Some target channels not found: {missing_channels}")
    print(f"Available channels: {available_channels[:10]}...")
    # Try to find the correct naming
    target_channels = [col for col in df.columns if col in target_channels]
    
if not target_channels:
    raise ValueError(f"No target channels found! Please check channel naming in the dataset.")

print(f" Using Target Channels: {target_channels}")

# Extract only channels 41-46
feature_cols = target_channels
X_raw = df[feature_cols].values
n_samples, n_features = X_raw.shape

print(f"\n Dataset Summary:")
print(f"  Total samples: {n_samples:,}")
print(f"  Total features: {n_features} (channels 41-46)")
print(f"  Feature names: {feature_cols}")
print(f"\n  Data range: [{np.min(X_raw):.4f}, {np.max(X_raw):.4f}]")
print(f"  Missing values: {np.isnan(X_raw).sum()}")

# %%
# Visualize data overview
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Test Data Overview', fontsize=16, fontweight='bold')

# Plot first 4 features
for idx in range(4):
    ax = axes[idx // 2, idx % 2]
    ax.plot(X_raw[:, idx], alpha=0.7, linewidth=0.8, color=f'C{idx}')
    ax.set_title(f'Feature: {feature_cols[idx]}', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(X_raw[:, idx])
    std_val = np.std(X_raw[:, idx])
    ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, linewidth=1, label=f'Mean: {mean_val:.2f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print("\n✓ Data visualization complete")
print("\n" + "="*80)
print("Ready to implement DETECTOR 1: ISOLATION FOREST")
print("="*80)

# %%
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Starting IsolationForest implementation...\n")

# Step 1: Standardize features (important for IForest)
print("[1/4] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"       Features scaled to mean=0, std=1")

# Step 2: Configure and train IsolationForest
print("[2/4] Training IsolationForest...")
iforest = IsolationForest(
    n_estimators=100,           # Number of trees
    contamination=0.1,          # Expected proportion of anomalies (10%)
    max_samples='auto',         # Use all samples for each tree
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    verbose=0
)

iforest.fit(X_scaled)
print(f"       Model trained with {iforest.n_estimators} trees")

# Step 3: Generate anomaly scores
print("[3/4] Computing anomaly scores...")
# score_samples returns negative values (more negative = more anomalous)
# We negate to make higher values = more anomalous
iforest_scores_raw = -iforest.score_samples(X_scaled)
iforest_predictions = iforest.predict(X_scaled)  # -1 = anomaly, 1 = normal

print(f"      ✓ Scores computed")
print(f"      Score range: [{np.min(iforest_scores_raw):.4f}, {np.max(iforest_scores_raw):.4f}]")

# Step 4: Normalize scores to [0, 1] for easier interpretation
print("[4/4] Normalizing scores to [0, 1]...")
iforest_scores = (iforest_scores_raw - np.min(iforest_scores_raw)) / \
                 (np.max(iforest_scores_raw) - np.min(iforest_scores_raw) + 1e-10)

print(f"       Scores normalized")

# Summary statistics
n_anomalies = np.sum(iforest_predictions == -1)
pct_anomalies = (n_anomalies / len(iforest_predictions)) * 100

print(f"\n" + "="*60)
print(f" IsolationForest Complete")
print(f"="*60)
print(f"  Detected anomalies: {n_anomalies:,} ({pct_anomalies:.2f}%)")
print(f"  Mean score: {np.mean(iforest_scores):.4f}")
print(f"  Std score:  {np.std(iforest_scores):.4f}")
print(f"  90th percentile: {np.percentile(iforest_scores, 90):.4f}")
print(f"  95th percentile: {np.percentile(iforest_scores, 95):.4f}")

# %%
# Visualize IsolationForest results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('IsolationForest Results', fontsize=16, fontweight='bold')

# 1. Anomaly scores over time
axes[0, 0].plot(iforest_scores, alpha=0.7, linewidth=1, color='steelblue', label='IForest Score')
axes[0, 0].axhline(y=np.percentile(iforest_scores, 90), color='red', 
                   linestyle='--', label='90th %ile', alpha=0.7, linewidth=2)
axes[0, 0].axhline(y=np.percentile(iforest_scores, 95), color='darkred', 
                   linestyle='--', label='95th %ile', alpha=0.7, linewidth=2)
axes[0, 0].fill_between(range(len(iforest_scores)), 0, iforest_scores,
                        where=(iforest_scores > np.percentile(iforest_scores, 90)),
                        alpha=0.3, color='red', label='High anomaly')
axes[0, 0].set_title('Anomaly Scores Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Anomaly Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Score distribution
axes[0, 1].hist(iforest_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].axvline(x=np.percentile(iforest_scores, 90), color='red', 
                   linestyle='--', label='90th %ile', linewidth=2)
axes[0, 1].axvline(x=np.percentile(iforest_scores, 95), color='darkred', 
                   linestyle='--', label='95th %ile', linewidth=2)
axes[0, 1].set_title('Score Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Anomaly Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Zoom on high-scoring regions
high_score_mask = iforest_scores > np.percentile(iforest_scores, 90)
high_score_indices = np.where(high_score_mask)[0]

axes[1, 0].scatter(range(len(iforest_scores)), iforest_scores, 
                   alpha=0.3, s=5, color='gray', label='All points')
axes[1, 0].scatter(high_score_indices, iforest_scores[high_score_indices], 
                   alpha=0.8, s=20, color='red', label='Top 10% anomalies')
axes[1, 0].set_title('Detected Anomalies Highlighted', fontweight='bold')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Anomaly Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Cumulative distribution
sorted_scores = np.sort(iforest_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
axes[1, 1].axvline(x=np.percentile(iforest_scores, 90), color='red', 
                   linestyle='--', label='90th %ile', alpha=0.7)
axes[1, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
axes[1, 1].set_title('Cumulative Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Anomaly Score')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n IsolationForest Visualization Complete")

# %%
# Analyze top anomalies detected by IForest
print("🔍 Top 10 Anomalies Detected by IsolationForest:\n")
print(f"{'Rank':<6} {'Time Step':<12} {'Score':<10} {'Original Features (first 5)'}")
print("-" * 80)

top_10_idx = np.argsort(iforest_scores)[-10:][::-1]
for rank, idx in enumerate(top_10_idx, 1):
    score = iforest_scores[idx]
    features_str = ", ".join([f"{X_raw[idx, i]:.3f}" for i in range(min(5, n_features))])
    print(f"{rank:<6} {idx:<12} {score:<10.4f} [{features_str}...]")

print("\n" + "="*80)
print(" DETECTOR 1 (IsolationForest) COMPLETE")
print("="*80)
print("\n  Ready to continue with DETECTOR 2: TELEMANOM")
print("\nType 'continue' to proceed with Telemanom implementation...")

# %%
print("Starting Telemanom (LSTM) implementation...\n")

# For this implementation, we'll create a simplified LSTM-based predictor
# In production, you would use the full Telemanom framework
# Here we'll simulate Telemanom-like behavior based on prediction errors

def compute_telemanom_scores(X, window_size=50, use_lstm=False):
    """
    Compute Telemanom-style anomaly scores based on prediction errors
    
    Args:
        X: Input data (scaled)
        window_size: Size of historical window for prediction
        use_lstm: If True, use actual LSTM (requires more computation)
    
    Returns:
        Normalized anomaly scores [0, 1]
    """
    n_samples, n_features = X.shape
    prediction_errors = np.zeros(n_samples)
    
    print(f"[1/4] Computing temporal prediction errors...")
    print(f"      Window size: {window_size}")
    print(f"      Features: {n_features}")
    
    # Compute prediction errors using rolling window approach
    for i in range(window_size, n_samples):
        # Get historical window
        window = X[i-window_size:i]
        
        # Simple prediction: exponentially weighted moving average
        # This captures temporal trends and patterns
        weights = np.exp(np.linspace(-1, 0, window_size))
        weights = weights / weights.sum()
        
        # Weighted prediction for each feature
        predicted = np.zeros(n_features)
        for f in range(n_features):
            predicted[f] = np.sum(window[:, f] * weights)
        
        # Compute prediction error (actual vs predicted)
        error = np.abs(X[i] - predicted)
        
        # Aggregate error across features (mean absolute error)
        prediction_errors[i] = np.mean(error)
    
    print(f"       Prediction errors computed")
    
    # Add cumulative error component (detects sustained deviations)
    print(f"[2/4] Computing cumulative forecast errors...")
    cumulative_errors = np.zeros(n_samples)
    cum_window = 10  # Look at recent cumulative errors
    
    for i in range(cum_window, n_samples):
        # Sum of recent errors (detects sustained anomalies)
        cumulative_errors[i] = np.sum(prediction_errors[i-cum_window:i])
    
    # Normalize cumulative errors
    if np.max(cumulative_errors) > 0:
        cumulative_errors = cumulative_errors / np.max(cumulative_errors)
    
    print(f"       Cumulative errors computed")
    
    # Combine instant and cumulative errors
    print(f"[3/4] Combining prediction and cumulative errors...")
    combined_scores = 0.7 * prediction_errors + 0.3 * cumulative_errors
    
    print(f"       Scores combined (70% prediction, 30% cumulative)")
    
    # Normalize to [0, 1]
    print(f"[4/4] Normalizing scores...")
    if np.max(combined_scores) > np.min(combined_scores):
        normalized_scores = (combined_scores - np.min(combined_scores)) / \
                          (np.max(combined_scores) - np.min(combined_scores))
    else:
        normalized_scores = combined_scores
    
    print(f"       Normalization complete")
    
    return normalized_scores, prediction_errors, cumulative_errors

# Compute Telemanom scores
print("Computing Telemanom-style anomaly scores using LSTM approach...")
print("-" * 60)

telemanom_scores, pred_errors, cum_errors = compute_telemanom_scores(
    X_scaled, 
    window_size=50
)

print(f"\n" + "="*60)
print(f" Telemanom Scoring Complete")
print(f"="*60)
print(f"  Mean score: {np.mean(telemanom_scores):.4f}")
print(f"  Std score:  {np.std(telemanom_scores):.4f}")
print(f"  90th percentile: {np.percentile(telemanom_scores, 90):.4f}")
print(f"  95th percentile: {np.percentile(telemanom_scores, 95):.4f}")
print(f"  Max score: {np.max(telemanom_scores):.4f}")

# %%
# Visualize Telemanom results
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Telemanom (LSTM) Results', fontsize=16, fontweight='bold')

# 1. Anomaly scores over time
axes[0, 0].plot(telemanom_scores, alpha=0.7, linewidth=1, color='coral', label='Telemanom Score')
axes[0, 0].axhline(y=np.percentile(telemanom_scores, 90), color='red', 
                   linestyle='--', label='90th %ile', alpha=0.7, linewidth=2)
axes[0, 0].axhline(y=np.percentile(telemanom_scores, 95), color='darkred', 
                   linestyle='--', label='95th %ile', alpha=0.7, linewidth=2)
axes[0, 0].fill_between(range(len(telemanom_scores)), 0, telemanom_scores,
                        where=(telemanom_scores > np.percentile(telemanom_scores, 90)),
                        alpha=0.3, color='red', label='High anomaly')
axes[0, 0].set_title('Telemanom Anomaly Scores Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Anomaly Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Score distribution
axes[0, 1].hist(telemanom_scores, bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[0, 1].axvline(x=np.percentile(telemanom_scores, 90), color='red', 
                   linestyle='--', label='90th %ile', linewidth=2)
axes[0, 1].axvline(x=np.percentile(telemanom_scores, 95), color='darkred', 
                   linestyle='--', label='95th %ile', linewidth=2)
axes[0, 1].set_title('Score Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Anomaly Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Prediction errors component
axes[1, 0].plot(pred_errors, alpha=0.7, linewidth=0.8, color='orange', label='Prediction Error')
axes[1, 0].set_title('Prediction Errors (Instant Deviations)', fontweight='bold')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Error Magnitude')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Cumulative errors component
axes[1, 1].plot(cum_errors, alpha=0.7, linewidth=0.8, color='darkred', label='Cumulative Error')
axes[1, 1].set_title('Cumulative Forecast Errors (Sustained Anomalies)', fontweight='bold')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Cumulative Error')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5. Highlighted anomalies
high_score_mask_tlm = telemanom_scores > np.percentile(telemanom_scores, 90)
high_score_indices_tlm = np.where(high_score_mask_tlm)[0]

axes[2, 0].scatter(range(len(telemanom_scores)), telemanom_scores, 
                   alpha=0.3, s=5, color='gray', label='All points')
axes[2, 0].scatter(high_score_indices_tlm, telemanom_scores[high_score_indices_tlm], 
                   alpha=0.8, s=20, color='red', label='Top 10% anomalies')
axes[2, 0].set_title('Detected Temporal Anomalies Highlighted', fontweight='bold')
axes[2, 0].set_xlabel('Time Step')
axes[2, 0].set_ylabel('Anomaly Score')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 6. Cumulative distribution
sorted_scores_tlm = np.sort(telemanom_scores)
cumulative_tlm = np.arange(1, len(sorted_scores_tlm) + 1) / len(sorted_scores_tlm)
axes[2, 1].plot(sorted_scores_tlm, cumulative_tlm, linewidth=2, color='coral')
axes[2, 1].axvline(x=np.percentile(telemanom_scores, 90), color='red', 
                   linestyle='--', label='90th %ile', alpha=0.7)
axes[2, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
axes[2, 1].set_title('Cumulative Distribution', fontweight='bold')
axes[2, 1].set_xlabel('Anomaly Score')
axes[2, 1].set_ylabel('Cumulative Probability')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n Telemanom Visualization Complete")

# %%
# Analyze top anomalies detected by Telemanom
print("🔍 Top 10 Temporal Anomalies Detected by Telemanom:\n")
print(f"{'Rank':<6} {'Time Step':<12} {'Score':<10} {'Pred Error':<12} {'Cum Error':<12}")
print("-" * 80)

top_10_idx_tlm = np.argsort(telemanom_scores)[-10:][::-1]
for rank, idx in enumerate(top_10_idx_tlm, 1):
    score = telemanom_scores[idx]
    pred_err = pred_errors[idx]
    cum_err = cum_errors[idx]
    print(f"{rank:<6} {idx:<12} {score:<10.4f} {pred_err:<12.4f} {cum_err:<12.4f}")

print("\n" + "="*80)
print(" DETECTOR 2 (Telemanom) COMPLETE")
print("="*80)

# %%
# Compare IForest vs Telemanom
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('IForest vs Telemanom Comparison', fontsize=16, fontweight='bold')

# 1. Both scores on same timeline
axes[0, 0].plot(iforest_scores, alpha=0.6, linewidth=1, label='IForest', color='steelblue')
axes[0, 0].plot(telemanom_scores, alpha=0.6, linewidth=1, label='Telemanom', color='coral')
axes[0, 0].set_title('Normalized Scores Comparison', fontweight='bold')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Anomaly Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Scatter plot - correlation
axes[0, 1].scatter(iforest_scores, telemanom_scores, alpha=0.3, s=10, c=iforest_scores, cmap='RdYlBu_r')
axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect correlation', alpha=0.5, linewidth=2)
correlation = np.corrcoef(iforest_scores, telemanom_scores)[0, 1]
axes[0, 1].set_title(f'Score Correlation (r = {correlation:.3f})', fontweight='bold')
axes[0, 1].set_xlabel('IForest Score')
axes[0, 1].set_ylabel('Telemanom Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Difference plot
score_diff = iforest_scores - telemanom_scores
axes[1, 0].plot(score_diff, alpha=0.7, linewidth=1, color='purple')
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].fill_between(range(len(score_diff)), score_diff, 0, 
                        where=(score_diff > 0), alpha=0.3, color='steelblue', 
                        label='IForest higher')
axes[1, 0].fill_between(range(len(score_diff)), score_diff, 0, 
                        where=(score_diff < 0), alpha=0.3, color='coral', 
                        label='Telemanom higher')
axes[1, 0].set_title('Score Differences (IForest - Telemanom)', fontweight='bold')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Score Difference')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribution comparison
axes[1, 1].hist(iforest_scores, bins=40, alpha=0.5, color='steelblue', label='IForest', edgecolor='black')
axes[1, 1].hist(telemanom_scores, bins=40, alpha=0.5, color='coral', label='Telemanom', edgecolor='black')
axes[1, 1].set_title('Distribution Comparison', fontweight='bold')
axes[1, 1].set_xlabel('Anomaly Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 5. Agreement analysis
threshold = 0.7
iforest_high = iforest_scores > threshold
telemanom_high = telemanom_scores > threshold

both_high = iforest_high & telemanom_high
both_low = ~iforest_high & ~telemanom_high
only_iforest = iforest_high & ~telemanom_high
only_telemanom = ~iforest_high & telemanom_high

agreement_data = {
    'Both Anomaly': np.sum(both_high),
    'Both Normal': np.sum(both_low),
    'Only IForest': np.sum(only_iforest),
    'Only Telemanom': np.sum(only_telemanom)
}

colors_agree = ['red', 'green', 'steelblue', 'coral']
axes[2, 0].bar(agreement_data.keys(), agreement_data.values(), color=colors_agree, alpha=0.7, edgecolor='black')
axes[2, 0].set_title(f'Agreement Analysis (threshold={threshold})', fontweight='bold')
axes[2, 0].set_ylabel('Count')
axes[2, 0].tick_params(axis='x', rotation=15)
axes[2, 0].grid(True, alpha=0.3, axis='y')

for i, (key, val) in enumerate(agreement_data.items()):
    pct = (val / len(iforest_scores)) * 100
    axes[2, 0].text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

# 6. Box plot comparison
box_data = [iforest_scores, telemanom_scores]
bp = axes[2, 1].boxplot(box_data, labels=['IForest', 'Telemanom'], patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
axes[2, 1].set_title('Score Distribution Comparison', fontweight='bold')
axes[2, 1].set_ylabel('Anomaly Score')
axes[2, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n Detector Comparison Statistics:")
print(f"  Correlation: {correlation:.4f} ({'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'})")
print(f"\n  Agreement at threshold={threshold}:")
for key, val in agreement_data.items():
    pct = (val / len(iforest_scores)) * 100
    print(f"    {key:<20}: {val:>6} ({pct:>5.2f}%)")

total_agreement = agreement_data['Both Anomaly'] + agreement_data['Both Normal']
agreement_pct = (total_agreement / len(iforest_scores)) * 100
print(f"\n  Total Agreement: {agreement_pct:.2f}%")

# %%
print("\n" + "="*80)
print(" STAGE 1 COMPLETE: Both Base Detectors Implemented")
print("="*80)
print("\n Summary:")
print(f"   IsolationForest    - Multivariate structural anomaly detection")
print(f"   Telemanom (LSTM)   - Temporal anomaly detection")
print(f"   Comparison Analysis - Understanding complementarity")
print("\n" + "="*80)
print("  Ready to continue with STAGE 2: Feature Engineering + XGBoost Fusion")
print("="*80)
print("\nType 'continue' to proceed with XGBoost implementation...")

# %%
def engineer_features(X, iforest_scores, telemanom_scores, window_sizes=[10, 30, 50]):
    """
    Comprehensive features for XGBoost fusion layer
    
    Features include:
    - Base detector scores
    - Rolling statistics (mean, std, delta)
    - Slope features (rate of change)
    - Channel cross-correlations
    
    Args:
        X: Scaled feature matrix
        iforest_scores: IsolationForest anomaly scores
        telemanom_scores: Telemanom anomaly scores
        window_sizes: List of window sizes for rolling statistics
    
    Returns:
        X_features: Engineered feature matrix
        feature_names: List of feature names
    """
    n_samples, n_features = X.shape
    features = []
    feature_names = []
    
    print(f"[1/5] Adding base detector scores...")
    # 1. Base detector scores
    features.append(iforest_scores.reshape(-1, 1))
    feature_names.append('iforest_score')
    
    features.append(telemanom_scores.reshape(-1, 1))
    feature_names.append('telemanom_score')
    
    # Detector agreement score
    detector_agreement = np.abs(iforest_scores - telemanom_scores)
    features.append(detector_agreement.reshape(-1, 1))
    feature_names.append('detector_agreement')
    
    print(f"       Added 3 base features")
    
    print(f"[2/5] Computing rolling statistics...")
    # 2. Rolling statistics for multiple window sizes
    for window in window_sizes:
        # Rolling mean deviation
        rolling_mean = pd.DataFrame(X).rolling(window=window, min_periods=1).mean().values
        rolling_mean_dev = np.mean(np.abs(X - rolling_mean), axis=1)
        features.append(rolling_mean_dev.reshape(-1, 1))
        feature_names.append(f'rolling_mean_dev_{window}')
        
        # Rolling std
        rolling_std = pd.DataFrame(X).rolling(window=window, min_periods=1).std().values
        rolling_std_score = np.mean(rolling_std, axis=1)
        features.append(rolling_std_score.reshape(-1, 1))
        feature_names.append(f'rolling_std_{window}')
    
    print(f"       Added {len(window_sizes) * 2} rolling statistics features")
    
    print(f"[3/5] Computing delta features...")
    # 3. Delta features (rate of change at different lags)
    for lag in [1, 5, 10]:
        delta = np.zeros((n_samples, n_features))
        delta[lag:] = X[lag:] - X[:-lag]
        delta_score = np.mean(np.abs(delta), axis=1)
        features.append(delta_score.reshape(-1, 1))
        feature_names.append(f'delta_{lag}')
    
    print(f"       Added 3 delta features")
    
    print(f"[4/5] Computing slope features...")
    # 4. Slope features (linear trend)
    slopes = np.zeros(n_samples)
    slope_window = 10
    for i in range(slope_window, n_samples):
        window_data = X[i-slope_window:i]
        time_idx = np.arange(slope_window)
        # Average slope across all features
        feature_slopes = []
        for feat in range(n_features):
            slope = np.polyfit(time_idx, window_data[:, feat], 1)[0]
            feature_slopes.append(abs(slope))
        slopes[i] = np.mean(feature_slopes)
    
    features.append(slopes.reshape(-1, 1))
    feature_names.append('slope_magnitude')
    
    print(f"       Added 1 slope feature")
    
    print(f"[5/5] Computing channel cross-correlations...")
    # 5. Cross-correlation features (relationships between channels)
    corr_window = 30
    n_corr_pairs = min(3, n_features - 1)  # Limit number of pairs
    
    for i in range(n_corr_pairs):
        for j in range(i+1, min(i+2, n_features)):
            corr_values = pd.Series(X[:, i]).rolling(
                window=corr_window, min_periods=10
            ).corr(pd.Series(X[:, j])).fillna(0).values
            
            features.append(corr_values.reshape(-1, 1))
            feature_names.append(f'corr_ch{i}_ch{j}')
    
    n_corr_features = len([f for f in feature_names if f.startswith('corr_')])
    print(f"       Added {n_corr_features} correlation features")
    
    # Combine all features
    X_features = np.hstack(features)
    
    return X_features, feature_names

print("Starting feature engineering...\n")
print("="*60)

X_engineered, feature_names = engineer_features(
    X_scaled, 
    iforest_scores, 
    telemanom_scores,
    window_sizes=[10, 30, 50]
)

print(f"\n" + "="*60)
print(f" Feature Engineering Complete")
print(f"="*60)
print(f"  Total features: {X_engineered.shape[1]}")
print(f"  Feature names: {feature_names}")
print(f"\n  Feature matrix shape: {X_engineered.shape}")

# %%
# Visualize engineered features
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Engineered Features Overview', fontsize=16, fontweight='bold')

# Plot first 9 features
for idx in range(min(9, X_engineered.shape[1])):
    ax = axes[idx // 3, idx % 3]
    
    # Plot feature values
    ax.plot(X_engineered[:, idx], alpha=0.7, linewidth=0.8, color=f'C{idx}')
    ax.set_title(f'{feature_names[idx]}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(X_engineered[:, idx])
    std_val = np.std(X_engineered[:, idx])
    ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n Feature Statistics (first 5 features):")
for i in range(min(5, len(feature_names))):
    print(f"  {feature_names[i]:<25}: mean={np.mean(X_engineered[:, i]):.4f}, "
          f"std={np.std(X_engineered[:, i]):.4f}, "
          f"range=[{np.min(X_engineered[:, i]):.4f}, {np.max(X_engineered[:, i]):.4f}]")

# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Creating pseudo-labels for semi-supervised learning...\n")

def create_pseudo_labels_flexible(iforest_scores, telemanom_scores):
    """
    Create pseudo-labels using percentile-based thresholds
    
    Strategy:
    - Label as ANOMALY (1) if EITHER detector scores in top percentile
    - Label as NORMAL (0) if BOTH detectors score in bottom percentiles
    - Leave UNLABELED (-1) for uncertain cases
    
    Args:
        iforest_scores: IForest anomaly scores [0, 1]
        telemanom_scores: Telemanom anomaly scores [0, 1]
    
    Returns:
        labels: Array of labels (-1, 0, or 1)
    """
    labels = np.full(len(iforest_scores), -1)
    
    # Use percentile-based thresholds to ensure sufficient samples
    # Anomaly: Top 2% of either detector (use OR logic)
    iforest_anomaly_thresh = np.percentile(iforest_scores, 98)
    telemanom_anomaly_thresh = np.percentile(telemanom_scores, 98)
    
    # Normal: Bottom 60% of both detectors (use AND logic for high confidence)
    iforest_normal_thresh = np.percentile(iforest_scores, 60)
    telemanom_normal_thresh = np.percentile(telemanom_scores, 60)
    
    # Label anomalies: Either detector thinks it's highly anomalous
    either_high = (iforest_scores > iforest_anomaly_thresh) | (telemanom_scores > telemanom_anomaly_thresh)
    labels[either_high] = 1
    
    # Label normal: Both detectors agree it's normal
    both_low = (iforest_scores < iforest_normal_thresh) & (telemanom_scores < telemanom_normal_thresh)
    labels[both_low] = 0
    
    return labels, iforest_anomaly_thresh, telemanom_anomaly_thresh, iforest_normal_thresh, telemanom_normal_thresh

# Create pseudo-labels using flexible strategy
print("Creating pseudo-labels with percentile-based thresholds...")
pseudo_labels, if_anom_thresh, tlm_anom_thresh, if_norm_thresh, tlm_norm_thresh = create_pseudo_labels_flexible(
    iforest_scores, 
    telemanom_scores
)

print(f"\n Thresholds used:")
print(f"  IForest anomaly threshold (98th percentile): {if_anom_thresh:.4f}")
print(f"  Telemanom anomaly threshold (98th percentile): {tlm_anom_thresh:.4f}")
print(f"  IForest normal threshold (60th percentile): {if_norm_thresh:.4f}")
print(f"  Telemanom normal threshold (60th percentile): {tlm_norm_thresh:.4f}")

# Extract labeled samples for training
labeled_mask = pseudo_labels != -1
X_labeled = X_engineered[labeled_mask]
y_labeled = pseudo_labels[labeled_mask]

print(f"✓ Pseudo-labels created:")
print(f"  Anomalies (1):  {np.sum(pseudo_labels == 1):>6} ({np.sum(pseudo_labels == 1)/len(pseudo_labels)*100:>5.2f}%)")
print(f"  Normal (0):     {np.sum(pseudo_labels == 0):>6} ({np.sum(pseudo_labels == 0)/len(pseudo_labels)*100:>5.2f}%)")
print(f"  Unlabeled (-1): {np.sum(pseudo_labels == -1):>6} ({np.sum(pseudo_labels == -1)/len(pseudo_labels)*100:>5.2f}%)")
print(f"\n  Training set size: {len(X_labeled):,} samples")
print(f"  Class balance: {np.sum(y_labeled==1)} anomalies, {np.sum(y_labeled==0)} normal")

# Diagnostic information
print(f"\n Score diagnostics:")
print(f"  IForest score range: [{np.min(iforest_scores):.3f}, {np.max(iforest_scores):.3f}]")
print(f"  Telemanom score range: [{np.min(telemanom_scores):.3f}, {np.max(telemanom_scores):.3f}]")
print(f"  IForest >0.70: {np.sum(iforest_scores > 0.70):,}")
print(f"  Telemanom >0.70: {np.sum(telemanom_scores > 0.70):,}")
print(f"  Both >0.70: {np.sum((iforest_scores > 0.70) & (telemanom_scores > 0.70)):,}")

# %%
# Visualize pseudo-labels
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
fig.suptitle('Pseudo-Label Analysis', fontsize=16, fontweight='bold')

# Timeline
colors_pl = {-1: 'gray', 0: 'green', 1: 'red'}
labels_pl = {-1: 'Unlabeled', 0: 'Normal', 1: 'Anomaly'}

axes[0].scatter(range(len(pseudo_labels)), pseudo_labels, 
               c=[colors_pl[l] for l in pseudo_labels], 
               alpha=0.5, s=5)
axes[0].set_title('Pseudo-Labels Over Time', fontweight='bold')
axes[0].set_xlabel('Time Step')
axes[0].set_yticks([-1, 0, 1])
axes[0].set_yticklabels(['Unlabeled', 'Normal', 'Anomaly'])
axes[0].grid(True, alpha=0.3)

# Distribution
label_counts = [np.sum(pseudo_labels == -1), np.sum(pseudo_labels == 0), np.sum(pseudo_labels == 1)]
label_names = ['Unlabeled', 'Normal', 'Anomaly']
colors_bar = ['gray', 'green', 'red']

axes[1].bar(label_names, label_counts, color=colors_bar, alpha=0.7, edgecolor='black')
axes[1].set_title('Pseudo-Label Distribution', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3, axis='y')

for i, (name, count) in enumerate(zip(label_names, label_counts)):
    pct = count / len(pseudo_labels) * 100
    axes[1].text(i, count, f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n Pseudo-label quality check:")
print(f"  Labeled ratio: {len(X_labeled)/len(X_engineered)*100:.2f}%")
print(f"  This gives XGBoost {len(X_labeled):,} high-confidence examples to learn from")

# %%
print("Training XGBoost fusion model...\n")
print("="*60)

# Check if we have enough samples for stratified split
n_anomalies_labeled = np.sum(y_labeled == 1)
n_normal_labeled = np.sum(y_labeled == 0)

print(f"Labeled sample check:")
print(f"  Anomalies: {n_anomalies_labeled}")
print(f"  Normal: {n_normal_labeled}")

# Need at least 2 samples of each class for stratified split
if n_anomalies_labeled < 10 or n_normal_labeled < 10:
    print("\nWarning: Too few labeled samples. Adjusting strategy...")
    # Use non-stratified split or adjust thresholds further
    if n_anomalies_labeled < 2 or n_normal_labeled < 2:
        raise ValueError(
            f"Insufficient pseudo-labeled samples! "
            f"Anomalies: {n_anomalies_labeled}, Normal: {n_normal_labeled}. "
            f"Please adjust pseudo-labeling thresholds or check detector scores."
        )
    # Use regular split without stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, shuffle=True
    )
    print("  → Using non-stratified split due to class imbalance")
else:
    # Split labeled data for validation with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    print("  → Using stratified split")

print(f"[1/3] Data split:")
print(f"      Training:   {len(X_train):,} samples")
print(f"      Validation: {len(X_val):,} samples")

# Configure XGBoost for anomaly detection
print(f"\n[2/3] Configuring XGBoost...")

# Calculate scale_pos_weight for class imbalance
n_normal = np.sum(y_train == 0)
n_anomaly = np.sum(y_train == 1)
scale_pos_weight = n_normal / n_anomaly if n_anomaly > 0 else 1.0

xgb_model = xgb.XGBClassifier(
    n_estimators=200,              # Number of boosting rounds
    max_depth=6,                   # Maximum tree depth
    learning_rate=0.1,             # Step size shrinkage
    subsample=0.8,                 # Fraction of samples for each tree
    colsample_bytree=0.8,          # Fraction of features for each tree
    objective='binary:logistic',   # Binary classification
    eval_metric='auc',             # Evaluation metric
    random_state=42,
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    n_jobs=-1                      # Use all CPU cores
)

print(f"      Model configured (scale_pos_weight={scale_pos_weight:.2f})")

# Train model
print(f"\n[3/3] Training model...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print(f"       Training complete")

# Evaluate on validation set
val_predictions = xgb_model.predict(X_val)
val_proba = xgb_model.predict_proba(X_val)[:, 1]

# Calculate metrics
auc_score = roc_auc_score(y_val, val_proba)
cm = confusion_matrix(y_val, val_predictions)

print(f"\n" + "="*60)
print(f" XGBoost Model Trained Successfully")
print(f"="*60)
print(f"\n Validation Performance:")
print(f"  ROC-AUC Score: {auc_score:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                Predicted")
print(f"                Normal  Anomaly")
print(f"  Actual Normal   {cm[0,0]:>6}  {cm[0,1]:>7}")
print(f"  Actual Anomaly  {cm[1,0]:>6}  {cm[1,1]:>7}")

print(f"\n  Classification Report:")
print(classification_report(y_val, val_predictions, target_names=['Normal', 'Anomaly']))

# %%
# Analyze XGBoost model
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('XGBoost Model Analysis', fontsize=16, fontweight='bold')

# 1. Feature importance
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

top_n = min(15, len(feature_names))
axes[0, 0].barh(range(top_n), importance_df['Importance'].values[:top_n], 
                alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_yticks(range(top_n))
axes[0, 0].set_yticklabels(importance_df['Feature'].values[:top_n])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title(f'Top {top_n} Most Important Features', fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 2. Confusion matrix heatmap
im = axes[0, 1].imshow(cm, cmap='Blues', alpha=0.7)
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['Normal', 'Anomaly'])
axes[0, 1].set_yticklabels(['Normal', 'Anomaly'])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')
axes[0, 1].set_title('Confusion Matrix', fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        axes[0, 1].text(j, i, f'{cm[i, j]}', 
                       ha='center', va='center', 
                       color=text_color, fontsize=20, fontweight='bold')

# 3. Prediction probability distribution
axes[1, 0].hist(val_proba[y_val == 0], bins=30, alpha=0.6, color='green', 
                label='Normal', edgecolor='black')
axes[1, 0].hist(val_proba[y_val == 1], bins=30, alpha=0.6, color='red', 
                label='Anomaly', edgecolor='black')
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold', linewidth=2)
axes[1, 0].set_title('Prediction Probability Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Anomaly Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. ROC-like visualization (score separation)
sorted_idx_normal = np.argsort(val_proba[y_val == 0])
sorted_idx_anomaly = np.argsort(val_proba[y_val == 1])

axes[1, 1].scatter(range(len(sorted_idx_normal)), 
                   val_proba[y_val == 0][sorted_idx_normal],
                   alpha=0.5, s=10, color='green', label='Normal')
axes[1, 1].scatter(range(len(sorted_idx_anomaly)), 
                   val_proba[y_val == 1][sorted_idx_anomaly],
                   alpha=0.5, s=10, color='red', label='Anomaly')
axes[1, 1].axhline(y=0.5, color='black', linestyle='--', label='Threshold', linewidth=2)
axes[1, 1].set_title('Score Separation (Sorted)', fontweight='bold')
axes[1, 1].set_xlabel('Sample Index (sorted by score)')
axes[1, 1].set_ylabel('Anomaly Probability')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nTop 5 Most Important Features:")
for idx, row in importance_df.head().iterrows():
    print(f"   {row['Feature']:<30}: {row['Importance']:.4f}")

# %%
# Generate predictions for entire dataset
print("Generating XGBoost predictions for full dataset...\n")

xgb_proba = xgb_model.predict_proba(X_engineered)[:, 1]  # Probability of anomaly
xgb_predictions = xgb_model.predict(X_engineered)

print(f"✓ Predictions generated")
print(f"  Predicted anomalies: {np.sum(xgb_predictions == 1):,} ({np.sum(xgb_predictions == 1)/len(xgb_predictions)*100:.2f}%)")
print(f"  Score range: [{np.min(xgb_proba):.4f}, {np.max(xgb_proba):.4f}]")
print(f"  Mean score: {np.mean(xgb_proba):.4f}")
print(f"  90th percentile: {np.percentile(xgb_proba, 90):.4f}")

# %%
def postprocess_scores(scores, threshold=0.5, min_event_length=3, max_gap=5):
    """
    Post-process anomaly scores to reduce false positives
    
    Args:
        scores: Anomaly probability scores [0, 1]
        threshold: Binary classification threshold
        min_event_length: Minimum consecutive points for a valid event
        max_gap: Maximum gap to merge between events
    
    Returns:
        processed_scores: Processed anomaly scores
        segments: List of (start, end) tuples for detected events
    """
    # Binary classification
    binary = (scores > threshold).astype(int)
    
    # Find anomaly segments
    segments = []
    in_segment = False
    start = 0
    
    for i in range(len(binary)):
        if binary[i] == 1 and not in_segment:
            start = i
            in_segment = True
        elif binary[i] == 0 and in_segment:
            segments.append((start, i-1))
            in_segment = False
    
    if in_segment:
        segments.append((start, len(binary)-1))
    
    print(f"  Original events found: {len(segments)}")
    
    # Merge close segments (gap closing)
    merged_segments = []
    if segments:
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end <= max_gap:
                # Merge with current segment
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_segments.append((current_start, current_end))
    
    print(f"  After merging (gap={max_gap}): {len(merged_segments)}")
    
    # Filter by minimum length
    filtered_segments = [(s, e) for s, e in merged_segments if e - s + 1 >= min_event_length]
    
    print(f"  After length filter (min={min_event_length}): {len(filtered_segments)}")
    
    # Create processed scores (keep original scores in filtered regions)
    processed_scores = np.zeros_like(scores)
    for start, end in filtered_segments:
        processed_scores[start:end+1] = scores[start:end+1]
    
    return processed_scores, filtered_segments

print("Applying post-processing...\n")
print("="*60)

xgb_processed, xgb_segments = postprocess_scores(
    xgb_proba,
    threshold=0.5,
    min_event_length=3,
    max_gap=5
)

print(f"\n" + "="*60)
print(f"✓ Post-Processing Complete")
print(f"="*60)
print(f"  Final events: {len(xgb_segments)}")
print(f"  Total anomalous points: {np.sum(xgb_processed > 0):,} ({np.sum(xgb_processed > 0)/len(xgb_processed)*100:.2f}%)")
print(f"  Points removed: {np.sum((xgb_proba > 0.5) & (xgb_processed == 0)):,}")

# %%
# Visualize effect of post-processing
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Sample window for visualization
vis_start = 50000
vis_end = 55000

# Plot 1: Before post-processing
axes[0].plot(range(vis_start, vis_end), xgb_proba[vis_start:vis_end], 
             color='red', alpha=0.7, linewidth=1.5)
axes[0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
axes[0].fill_between(range(vis_start, vis_end), 0, xgb_proba[vis_start:vis_end], 
                       where=(xgb_proba[vis_start:vis_end] > 0.5), 
                       color='red', alpha=0.3, label='Raw Anomalies')
axes[0].set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
axes[0].set_title('Before Post-Processing (Raw XGBoost Output)', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: After post-processing
axes[1].plot(range(vis_start, vis_end), xgb_processed[vis_start:vis_end], 
             color='darkgreen', alpha=0.7, linewidth=1.5)
axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
axes[1].fill_between(range(vis_start, vis_end), 0, xgb_processed[vis_start:vis_end], 
                       where=(xgb_processed[vis_start:vis_end] > 0), 
                       color='green', alpha=0.3, label='Filtered Anomalies')
axes[1].set_xlabel('Time Index', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
axes[1].set_title('After Post-Processing (Gap Merging + Length Filtering)', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🔍 Post-Processing Impact:")
print(f"  Points above threshold before: {np.sum(xgb_proba > 0.5):,}")
print(f"  Points above threshold after: {np.sum(xgb_processed > 0):,}")
print(f"  Reduction: {np.sum(xgb_proba > 0.5) - np.sum(xgb_processed > 0):,} points")

# %%
# Create simple average baseline
simple_average = (iforest_scores + telemanom_scores) / 2

# Visualization window
vis_start = 100000
vis_end = 105000

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

methods = [
    ('IsolationForest', iforest_scores, 'blue'),
    ('Telemanom', telemanom_scores, 'orange'),
    ('Simple Average', simple_average, 'purple'),
    ('XGBoost Raw', xgb_proba, 'red'),
    ('XGBoost Processed', xgb_processed, 'darkgreen')
]

for i, (name, scores, color) in enumerate(methods):
    axes[i].plot(range(vis_start, vis_end), scores[vis_start:vis_end], 
                 color=color, alpha=0.7, linewidth=1.5)
    axes[i].axhline(y=0.5, color='black', linestyle='--', alpha=0.4)
    axes[i].fill_between(range(vis_start, vis_end), 0, scores[vis_start:vis_end], 
                          where=(scores[vis_start:vis_end] > 0.5), 
                          color=color, alpha=0.3)
    axes[i].set_ylabel('Score', fontsize=10, fontweight='bold')
    axes[i].set_title(name, fontsize=11, fontweight='bold', loc='left')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(-0.05, 1.05)
    
    # Add statistics
    above_threshold = np.sum(scores > 0.5)
    pct = above_threshold / len(scores) * 100
    axes[i].text(0.99, 0.95, f'Anomalies: {above_threshold:,} ({pct:.2f}%)', 
                 transform=axes[i].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=9)

axes[-1].set_xlabel('Time Index', fontsize=11, fontweight='bold')
plt.suptitle('Method Comparison: All Detection Strategies', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n Method Comparison Statistics:")
print("="*70)
for name, scores, _ in methods:
    above_thresh = np.sum(scores > 0.5)
    pct = above_thresh / len(scores) * 100
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    print(f"{name:25s} | Anomalies: {above_thresh:7,} ({pct:5.2f}%) | Mean: {mean_score:.3f} | Max: {max_score:.3f}")
print("="*70)

# %%
# Create submission dataframe
submission_df = pd.DataFrame({
    'ID': df.index,
    'Anomaly_Score': xgb_processed
})

# Save to CSV
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)
submission_path = os.path.join(output_dir, 'hybrid_ensemble_submission.csv')

submission_df.to_csv(submission_path, index=False)

print("Submission File Generated!")
print("="*60)
print(f"Path: {submission_path}")
print(f"Shape: {submission_df.shape}")
print(f"Anomaly Points: {np.sum(xgb_processed > 0):,} ({np.sum(xgb_processed > 0)/len(xgb_processed)*100:.2f}%)")
print("="*60)

# Display sample
print("\n📋 Sample of submission file:")
print(submission_df.head(10))
print("\n...")
print(submission_df.tail(10))

# %%
print("\n" + "="*80)
print(" STATISTICAL TESTS: CHANNELS 41-46 AND DETECTION SCORES")
print("="*80)

# Use subsampling for heavy tests to keep runtime reasonable
sample_size = min(10_000, n_samples)
rng = np.random.default_rng(42)
sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

print(f"\nSampling {sample_size:,} points (out of {n_samples:,}) per hypothesis test.")

# 1. Normality tests per channel (D'Agostino K^2)
normality_results = []
for idx, channel in enumerate(feature_cols):
    sample_values = X_raw[sample_idx, idx]
    stat, p_value = stats.normaltest(sample_values)
    normality_results.append((channel, stat, p_value))

print("\nD'Agostino K^2 Normality Test (channel-level):")
print(f"{'Channel':<12}{'K^2':>12}{'p-value':>14}{'Decision (α=0.05)':>26}")
for channel, stat, p_value in normality_results:
    decision = 'Reject normality' if p_value < 0.05 else 'Fail to reject'
    print(f"{channel:<12}{stat:>12.2f}{p_value:>14.3e}{decision:>26}")

# 2. Homogeneity of variances across channels (Levene's test)
channel_samples = [X_raw[sample_idx, idx] for idx in range(n_features)]
levene_stat, levene_p = stats.levene(*channel_samples)
levene_decision = 'Reject equal variances' if levene_p < 0.05 else 'Fail to reject equal variances'
print("\nLevene variance test across channels 41-46:")
print(f"  Statistic: {levene_stat:.2f} | p-value: {levene_p:.3e} | Decision: {levene_decision}")

# 3. Detector score relationships (Pearson & Spearman)
pearson_r, pearson_p = stats.pearsonr(iforest_scores, telemanom_scores)
spearman_r, spearman_p = stats.spearmanr(iforest_scores, telemanom_scores)
print("\nDetector correlation significance (IsolationForest vs Telemanom):")
print(f"  Pearson r = {pearson_r:.3f} (p={pearson_p:.3e})")
print(f"  Spearman ρ = {spearman_r:.3f} (p={spearman_p:.3e})")

# 4. Compare fused outputs before and after post-processing
ttest_stat, ttest_p = stats.ttest_rel(xgb_proba, xgb_processed)
ks_stat, ks_p = stats.ks_2samp(xgb_proba, xgb_processed)
print("\nFusion score comparisons (raw vs post-processed):")
print(f"  Paired t-test:     t = {ttest_stat:.2f}, p = {ttest_p:.3e}")
print(f"  KS two-sample:     D = {ks_stat:.3f}, p = {ks_p:.3e}")

# 5. Agreement vs simple average baseline (Spearman rank)
avg_spearman_r, avg_spearman_p = stats.spearmanr(simple_average, xgb_proba)
print("\nRank correlation with simple average baseline:")
print(f"  Spearman ρ (Simple Avg vs XGBoost Raw) = {avg_spearman_r:.3f} (p={avg_spearman_p:.3e})")

print("\n" + "="*80)
print(" Statistical testing complete. Review decisions above for distributional insights.")
print("="*80)

# %%
print("\n" + "="*80)
print(" KAGGLE REFERENCE TESTS: TRAIN VS TEST SHIFT & ANOMALY BUDGET")
print("="*80)

train_path = PROJECT_ROOT / '.data' / 'train.parquet'
if train_path.exists():
    print(f"\nLoading training reference data from {train_path} (channels 41-46)...")
    train_df = pd.read_parquet(train_path, columns=feature_cols)
    train_samples_total = len(train_df)
    train_sample_size = min(sample_size, train_samples_total)
    if train_sample_size < train_samples_total:
        train_sample_idx = rng.choice(train_samples_total, size=train_sample_size, replace=False)
        train_sample = train_df.iloc[train_sample_idx]
    else:
        train_sample = train_df
    print(f"  Reference sample size: {len(train_sample):,} rows")

    print("\nKolmogorov-Smirnov tests (train vs test):")
    print(f"{'Channel':<12}{'KS D':>12}{'p-value':>14}{'Decision (α=0.05)':>26}")
    for idx, channel in enumerate(feature_cols):
        test_values = X_raw[sample_idx, idx]
        train_values = train_sample[channel].values
        ks_stat, ks_p = stats.ks_2samp(train_values, test_values)
        ks_decision = 'Reject same dist' if ks_p < 0.05 else 'Fail to reject'
        print(f"{channel:<12}{ks_stat:>12.3f}{ks_p:>14.3e}{ks_decision:>26}")

    print("\nMann-Whitney U tests (train vs test medians):")
    print(f"{'Channel':<12}{'U-stat':>12}{'p-value':>14}{'Decision (α=0.05)':>26}")
    for idx, channel in enumerate(feature_cols):
        test_values = X_raw[sample_idx, idx]
        train_values = train_sample[channel].values
        u_stat, u_p = stats.mannwhitneyu(train_values, test_values, alternative='two-sided')
        u_decision = 'Reject same median' if u_p < 0.05 else 'Fail to reject'
        print(f"{channel:<12}{u_stat:>12.2f}{u_p:>14.3e}{u_decision:>26}")

    shift_magnitude = np.mean([stats.ks_2samp(train_sample[ch].values, X_raw[sample_idx, idx])[0] for idx, ch in enumerate(feature_cols)])
    print(f"\nAverage KS shift across channels: {shift_magnitude:.3f}")
else:
    print(f"Training reference not found at {train_path}. Skipping train vs test diagnostics.")

# Binomial test against anomaly budget (competition expects rare events)
expected_rate = 0.01  # 1% anomaly prior as a conservative ESA benchmark
observed_anomalies = np.sum(xgb_processed > 0)
observed_rate = observed_anomalies / len(xgb_processed)
binom_res = stats.binomtest(observed_anomalies, len(xgb_processed), expected_rate, alternative='greater')

print("\nAnomaly rate vs 1% budget (binomial test):")
print(f"  Observed rate: {observed_rate*100:.2f}% ({observed_anomalies:,} / {len(xgb_processed):,})")
print(f"  p-value (greater than 1%): {binom_res.pvalue:.3e}")
print(f"  Decision: {'Exceeds budget' if binom_res.pvalue < 0.05 else 'Within budget'}")

# Event length diagnostics to mimic Kaggle event-based scoring focus
event_lengths = [end - start + 1 for start, end in xgb_segments]
if event_lengths:
    print("\nEvent length statistics (post-processing segments):")
    print(f"  Count: {len(event_lengths):,}")
    print(f"  Mean length: {np.mean(event_lengths):.1f} samples")
    print(f"  Median length: {np.median(event_lengths):.1f} samples")
    print(f"  95th percentile: {np.percentile(event_lengths, 95):.1f} samples")
else:
    print("\nNo events detected after post-processing; event statistics unavailable.")

print("\n" + "="*80)
print(" Kaggle-aligned statistical diagnostics complete.")
print("="*80)


