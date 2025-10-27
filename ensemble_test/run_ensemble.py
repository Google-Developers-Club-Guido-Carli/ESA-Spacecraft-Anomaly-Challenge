"""
Demo script to run ensemble on ESA spacecraft data.
Loads train/test from .data folder and runs ensemble detector.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ensemble import EnsembleDetector
from ensemble_utils import (
    eventwise_precision_recall_f05,
    pointwise_metrics,
    extract_events
)
from ensemble_plots import plot_comprehensive_report


def load_data(data_dir: str = None):
    """Load train and test data."""
    print("Loading data...")
    
    # Auto-detect data directory relative to script location
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), '.data')
    
    print(f"Data directory: {data_dir}")
    
    # Use parquet files (CSV files are empty)
    train_path = os.path.join(data_dir, 'train.parquet')
    test_path = os.path.join(data_dir, 'test.parquet')
    
    # Load with relevant columns only (channels 41-46 + id + is_anomaly)
    channel_names = [f'channel_{i}' for i in range(41, 47)]
    
    # Try to load train with is_anomaly column
    try:
        # Load only needed columns to save memory
        cols_to_load = ['id'] + channel_names + ['is_anomaly']
        train_df = pd.read_parquet(train_path, columns=cols_to_load)
        print(f"Train shape: {train_df.shape}")
        print(f"Train columns: {train_df.columns.tolist()}")
        
        # Check if is_anomaly exists
        has_labels = 'is_anomaly' in train_df.columns
        if has_labels:
            print(f"Anomaly rate: {train_df['is_anomaly'].mean():.6f}")
        else:
            print("Warning: No 'is_anomaly' column in train data")
    except Exception as e:
        print(f"Error loading train: {e}")
        return None, None
    
    # Load test
    try:
        # Test might not have is_anomaly
        cols_to_load_test = ['id'] + channel_names
        test_df = pd.read_parquet(test_path, columns=cols_to_load_test)
        print(f"Test shape: {test_df.shape}")
        print(f"Test columns: {test_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading test: {e}")
        return train_df, None
    
    return train_df, test_df


def split_train_val(train_df: pd.DataFrame, val_fraction: float = 0.15):
    """Split train into train/val (time-aware, no shuffle)."""
    n = len(train_df)
    split_idx = int(n * (1 - val_fraction))
    
    train_subset = train_df.iloc[:split_idx].copy()
    val_subset = train_df.iloc[split_idx:].copy()
    
    print(f"Train subset: {len(train_subset)} samples")
    print(f"Val subset: {len(val_subset)} samples")
    
    return train_subset, val_subset


def main():
    """Main execution."""
    print("="*60)
    print("ESA Spacecraft Anomaly Detection - Ensemble Demo")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data()
    if train_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Check for required columns
    channel_names = [f'channel_{i}' for i in range(41, 47)]
    missing_channels = [c for c in channel_names if c not in train_df.columns]
    if missing_channels:
        print(f"Error: Missing channels: {missing_channels}")
        return
    
    # Split train/val
    train_subset, val_subset = split_train_val(train_df, val_fraction=0.15)
    
    # Estimate contamination if labels exist
    has_labels = 'is_anomaly' in train_subset.columns
    if has_labels:
        contamination = train_subset['is_anomaly'].mean()
        print(f"Using contamination={contamination:.4f} for IsolationForest")
    else:
        contamination = 0.01
        print(f"No labels found, using default contamination={contamination}")
    
    # Initialize ensemble (removed explicit weights to use new defaults)
    print("\n" + "="*60)
    print("Initializing Ensemble Detector")
    print("="*60)
    
    ensemble = EnsembleDetector(
        channel_names=channel_names
    )
    
    # Fit on training data
    print("\n" + "="*60)
    print("Fitting Ensemble")
    print("="*60)
    ensemble.fit(train_subset, contamination=contamination)
    
    # Calibrate threshold on validation if labels exist
    if has_labels and 'is_anomaly' in val_subset.columns:
        print("\n" + "="*60)
        print("Calibrating Threshold on Validation Set")
        print("="*60)
        
        y_val = val_subset['is_anomaly'].values
        best_threshold = ensemble.calibrate_threshold(
            val_subset,
            y_val,
            metric='f05',
            n_thresholds=200
        )
        
        # Evaluate on validation
        print("\n" + "="*60)
        print("Validation Results")
        print("="*60)
        
        scores_val = ensemble.score(val_subset)
        predictions_val = ensemble.predict(val_subset, threshold=best_threshold)
        
        # Show detector score statistics
        print(f"\nDetector Score Statistics:")
        print(f"  Global STD - Mean: {scores_val['s_global'].mean():.4f}, Max: {scores_val['s_global'].max():.4f}, >0.5: {(scores_val['s_global'] > 0.5).sum()}")
        print(f"  Local STD  - Mean: {scores_val['s_local'].mean():.4f}, Max: {scores_val['s_local'].max():.4f}, >0.5: {(scores_val['s_local'] > 0.5).sum()}")
        print(f"  IForest    - Mean: {scores_val['s_iforest'].mean():.4f}, Max: {scores_val['s_iforest'].max():.4f}, >0.5: {(scores_val['s_iforest'] > 0.5).sum()}")
        print(f"  Ensemble   - Mean: {scores_val['score'].mean():.4f}, Max: {scores_val['score'].max():.4f}, >{best_threshold:.2f}: {(scores_val['score'] > best_threshold).sum()}")
        
        # Event-wise metrics
        precision_ew, recall_ew, f05_ew = eventwise_precision_recall_f05(y_val, predictions_val)
        print(f"Event-wise Precision: {precision_ew:.4f}")
        print(f"Event-wise Recall: {recall_ew:.4f}")
        print(f"Event-wise F0.5: {f05_ew:.4f}")
        
        # Point-wise metrics
        pointwise = pointwise_metrics(y_val, predictions_val)
        print(f"\nPoint-wise Precision: {pointwise['precision']:.4f}")
        print(f"Point-wise Recall: {pointwise['recall']:.4f}")
        print(f"Point-wise F0.5: {pointwise['f_beta']:.4f}")
        print(f"TP: {pointwise['tp']}, FP: {pointwise['fp']}, FN: {pointwise['fn']}")
        
        # Show detected events
        pred_events = extract_events(predictions_val)
        true_events = extract_events(y_val)
        print(f"\nDetected {len(pred_events)} events (ground truth: {len(true_events)})")
        if len(pred_events) > 0:
            print(f"Example events: {pred_events[:5]}")
        
        # Generate visualization plots
        print("\n" + "="*60)
        print("Generating Visualization Plots")
        print("="*60)
        
        # Choose a sample range with anomalies for detailed view
        if len(true_events) > 0:
            # Pick middle event and show context around it
            mid_event = true_events[len(true_events) // 2]
            event_start, event_end = mid_event
            context = 5000  # Show 5000 samples before and after
            sample_start = max(0, event_start - context)
            sample_end = min(len(y_val), event_end + context)
            sample_range = (sample_start, sample_end)
        else:
            # Just show first 10000 samples
            sample_range = (0, min(10000, len(y_val)))
        
        plot_comprehensive_report(
            scores_df=scores_val,
            predictions=predictions_val,
            ground_truth=y_val,
            threshold=best_threshold,
            sample_range=sample_range,
            output_dir="outputs/plots"
        )
    else:
        best_threshold = 0.5
        print(f"\nNo validation labels, using default threshold={best_threshold}")
    
    # Predict on test set if available
    if test_df is not None:
        print("\n" + "="*60)
        print("Predicting on Test Set")
        print("="*60)
        
        scores_test = ensemble.score(test_df)
        predictions_test = ensemble.predict(test_df, threshold=best_threshold)
        
        print(f"Test predictions shape: {predictions_test.shape}")
        print(f"Anomalies detected: {predictions_test.sum()} / {len(predictions_test)} ({predictions_test.mean():.4f})")
        
        # Count events
        test_events = extract_events(predictions_test)
        print(f"Number of anomaly events: {len(test_events)}")
        
        # Save submission
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        submission_df = pd.DataFrame({
            'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
            'is_anomaly': predictions_test
        })
        
        submission_path = os.path.join(output_dir, 'ensemble_submission.csv')
        submission_df.to_csv(submission_path, index=False)
        print(f"\nSubmission saved to: {submission_path}")
        
        # Save detailed scores
        scores_path = os.path.join(output_dir, 'ensemble_scores.csv')
        scores_test['id'] = submission_df['id']
        scores_test.to_csv(scores_path, index=False)
        print(f"Detailed scores saved to: {scores_path}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
