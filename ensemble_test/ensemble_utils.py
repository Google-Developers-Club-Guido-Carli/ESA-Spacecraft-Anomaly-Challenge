"""
Ensemble Anomaly Detection Utilities
Provides score transforms, event postprocessing, and event-wise metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def z_to_score(z: np.ndarray, z0: float = 3.0) -> np.ndarray:
    """
    Transform z-scores to anomaly scores in [0,1] using exponential mapping.
    
    s = 1 - exp(-z/z0)
    
    Args:
        z: Array of z-scores (non-negative, absolute values)
        z0: Scale parameter (higher = slower rise)
    
    Returns:
        Anomaly scores in [0,1]
    """
    return 1 - np.exp(-np.abs(z) / z0)


def percentile_normalize(raw_scores: np.ndarray, p5: float, p95: float) -> np.ndarray:
    """
    Normalize raw scores using percentile clipping to [0,1].
    
    Args:
        raw_scores: Raw anomaly scores
        p5: 5th percentile (lower bound)
        p95: 95th percentile (upper bound)
    
    Returns:
        Normalized scores clipped to [0,1]
    """
    if p95 <= p5:
        # Handle edge case where all scores are similar
        return np.clip((raw_scores - p5) / max(1e-6, p95 - p5), 0, 1)
    normalized = (raw_scores - p5) / (p95 - p5)
    return np.clip(normalized, 0, 1)


def compute_percentiles(scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute 5th and 95th percentiles for normalization.
    
    Args:
        scores: Array of scores
    
    Returns:
        (p5, p95) tuple
    """
    p5 = np.percentile(scores, 5)
    p95 = np.percentile(scores, 95)
    return p5, p95


def extract_events(binary_series: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract contiguous event segments from binary series.
    
    Args:
        binary_series: Binary array (0/1)
    
    Returns:
        List of (start_idx, end_idx) tuples for each event (inclusive)
    """
    events = []
    in_event = False
    start = 0
    
    for i, val in enumerate(binary_series):
        if val == 1 and not in_event:
            start = i
            in_event = True
        elif val == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    
    # Handle case where series ends in event
    if in_event:
        events.append((start, len(binary_series) - 1))
    
    return events


def prune_short_events(binary_series: np.ndarray, L_min: int) -> np.ndarray:
    """
    Remove events shorter than L_min samples.
    
    Args:
        binary_series: Binary array
        L_min: Minimum event length
    
    Returns:
        Pruned binary array
    """
    events = extract_events(binary_series)
    pruned = np.zeros_like(binary_series)
    
    for start, end in events:
        event_length = end - start + 1
        if event_length >= L_min:
            pruned[start:end+1] = 1
    
    return pruned


def merge_nearby_events(binary_series: np.ndarray, gap_merge: int) -> np.ndarray:
    """
    Merge events separated by gap <= gap_merge samples.
    
    Args:
        binary_series: Binary array
        gap_merge: Maximum gap to merge
    
    Returns:
        Merged binary array
    """
    events = extract_events(binary_series)
    if len(events) == 0:
        return binary_series
    
    merged = np.zeros_like(binary_series)
    merged_events = []
    current_start, current_end = events[0]
    
    for i in range(1, len(events)):
        next_start, next_end = events[i]
        gap = next_start - current_end - 1
        
        if gap <= gap_merge:
            # Merge by extending current event
            current_end = next_end
        else:
            # Save current and start new
            merged_events.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Add last event
    merged_events.append((current_start, current_end))
    
    # Fill merged binary array
    for start, end in merged_events:
        merged[start:end+1] = 1
    
    return merged


def postprocess_events(binary_series: np.ndarray, L_min: int = 3, gap_merge: int = 2) -> np.ndarray:
    """
    Complete event postprocessing pipeline:
    1. Prune short events
    2. Merge nearby events
    
    Args:
        binary_series: Binary predictions
        L_min: Minimum event length
        gap_merge: Maximum gap to merge
    
    Returns:
        Postprocessed binary array
    """
    pruned = prune_short_events(binary_series, L_min)
    merged = merge_nearby_events(pruned, gap_merge)
    return merged


def eventwise_precision_recall_f05(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute event-wise precision, recall, and F-beta score.
    
    A predicted event is a true positive if it overlaps with at least one ground truth event.
    
    Args:
        y_true: Ground truth binary array
        y_pred: Predicted binary array
        beta: F-beta parameter (0.5 for F0.5)
    
    Returns:
        (precision, recall, f_beta) tuple
    """
    true_events = extract_events(y_true)
    pred_events = extract_events(y_pred)
    
    if len(pred_events) == 0:
        # No predictions
        if len(true_events) == 0:
            return 1.0, 1.0, 1.0  # Perfect if no anomalies
        else:
            return 0.0, 0.0, 0.0  # Missed all anomalies
    
    if len(true_events) == 0:
        # Predicted anomalies but none exist
        return 0.0, 0.0, 0.0
    
    # Count true positive predicted events (those that overlap with any true event)
    tp_pred = 0
    for pred_start, pred_end in pred_events:
        for true_start, true_end in true_events:
            # Check overlap
            if not (pred_end < true_start or pred_start > true_end):
                tp_pred += 1
                break  # Count this predicted event once
    
    precision = tp_pred / len(pred_events) if len(pred_events) > 0 else 0.0
    recall = tp_pred / len(true_events) if len(true_events) > 0 else 0.0
    
    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    return precision, recall, f_beta


def pointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> dict:
    """
    Compute point-wise metrics for debugging.
    
    Args:
        y_true: Ground truth binary
        y_pred: Predicted binary
        beta: F-beta parameter
    
    Returns:
        Dictionary with precision, recall, f_beta
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f_beta': f_beta,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }
