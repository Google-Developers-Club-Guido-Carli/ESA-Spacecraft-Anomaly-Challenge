"""
Ensemble Detector Visualization Module
Creates plots for interpreting ensemble detector outputs and individual detector scores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from ensemble_utils import extract_events

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def plot_detector_scores(
    scores_df: pd.DataFrame,
    ground_truth: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    time_range: Optional[Tuple[int, int]] = None,
    title: str = "Ensemble Detector Scores",
    save_path: Optional[str] = None
):
    """
    Plot individual detector scores and ensemble output over time.
    
    Args:
        scores_df: DataFrame with columns ['score', 's_global', 's_local', 's_iforest']
        ground_truth: Optional binary ground truth array
        predictions: Optional binary predictions array
        time_range: Optional (start, end) indices to plot
        title: Plot title
        save_path: Optional path to save figure
    """
    if time_range is not None:
        start, end = time_range
        scores_subset = scores_df.iloc[start:end].copy()
        scores_subset.index = range(len(scores_subset))
        if ground_truth is not None:
            ground_truth = ground_truth[start:end]
        if predictions is not None:
            predictions = predictions[start:end]
    else:
        scores_subset = scores_df.copy()
    
    n_samples = len(scores_subset)
    time_idx = np.arange(n_samples)
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: GlobalSTD detector
    axes[0].plot(time_idx, scores_subset['s_global'], color='blue', alpha=0.7, linewidth=0.8)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].set_ylabel('GlobalSTD\nScore', fontsize=10)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    if ground_truth is not None:
        axes[0].fill_between(time_idx, 0, 1, where=ground_truth > 0, alpha=0.2, color='red', label='True Anomaly')
    
    # Plot 2: LocalSTD detector
    axes[1].plot(time_idx, scores_subset['s_local'], color='green', alpha=0.7, linewidth=0.8)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_ylabel('LocalSTD\nScore', fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    if ground_truth is not None:
        axes[1].fill_between(time_idx, 0, 1, where=ground_truth > 0, alpha=0.2, color='red', label='True Anomaly')
    
    # Plot 3: IsolationForest detector
    axes[2].plot(time_idx, scores_subset['s_iforest'], color='orange', alpha=0.7, linewidth=0.8)
    axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[2].set_ylabel('IForest\nScore', fontsize=10)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    if ground_truth is not None:
        axes[2].fill_between(time_idx, 0, 1, where=ground_truth > 0, alpha=0.2, color='red', label='True Anomaly')
    
    # Plot 4: Ensemble score
    axes[3].plot(time_idx, scores_subset['score'], color='purple', alpha=0.7, linewidth=0.8)
    axes[3].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[3].set_ylabel('Ensemble\nScore', fontsize=10)
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)
    if ground_truth is not None:
        axes[3].fill_between(time_idx, 0, 1, where=ground_truth > 0, alpha=0.2, color='red', label='True Anomaly')
    
    # Plot 5: Binary predictions vs ground truth
    if predictions is not None:
        axes[4].fill_between(time_idx, 0, 1, where=predictions > 0, alpha=0.5, color='blue', label='Predicted Anomaly')
    if ground_truth is not None:
        axes[4].fill_between(time_idx, 0, 1, where=ground_truth > 0, alpha=0.3, color='red', label='True Anomaly')
    axes[4].set_ylabel('Binary\nPredictions', fontsize=10)
    axes[4].set_xlabel('Time Index', fontsize=11)
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_yticks([0, 1])
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='upper right', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_score_distributions(
    scores_df: pd.DataFrame,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Detector Score Distributions",
    save_path: Optional[str] = None
):
    """
    Plot histograms and distributions of detector scores.
    
    Args:
        scores_df: DataFrame with detector scores
        ground_truth: Optional binary ground truth for normal vs anomaly comparison
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    detectors = [
        ('s_global', 'GlobalSTD', 'blue'),
        ('s_local', 'LocalSTD', 'green'),
        ('s_iforest', 'IForest', 'orange'),
        ('score', 'Ensemble', 'purple')
    ]
    
    for idx, (col, name, color) in enumerate(detectors):
        ax = axes[idx // 2, idx % 2]
        
        if ground_truth is not None:
            # Separate normal and anomaly scores
            normal_scores = scores_df[col][ground_truth == 0]
            anomaly_scores = scores_df[col][ground_truth == 1]
            
            ax.hist(normal_scores, bins=50, alpha=0.6, color='gray', label='Normal', density=True)
            ax.hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly', density=True)
            ax.legend(fontsize=10)
        else:
            ax.hist(scores_df[col], bins=50, alpha=0.7, color=color, density=True)
        
        ax.set_xlabel(f'{name} Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name} Score Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = scores_df[col].mean()
        median_score = scores_df[col].median()
        max_score = scores_df[col].max()
        ax.axvline(mean_score, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_score:.3f}')
        ax.text(0.98, 0.95, f'Mean: {mean_score:.3f}\nMedian: {median_score:.3f}\nMax: {max_score:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_detector_agreement(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
    title: str = "Detector Agreement Analysis",
    save_path: Optional[str] = None
):
    """
    Visualize agreement between different detectors.
    
    Args:
        scores_df: DataFrame with detector scores
        threshold: Threshold for binary detector votes
        title: Plot title
        save_path: Optional path to save figure
    """
    # Compute binary votes
    vote_global = (scores_df['s_global'] >= threshold).astype(int)
    vote_local = (scores_df['s_local'] >= threshold).astype(int)
    vote_iforest = (scores_df['s_iforest'] >= threshold).astype(int)
    
    # Count agreement levels
    votes_sum = vote_global + vote_local + vote_iforest
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Agreement counts
    agreement_counts = pd.Series(votes_sum).value_counts().sort_index()
    axes[0, 0].bar(agreement_counts.index, agreement_counts.values, color=['gray', 'yellow', 'orange', 'red'], alpha=0.7)
    axes[0, 0].set_xlabel('Number of Detectors Agreeing', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Detector Agreement Levels', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = len(votes_sum)
    for i, count in agreement_counts.items():
        pct = 100 * count / total
        axes[0, 0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Correlation heatmap
    corr_matrix = scores_df[['s_global', 's_local', 's_iforest', 'score']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, ax=axes[0, 1], cbar_kws={'label': 'Correlation'})
    axes[0, 1].set_title('Detector Score Correlations', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticklabels(['Global', 'Local', 'IForest', 'Ensemble'], rotation=45)
    axes[0, 1].set_yticklabels(['Global', 'Local', 'IForest', 'Ensemble'], rotation=0)
    
    # Plot 3: Scatter plot - Global vs Local
    axes[1, 0].scatter(scores_df['s_global'], scores_df['s_local'], alpha=0.3, s=5, color='blue')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('GlobalSTD Score', fontsize=11)
    axes[1, 0].set_ylabel('LocalSTD Score', fontsize=11)
    axes[1, 0].set_title('Global vs Local Detector', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)
    
    # Plot 4: Scatter plot - Ensemble vs Max detector
    max_detector = scores_df[['s_global', 's_local', 's_iforest']].max(axis=1)
    axes[1, 1].scatter(max_detector, scores_df['score'], alpha=0.3, s=5, color='purple')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel('Max Individual Detector Score', fontsize=11)
    axes[1, 1].set_ylabel('Ensemble Score', fontsize=11)
    axes[1, 1].set_title('Ensemble vs Max Detector', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_event_analysis(
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    title: str = "Anomaly Event Analysis",
    save_path: Optional[str] = None
):
    """
    Analyze detected anomaly events.
    
    Args:
        predictions: Binary predictions
        ground_truth: Optional ground truth
        scores: Optional continuous scores
        title: Plot title
        save_path: Optional path to save figure
    """
    pred_events = extract_events(predictions)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Event lengths
    if len(pred_events) > 0:
        event_lengths = [end - start + 1 for start, end in pred_events]
        axes[0].hist(event_lengths, bins=min(50, len(event_lengths)), color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Event Length (samples)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title(f'Predicted Event Length Distribution (n={len(pred_events)} events)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_len = np.mean(event_lengths)
        median_len = np.median(event_lengths)
        axes[0].axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
        axes[0].axvline(median_len, color='green', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
        axes[0].legend(fontsize=10)
    else:
        axes[0].text(0.5, 0.5, 'No events detected', ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
        axes[0].set_title('Predicted Event Length Distribution (n=0 events)', fontsize=12, fontweight='bold')
    
    # Plot 2: Event comparison if ground truth available
    if ground_truth is not None:
        true_events = extract_events(ground_truth)
        
        # Create timeline visualization
        time_idx = np.arange(len(predictions))
        
        axes[1].fill_between(time_idx, 0, 0.4, where=predictions > 0, alpha=0.5, color='blue', label='Predicted')
        axes[1].fill_between(time_idx, 0.6, 1.0, where=ground_truth > 0, alpha=0.5, color='red', label='Ground Truth')
        
        axes[1].set_xlabel('Time Index', fontsize=11)
        axes[1].set_ylabel('Event Type', fontsize=11)
        axes[1].set_title(f'Event Timeline (Predicted: {len(pred_events)}, True: {len(true_events)})', 
                         fontsize=12, fontweight='bold')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_yticks([0.2, 0.8])
        axes[1].set_yticklabels(['Predicted', 'True'])
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='x')
    else:
        # Just show predicted events timeline
        time_idx = np.arange(len(predictions))
        axes[1].fill_between(time_idx, 0, 1, where=predictions > 0, alpha=0.5, color='blue')
        axes[1].set_xlabel('Time Index', fontsize=11)
        axes[1].set_ylabel('Anomaly', fontsize=11)
        axes[1].set_title(f'Predicted Events Timeline (n={len(pred_events)} events)', fontsize=12, fontweight='bold')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_comprehensive_report(
    scores_df: pd.DataFrame,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    sample_range: Optional[Tuple[int, int]] = None,
    output_dir: str = "plots"
):
    """
    Generate a comprehensive visualization report.
    
    Args:
        scores_df: DataFrame with all detector scores
        predictions: Binary predictions
        ground_truth: Optional ground truth
        threshold: Detection threshold
        sample_range: Optional (start, end) for time series plots
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating comprehensive visualization report...")
    print(f"Output directory: {output_dir}")
    
    # Plot 1: Time series of all detector scores
    plot_detector_scores(
        scores_df,
        ground_truth=ground_truth,
        predictions=predictions,
        time_range=sample_range,
        title="Ensemble Detector Scores Over Time",
        save_path=os.path.join(output_dir, "detector_scores_timeline.png")
    )
    plt.close()
    
    # Plot 2: Score distributions
    plot_score_distributions(
        scores_df,
        ground_truth=ground_truth,
        title="Detector Score Distributions",
        save_path=os.path.join(output_dir, "score_distributions.png")
    )
    plt.close()
    
    # Plot 3: Detector agreement
    plot_detector_agreement(
        scores_df,
        threshold=threshold,
        title="Detector Agreement Analysis",
        save_path=os.path.join(output_dir, "detector_agreement.png")
    )
    plt.close()
    
    # Plot 4: Event analysis
    plot_event_analysis(
        predictions,
        ground_truth=ground_truth,
        scores=scores_df['score'].values,
        title="Anomaly Event Analysis",
        save_path=os.path.join(output_dir, "event_analysis.png")
    )
    plt.close()
    
    print(f"✓ Generated 4 visualization plots in {output_dir}/")
    print(f"  - detector_scores_timeline.png")
    print(f"  - score_distributions.png")
    print(f"  - detector_agreement.png")
    print(f"  - event_analysis.png")


if __name__ == "__main__":
    print("Ensemble plotting module - import and use functions for visualization")
    print("\nAvailable functions:")
    print("  - plot_detector_scores(): Time series plot of all detector outputs")
    print("  - plot_score_distributions(): Histogram distributions of scores")
    print("  - plot_detector_agreement(): Detector correlation and agreement analysis")
    print("  - plot_event_analysis(): Anomaly event statistics and timeline")
    print("  - plot_comprehensive_report(): Generate all plots at once")
