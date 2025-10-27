"""
Ensemble Anomaly Detector
Combines GlobalSTD, LocalSTD, and Isolation Forest detectors.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from sklearn.ensemble import IsolationForest
import warnings

from ensemble_utils import (
    z_to_score,
    percentile_normalize,
    compute_percentiles,
    postprocess_events
)


class EnsembleDetector:
    """
    Ensemble anomaly detector combining multiple detection algorithms.
    
    Detectors:
    - GlobalSTD: Global mean/std deviation detection
    - LocalSTD: Rolling window local statistics
    - IsolationForest: Multivariate isolation forest
    
    Weights sum to 1.0 and can be adjusted dynamically.
    """
    
    def __init__(
        self,
        channel_names: List[str] = None,
        weights: Dict[str, float] = None,
        per_detector_thresholds: Dict[str, float] = None,
        L_min: int = 2,
        gap_merge: int = 8,
        z0_global: float = 2.0,
        z0_local: float = 1.8,
        window_size: int = 100,
        min_std: float = 1e-6,
        consensus_bonus: float = 0.15,
        consensus_min_votes: int = 2
    ):
        """
        Initialize ensemble detector.
        
        Args:
            channel_names: List of channel column names (e.g., ['channel_41', ...])
            weights: Detector weights {'global': w_g, 'local': w_l, 'iforest': w_i}
            per_detector_thresholds: Vote thresholds for each detector
            L_min: Minimum event length for postprocessing
            gap_merge: Maximum gap to merge events
            z0_global: Scale parameter for GlobalSTD z-score transform
            z0_local: Scale parameter for LocalSTD z-score transform
            window_size: Rolling window size for LocalSTD
            min_std: Minimum std to prevent division by zero
            consensus_bonus: Bonus added when multiple detectors agree
            consensus_min_votes: Minimum votes needed for consensus bonus
        """
        if channel_names is None:
            channel_names = [f'channel_{i}' for i in range(41, 47)]
        self.channel_names = channel_names
        
        # Default weights - favor local and iforest for better sensitivity
        if weights is None:
            weights = {'global': 0.20, 'local': 0.45, 'iforest': 0.35}
        self.weights = self._normalize_weights(weights)
        
        # Default per-detector vote thresholds - lower for better recall
        if per_detector_thresholds is None:
            per_detector_thresholds = {'global': 0.35, 'local': 0.30, 'iforest': 0.40}
        self.per_detector_thresholds = per_detector_thresholds
        
        # Postprocessing params
        self.L_min = L_min
        self.gap_merge = gap_merge
        
        # Detector-specific params
        self.z0_global = z0_global
        self.z0_local = z0_local
        self.window_size = window_size
        self.min_std = min_std
        self.consensus_bonus = consensus_bonus
        self.consensus_min_votes = consensus_min_votes
        
        # Fitted state
        self.global_means_ = None
        self.global_stds_ = None
        self.iforest_model_ = None
        self.iforest_p5_ = None
        self.iforest_p95_ = None
        self.fitted_ = False
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Weights sum to zero")
        return {k: v / total for k, v in weights.items()}
    
    def fit(self, X_train: pd.DataFrame, contamination: float = 0.01):
        """
        Fit the ensemble on training data.
        
        Args:
            X_train: Training DataFrame with channel columns
            contamination: Anomaly contamination rate for IsolationForest
        """
        # Extract channel data
        X_channels = X_train[self.channel_names].values
        
        # Fit GlobalSTD: compute mean and std on training data
        self.global_means_ = np.mean(X_channels, axis=0)
        self.global_stds_ = np.std(X_channels, axis=0)
        self.global_stds_ = np.maximum(self.global_stds_, self.min_std)
        
        # Fit IsolationForest
        print(f"Fitting IsolationForest with contamination={contamination}...")
        self.iforest_model_ = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.iforest_model_.fit(X_channels)
        
        # Compute percentiles for iForest score normalization
        iforest_scores_train = self.iforest_model_.decision_function(X_channels)
        self.iforest_p5_, self.iforest_p95_ = compute_percentiles(iforest_scores_train)
        
        self.fitted_ = True
        print("Ensemble fitted successfully.")
    
    def _detect_globalstd(self, X: np.ndarray) -> np.ndarray:
        """
        GlobalSTD detector: compare each point to global mean ± N*std.
        
        Returns:
            Anomaly scores [0,1] for each timepoint
        """
        # Compute z-scores per channel
        z_scores = np.abs(X - self.global_means_) / self.global_stds_
        
        # Use 95th percentile of z-scores across channels for better sensitivity
        z_p95 = np.percentile(z_scores, 95, axis=1)
        
        # Transform to [0,1]
        scores = z_to_score(z_p95, self.z0_global)
        return scores
    
    def _detect_localstd(self, X: pd.DataFrame) -> np.ndarray:
        """
        LocalSTD detector: rolling window local z-scores.
        
        Returns:
            Anomaly scores [0,1] for each timepoint
        """
        # Compute rolling mean and std for each channel
        rolling_mean = X.rolling(
            window=self.window_size,
            center=True,
            min_periods=max(1, self.window_size // 4)
        ).mean()
        
        rolling_std = X.rolling(
            window=self.window_size,
            center=True,
            min_periods=max(1, self.window_size // 4)
        ).std()
        
        # Enforce minimum std
        rolling_std = rolling_std.fillna(self.min_std)
        rolling_std = np.maximum(rolling_std.values, self.min_std)
        
        # Compute local z-scores
        z_local = np.abs(X.values - rolling_mean.values) / rolling_std
        
        # Use 90th percentile across channels for better sensitivity
        z_p90 = np.percentile(z_local, 90, axis=1)
        
        # Transform to [0,1]
        scores = z_to_score(z_p90, self.z0_local)
        return scores
    
    def _detect_iforest(self, X: np.ndarray) -> np.ndarray:
        """
        IsolationForest detector.
        
        Returns:
            Anomaly scores [0,1] for each timepoint
        """
        # Get decision function (higher = more normal in sklearn IsolationForest)
        # We want higher = more anomalous, so negate
        raw_scores = -self.iforest_model_.decision_function(X)
        
        # Normalize using training percentiles
        scores = percentile_normalize(raw_scores, self.iforest_p5_, self.iforest_p95_)
        return scores
    
    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ensemble anomaly scores.
        
        Args:
            X: DataFrame with channel columns
        
        Returns:
            DataFrame with columns: ['score', 's_global', 's_local', 's_iforest']
        """
        if not self.fitted_:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Extract channel data
        X_channels = X[self.channel_names]
        X_array = X_channels.values
        
        # Compute per-detector scores
        s_global = self._detect_globalstd(X_array)
        s_local = self._detect_localstd(X_channels)
        s_iforest = self._detect_iforest(X_array)
        
        # Weighted aggregation
        S_weighted = (
            self.weights['global'] * s_global +
            self.weights['local'] * s_local +
            self.weights['iforest'] * s_iforest
        )
        
        # Also consider max score for high-confidence anomalies
        S_max = np.maximum.reduce([s_global, s_local, s_iforest])
        
        # Blend weighted average with max (60% weighted, 40% max for more sensitivity)
        S_raw = 0.6 * S_weighted + 0.4 * S_max
        
        # Consensus bonus: count votes
        votes = (
            (s_global >= self.per_detector_thresholds['global']).astype(int) +
            (s_local >= self.per_detector_thresholds['local']).astype(int) +
            (s_iforest >= self.per_detector_thresholds['iforest']).astype(int)
        )
        
        consensus_mask = votes >= self.consensus_min_votes
        bonus = self.consensus_bonus * consensus_mask
        
        # Final score
        S = np.clip(S_raw + bonus, 0, 1)
        
        # Return as DataFrame
        result = pd.DataFrame({
            'score': S,
            's_global': s_global,
            's_local': s_local,
            's_iforest': s_iforest
        })
        
        return result
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary anomalies with postprocessing.
        
        Args:
            X: DataFrame with channel columns
            threshold: Final threshold for binarization
        
        Returns:
            Binary array (0/1) after temporal postprocessing
        """
        scores_df = self.score(X)
        
        # Binarize
        binary = (scores_df['score'].values >= threshold).astype(int)
        
        # Postprocess
        binary_postprocessed = postprocess_events(binary, self.L_min, self.gap_merge)
        
        return binary_postprocessed
    
    def calibrate_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        metric: str = 'f05',
        n_thresholds: int = 200
    ) -> float:
        """
        Calibrate final threshold by maximizing validation metric.
        
        Args:
            X_val: Validation DataFrame
            y_val: Validation ground truth (binary)
            metric: Metric to optimize ('f05', 'precision', 'recall')
            n_thresholds: Number of thresholds to try
        
        Returns:
            Best threshold value
        """
        from ensemble_utils import eventwise_precision_recall_f05
        
        scores_df = self.score(X_val)
        thresholds = np.linspace(0, 1, n_thresholds)
        
        best_threshold = 0.5
        best_metric_value = 0.0
        
        print(f"Calibrating threshold (optimizing {metric})...")
        
        for thresh in thresholds:
            binary = (scores_df['score'].values >= thresh).astype(int)
            binary_postprocessed = postprocess_events(binary, self.L_min, self.gap_merge)
            
            precision, recall, f05 = eventwise_precision_recall_f05(y_val, binary_postprocessed, beta=0.5)
            
            if metric == 'f05':
                metric_value = f05
            elif metric == 'precision':
                metric_value = precision
            elif metric == 'recall':
                metric_value = recall
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = thresh
        
        print(f"Best threshold: {best_threshold:.4f} ({metric}={best_metric_value:.4f})")
        return best_threshold
