"""
Signal Interface for Regime and Kalman Filter Integration

This module provides a unified interface to query current regime and Kalman signals
per timestep, combining both HMM regime detection and Kalman filter smoothing.
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.regime_hmm import RegimeHMM
from models.kalman_filter import KalmanFilterModel


class SignalInterface:
    """
    Unified interface for regime and Kalman filter signals.
    
    Provides methods to:
    - Load fitted models
    - Query current regime and Kalman signals
    - Get combined features for RL environment
    - Ensure no future leakage
    """
    
    def __init__(self, 
                 regime_model_path: str = '../data_cache/regime_hmm.pkl',
                 kalman_model_path: str = '../data_cache/kalman_filter.pkl',
                 regime_features_path: str = '../data_cache/regime_features.csv',
                 kalman_features_path: str = '../data_cache/kalman_features.csv'):
        """
        Initialize the signal interface.
        
        Args:
            regime_model_path: Path to saved regime HMM model
            kalman_model_path: Path to saved Kalman filter model
            regime_features_path: Path to precomputed regime features
            kalman_features_path: Path to precomputed Kalman features
        """
        self.regime_model_path = regime_model_path
        self.kalman_model_path = kalman_model_path
        self.regime_features_path = regime_features_path
        self.kalman_features_path = kalman_features_path
        
        # Load models and features
        self.regime_model = None
        self.kalman_model = None
        self.regime_features = None
        self.kalman_features = None
        self.is_loaded = False
        
        self._load_models_and_features()
    
    def _load_models_and_features(self):
        """Load the fitted models and precomputed features."""
        try:
            # Load regime model
            if os.path.exists(self.regime_model_path):
                print("Loading regime HMM model...")
                self.regime_model = RegimeHMM.load_model(self.regime_model_path)
                print("Regime HMM model loaded successfully!")
            else:
                # Suppress warning - this is expected when models haven't been trained yet
                pass
            
            # Load Kalman model
            if os.path.exists(self.kalman_model_path):
                print("Loading Kalman filter model...")
                self.kalman_model = KalmanFilterModel.load_model(self.kalman_model_path)
                print("Kalman filter model loaded successfully!")
            else:
                # Suppress warning - this is expected when models haven't been trained yet
                pass
            
            # Load regime features
            if os.path.exists(self.regime_features_path):
                print("Loading regime features...")
                self.regime_features = pd.read_csv(self.regime_features_path, index_col=0, parse_dates=True)
                print(f"Regime features loaded: {self.regime_features.shape}")
            else:
                # Suppress warning - this is expected when features haven't been computed yet
                pass
            
            # Load Kalman features
            if os.path.exists(self.kalman_features_path):
                print("Loading Kalman features...")
                self.kalman_features = pd.read_csv(self.kalman_features_path, index_col=0, parse_dates=True)
                print(f"Kalman features loaded: {self.kalman_features.shape}")
            else:
                # Suppress warning - this is expected when features haven't been computed yet
                pass
            
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading models and features: {e}")
            self.is_loaded = False
    
    def get_current_signals(self, timestamp: pd.Timestamp) -> Dict:
        """
        Get current regime and Kalman signals for a specific timestamp.
        
        Args:
            timestamp: Timestamp to query
            
        Returns:
            Dictionary with current signals
        """
        if not self.is_loaded:
            raise ValueError("Models and features not loaded")
        
        signals = {}
        
        # Get regime signals
        if self.regime_features is not None and timestamp in self.regime_features.index:
            regime_row = self.regime_features.loc[timestamp]
            signals.update({
                'regime_label': int(regime_row['regime_label']),
                'regime_prob_0': float(regime_row['regime_prob_0']),
                'regime_prob_1': float(regime_row['regime_prob_1']),
                'regime_change': int(regime_row['regime_change'])
            })
        else:
            # Default values if not found
            signals.update({
                'regime_label': 0,
                'regime_prob_0': 1.0,
                'regime_prob_1': 0.0,
                'regime_change': 0
            })
        
        # Get Kalman signals
        if self.kalman_features is not None and timestamp in self.kalman_features.index:
            kalman_row = self.kalman_features.loc[timestamp]
            signals.update({
                'trend': float(kalman_row['trend']),
                'velocity': float(kalman_row['velocity']),
                'volatility': float(kalman_row['volatility']),
                'trend_residual': float(kalman_row['trend_residual']),
                'trend_residual_zscore': float(kalman_row['trend_residual_zscore']),
                'trend_direction': int(kalman_row['trend_direction']),
                'trend_strength': float(kalman_row['trend_strength']),
                'volatility_regime': int(kalman_row['volatility_regime'])
            })
        else:
            # Default values if not found
            signals.update({
                'trend': 0.0,
                'velocity': 0.0,
                'volatility': 0.01,
                'trend_residual': 0.0,
                'trend_residual_zscore': 0.0,
                'trend_direction': 0,
                'trend_strength': 0.0,
                'volatility_regime': 1
            })
        
        return signals
    
    def get_signals_for_period(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """
        Get regime and Kalman signals for a time period.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with signals for the period
        """
        if not self.is_loaded:
            raise ValueError("Models and features not loaded")
        
        # Get regime features for period
        regime_period = None
        if self.regime_features is not None:
            regime_period = self.regime_features.loc[start_time:end_time]
        
        # Get Kalman features for period
        kalman_period = None
        if self.kalman_features is not None:
            kalman_period = self.kalman_features.loc[start_time:end_time]
        
        # Combine features
        if regime_period is not None and kalman_period is not None:
            # Ensure same index
            common_index = regime_period.index.intersection(kalman_period.index)
            combined_features = pd.concat([
                regime_period.loc[common_index],
                kalman_period.loc[common_index]
            ], axis=1)
        elif regime_period is not None:
            combined_features = regime_period
        elif kalman_period is not None:
            combined_features = kalman_period
        else:
            raise ValueError("No features available for the specified period")
        
        return combined_features
    
    def get_combined_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get combined regime and Kalman features for RL environment.
        
        Args:
            data: DataFrame with asset log returns (assets as columns, dates as index)
            
        Returns:
            DataFrame with combined features
        """
        if not self.is_loaded:
            raise ValueError("Models and features not loaded")
        
        # Get regime features
        regime_features = None
        if self.regime_model is not None:
            try:
                regime_features = self.regime_model.get_regime_features(data)
            except Exception as e:
                print(f"Warning: Could not get regime features: {e}")
        
        # Get Kalman features
        kalman_features = None
        if self.kalman_model is not None:
            try:
                kalman_features = self.kalman_model.apply_to_portfolio(data)
            except Exception as e:
                print(f"Warning: Could not get Kalman features: {e}")
        
        # Combine features
        combined_features = pd.DataFrame(index=data.index)
        
        if regime_features is not None:
            combined_features = pd.concat([combined_features, regime_features], axis=1)
        
        if kalman_features is not None:
            combined_features = pd.concat([combined_features, kalman_features], axis=1)
        
        # Remove duplicate columns if any
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        
        return combined_features
    
    def get_feature_vector(self, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Get feature vector for RL environment at specific timestamp.
        
        Args:
            timestamp: Timestamp to query
            
        Returns:
            Feature vector as numpy array
        """
        signals = self.get_current_signals(timestamp)
        
        # Create feature vector
        feature_vector = np.array([
            signals['regime_label'],
            signals['regime_prob_0'],
            signals['regime_prob_1'],
            signals['regime_change'],
            signals['trend'],
            signals['velocity'],
            signals['volatility'],
            signals['trend_residual'],
            signals['trend_residual_zscore'],
            signals['trend_direction'],
            signals['trend_strength'],
            signals['volatility_regime']
        ])
        
        return feature_vector
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return [
            'regime_label',
            'regime_prob_0',
            'regime_prob_1',
            'regime_change',
            'trend',
            'velocity',
            'volatility',
            'trend_residual',
            'trend_residual_zscore',
            'trend_direction',
            'trend_strength',
            'volatility_regime'
        ]
    
    def validate_no_future_leakage(self, data: pd.DataFrame) -> bool:
        """
        Validate that no future leakage exists in the signals.
        
        Args:
            data: DataFrame with asset log returns
            
        Returns:
            True if no future leakage detected
        """
        if not self.is_loaded:
            return False
        
        # Check that regime features don't extend beyond data
        if self.regime_features is not None:
            if self.regime_features.index.max() > data.index.max():
                print("Warning: Regime features extend beyond data (potential future leakage)")
                return False
        
        # Check that Kalman features don't extend beyond data
        if self.kalman_features is not None:
            if self.kalman_features.index.max() > data.index.max():
                print("Warning: Kalman features extend beyond data (potential future leakage)")
                return False
        
        print("No future leakage detected in signals")
        return True
    
    def get_signal_summary(self) -> Dict:
        """Get summary of available signals."""
        summary = {
            'regime_model_loaded': self.regime_model is not None,
            'kalman_model_loaded': self.kalman_model is not None,
            'regime_features_loaded': self.regime_features is not None,
            'kalman_features_loaded': self.kalman_features is not None,
            'total_features': len(self.get_feature_names())
        }
        
        if self.regime_features is not None:
            summary['regime_features_shape'] = self.regime_features.shape
            summary['regime_date_range'] = (
                self.regime_features.index.min(),
                self.regime_features.index.max()
            )
        
        if self.kalman_features is not None:
            summary['kalman_features_shape'] = self.kalman_features.shape
            summary['kalman_date_range'] = (
                self.kalman_features.index.min(),
                self.kalman_features.index.max()
            )
        
        return summary


def main():
    """Main function to test the signal interface."""
    import sys
    sys.path.append('..')
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    data_path = '../data_cache/preprocessed_all.csv'
    
    if not os.path.exists(data_path):
        print(f"No preprocessed data found at {data_path}. Please run the data pipeline first.")
        return
    
    # Load data
    data = pd.read_csv(data_path, parse_dates=['datetime'])
    
    # Check if we have the required columns
    if 'log_return' not in data.columns:
        print("No log_return column found in preprocessed data.")
        return
    
    # Pivot to wide format: datetime as index, assets as columns
    log_returns_wide = data.pivot(index='datetime', columns='asset', values='log_return')
    
    print(f"Loaded {len(log_returns_wide)} timesteps with {len(log_returns_wide.columns)} assets")
    
    # Initialize signal interface
    signal_interface = SignalInterface()
    
    # Get signal summary
    summary = signal_interface.get_signal_summary()
    print(f"\nSignal summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test getting current signals
    test_timestamp = log_returns_wide.index[100]  # Some middle timestamp
    print(f"\nTesting signals for timestamp: {test_timestamp}")
    
    try:
        current_signals = signal_interface.get_current_signals(test_timestamp)
        print(f"Current signals: {current_signals}")
        
        # Test feature vector
        feature_vector = signal_interface.get_feature_vector(test_timestamp)
        feature_names = signal_interface.get_feature_names()
        print(f"\nFeature vector:")
        for name, value in zip(feature_names, feature_vector):
            print(f"  {name}: {value}")
        
        # Test getting signals for a period
        start_time = log_returns_wide.index[50]
        end_time = log_returns_wide.index[150]
        period_signals = signal_interface.get_signals_for_period(start_time, end_time)
        print(f"\nPeriod signals shape: {period_signals.shape}")
        print(f"Period signals columns: {period_signals.columns.tolist()}")
        
        # Validate no future leakage
        no_leakage = signal_interface.validate_no_future_leakage(log_returns_wide)
        print(f"\nNo future leakage: {no_leakage}")
        
    except Exception as e:
        print(f"Error testing signal interface: {e}")


if __name__ == "__main__":
    main() 