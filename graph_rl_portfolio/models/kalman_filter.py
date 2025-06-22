"""
Kalman Filter for Latent Trend and Volatility Estimation

This module implements a Kalman filter to estimate latent trends and volatility
from financial time series data.
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pickle
import os
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class KalmanFilterModel:
    """
    Kalman Filter for estimating latent trends and volatility in financial data.
    
    Uses a state-space model to estimate:
    - Latent trend (smoothed price/return)
    - Volatility (time-varying)
    - Trend velocity (rate of change)
    """
    
    def __init__(self, 
                 n_states: int = 3,
                 random_state: int = 42):
        """
        Initialize the Kalman filter model.
        
        Args:
            n_states: Number of state variables (default: 3 for trend, velocity, volatility)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.is_fitted = False
        
        # State variables: [trend, velocity, volatility]
        self.kf = None
        self.smoothed_states = None
        self.smoothed_covariances = None
        
    def _build_kalman_filter(self, initial_obs: float) -> KalmanFilter:
        """
        Build the Kalman filter with appropriate state transition and observation matrices.
        
        Args:
            initial_obs: Initial observation value
            
        Returns:
            Configured KalmanFilter instance
        """
        # State transition matrix (how states evolve)
        # [trend, velocity, volatility]
        transition_matrices = np.array([
            [1, 1, 0],    # trend_t = trend_{t-1} + velocity_{t-1}
            [0, 0.95, 0], # velocity_t = 0.95 * velocity_{t-1} (mean reversion)
            [0, 0, 0.95]  # volatility_t = 0.95 * volatility_{t-1} (persistence)
        ])
        
        # Observation matrix (what we observe)
        observation_matrices = np.array([[1, 0, 0]])  # observe trend + noise
        
        # Initial state mean
        initial_state_mean = np.array([initial_obs, 0, 0.01])
        
        # Initial state covariance
        initial_state_covariance = np.array([
            [0.1, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.001]
        ])
        
        # State transition noise covariance
        transition_covariance = np.array([
            [0.001, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.0001]
        ])
        
        # Observation noise covariance
        observation_covariance = np.array([[0.01]])
        
        # Create Kalman filter
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )
        
        return kf
    
    def fit(self, data: pd.Series) -> 'KalmanFilterModel':
        """
        Fit the Kalman filter on time series data.
        
        Args:
            data: Time series data (prices or returns)
            
        Returns:
            Self for chaining
        """
        print(f"Fitting Kalman filter on {len(data)} observations...")
        
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            raise ValueError("Insufficient data for Kalman filter fitting")
        
        # Build Kalman filter
        self.kf = self._build_kalman_filter(clean_data.iloc[0])
        
        # Fit the filter
        observations = clean_data.values.reshape(-1, 1)
        
        try:
            # Get smoothed estimates
            self.smoothed_states, self.smoothed_covariances = self.kf.smooth(observations)
            
            self.is_fitted = True
            print("Kalman filter fitted successfully!")
            
            # Print some statistics
            trend_std = np.std(self.smoothed_states[:, 0])
            velocity_std = np.std(self.smoothed_states[:, 1])
            volatility_mean = np.mean(self.smoothed_states[:, 2])
            
            print(f"Trend std: {trend_std:.6f}")
            print(f"Velocity std: {velocity_std:.6f}")
            print(f"Mean volatility: {volatility_mean:.6f}")
            
        except Exception as e:
            print(f"Warning: Kalman filter fitting failed: {e}")
            # Fall back to simple moving average
            return self._simple_trend_estimation(clean_data)
        
        return self
    
    def _simple_trend_estimation(self, data: pd.Series) -> 'KalmanFilterModel':
        """
        Fallback simple trend estimation using moving averages.
        
        Args:
            data: Time series data
            
        Returns:
            Self for chaining
        """
        print("Using simple trend estimation as fallback...")
        
        # Simple moving average approach
        trend = data.rolling(window=30, min_periods=10).mean()
        velocity = data.diff().rolling(window=30, min_periods=10).mean()
        volatility = data.rolling(window=30, min_periods=10).std()
        
        # Create state matrix
        self.smoothed_states = np.column_stack([trend, velocity, volatility])
        self.smoothed_covariances = np.zeros((len(data), self.n_states, self.n_states))
        
        self.is_fitted = True
        print("Simple trend estimation completed!")
        
        return self
    
    def predict(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict latent states for given data.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (smoothed_states, smoothed_covariances)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.kf is None:
            # Use simple estimation
            return self._simple_trend_estimation(data)
        
        # Remove NaN values
        clean_data = data.dropna()
        observations = clean_data.values.reshape(-1, 1)
        
        # Get smoothed estimates
        smoothed_states, smoothed_covariances = self.kf.smooth(observations)
        
        return smoothed_states, smoothed_covariances
    
    def get_kalman_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Get Kalman filter features for each timestep.
        
        Args:
            data: Time series data
            
        Returns:
            DataFrame with Kalman features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting features")
        
        # Get predictions
        smoothed_states, smoothed_covariances = self.predict(data)
        
        # Create features DataFrame
        kalman_features = pd.DataFrame(index=data.index)
        
        # Ensure lengths match
        if len(smoothed_states) != len(data):
            print(f"Warning: Smoothed states length ({len(smoothed_states)}) != data length ({len(data)})")
            # Truncate or pad to match
            if len(smoothed_states) < len(data):
                # Pad with the last state
                last_state = smoothed_states[-1]
                padding = np.tile(last_state, (len(data) - len(smoothed_states), 1))
                smoothed_states = np.vstack([smoothed_states, padding])
            else:
                # Truncate to match data length
                smoothed_states = smoothed_states[:len(data)]
        
        # Add state estimates
        kalman_features['trend'] = smoothed_states[:, 0]
        kalman_features['velocity'] = smoothed_states[:, 1]
        kalman_features['volatility'] = smoothed_states[:, 2]
        
        # Add derived features
        kalman_features['trend_residual'] = data - kalman_features['trend']
        kalman_features['trend_residual_zscore'] = kalman_features['trend_residual'] / (kalman_features['volatility'] + 1e-8)
        
        # Add trend direction (1 if velocity > 0, 0 otherwise)
        kalman_features['trend_direction'] = (kalman_features['velocity'] > 0).astype(int)
        
        # Add trend strength (absolute velocity normalized by volatility)
        kalman_features['trend_strength'] = np.abs(kalman_features['velocity']) / (kalman_features['volatility'] + 1e-8)
        
        # Add volatility regime (high/medium/low)
        vol_quantiles = kalman_features['volatility'].quantile([0.33, 0.67])
        kalman_features['volatility_regime'] = pd.cut(
            kalman_features['volatility'],
            bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
            labels=[0, 1, 2]  # low, medium, high
        ).astype(int)
        
        return kalman_features
    
    def apply_to_portfolio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman filter to portfolio returns and get features.
        
        Args:
            data: DataFrame with asset log returns (assets as columns, dates as index)
            
        Returns:
            DataFrame with portfolio-level Kalman features
        """
        # Compute portfolio returns
        portfolio_returns = data.mean(axis=1)
        
        # Get Kalman features
        kalman_features = self.get_kalman_features(portfolio_returns)
        
        return kalman_features
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'n_states': self.n_states,
            'random_state': self.random_state,
            'kf': self.kf,
            'smoothed_states': self.smoothed_states,
            'smoothed_covariances': self.smoothed_covariances,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'KalmanFilterModel':
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            n_states=model_data['n_states'],
            random_state=model_data['random_state']
        )
        
        # Restore model state
        instance.kf = model_data['kf']
        instance.smoothed_states = model_data['smoothed_states']
        instance.smoothed_covariances = model_data['smoothed_covariances']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


def main():
    """Main function to test the Kalman filter model."""
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
    
    # Initialize Kalman filter
    kalman_model = KalmanFilterModel(
        n_states=3,  # trend, velocity, volatility
        random_state=42
    )
    
    # Fit on portfolio returns
    portfolio_returns = log_returns_wide.mean(axis=1)
    kalman_model.fit(portfolio_returns)
    
    # Get Kalman features
    kalman_features = kalman_model.get_kalman_features(portfolio_returns)
    
    # Save model
    os.makedirs('../data_cache', exist_ok=True)
    kalman_model.save_model('../data_cache/kalman_filter.pkl')
    
    # Save Kalman features
    kalman_features.to_csv('../data_cache/kalman_features.csv')
    
    print(f"\nKalman features shape: {kalman_features.shape}")
    print(f"Kalman features columns: {kalman_features.columns.tolist()}")
    
    # Show sample of Kalman features
    print(f"\nSample Kalman features:")
    print(kalman_features.head(10))
    
    # Show some statistics
    print(f"\nKalman features statistics:")
    print(kalman_features.describe())


if __name__ == "__main__":
    main() 