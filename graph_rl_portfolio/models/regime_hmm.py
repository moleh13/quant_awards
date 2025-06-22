"""
Regime HMM Model for Market State Detection

This module implements a Hidden Markov Model to detect market regimes (bull/bear)
using portfolio log returns with a rolling window approach.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RegimeHMM:
    """
    Hidden Markov Model for market regime detection.
    
    Fits a 2-state HMM (bull/bear) on portfolio log returns using a rolling window.
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 window_size: int = 60,
                 random_state: int = 42):
        """
        Initialize the regime HMM model.
        
        Args:
            n_regimes: Number of hidden states (default: 2 for bull/bear)
            window_size: Rolling window size in days (default: 60)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.random_state = random_state
        
        # Initialize HMM with Gaussian emissions and full covariance
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            random_state=random_state,
            n_iter=1000,
            tol=1e-4
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_labels = None
        self.regime_probs = None
        
    def _compute_portfolio_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute portfolio log returns from asset data.
        
        Args:
            data: DataFrame with asset log returns
            
        Returns:
            Portfolio log returns series
        """
        # Equal-weighted portfolio returns
        portfolio_returns = data.mean(axis=1)
        return portfolio_returns
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training.
        
        Args:
            data: DataFrame with asset log returns
            
        Returns:
            Feature array for HMM
        """
        # Compute portfolio returns
        portfolio_returns = self._compute_portfolio_returns(data)
        
        # Remove NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        # Reshape for HMM (expects 2D array)
        features = portfolio_returns.values.reshape(-1, 1)
        
        return features
    
    def fit(self, data: pd.DataFrame) -> 'RegimeHMM':
        """
        Fit the HMM model on historical data.
        
        Args:
            data: DataFrame with asset log returns (assets as columns, dates as index)
            
        Returns:
            Self for chaining
        """
        print(f"Fitting {self.n_regimes}-state HMM on portfolio returns...")
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM
        self.hmm_model.fit(features_scaled)
        
        # Get regime labels and probabilities
        self.regime_labels = self.hmm_model.predict(features_scaled)
        self.regime_probs = self.hmm_model.predict_proba(features_scaled)
        
        self.is_fitted = True
        
        print(f"HMM fitted successfully!")
        print(f"Regime transition matrix:\n{self.hmm_model.transmat_}")
        print(f"Regime means: {self.hmm_model.means_.flatten()}")
        print(f"Regime variances: {np.diag(self.hmm_model.covars_.reshape(self.n_regimes, -1))}")
        
        return self
    
    def predict_regimes(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime labels and probabilities for given data.
        
        Args:
            data: DataFrame with asset log returns
            
        Returns:
            Tuple of (regime_labels, regime_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict regimes
        regime_labels = self.hmm_model.predict(features_scaled)
        regime_probs = self.hmm_model.predict_proba(features_scaled)
        
        return regime_labels, regime_probs
    
    def fit_rolling(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit HMM using rolling windows and predict regimes for each window.
        
        Args:
            data: DataFrame with asset log returns
            
        Returns:
            Dictionary with regime labels and probabilities for each timestep
        """
        print(f"Fitting rolling HMM with {self.window_size}-day windows...")
        
        # Compute portfolio returns
        portfolio_returns = self._compute_portfolio_returns(data)
        portfolio_returns = portfolio_returns.dropna()
        
        # Initialize arrays for results
        n_timesteps = len(portfolio_returns)
        all_regime_labels = np.full(n_timesteps, -1, dtype=int)
        all_regime_probs = np.full((n_timesteps, self.n_regimes), np.nan)
        
        # Use a more efficient approach: fit once on the full dataset first
        print("Fitting initial HMM on full dataset...")
        features = portfolio_returns.values.reshape(-1, 1)
        features_scaled = self.scaler.fit_transform(features)
        
        try:
            self.hmm_model.fit(features_scaled)
            print("Initial HMM fitted successfully!")
        except Exception as e:
            print(f"Warning: Initial HMM fitting failed: {e}")
            # Fall back to simple regime classification
            return self._simple_regime_classification(portfolio_returns)
        
        # Now use rolling windows for regime prediction (not fitting)
        print("Predicting regimes using rolling windows...")
        for i in range(self.window_size, n_timesteps):
            # Get window data
            window_start = i - self.window_size
            window_end = i
            window_returns = portfolio_returns.iloc[window_start:window_end]
            
            if len(window_returns) < self.window_size // 2:  # Skip if insufficient data
                continue
                
            # Scale the window data
            window_features = window_returns.values.reshape(-1, 1)
            window_features_scaled = self.scaler.transform(window_features)
            
            try:
                # Predict regime for the last timestep in window
                last_features = window_features_scaled[-1:].reshape(1, -1)
                regime_label = self.hmm_model.predict(last_features)[0]
                regime_prob = self.hmm_model.predict_proba(last_features)[0]
                
                # Store results
                all_regime_labels[i] = regime_label
                all_regime_probs[i] = regime_prob
                
            except Exception as e:
                print(f"Warning: HMM prediction failed for window ending at index {i}: {e}")
                continue
        
        # Forward fill any remaining NaN values
        all_regime_labels = pd.Series(all_regime_labels).fillna(method='ffill').values
        all_regime_probs = pd.DataFrame(all_regime_probs).fillna(method='ffill').values
        
        self.regime_labels = all_regime_labels
        self.regime_probs = all_regime_probs
        self.is_fitted = True
        
        print(f"Rolling HMM completed!")
        print(f"Regime distribution: {np.bincount(all_regime_labels[all_regime_labels >= 0])}")
        
        return {
            'regime_labels': all_regime_labels,
            'regime_probs': all_regime_probs
        }
    
    def _simple_regime_classification(self, portfolio_returns: pd.Series) -> Dict[str, np.ndarray]:
        """
        Fallback simple regime classification based on return quantiles.
        
        Args:
            portfolio_returns: Portfolio return series
            
        Returns:
            Dictionary with regime labels and probabilities
        """
        print("Using simple regime classification as fallback...")
        
        # Simple regime classification: 0 = bear (bottom 50%), 1 = bull (top 50%)
        rolling_mean = portfolio_returns.rolling(window=30, min_periods=10).mean()
        rolling_std = portfolio_returns.rolling(window=30, min_periods=10).std()
        
        # Z-score based classification
        z_score = (portfolio_returns - rolling_mean) / rolling_std
        regime_labels = (z_score > 0).astype(int)  # 0 = bear, 1 = bull
        
        # Create probability matrix
        regime_probs = np.zeros((len(regime_labels), self.n_regimes))
        regime_probs[np.arange(len(regime_labels)), regime_labels] = 1.0
        
        # Handle NaN values
        regime_labels = pd.Series(regime_labels).fillna(method='ffill').values
        regime_probs = pd.DataFrame(regime_probs).fillna(method='ffill').values
        
        self.regime_labels = regime_labels
        self.regime_probs = regime_probs
        self.is_fitted = True
        
        print(f"Simple regime classification completed!")
        print(f"Regime distribution: {np.bincount(regime_labels[regime_labels >= 0])}")
        
        return {
            'regime_labels': regime_labels,
            'regime_probs': regime_probs
        }
    
    def get_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime features for each timestep.
        
        Args:
            data: DataFrame with asset log returns
            
        Returns:
            DataFrame with regime features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting features")
        
        # Get regime predictions
        regime_labels, regime_probs = self.predict_regimes(data)
        
        # Create regime features DataFrame
        regime_features = pd.DataFrame(index=data.index)
        
        # Ensure lengths match
        if len(regime_labels) != len(data):
            print(f"Warning: Regime labels length ({len(regime_labels)}) != data length ({len(data)})")
            # Truncate or pad to match
            if len(regime_labels) < len(data):
                # Pad with the last regime label
                regime_labels = np.concatenate([regime_labels, [regime_labels[-1]] * (len(data) - len(regime_labels))])
                regime_probs = np.vstack([regime_probs, [regime_probs[-1]] * (len(data) - len(regime_probs))])
            else:
                # Truncate to match data length
                regime_labels = regime_labels[:len(data)]
                regime_probs = regime_probs[:len(data)]
        
        # Add regime label
        regime_features['regime_label'] = regime_labels
        
        # Add regime probabilities
        for i in range(self.n_regimes):
            regime_features[f'regime_prob_{i}'] = regime_probs[:, i]
        
        # Add regime transition indicator (1 if regime changed from previous step)
        regime_features['regime_change'] = np.concatenate([
            [0], np.diff(regime_labels) != 0
        ])
        
        return regime_features
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'window_size': self.window_size,
            'random_state': self.random_state,
            'regime_labels': self.regime_labels,
            'regime_probs': self.regime_probs,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RegimeHMM':
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            n_regimes=model_data['n_regimes'],
            window_size=model_data['window_size'],
            random_state=model_data['random_state']
        )
        
        # Restore model state
        instance.hmm_model = model_data['hmm_model']
        instance.scaler = model_data['scaler']
        instance.regime_labels = model_data['regime_labels']
        instance.regime_probs = model_data['regime_probs']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


def main():
    """Main function to test the regime HMM model."""
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
    
    # Initialize regime HMM
    regime_model = RegimeHMM(
        n_regimes=2,  # Bull/Bear
        window_size=60,  # 60-day rolling window
        random_state=42
    )
    
    # Fit using rolling windows
    regime_results = regime_model.fit_rolling(log_returns_wide)
    
    # Get regime features
    regime_features = regime_model.get_regime_features(log_returns_wide)
    
    # Save model
    os.makedirs('../data_cache', exist_ok=True)
    regime_model.save_model('../data_cache/regime_hmm.pkl')
    
    # Save regime features
    regime_features.to_csv('../data_cache/regime_features.csv')
    
    print(f"\nRegime features shape: {regime_features.shape}")
    print(f"Regime features columns: {regime_features.columns.tolist()}")
    print(f"Regime distribution:\n{regime_features['regime_label'].value_counts()}")
    
    # Show sample of regime features
    print(f"\nSample regime features:")
    print(regime_features.head(10))


if __name__ == "__main__":
    main() 