import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import os
from models.signal_interface import SignalInterface

class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio environment with shorting and leverage.
    - Initial cash: 1,000,000 USDT
    - Shorting supported
    - Transaction cost: 0.05% per trade
    - Max leverage: 5x (sum of abs(weights) <= 5)
    - Action: portfolio weights (can be negative for short)
    - Observation: features per asset + cash + regime/Kalman signals
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path=None, asset_list_path=None):
        super().__init__()
        # Load data
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '../data_cache/preprocessed_all.csv')
        self.data = pd.read_csv(data_path, parse_dates=['datetime'])
        # Load filtered asset list
        if asset_list_path is None:
            asset_list_path = os.path.join(os.path.dirname(__file__), '../data_cache/top_50_assets.csv')
        asset_list = pd.read_csv(asset_list_path)['asset'].tolist()
        self.assets = [a for a in asset_list if a in self.data['asset'].unique()]
        self.n_assets = len(self.assets)
        self.features = [
            'close', 'log_return', 'sma_7', 'sma_21', 'ema_7', 'ema_21',
            'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'bb_width'
        ]
        self.n_features = len(self.features)
        # --- SignalInterface integration ---
        self.signal_interface = SignalInterface()
        self.signal_feature_names = self.signal_interface.get_feature_names()
        self.n_signal_features = len(self.signal_feature_names)
        # Build observation space: (n_assets, n_features) + cash + signal features
        obs_low = np.full((self.n_assets, self.n_features), -np.inf)
        obs_high = np.full((self.n_assets, self.n_features), np.inf)
        signal_low = np.full((self.n_signal_features,), -np.inf)
        signal_high = np.full((self.n_signal_features,), np.inf)
        self.observation_space = spaces.Dict({
            'features': spaces.Box(obs_low, obs_high, dtype=np.float32),
            'cash': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'positions': spaces.Box(low=-5, high=5, shape=(self.n_assets,), dtype=np.float32),
            'signals': spaces.Box(signal_low, signal_high, dtype=np.float32),
        })
        # Action space: weights per asset, can be negative, sum(abs(weights)) <= 5
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n_assets,), dtype=np.float32)
        # Constants
        self.initial_cash = 1_000_000
        self.transaction_cost = 0.0005  # 0.05%
        self.max_leverage = 5.0
        # Track warned assets to avoid repetitive warnings
        self.warned_assets = set()
        # Store chronologically sorted dates to ensure consistent indexing
        self.sorted_dates = sorted(self.data['datetime'].unique())
        # Find the first date when all assets have data
        self.start_step = self._find_first_complete_date()
        # Internal state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_step
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_assets)  # Now represents number of shares
        self.asset_prices = self._get_prices(self.current_step)
        self.history = []
        return self._get_obs(), {}

    def _get_prices(self, step):
        row = self.data[self.data['datetime'] == self.sorted_dates[step]]
        return row.set_index('asset')['close'].reindex(self.assets).values

    def _get_obs(self):
        dt = self.sorted_dates[self.current_step]
        obs_df = self.data[self.data['datetime'] == dt].set_index('asset').reindex(self.assets)
        features = obs_df[self.features].values.astype(np.float32)
        
        # For any asset with NaNs in features, fill with zeros and warn (only once per asset)
        nan_rows = np.isnan(features).any(axis=1)
        if np.any(nan_rows):
            dropped = np.array(self.assets)[nan_rows]
            # Only warn about assets we haven't warned about before
            new_warnings = [asset for asset in dropped if asset not in self.warned_assets]
            if new_warnings:
                # Check if this is just the expected first log_return NaN
                is_expected_nan = True
                for asset in new_warnings:
                    asset_idx = self.assets.index(asset)
                    asset_features = features[asset_idx]
                    # Only the log_return should be NaN at the first step
                    log_return_idx = self.features.index('log_return')
                    other_nans = np.isnan(asset_features)
                    other_nans[log_return_idx] = False  # Ignore log_return NaN
                    if np.any(other_nans):
                        is_expected_nan = False
                        break
                
                if not is_expected_nan:
                    print(f"Warning: Assets with unexpected NaNs detected (will be filled with zeros): {new_warnings}")
                self.warned_assets.update(new_warnings)
            features[nan_rows] = 0.0
        
        # Normalize and clip features
        features = np.clip(features, -100, 100)
        
        # --- Add regime/Kalman signals ---
        signal_vec = self.signal_interface.get_feature_vector(pd.Timestamp(dt)).astype(np.float32)
        if np.any(np.isnan(signal_vec)):
            print(f"Warning: NaN values found in signals at step {self.current_step}, replacing with 0")
            signal_vec = np.nan_to_num(signal_vec, nan=0.0)
        signal_vec = np.clip(signal_vec, -100, 100)
        
        # --- Calculate current portfolio weights for the observation ---
        current_prices = self._get_prices(self.current_step)
        valid_prices = ~np.isnan(current_prices) & (current_prices > 0)
        
        asset_values = np.zeros_like(self.positions)
        asset_values[valid_prices] = self.positions[valid_prices] * current_prices[valid_prices]
        
        portfolio_value = self.cash + np.sum(asset_values)
        
        if portfolio_value > 0:
            current_weights = asset_values / portfolio_value
        else:
            current_weights = np.zeros_like(self.positions)
        
        # Final safety check - ensure no NaNs anywhere
        features = np.nan_to_num(features, nan=0.0)
        current_weights = np.nan_to_num(current_weights, nan=0.0)
        signal_vec = np.nan_to_num(signal_vec, nan=0.0)
        
        obs = {
            'features': features,
            'cash': np.array([self.cash / self.initial_cash], dtype=np.float32), # Normalized cash
            'positions': current_weights.astype(np.float32),
            'signals': signal_vec
        }
        
        # Final validation
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and np.any(np.isnan(value)):
                print(f"ERROR: NaN found in {key} at step {self.current_step}, replacing with zeros")
                obs[key] = np.nan_to_num(value, nan=0.0)
        
        return obs

    def step(self, action):
        # 1. Get portfolio value BEFORE rebalancing
        current_prices = self._get_prices(self.current_step)
        if np.any(np.isnan(current_prices)) or np.any(current_prices <= 0):
            # If prices are invalid, we can't trade. End episode.
            print(f"Warning: Invalid prices at step {self.current_step}. Ending episode.")
            return self._get_obs(), 0.0, True, False, {'portfolio_value': self.cash}

        portfolio_value_before = self.cash + np.sum(self.positions * current_prices)
        if portfolio_value_before <= 0:
            return self._get_obs(), 0, True, False, {'portfolio_value': 0}

        # 2. Normalize action to get target weights
        target_weights = np.clip(action, -self.max_leverage, self.max_leverage)
        if np.sum(np.abs(target_weights)) > self.max_leverage:
            # Normalize to preserve relative weights while respecting leverage
            target_weights *= self.max_leverage / np.sum(np.abs(target_weights))
        
        # 3. Calculate target asset holdings in shares
        target_asset_values = portfolio_value_before * target_weights
        
        # Avoid division by zero for assets with invalid prices
        target_positions = np.zeros_like(self.positions)
        valid_prices_mask = current_prices > 0
        target_positions[valid_prices_mask] = target_asset_values[valid_prices_mask] / current_prices[valid_prices_mask]

        # 4. Calculate trades (in shares) and transaction costs
        trades = target_positions - self.positions
        trade_value = np.sum(np.abs(trades) * current_prices)
        transaction_cost = trade_value * self.transaction_cost

        # 5. Update cash and positions (shares)
        self.cash -= np.sum(trades * current_prices)  # Cash flow from trades
        self.cash -= transaction_cost
        self.positions = target_positions.copy()

        # 6. Move to next step and get new portfolio value
        self.current_step += 1
        done = self.current_step >= len(self.sorted_dates) - 1
        
        # Use new prices to value the portfolio
        new_prices = self._get_prices(self.current_step)
        if np.any(np.isnan(new_prices)) or np.any(new_prices <= 0):
            # If new prices are invalid, assume no change from previous valid prices for valuation
            print(f"Warning: Invalid new prices at step {self.current_step}, using previous prices for valuation.")
            new_prices = current_prices

        portfolio_value_after = self.cash + np.sum(self.positions * new_prices)

        # 7. Calculate reward
        reward = np.log(portfolio_value_after / portfolio_value_before) if portfolio_value_before > 0 else 0.0
        if np.isnan(reward): reward = 0.0

        # 8. Get next observation and info
        obs = self._get_obs()
        info = {
            'portfolio_value': portfolio_value_after,
            'cash': self.cash,
            'positions': self.positions.copy(),  # In shares
            'step': self.current_step
        }
        
        return obs, reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.cash + np.sum(self.positions * self._get_prices(self.current_step)):.2f}")

    def _find_first_complete_date(self):
        """Find the first date when all assets have valid prices."""
        all_assets_set = set(self.assets)
        
        for step in range(len(self.sorted_dates)):
            date = self.sorted_dates[step]
            data_at_date = self.data[self.data['datetime'] == date]
            assets_at_date = set(data_at_date['asset'].values)
            
            # Check if all required assets are present
            if all_assets_set.issubset(assets_at_date):
                # Double-check that all prices are valid
                prices = self._get_prices(step)
                if np.all(prices > 0) and not np.any(np.isnan(prices)):
                    print(f"Starting environment from step {step} ({date}) where all assets have data")
                    return step
        
        # If no complete date found, return the last step
        print(f"Warning: No complete date found, starting from last step")
        return len(self.sorted_dates) - 1

if __name__ == "__main__":
    env = PortfolioEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print(f"Test run complete. Total reward: {total_reward:.4f}") 