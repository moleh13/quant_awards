"""
Minimal test environment to isolate the NaN issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

class MinimalPortfolioEnv(gym.Env):
    """Minimal portfolio environment for testing."""
    
    def __init__(self):
        super().__init__()
        self.n_assets = 5  # Just 5 assets
        self.n_features = 3  # Just 3 features
        
        # Simple observation space
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-1, high=1, shape=(self.n_assets, self.n_features), dtype=np.float32),
            'cash': spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32),
            'positions': spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32),
        })
        
        # Simple action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Generate simple random features
        features = np.random.uniform(-1, 1, (self.n_assets, self.n_features)).astype(np.float32)
        cash = np.array([1.0], dtype=np.float32)
        positions = np.zeros(self.n_assets, dtype=np.float32)
        
        return {
            'features': features,
            'cash': cash,
            'positions': positions,
        }
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Simple reward
        reward = np.sum(action) * 0.01
        
        obs = self._get_obs()
        info = {'step': self.current_step}
        
        return obs, reward, done, False, info

def test_minimal():
    print("Testing minimal environment...")
    
    # Create minimal environment
    env = MinimalPortfolioEnv()
    
    # Test environment
    obs, _ = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape: {value.shape}, dtype: {value.dtype}")
            print(f"{key} min: {np.min(value)}, max: {np.max(value)}")
            print(f"{key} has NaN: {np.any(np.isnan(value))}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, done={done}")
        if done:
            break
    
    # Try training
    print("Starting minimal training...")
    policy_kwargs = dict(
        net_arch=[16],  # Very small network
        activation_fn=torch.nn.Tanh,
    )
    
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=1e-4,
        n_steps=32,
        batch_size=8,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    
    try:
        model.learn(total_timesteps=100)
        print("Minimal training completed successfully!")
    except Exception as e:
        print(f"Minimal training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal() 