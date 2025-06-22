"""
Test that ALL assets are working correctly, not just VET.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.portfolio_env import PortfolioEnv

def test_all_assets():
    """Test environment with assets starting from 2019-01-01"""
    
    print("Testing environment with assets from 2019-01-01...")
    
    # Create environment with 2019 asset list
    env = PortfolioEnv(asset_list_path='data_cache/assets_2019.csv')
    
    print(f"Environment loaded with {env.n_assets} assets")
    print(f"Start step: {env.start_step}")
    print(f"Start date: {env.sorted_dates[env.start_step]}")
    
    # Test a few steps to make sure it works
    print("\nTesting environment functionality...")
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs['features'].shape}")
    print(f"Features min: {np.min(obs['features'])}, max: {np.max(obs['features'])}")
    print(f"Features has NaN: {np.any(np.isnan(obs['features']))}")
    
    # Test a few steps
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i}: reward={reward:.6f}, portfolio_value={info['portfolio_value']:.2f}")
        if done:
            break
    
    print(f"\nTotal reward over 5 steps: {total_reward:.6f}")
    print(f"Environment working successfully with {env.n_assets} assets!")

if __name__ == "__main__":
    test_all_assets() 