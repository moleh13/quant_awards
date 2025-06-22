"""
Test that the environment fix resolves the VET warnings and invalid price issues.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def test_environment_fix():
    print("=== Testing Environment Fix ===\n")
    
    # Create environment
    env = PortfolioEnv()
    obs, _ = env.reset()
    
    print(f"Environment initialized with {env.n_assets} assets")
    print(f"Assets: {env.assets}")
    
    # Run a few steps to check for warnings
    print(f"\n--- Running environment steps ---")
    
    for step in range(5):
        print(f"\nStep {step}:")
        
        # Get current observation
        obs = env._get_obs()
        
        # Check for any NaN values in features
        features = obs['features']
        nan_count = np.isnan(features).sum()
        print(f"  NaN count in features: {nan_count}")
        
        # Check VET specifically
        vet_index = env.assets.index('VET/USDT')
        vet_features = features[vet_index]
        vet_has_nan = np.any(np.isnan(vet_features))
        print(f"  VET has NaN: {vet_has_nan}")
        
        # Get prices
        prices = env._get_prices(env.current_step)
        vet_price = prices[vet_index]
        print(f"  VET price: {vet_price}")
        
        # Take a random action and step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  Reward: {reward:.6f}")
        print(f"  Done: {done}")
        
        if done:
            print("Episode ended early!")
            break
    
    # Test specific problematic steps (480-482)
    print(f"\n--- Testing problematic steps 480-482 ---")
    
    # Reset and step to 480
    env.reset()
    for step in range(480):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print(f"Episode ended at step {step}")
            break
    
    # Now test steps 480-482
    for step in range(480, 483):
        print(f"\nStep {step}:")
        
        # Get prices
        prices = env._get_prices(env.current_step)
        vet_index = env.assets.index('VET/USDT')
        vet_price = prices[vet_index]
        print(f"  VET price: {vet_price}")
        print(f"  VET price is NaN: {np.isnan(vet_price)}")
        print(f"  VET price is valid: {not np.isnan(vet_price) and vet_price > 0}")
        
        # Check if any prices are invalid
        invalid_prices = np.isnan(prices) | (prices <= 0)
        invalid_count = invalid_prices.sum()
        print(f"  Invalid prices count: {invalid_count}")
        
        if invalid_count > 0:
            invalid_assets = np.array(env.assets)[invalid_prices]
            print(f"  Invalid assets: {invalid_assets}")
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  Reward: {reward:.6f}")
        print(f"  Done: {done}")
        
        if done:
            print("Episode ended!")
            break
    
    print(f"\n=== Test Complete ===")
    print("If no 'Invalid prices' or 'VET warnings' appeared above, the fix is working!")

if __name__ == "__main__":
    test_environment_fix() 