"""
Debug the environment step by step to understand VET warnings.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from environment.portfolio_env import PortfolioEnv

def debug_environment():
    print("=== Debugging Environment Step by Step ===\n")
    
    # Create environment
    env = PortfolioEnv()
    obs, _ = env.reset()
    
    print(f"Environment initialized with {env.n_assets} assets")
    print(f"Assets: {env.assets[:10]}...")  # Show first 10
    print(f"Features: {env.features}")
    print(f"Signal features: {env.signal_feature_names}")
    
    # Check if VET is in the assets
    vet_index = None
    for i, asset in enumerate(env.assets):
        if 'VET' in asset:
            vet_index = i
            print(f"VET found at index {i}: {asset}")
            break
    
    if vet_index is None:
        print("VET not found in assets list!")
        return
    
    # Run a few steps and monitor VET specifically
    print(f"\n=== Monitoring VET at index {vet_index} ===\n")
    
    for step in range(10):  # Check first 10 steps
        print(f"\n--- Step {step} ---")
        
        # Get current observation
        obs = env._get_obs()
        vet_features = obs['features'][vet_index]
        
        print(f"VET features: {vet_features}")
        print(f"VET has NaN: {np.any(np.isnan(vet_features))}")
        
        if np.any(np.isnan(vet_features)):
            print(f"VET NaN positions: {np.where(np.isnan(vet_features))}")
            print(f"VET feature names with NaN: {[env.features[i] for i in np.where(np.isnan(vet_features))[0]]}")
        
        # Check raw data for this step
        dt = env.data['datetime'].unique()[step]
        vet_raw = env.data[(env.data['datetime'] == dt) & (env.data['asset'] == env.assets[vet_index])]
        
        if not vet_raw.empty:
            print(f"VET raw data at step {step}:")
            for feature in env.features:
                value = vet_raw[feature].iloc[0]
                print(f"  {feature}: {value} (NaN: {pd.isna(value)})")
        else:
            print(f"No VET data found at step {step}")
        
        # Take a random action and step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            print("Episode ended early!")
            break
    
    # Check specific problematic step (478)
    print(f"\n=== Checking Step 478 specifically ===\n")
    
    # Reset and step to 478
    env.reset()
    for step in range(478):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print(f"Episode ended at step {step}")
            break
    
    # Now check step 478
    print(f"Current step: {env.current_step}")
    obs = env._get_obs()
    vet_features = obs['features'][vet_index]
    print(f"VET features at step 478: {vet_features}")
    print(f"VET has NaN: {np.any(np.isnan(vet_features))}")
    
    # Check raw data
    dt = env.data['datetime'].unique()[env.current_step]
    vet_raw = env.data[(env.data['datetime'] == dt) & (env.data['asset'] == env.assets[vet_index])]
    
    if not vet_raw.empty:
        print(f"VET raw data at step 478:")
        for feature in env.features:
            value = vet_raw[feature].iloc[0]
            print(f"  {feature}: {value} (NaN: {pd.isna(value)})")
    
    # Check signal interface
    print(f"\n=== Checking Signal Interface ===\n")
    signal_vec = env.signal_interface.get_feature_vector(pd.Timestamp(dt))
    print(f"Signal vector shape: {signal_vec.shape}")
    print(f"Signal vector has NaN: {np.any(np.isnan(signal_vec))}")
    if np.any(np.isnan(signal_vec)):
        print(f"Signal NaN positions: {np.where(np.isnan(signal_vec))}")
        print(f"Signal feature names with NaN: {[env.signal_feature_names[i] for i in np.where(np.isnan(signal_vec))[0]]}")

if __name__ == "__main__":
    debug_environment() 