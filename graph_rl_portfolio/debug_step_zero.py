"""
Debug what's happening at step 0.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def debug_step_zero():
    print("=== Debugging Step 0 ===\n")
    
    # Create environment
    env = PortfolioEnv()
    
    # Check step 0
    step = 0
    date = env.sorted_dates[step]
    print(f"Step 0 date: {date}")
    
    # Check what data exists for this date
    data_at_step0 = env.data[env.data['datetime'] == date]
    print(f"Data at step 0 shape: {data_at_step0.shape}")
    print(f"Assets at step 0: {data_at_step0['asset'].tolist()}")
    
    # Check which assets are missing
    missing_assets = set(env.assets) - set(data_at_step0['asset'].values)
    print(f"Missing assets at step 0: {missing_assets}")
    
    # Check VET specifically
    vet_at_step0 = data_at_step0[data_at_step0['asset'] == 'VET/USDT']
    print(f"VET data at step 0: {len(vet_at_step0)} rows")
    
    # Check when VET data starts
    vet_data = env.data[env.data['asset'] == 'VET/USDT']
    vet_dates = sorted(vet_data['datetime'].unique())
    print(f"VET first date: {vet_dates[0]}")
    print(f"VET last date: {vet_dates[-1]}")
    
    # Find the first date that has all assets
    print(f"\n--- Finding first complete date ---")
    all_assets_set = set(env.assets)
    
    for i, dt in enumerate(env.sorted_dates[:100]):  # Check first 100 dates
        data_at_dt = env.data[env.data['datetime'] == dt]
        assets_at_dt = set(data_at_dt['asset'].values)
        
        if all_assets_set.issubset(assets_at_dt):
            print(f"First complete date found at step {i}: {dt}")
            print(f"Assets at this date: {len(data_at_dt)}")
            break
    else:
        print("No complete date found in first 100 steps")
    
    # Check what the environment should do for incomplete dates
    print(f"\n--- Testing _get_prices at step 0 ---")
    try:
        prices = env._get_prices(0)
        print(f"Prices at step 0: {prices}")
        print(f"Any NaN prices: {np.any(np.isnan(prices))}")
        if np.any(np.isnan(prices)):
            nan_indices = np.where(np.isnan(prices))[0]
            nan_assets = [env.assets[i] for i in nan_indices]
            print(f"Assets with NaN prices: {nan_assets}")
    except Exception as e:
        print(f"Error in _get_prices at step 0: {e}")

if __name__ == "__main__":
    debug_step_zero() 