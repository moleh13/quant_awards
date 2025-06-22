"""
Debug what's happening at the start step (342) where VET prices are showing as NaN.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def debug_start_step():
    print("=== Debugging Start Step (342) ===\n")
    
    # Create environment
    env = PortfolioEnv()
    
    # Check the start step
    start_step = env.start_step
    start_date = env.sorted_dates[start_step]
    print(f"Start step: {start_step}")
    print(f"Start date: {start_date}")
    
    # Check data at start step
    data_at_start = env.data[env.data['datetime'] == start_date]
    print(f"Data at start step shape: {data_at_start.shape}")
    print(f"Assets at start step: {data_at_start['asset'].tolist()}")
    
    # Check VET specifically
    vet_at_start = data_at_start[data_at_start['asset'] == 'VET/USDT']
    print(f"VET data at start step: {len(vet_at_start)} rows")
    if len(vet_at_start) > 0:
        print(f"VET close price: {vet_at_start['close'].iloc[0]}")
    
    # Check if all assets are present
    missing_assets = set(env.assets) - set(data_at_start['asset'].values)
    print(f"Missing assets at start step: {missing_assets}")
    
    # Test _get_prices at start step
    print(f"\n--- Testing _get_prices at start step ---")
    prices = env._get_prices(start_step)
    print(f"Prices at start step: {prices}")
    print(f"Any NaN prices: {np.any(np.isnan(prices))}")
    if np.any(np.isnan(prices)):
        nan_indices = np.where(np.isnan(prices))[0]
        nan_assets = [env.assets[i] for i in nan_indices]
        print(f"Assets with NaN prices: {nan_assets}")
    
    # Check VET price specifically
    vet_index = env.assets.index('VET/USDT')
    vet_price = prices[vet_index]
    print(f"VET price at start step: {vet_price}")
    
    # Check a few steps after start
    print(f"\n--- Checking steps after start ---")
    for step in range(start_step, start_step + 5):
        prices = env._get_prices(step)
        vet_price = prices[vet_index]
        date = env.sorted_dates[step]
        print(f"Step {step} ({date}): VET price = {vet_price}")
    
    # Check if there's an issue with the reindexing
    print(f"\n--- Testing reindexing at start step ---")
    row = env.data[env.data['datetime'] == start_date]
    row_indexed = row.set_index('asset')
    reindexed = row_indexed['close'].reindex(env.assets)
    print(f"Reindexed values: {reindexed.values}")
    print(f"VET reindexed value: {reindexed.get('VET/USDT', 'Not found')}")

if __name__ == "__main__":
    debug_start_step() 