"""
Debug the _get_prices method to understand why VET gets NaN values.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def debug_price_method():
    print("=== Debugging _get_prices Method ===\n")
    
    # Create environment
    env = PortfolioEnv()
    
    # Get all dates
    all_dates = sorted(env.data['datetime'].unique())
    
    # Check specific problematic steps
    for step in [480, 481, 482]:
        if step < len(all_dates):
            date = all_dates[step]
            print(f"\n--- Step {step} ({date}) ---")
            
            # Get the row for this step
            row = env.data[env.data['datetime'] == date]
            print(f"Row shape: {row.shape}")
            print(f"Assets in row: {row['asset'].tolist()}")
            
            # Check if VET is in the row
            vet_in_row = 'VET/USDT' in row['asset'].values
            print(f"VET in row: {vet_in_row}")
            
            if vet_in_row:
                vet_price = row[row['asset'] == 'VET/USDT']['close'].iloc[0]
                print(f"VET price in row: {vet_price}")
            
            # Set index and check
            row_indexed = row.set_index('asset')
            print(f"Indexed row assets: {row_indexed.index.tolist()}")
            
            # Check if VET is in indexed row
            vet_in_indexed = 'VET/USDT' in row_indexed.index
            print(f"VET in indexed row: {vet_in_indexed}")
            
            if vet_in_indexed:
                vet_price_indexed = row_indexed.loc['VET/USDT', 'close']
                print(f"VET price in indexed row: {vet_price_indexed}")
            
            # Check environment assets list
            print(f"Environment assets: {env.assets}")
            print(f"VET in env assets: {'VET/USDT' in env.assets}")
            if 'VET/USDT' in env.assets:
                vet_index = env.assets.index('VET/USDT')
                print(f"VET index in env assets: {vet_index}")
            
            # Try reindexing
            try:
                reindexed = row_indexed['close'].reindex(env.assets)
                print(f"Reindexed shape: {reindexed.shape}")
                print(f"Reindexed values: {reindexed.values}")
                print(f"VET reindexed value: {reindexed.loc['VET/USDT'] if 'VET/USDT' in reindexed.index else 'Not found'}")
                print(f"Any NaN in reindexed: {reindexed.isna().any()}")
                if reindexed.isna().any():
                    nan_assets = reindexed[reindexed.isna()].index.tolist()
                    print(f"Assets with NaN after reindex: {nan_assets}")
            except Exception as e:
                print(f"Error in reindexing: {e}")
            
            # Check the actual _get_prices method
            print(f"\n--- Testing _get_prices method ---")
            try:
                prices = env._get_prices(step)
                print(f"Prices from _get_prices: {prices}")
                print(f"VET price from _get_prices: {prices[vet_index] if 'VET/USDT' in env.assets else 'N/A'}")
            except Exception as e:
                print(f"Error in _get_prices: {e}")

if __name__ == "__main__":
    debug_price_method() 