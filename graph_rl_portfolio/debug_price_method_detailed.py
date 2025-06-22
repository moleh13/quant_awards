"""
Detailed debug of the _get_prices method.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def debug_price_method_detailed():
    print("=== Detailed Debug of _get_prices Method ===\n")
    
    # Create environment
    env = PortfolioEnv()
    
    # Get all dates
    all_dates = sorted(env.data['datetime'].unique())
    
    # Check step 480 specifically
    step = 480
    date = all_dates[step]
    print(f"--- Step {step} ({date}) ---")
    
    # Check the exact data the environment is using
    print(f"Environment data shape: {env.data.shape}")
    print(f"Environment data columns: {env.data.columns.tolist()}")
    print(f"Environment assets: {env.assets}")
    print(f"Environment n_assets: {env.n_assets}")
    
    # Get the exact row the environment would use
    row = env.data[env.data['datetime'] == date]
    print(f"\nRow from environment data:")
    print(f"Row shape: {row.shape}")
    print(f"Row assets: {row['asset'].tolist()}")
    
    # Check if there are any duplicates
    duplicates = row['asset'].duplicated()
    if duplicates.any():
        print(f"Duplicate assets found: {row[duplicates]['asset'].tolist()}")
    
    # Check VET specifically
    vet_data = row[row['asset'] == 'VET/USDT']
    print(f"\nVET data in row:")
    print(f"VET rows found: {len(vet_data)}")
    if len(vet_data) > 0:
        print(f"VET close price: {vet_data['close'].iloc[0]}")
        print(f"VET close price type: {type(vet_data['close'].iloc[0])}")
        print(f"VET close price is NaN: {pd.isna(vet_data['close'].iloc[0])}")
    
    # Test the exact _get_prices logic step by step
    print(f"\n--- Testing _get_prices logic step by step ---")
    
    # Step 1: Get the row
    step1_row = env.data[env.data['datetime'] == env.data['datetime'].unique()[step]]
    print(f"Step 1 - Row shape: {step1_row.shape}")
    print(f"Step 1 - Row assets: {step1_row['asset'].tolist()}")
    
    # Step 2: Set index
    step2_indexed = step1_row.set_index('asset')
    print(f"Step 2 - Indexed shape: {step2_indexed.shape}")
    print(f"Step 2 - Indexed index: {step2_indexed.index.tolist()}")
    
    # Step 3: Get close column
    step3_close = step2_indexed['close']
    print(f"Step 3 - Close series shape: {step3_close.shape}")
    print(f"Step 3 - Close series index: {step3_close.index.tolist()}")
    print(f"Step 3 - VET close value: {step3_close.get('VET/USDT', 'Not found')}")
    
    # Step 4: Reindex with environment assets
    step4_reindexed = step3_close.reindex(env.assets)
    print(f"Step 4 - Reindexed shape: {step4_reindexed.shape}")
    print(f"Step 4 - Reindexed index: {step4_reindexed.index.tolist()}")
    print(f"Step 4 - Reindexed values: {step4_reindexed.values}")
    print(f"Step 4 - VET reindexed value: {step4_reindexed.get('VET/USDT', 'Not found')}")
    
    # Step 5: Convert to numpy array
    step5_values = step4_reindexed.values
    print(f"Step 5 - Values shape: {step5_values.shape}")
    print(f"Step 5 - Values: {step5_values}")
    print(f"Step 5 - VET value (index 10): {step5_values[10]}")
    
    # Compare with environment method
    print(f"\n--- Environment method result ---")
    env_prices = env._get_prices(step)
    print(f"Environment prices: {env_prices}")
    print(f"Environment VET price (index 10): {env_prices[10]}")
    
    # Check if there's a difference in the data being used
    print(f"\n--- Data consistency check ---")
    print(f"Environment data ID: {id(env.data)}")
    print(f"Environment data memory location: {env.data._data}")
    
    # Check if the environment is using a different data source
    print(f"\n--- Environment data source ---")
    print(f"Environment data path: {os.path.join(os.path.dirname(__file__), 'environment', '../data_cache/preprocessed_all.csv')}")
    
    # Load data directly to compare
    direct_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data_cache/preprocessed_all.csv'), parse_dates=['datetime'])
    print(f"Direct data shape: {direct_data.shape}")
    print(f"Environment data shape: {env.data.shape}")
    print(f"Data shapes match: {direct_data.shape == env.data.shape}")

if __name__ == "__main__":
    debug_price_method_detailed() 