"""
Debug VET warnings in the environment.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.portfolio_env import PortfolioEnv

def debug_vet_data():
    """Debug VET data specifically at the start step"""
    
    # Load data using the same path as the environment
    data_path = 'data_cache/preprocessed_all.csv'
    data = pd.read_csv(data_path, parse_dates=['datetime'])
    
    # Get asset list
    asset_list_path = 'data_cache/top_50_assets.csv'
    asset_list = pd.read_csv(asset_list_path)['asset'].tolist()
    assets = [a for a in asset_list if a in data['asset'].unique()]
    
    print(f"Total assets: {len(assets)}")
    print(f"VET/USDT in assets: {'VET/USDT' in assets}")
    
    # Get VET data
    vet_data = data[data['asset'] == 'VET/USDT'].copy()
    print(f"\nVET data shape: {vet_data.shape}")
    print(f"VET data date range: {vet_data['datetime'].min()} to {vet_data['datetime'].max()}")
    
    # Check for NaNs in VET data
    features = ['close', 'log_return', 'sma_7', 'sma_21', 'ema_7', 'ema_21',
                'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'bb_width']
    
    print(f"\nVET NaN check:")
    for feature in features:
        nan_count = vet_data[feature].isna().sum()
        if nan_count > 0:
            print(f"  {feature}: {nan_count} NaNs")
    
    # Check the start step data
    env = PortfolioEnv()
    start_step = env.start_step
    start_date = env.sorted_dates[start_step]
    
    print(f"\nStart step: {start_step}")
    print(f"Start date: {start_date}")
    
    # Get VET data at start date
    vet_at_start = vet_data[vet_data['datetime'] == start_date]
    print(f"\nVET data at start date:")
    if len(vet_at_start) > 0:
        for feature in features:
            value = vet_at_start[feature].iloc[0]
            print(f"  {feature}: {value}")
    else:
        print("  No VET data at start date!")
    
    # Check what assets have data at start date
    data_at_start = data[data['datetime'] == start_date]
    assets_at_start = data_at_start['asset'].tolist()
    print(f"\nAssets with data at start date: {len(assets_at_start)}")
    print(f"VET/USDT at start: {'VET/USDT' in assets_at_start}")
    
    # Check if VET has any data before start date
    vet_before_start = vet_data[vet_data['datetime'] < start_date]
    print(f"\nVET data points before start date: {len(vet_before_start)}")
    if len(vet_before_start) > 0:
        print(f"First VET data: {vet_before_start['datetime'].min()}")
        print(f"Last VET data before start: {vet_before_start['datetime'].max()}")
    
    # Check the actual observation creation
    print(f"\nTesting observation creation...")
    obs, _ = env.reset()
    
    # Check VET index in assets
    vet_idx = assets.index('VET/USDT')
    print(f"VET index in assets: {vet_idx}")
    
    # Check VET features in observation
    vet_features = obs['features'][vet_idx]
    print(f"VET features in observation:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {vet_features[i]}")

if __name__ == "__main__":
    debug_vet_data() 