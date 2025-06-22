"""
Debug specific steps 480-482 where VET shows NaN issues and invalid prices.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def debug_specific_steps():
    print("=== Debugging Steps 480-482 ===\n")
    
    # Create environment
    env = PortfolioEnv()
    
    # Get all dates
    all_dates = sorted(env.data['datetime'].unique())
    print(f"Total dates: {len(all_dates)}")
    
    # Check specific problematic steps
    for step in [480, 481, 482]:
        if step < len(all_dates):
            date = all_dates[step]
            print(f"\n--- Step {step} ({date}) ---")
            
            # Check VET data for this specific date
            vet_data = env.data[(env.data['datetime'] == date) & (env.data['asset'] == 'VET/USDT')]
            print(f"VET data found: {not vet_data.empty}")
            
            if not vet_data.empty:
                print("VET features:")
                for feature in env.features:
                    value = vet_data[feature].iloc[0]
                    print(f"  {feature}: {value} (NaN: {pd.isna(value)})")
                
                # Check if close price is valid
                close_price = vet_data['close'].iloc[0]
                print(f"Close price: {close_price} (valid: {not pd.isna(close_price) and close_price > 0})")
            else:
                print("No VET data found for this date!")
            
            # Check all assets for this date
            all_assets_data = env.data[env.data['datetime'] == date]
            print(f"Total assets for this date: {len(all_assets_data)}")
            
            # Check for any assets with NaN close prices
            nan_close_assets = all_assets_data[all_assets_data['close'].isna()]['asset'].tolist()
            if nan_close_assets:
                print(f"Assets with NaN close prices: {nan_close_assets}")
            
            # Check for any assets with zero or negative close prices
            invalid_price_assets = all_assets_data[(all_assets_data['close'] <= 0) | all_assets_data['close'].isna()]['asset'].tolist()
            if invalid_price_assets:
                print(f"Assets with invalid close prices: {invalid_price_assets}")
    
    # Check the _get_prices method specifically
    print(f"\n=== Testing _get_prices method ===\n")
    
    for step in [480, 481, 482]:
        if step < len(all_dates):
            print(f"\n--- Step {step} ---")
            
            # Set current step
            env.current_step = step
            
            # Get prices using the environment method
            prices = env._get_prices(step)
            print(f"Prices shape: {prices.shape}")
            print(f"Prices: {prices}")
            print(f"Any NaN prices: {np.any(np.isnan(prices))}")
            print(f"Any zero/negative prices: {np.any(prices <= 0)}")
            
            if np.any(np.isnan(prices)) or np.any(prices <= 0):
                print("Invalid prices detected!")
                for i, (asset, price) in enumerate(zip(env.assets, prices)):
                    if np.isnan(price) or price <= 0:
                        print(f"  {asset}: {price}")
    
    # Check if VET is in the assets list
    print(f"\n=== Checking VET in assets list ===\n")
    vet_in_assets = 'VET/USDT' in env.assets
    print(f"VET/USDT in assets list: {vet_in_assets}")
    if vet_in_assets:
        vet_index = env.assets.index('VET/USDT')
        print(f"VET index: {vet_index}")
        
        # Check VET prices across all steps
        print(f"\nVET prices across all steps:")
        for step in range(min(485, len(all_dates))):
            date = all_dates[step]
            vet_data = env.data[(env.data['datetime'] == date) & (env.data['asset'] == 'VET/USDT')]
            if not vet_data.empty:
                price = vet_data['close'].iloc[0]
                if pd.isna(price) or price <= 0:
                    print(f"  Step {step} ({date}): {price} - INVALID!")
                elif step >= 480 and step <= 482:
                    print(f"  Step {step} ({date}): {price} - CHECKING")
            else:
                print(f"  Step {step} ({date}): No data found!")

if __name__ == "__main__":
    debug_specific_steps() 