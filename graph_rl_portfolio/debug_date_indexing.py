"""
Debug the date indexing issue that's causing VET to be missing.
"""

import numpy as np
import pandas as pd
import os
from environment.portfolio_env import PortfolioEnv

def debug_date_indexing():
    print("=== Debugging Date Indexing Issue ===\n")
    
    # Create environment
    env = PortfolioEnv()
    
    # Get all dates
    all_dates = sorted(env.data['datetime'].unique())
    print(f"Total dates: {len(all_dates)}")
    
    # Check step 480 specifically
    step = 480
    if step < len(all_dates):
        date_from_unique = all_dates[step]
        date_from_env = env.data['datetime'].unique()[step]
        
        print(f"Step {step}:")
        print(f"Date from sorted unique: {date_from_unique}")
        print(f"Date from env.data['datetime'].unique()[step]: {date_from_env}")
        print(f"Dates match: {date_from_unique == date_from_env}")
        
        # Check what data exists for each date
        print(f"\nData for date_from_unique ({date_from_unique}):")
        data_unique = env.data[env.data['datetime'] == date_from_unique]
        print(f"Shape: {data_unique.shape}")
        print(f"Assets: {data_unique['asset'].tolist()}")
        print(f"VET in data: {'VET/USDT' in data_unique['asset'].values}")
        
        print(f"\nData for date_from_env ({date_from_env}):")
        data_env = env.data[env.data['datetime'] == date_from_env]
        print(f"Shape: {data_env.shape}")
        print(f"Assets: {data_env['asset'].tolist()}")
        print(f"VET in data: {'VET/USDT' in data_env['asset'].values}")
        
        # Check the unique dates in the environment data
        print(f"\nEnvironment unique dates (first 10):")
        env_unique_dates = env.data['datetime'].unique()
        for i, dt in enumerate(env_unique_dates[:10]):
            print(f"  {i}: {dt}")
        
        print(f"\nSorted unique dates (first 10):")
        for i, dt in enumerate(all_dates[:10]):
            print(f"  {i}: {dt}")
        
        # Check if there are any missing dates for VET
        print(f"\nChecking VET data availability:")
        vet_data = env.data[env.data['asset'] == 'VET/USDT']
        vet_dates = sorted(vet_data['datetime'].unique())
        print(f"VET data dates: {len(vet_dates)}")
        print(f"VET first date: {vet_dates[0]}")
        print(f"VET last date: {vet_dates[-1]}")
        
        # Check if the problematic date exists for VET
        print(f"\nChecking if {date_from_env} exists for VET:")
        vet_at_date = vet_data[vet_data['datetime'] == date_from_env]
        print(f"VET data at {date_from_env}: {len(vet_at_date)} rows")
        
        # Check what dates are missing VET data
        print(f"\nChecking for dates missing VET data:")
        all_dates_set = set(all_dates)
        vet_dates_set = set(vet_dates)
        missing_vet_dates = all_dates_set - vet_dates_set
        print(f"Dates missing VET data: {len(missing_vet_dates)}")
        if len(missing_vet_dates) > 0:
            print(f"First few missing dates: {sorted(missing_vet_dates)[:5]}")
        
        # Check if the problematic date is in the missing dates
        if date_from_env in missing_vet_dates:
            print(f"PROBLEM FOUND: {date_from_env} is missing VET data!")
            
            # Find the closest date with VET data
            vet_dates_array = np.array(vet_dates)
            date_from_env_np = np.array([date_from_env])
            closest_idx = np.argmin(np.abs(vet_dates_array - date_from_env_np))
            closest_date = vet_dates_array[closest_idx]
            print(f"Closest date with VET data: {closest_date}")

if __name__ == "__main__":
    debug_date_indexing() 