"""
Investigate VET/USDT data issues in the pipeline.
"""

import pandas as pd
import numpy as np
import os

def investigate_vet_data():
    print("=== Investigating VET/USDT Data Issues ===\n")
    
    # 1. Check raw data
    print("1. Checking raw data...")
    raw_path = "../data_cache/raw/"
    vet_files = [f for f in os.listdir(raw_path) if "VET" in f]
    print(f"VET files found: {vet_files}")
    
    if vet_files:
        vet_file = os.path.join(raw_path, vet_files[0])
        vet_raw = pd.read_csv(vet_file)
        print(f"VET raw data shape: {vet_raw.shape}")
        print(f"VET raw data columns: {vet_raw.columns.tolist()}")
        print(f"VET raw data date range: {vet_raw['datetime'].min()} to {vet_raw['datetime'].max()}")
        print(f"VET raw data NaN count: {vet_raw.isna().sum().sum()}")
        print(f"VET raw data sample:\n{vet_raw.head()}\n")
    
    # 2. Check preprocessed data
    print("2. Checking preprocessed data...")
    preprocessed_path = "../data_cache/preprocessed_all.csv"
    if os.path.exists(preprocessed_path):
        df = pd.read_csv(preprocessed_path, parse_dates=['datetime'])
        vet_data = df[df['asset'] == 'VET/USDT'].copy()
        print(f"VET preprocessed data shape: {vet_data.shape}")
        print(f"VET preprocessed date range: {vet_data['datetime'].min()} to {vet_data['datetime'].max()}")
        print(f"VET preprocessed NaN count by column:")
        for col in vet_data.columns:
            nan_count = vet_data[col].isna().sum()
            if nan_count > 0:
                print(f"  {col}: {nan_count} NaNs")
        
        # Check specific problematic dates
        print(f"\nVET data around step 478 (problematic area):")
        all_dates = sorted(df['datetime'].unique())
        if len(all_dates) > 478:
            problem_date = all_dates[478]
            print(f"Step 478 date: {problem_date}")
            vet_at_478 = vet_data[vet_data['datetime'] == problem_date]
            print(f"VET data at step 478: {vet_at_478.shape}")
            if not vet_at_478.empty:
                print(f"VET features at step 478:\n{vet_at_478.iloc[0]}")
        
        # Check consecutive NaN patterns
        print(f"\nVET consecutive NaN patterns:")
        vet_data_sorted = vet_data.sort_values('datetime')
        nan_mask = vet_data_sorted.isna().any(axis=1)
        nan_streaks = []
        current_streak = 0
        for is_nan in nan_mask:
            if is_nan:
                current_streak += 1
            else:
                if current_streak > 0:
                    nan_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            nan_streaks.append(current_streak)
        
        print(f"NaN streak lengths: {nan_streaks}")
        if nan_streaks:
            print(f"Max consecutive NaN days: {max(nan_streaks)}")
            print(f"Total NaN days: {sum(nan_streaks)}")
    
    # 3. Check train/val/test splits
    print("\n3. Checking train/val/test splits...")
    for split_name, split_path in [
        ("train", "../data_cache/train.csv"),
        ("val", "../data_cache/val.csv"), 
        ("test", "../data_cache/test.csv")
    ]:
        if os.path.exists(split_path):
            split_df = pd.read_csv(split_path, parse_dates=['datetime'])
            vet_split = split_df[split_df['asset'] == 'VET/USDT']
            print(f"{split_name}: VET shape {vet_split.shape}, NaN count {vet_split.isna().sum().sum()}")
    
    # 4. Check top 50 assets list
    print("\n4. Checking top 50 assets list...")
    top_50_path = "../data_cache/top_50_assets.csv"
    if os.path.exists(top_50_path):
        top_50 = pd.read_csv(top_50_path)
        vet_in_top_50 = 'VET/USDT' in top_50['asset'].values
        print(f"VET/USDT in top 50 assets: {vet_in_top_50}")
        if vet_in_top_50:
            vet_rank = top_50[top_50['asset'] == 'VET/USDT'].index[0]
            print(f"VET/USDT rank in top 50: {vet_rank + 1}")
    
    # 5. Compare with other assets
    print("\n5. Comparing with other assets...")
    if os.path.exists(preprocessed_path):
        df = pd.read_csv(preprocessed_path, parse_dates=['datetime'])
        asset_nan_counts = {}
        for asset in df['asset'].unique():
            asset_data = df[df['asset'] == asset]
            nan_count = asset_data.isna().sum().sum()
            asset_nan_counts[asset] = nan_count
        
        print("Assets with most NaN values:")
        sorted_assets = sorted(asset_nan_counts.items(), key=lambda x: x[1], reverse=True)
        for asset, nan_count in sorted_assets[:10]:
            print(f"  {asset}: {nan_count} NaNs")

if __name__ == "__main__":
    investigate_vet_data() 