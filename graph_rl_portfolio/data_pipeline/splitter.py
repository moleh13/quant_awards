import os
import pandas as pd
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
INPUT_PATH = os.path.join(CACHE_DIR, 'preprocessed_all.csv')
TRAIN_PATH = os.path.join(CACHE_DIR, 'train.csv')
VAL_PATH = os.path.join(CACHE_DIR, 'val.csv')
TEST_PATH = os.path.join(CACHE_DIR, 'test.csv')

# Parameters
MIN_START_DATE = pd.Timestamp('2019-01-01')
TRAIN_END = pd.Timestamp('2022-12-31')
VAL_END = pd.Timestamp('2023-12-31')

# Load preprocessed data
full_df = pd.read_csv(INPUT_PATH, parse_dates=['datetime'])
full_df.set_index(['datetime', 'asset'], inplace=True)

# Filter assets with data starting from at least 2019-01-01
assets = full_df.index.get_level_values('asset').unique()
qualified_assets = []
for asset in assets:
    asset_df = full_df.xs(asset, level='asset')
    if asset_df.index.min() <= MIN_START_DATE:
        qualified_assets.append(asset)

filtered_df = full_df[full_df.index.get_level_values('asset').isin(qualified_assets)]
# Remove any rows before 2019-01-01
filtered_df = filtered_df.reset_index()
filtered_df = filtered_df[filtered_df['datetime'] >= MIN_START_DATE]
filtered_df.set_index(['datetime', 'asset'], inplace=True)

# Split by date
train_df = filtered_df.loc[(slice(None), slice(None)), :].reset_index()
train_df = train_df[train_df['datetime'] <= TRAIN_END]
val_df = filtered_df.loc[(slice(None), slice(None)), :].reset_index()
val_df = val_df[(val_df['datetime'] > TRAIN_END) & (val_df['datetime'] <= VAL_END)]
test_df = filtered_df.loc[(slice(None), slice(None)), :].reset_index()
test_df = test_df[test_df['datetime'] > VAL_END]

# Save splits
train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

def filter_top_assets(input_csv, output_csv, n_assets=50, min_start_date='2019-01-01'):
    df = pd.read_csv(input_csv, parse_dates=['datetime'])
    # Only keep assets with data starting from min_start_date
    asset_groups = df.groupby('asset')
    eligible_assets = []
    for asset, group in asset_groups:
        if group['datetime'].min() <= pd.Timestamp(min_start_date):
            eligible_assets.append(asset)
    # Count NaNs per asset
    nan_counts = {asset: df[df['asset'] == asset].isna().sum().sum() for asset in eligible_assets}
    # Sort by fewest NaNs
    top_assets = sorted(nan_counts, key=nan_counts.get)[:n_assets]
    # Save filtered asset list
    pd.Series(top_assets).to_csv(output_csv, index=False, header=['asset'])
    print(f"Saved top {n_assets} assets to {output_csv}")

if __name__ == "__main__":
    print(f"Assets with data from at least {MIN_START_DATE.date()}: {len(qualified_assets)}")
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print(f"Saved to {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")
    input_csv = os.path.join(os.path.dirname(__file__), '../data_cache/preprocessed_all.csv')
    output_csv = os.path.join(os.path.dirname(__file__), '../data_cache/top_50_assets.csv')
    filter_top_assets(input_csv, output_csv, n_assets=50) 