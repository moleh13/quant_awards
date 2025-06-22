import os
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), '../data_cache')
PREPROCESSED = os.path.join(CACHE_DIR, 'preprocessed_all.csv')
TRAIN = os.path.join(CACHE_DIR, 'train.csv')
VAL = os.path.join(CACHE_DIR, 'val.csv')
TEST = os.path.join(CACHE_DIR, 'test.csv')

# 1. Check files exist
for path in [PREPROCESSED, TRAIN, VAL, TEST]:
    assert os.path.exists(path), f"Missing file: {path}"

# 2. Check DataFrames are not empty and have expected columns
expected_cols = [
    'open', 'high', 'low', 'close', 'volume', 'log_close', 'log_return',
    'sma_7', 'sma_21', 'ema_7', 'ema_21', 'rsi_14',
    'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'bb_width', 'asset'
]
for path in [PREPROCESSED, TRAIN, VAL, TEST]:
    df = pd.read_csv(path)
    assert not df.empty, f"DataFrame is empty: {path}"
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {path}"

# 3. Check for NaNs in critical columns
critical_cols = ['close', 'log_return']
for path in [PREPROCESSED, TRAIN, VAL, TEST]:
    df = pd.read_csv(path)
    for col in critical_cols:
        n_nans = df[col].isna().sum()
        assert n_nans < 0.01 * len(df), f"Too many NaNs in {col} of {path}: {n_nans}"

# 4. Check date ranges and asset counts are reasonable
for path in [TRAIN, VAL, TEST]:
    df = pd.read_csv(path)
    assert df['datetime'].min() >= '2019-01-01', f"Dates before 2019 in {path}"
    assert df['asset'].nunique() >= 5, f"Too few assets in {path}"

print("All data pipeline sanity checks passed.") 