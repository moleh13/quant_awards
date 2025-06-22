import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
RAW_DIR = os.path.join(CACHE_DIR, 'raw')
OUTPUT_PATH = os.path.join(CACHE_DIR, 'preprocessed_all.csv')

# List all asset CSVs in raw dir
csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]

all_dfs = []
for csv_file in csv_files:
    asset = csv_file.replace('_1d.csv', '').replace('_', '/')
    df = pd.read_csv(os.path.join(RAW_DIR, csv_file), parse_dates=['datetime'], index_col='datetime')
    # Compute log-price
    df['log_close'] = np.log(df['close'])
    # Compute log return
    df['log_return'] = np.log(df['close']).diff()
    # SMA/EMA
    df['sma_7'] = SMAIndicator(df['close'], window=7, fillna=True).sma_indicator()
    df['sma_21'] = SMAIndicator(df['close'], window=21, fillna=True).sma_indicator()
    df['ema_7'] = EMAIndicator(df['close'], window=7, fillna=True).ema_indicator()
    df['ema_21'] = EMAIndicator(df['close'], window=21, fillna=True).ema_indicator()
    # RSI
    df['rsi_14'] = RSIIndicator(df['close'], window=14, fillna=True).rsi()
    # MACD
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # Bollinger Bands
    bb = BollingerBands(df['close'], window=20, window_dev=2, fillna=True)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    # Add asset column
    df['asset'] = asset
    all_dfs.append(df)

# Concatenate all assets
full_df = pd.concat(all_dfs)
full_df.set_index('asset', append=True, inplace=True)
full_df = full_df.reorder_levels(['datetime', 'asset'])

# Save to CSV
full_df.to_csv(OUTPUT_PATH)

if __name__ == "__main__":
    print(f"Preprocessing complete. Output saved to {OUTPUT_PATH}")
    print(full_df.head()) 