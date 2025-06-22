import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
RAW_DIR = os.path.join(CACHE_DIR, 'raw')
if not os.path.exists(RAW_DIR):
    os.makedirs(RAW_DIR)

EXCHANGE = 'binance'
OHLCV_TIMEFRAME = '1d'


def get_top_200_symbols(exchange):
    markets = exchange.load_markets()
    # Filter for USDT pairs, as most liquid
    usdt_pairs = [s for s in markets if s.endswith('/USDT') and not s.startswith('USD')]
    # Fetch all tickers at once (no argument)
    if hasattr(exchange, 'fetch_tickers'):
        tickers = exchange.fetch_tickers()
        filtered_tickers = {k: v for k, v in tickers.items() if k in usdt_pairs}
        sorted_pairs = sorted(
            filtered_tickers.items(),
            key=lambda x: x[1].get('quoteVolume', 0),
            reverse=True
        )
        top_200 = [k for k, v in sorted_pairs[:200]]
    else:
        top_200 = usdt_pairs[:200]
    return top_200


def fetch_ohlcv_for_symbol(exchange, symbol, since, until):
    all_ohlcv = []
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    while since_ms < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=OHLCV_TIMEFRAME, since=since_ms, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            # Add one ms to avoid overlap
            since_ms = last_ts + 1
            # Sleep to avoid rate limits
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df


def load_or_fetch_ohlcv(symbol, since=None, until=None, cache_dir=RAW_DIR):
    if since is None:
        since = datetime(2017, 1, 1)  # Binance spot launch
    if until is None:
        until = datetime.utcnow()
    safe_symbol = symbol.replace('/', '_')
    cache_path = os.path.join(cache_dir, f'{safe_symbol}_{OHLCV_TIMEFRAME}.csv')
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=['datetime'], index_col='datetime')
        # Check if cache covers the required range
        if df.index.min() <= since and df.index.max() >= until:
            return df.loc[since:until]
    # Fetch from exchange
    exchange = getattr(ccxt, EXCHANGE)({'enableRateLimit': True})
    df = fetch_ohlcv_for_symbol(exchange, symbol, since, until)
    df.to_csv(cache_path)
    return df


def get_ohlcv_dataframe(since=None, until=None, symbols=None):
    exchange = getattr(ccxt, EXCHANGE)({'enableRateLimit': True})
    if symbols is None:
        symbols = get_top_200_symbols(exchange)
    all_dfs = []
    for symbol in symbols:
        print(f"Processing {symbol}")
        df = load_or_fetch_ohlcv(symbol, since, until)
        # Add asset column for stacking
        df['asset'] = symbol
        all_dfs.append(df)
    # Concatenate all assets
    full_df = pd.concat(all_dfs)
    # MultiIndex: (datetime, asset)
    full_df.set_index('asset', append=True, inplace=True)
    full_df = full_df.reorder_levels(['datetime', 'asset'])
    return full_df


if __name__ == "__main__":
    # Example usage: fetch maximum available range
    df = get_ohlcv_dataframe()
    print(df.head())
    print(df.index.names) 