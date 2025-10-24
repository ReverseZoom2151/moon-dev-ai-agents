'''
nice functions from hyper liquid i can use

🌙 Moon Dev's Hyperliquid Functions
Built with love by Moon Dev 🚀
'''

# Standard library imports
import sys
import time
import traceback

# Fix Windows console encoding for emojis (must be done early)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Python < 3.7 or already configured
        pass

# Third-party imports
import numpy as np
import pandas as pd
import requests
import talib

# Standard library from imports
from datetime import datetime, timedelta, timezone

# Constants
BATCH_SIZE = 5000  # MAX IS 5000 FOR HYPERLIQUID
MAX_RETRIES = 3
MAX_ROWS = 5000
BASE_URL = 'https://api.hyperliquid.xyz/info'

# Global variable to store timestamp offset
timestamp_offset = None

def adjust_timestamp(dt):
    """Adjust API timestamps by subtracting the timestamp offset."""
    if timestamp_offset is not None:
        corrected_dt = dt - timestamp_offset
        return corrected_dt
    return dt

def _get_ohlcv(symbol, interval, start_time, end_time, batch_size=BATCH_SIZE):
    """Internal function to fetch OHLCV data from Hyperliquid"""
    global timestamp_offset
    print(f'\n🔍 Requesting data for {symbol}:')
    print(f'📊 Batch Size: {batch_size}')
    print(f'🚀 Start: {start_time.strftime("%Y-%m-%d %H:%M:%S")} UTC')
    print(f'🎯 End: {end_time.strftime("%Y-%m-%d %H:%M:%S")} UTC')

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                BASE_URL,
                headers={'Content-Type': 'application/json'},
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": interval,
                        "startTime": start_ts,
                        "endTime": end_ts,
                        "limit": batch_size
                    }
                },
                timeout=10
            )

            if response.status_code == 200:
                snapshot_data = response.json()
                if snapshot_data:
                    # Handle timestamp offset
                    if timestamp_offset is None:
                        latest_api_timestamp = datetime.fromtimestamp(snapshot_data[-1]['t'] / 1000, tz=timezone.utc)
                        system_current_date = datetime.now(timezone.utc)
                        expected_latest_timestamp = system_current_date
                        timestamp_offset = latest_api_timestamp - expected_latest_timestamp
                        print(f"⏱️ Calculated timestamp offset: {timestamp_offset}")

                    # Adjust timestamps
                    for candle in snapshot_data:
                        dt = datetime.fromtimestamp(candle['t'] / 1000, tz=timezone.utc)
                        adjusted_dt = adjust_timestamp(dt)
                        candle['t'] = int(adjusted_dt.timestamp() * 1000)

                    first_time = datetime.fromtimestamp(snapshot_data[0]['t'] / 1000, tz=timezone.utc)
                    last_time = datetime.fromtimestamp(snapshot_data[-1]['t'] / 1000, tz=timezone.utc)
                    print(f'✨ Received {len(snapshot_data)} candles')
                    print(f'📈 First: {first_time}')
                    print(f'📉 Last: {last_time}')
                    return snapshot_data
                print('❌ No data returned by API')
                return None
            print(f'⚠️ HTTP Error {response.status_code}: {response.text}')
        except requests.exceptions.RequestException as e:
            print(f'⚠️ Request failed (attempt {attempt + 1}): {e}')
            time.sleep(1)
    return None

def _process_data_to_df(snapshot_data):
    """Convert raw API data to DataFrame"""
    if snapshot_data:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000, tz=timezone.utc)
            # Convert all numeric values to float
            data.append([
                timestamp,
                float(snapshot['o']),
                float(snapshot['h']),
                float(snapshot['l']),
                float(snapshot['c']),
                float(snapshot['v'])
            ])
        df = pd.DataFrame(data, columns=columns)
        
        # Ensure numeric columns are float64
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype('float64')
        
        print("\n📊 OHLCV Data Types:")
        print(df.dtypes)
        
        return df
    return pd.DataFrame()

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    if df.empty:
        return df
        
    try:
        print("\n🔧 Adding technical indicators...")
        
        # Ensure numeric columns are float64
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype('float64')

        # Add basic indicators using talib (more reliable than pandas_ta)
        df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

        # Add MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
        df['MACD_12_26_9'] = macd
        df['MACDs_12_26_9'] = macd_signal
        df['MACDh_12_26_9'] = macd_hist

        # Add Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
        df['BBU_20_2.0'] = upper
        df['BBM_20_2.0'] = middle
        df['BBL_20_2.0'] = lower
        
        print("✅ Technical indicators added successfully")
        return df
        
    except Exception as e:
        print(f"❌ Error adding technical indicators: {str(e)}")
        traceback.print_exc()
        return df

def get_data(symbol, timeframe='15m', bars=100, add_indicators=True):
    """
    🌙 Moon Dev's Hyperliquid Data Fetcher
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC', 'ETH')
        timeframe (str): Candle timeframe (default: '15m')
        bars (int): Number of bars to fetch (default: 100, max: 5000)
        add_indicators (bool): Whether to add technical indicators
    
    Returns:
        pd.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]
                     and technical indicators if requested
    """
    print("\n🌙 Moon Dev's Hyperliquid Data Fetcher")
    print(f"🎯 Symbol: {symbol}")
    print(f"⏰ Timeframe: {timeframe}")
    print(f"📊 Requested bars: {min(bars, MAX_ROWS)}")

    # Ensure we don't exceed max rows
    bars = min(bars, MAX_ROWS)

    # Calculate time window
    end_time = datetime.now(timezone.utc)
    # Add extra time to ensure we get enough bars
    start_time = end_time - timedelta(days=60)

    data = _get_ohlcv(symbol, timeframe, start_time, end_time, batch_size=bars)
    
    if not data:
        print("❌ No data available.")
        return pd.DataFrame()

    df = _process_data_to_df(data)

    if not df.empty:
        # Get the most recent bars
        df = df.sort_values('timestamp', ascending=False).head(bars).sort_values('timestamp')
        df = df.reset_index(drop=True)
        
        # Add technical indicators if requested
        if add_indicators:
            df = add_technical_indicators(df)

        print("\n📊 Data summary:")
        print(f"📈 Total candles: {len(df)}")
        print(f"📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print("✨ Thanks for using Moon Dev's Data Fetcher! ✨")

    return df

def get_market_info():
    """Get current market info for all coins on Hyperliquid"""
    try:
        print("\n🔄 Sending request to Hyperliquid API...")
        response = requests.post(
            BASE_URL,
            headers={'Content-Type': 'application/json'},
            json={"type": "allMids"}
        )
        
        print(f"📡 Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📦 Raw response data: {data}")
            return data
        print(f"❌ Bad status code: {response.status_code}")
        print(f"📄 Response text: {response.text}")
        return None
    except Exception as e:
        print(f"❌ Error getting market info: {str(e)}")
        traceback.print_exc()  # Print full error traceback
        return None

def test_market_info():
    print("\n💹 Testing Market Info...")
    try:
        print("🎯 Fetching current market prices...")
        info = get_market_info()
        
        print(f"\n📊 Response type: {type(info)}")
        if info is not None:
            print(f"📝 Response content: {info}")
        
        if info and isinstance(info, dict):
            print("\n💰 Current Market Prices:")
            print("=" * 50)
            # Target symbols we're interested in
            target_symbols = ["BTC", "ETH", "SOL", "ARB", "OP", "MATIC"]
            
            for symbol in target_symbols:
                if symbol in info:
                    try:
                        price = float(info[symbol])
                        print(f"Symbol: {symbol:8} | Price: ${price:,.2f}")
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Error processing price for {symbol}: {str(e)}")
                else:
                    print(f"⚠️ No price data for {symbol}")
        else:
            print("❌ No valid market info received")
            if info is None:
                print("📛 Response was None")
            else:
                print(f"❓ Unexpected response type: {type(info)}")
    except Exception as e:
        print(f"❌ Error in market info test: {str(e)}")
        print(f"🔍 Full error traceback:")
        traceback.print_exc()

def get_funding_rates(symbol):
    """
    Get current funding rate for a specific coin on Hyperliquid
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC', 'ETH', 'FART')
        
    Returns:
        dict: Funding data including rate, mark price, and open interest
    """
    try:
        print(f"\n🔄 Fetching funding rate for {symbol}...")
        response = requests.post(
            BASE_URL,
            headers={'Content-Type': 'application/json'},
            json={"type": "metaAndAssetCtxs"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if len(data) >= 2 and isinstance(data[0], dict) and isinstance(data[1], list):
                # Get universe (symbols) from first element
                universe = {coin['name']: i for i, coin in enumerate(data[0]['universe'])}
                
                # Check if symbol exists
                if symbol not in universe:
                    print(f"❌ Symbol {symbol} not found in Hyperliquid universe")
                    print(f"📝 Available symbols: {', '.join(universe.keys())}")
                    return None
                
                # Get funding data from second element
                funding_data = data[1]
                idx = universe[symbol]
                
                if idx < len(funding_data):
                    asset_data = funding_data[idx]
                    return {
                        'funding_rate': float(asset_data['funding']),
                        'mark_price': float(asset_data['markPx']),
                        'open_interest': float(asset_data['openInterest'])
                    }
                    
            print("❌ Unexpected response format")
            return None
        print(f"❌ Bad status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Error getting funding rate for {symbol}: {str(e)}")
        traceback.print_exc()
        return None

def test_funding_rates():
    print("\n💸 Testing Funding Rates...")
    try:
        # Test with some interesting symbols
        test_symbols = ["BTC", "ETH", "FARTCOIN"]
        
        for symbol in test_symbols:
            print(f"\n📊 Testing {symbol}:")
            print("=" * 50)
            data = get_funding_rates(symbol)
            
            if data:
                # The API returns the 8-hour funding rate
                # To get hourly rate: funding_rate
                # To get annual rate: hourly * 24 * 365
                hourly_rate = float(data['funding_rate']) * 100  # Convert to percentage
                annual_rate = hourly_rate * 24 * 365  # Convert hourly to annual
                
                print(f"Symbol: {symbol:8} | Hourly: {hourly_rate:7.4f}% | Annual: {annual_rate:7.2f}% | OI: {data['open_interest']:10.2f}")
            else:
                print(f"❌ No funding data received for {symbol}")
                
    except Exception as e:
        print(f"❌ Error in funding rates test: {str(e)}")
        print(f"🔍 Full error traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🌙 Moon Dev's Hyperliquid Function Tester")
    print("=" * 50)
    
    def test_btc_data():
        print("\n🔍 Testing BTC Data Retrieval...")
        try:
            # Test with BTC on 15m timeframe
            df = get_data("BTC", timeframe="15m", bars=100, add_indicators=True)
            
            if not df.empty:
                print("\n📊 Last 5 candles:")
                print("=" * 80)
                for idx, row in df.tail().iterrows():
                    print(f"Time: {row['timestamp'].strftime('%H:%M:%S')} | Open: ${row['open']:,.2f} | High: ${row['high']:,.2f} | Low: ${row['low']:,.2f} | Close: ${row['close']:,.2f} | Vol: ${row['volume']:,.2f}")
                
                print("\n📈 Technical Indicators (Last Candle):")
                print("=" * 50)
                last_row = df.iloc[-1]
                print(f"SMA20: ${last_row['sma_20']:,.2f}")
                print(f"SMA50: ${last_row['sma_50']:,.2f}")
                print(f"RSI: {last_row['rsi']:.2f}")
                print(f"MACD: {last_row['MACD_12_26_9']:,.2f}")
                
            else:
                print("❌ No data received")
                
        except Exception as e:
            print(f"❌ Error in BTC test: {str(e)}")
    
    # Run tests
    print("\n🧪 Running function tests...")
    
    test_btc_data()
    test_market_info()
    test_funding_rates()  # Now tests individual symbols
    
    print("\n✨ Testing complete! Moon Dev out! 🌙") 