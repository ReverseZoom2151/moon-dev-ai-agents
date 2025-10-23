"""
Quick test script for Exchange Orchestrator with CoinGecko and Birdeye
"""

import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_exchange_manager import MultiExchangeManager

def test_orchestrator():
    """Test the complete Exchange Orchestrator with all data sources"""

    print("\n" + "="*70)
    print("🧪 Testing Exchange Orchestrator with CoinGecko & Birdeye")
    print("="*70 + "\n")

    # Initialize orchestrator
    manager = MultiExchangeManager()

    print(f"\n📋 Active exchanges: {list(manager.active_exchanges.keys())}\n")

    # Test 1: Major token via CoinGecko (should work without any API keys)
    print("=" * 70)
    print("TEST 1: Fetching BTC data (should use CoinGecko)")
    print("=" * 70)
    df = manager.get_ohlcv('BTC', timeframe='1d', limit=7)
    if df is not None:
        print(f"\n✅ SUCCESS: Got {len(df)} candles")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst row:")
        print(df.head(1))
    else:
        print("❌ FAILED: No data returned")

    # Test 2: Trading pair format
    print("\n" + "=" * 70)
    print("TEST 2: Fetching ETH/USD (should try exchanges)")
    print("=" * 70)
    df = manager.get_ohlcv('ETH/USD', timeframe='1h', limit=24)
    if df is not None:
        print(f"\n✅ SUCCESS: Got {len(df)} candles")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("❌ FAILED: No data returned")

    # Test 3: Symbol without slash
    print("\n" + "=" * 70)
    print("TEST 3: Fetching SOL (should use CoinGecko)")
    print("=" * 70)
    df = manager.get_ohlcv('SOL', timeframe='1d', limit=30)
    if df is not None:
        print(f"\n✅ SUCCESS: Got {len(df)} candles")
        print(f"Last price: ${df.iloc[-1]['close']:.2f}")
    else:
        print("❌ FAILED: No data returned")

    print("\n" + "="*70)
    print("🎉 Exchange Orchestrator Testing Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_orchestrator()
