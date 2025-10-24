"""
Birdeye Exchange Implementation
Data-only source for Solana token market data (no trading)
Specialized for Solana ecosystem tokens
"""

# Standard library imports
import os

# Third-party imports
import pandas as pd
import requests

# Standard library from imports
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Local from imports
try:
    from .base_exchange import BaseExchange
except ImportError:
    from base_exchange import BaseExchange


class BirdeyeExchange(BaseExchange):
    """Birdeye data source for Solana tokens (no trading, data only)"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv('BIRDEYE_API_KEY', '')

        self.base_url = "https://public-api.birdeye.so/defi"

    def _is_solana_address(self, symbol: str) -> bool:
        """Check if string looks like a Solana token address"""
        # Solana addresses are base58 encoded, typically 32-44 characters
        return len(symbol) >= 32 and symbol.isalnum() and '/' not in symbol

    def _get_time_range(self, days_back: int):
        """Calculate time_from and time_to for API request"""
        now = datetime.now()
        days_earlier = now - timedelta(days=days_back)
        time_to = int(now.timestamp())
        time_from = int(days_earlier.timestamp())
        return time_from, time_to

    def _timeframe_to_days(self, timeframe: str, limit: int) -> int:
        """Convert timeframe and limit to approximate days"""
        # Map timeframe to hours
        timeframe_hours = {
            '1m': 1/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '12h': 12,
            '1d': 24, '1w': 168
        }

        hours_per_candle = timeframe_hours.get(timeframe, 1)
        total_hours = hours_per_candle * limit
        days = max(1, int(total_hours / 24))

        return days

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Fetch OHLCV data from Birdeye for Solana tokens

        Args:
            symbol: Solana token address (e.g., 'So11111111111111111111111111111111111111112')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch

        Returns:
            List of OHLCV data: [[timestamp, open, high, low, close, volume], ...]
            Or empty list on error
        """
        try:
            # Check if API key is available
            if not self.api_key:
                print("Birdeye: API key required")
                return []

            # Verify this looks like a Solana address
            if not self._is_solana_address(symbol):
                print(f"Birdeye: Invalid Solana address format: {symbol}")
                return []

            # Calculate days_back from timeframe and limit
            days_back = self._timeframe_to_days(timeframe, limit)
            time_from, time_to = self._get_time_range(days_back)

            # Build API request
            url = f"{self.base_url}/ohlcv"
            params = {
                'address': symbol,
                'type': timeframe,
                'time_from': time_from,
                'time_to': time_to
            }
            headers = {"X-API-KEY": self.api_key}

            # Make request
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Birdeye API error: {response.status_code}")
                return []

            json_response = response.json()
            items = json_response.get('data', {}).get('items', [])

            if not items:
                return []

            # Format data: Birdeye returns {unixTime, o, h, l, c, v}
            # We need: [timestamp_ms, open, high, low, close, volume]
            formatted_data = []
            for item in items:
                formatted_data.append([
                    item['unixTime'] * 1000,  # Convert to milliseconds
                    item['o'],  # open
                    item['h'],  # high
                    item['l'],  # low
                    item['c'],  # close
                    item['v']   # volume
                ])

            # Filter out future dates
            current_time = datetime.now().timestamp() * 1000
            formatted_data = [d for d in formatted_data if d[0] <= current_time]

            return formatted_data[:limit]

        except Exception as e:
            print(f"Birdeye get_ohlcv error for {symbol}: {e}")
            return []

    def supports_symbol(self, symbol: str) -> bool:
        """Check if this looks like a Solana address"""
        return self._is_solana_address(symbol) and bool(self.api_key)

    # ============================================================================
    # DATA-ONLY SOURCE: Trading methods not supported
    # ============================================================================

    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def get_balance(self, asset: str = None) -> Dict:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def get_position(self, symbol: str) -> Dict:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def get_all_positions(self) -> List[Dict]:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def get_ticker(self, symbol: str) -> Dict:
        """Get current price data for Solana token (supported)"""
        try:
            if not self.api_key or not self._is_solana_address(symbol):
                return {}

            url = f"{self.base_url}/token/price"
            params = {'address': symbol}
            headers = {"X-API-KEY": self.api_key}

            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                return {}

            data = response.json().get('data', {})

            if data:
                return {
                    'symbol': symbol,
                    'last': data.get('value', 0),
                    'bid': data.get('value', 0),  # Approximation
                    'ask': data.get('value', 0),  # Approximation
                    'volume': data.get('volume24h', 0),
                    'timestamp': int(data.get('updateUnixTime', 0) * 1000)
                }

            return {}

        except Exception as e:
            print(f"Birdeye get_ticker error: {e}")
            return {}

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        raise NotImplementedError("Birdeye is a data-only source. Order book not available.")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        raise NotImplementedError("Birdeye is a data-only source. Trading not supported.")

    def normalize_symbol(self, symbol: str) -> str:
        """Birdeye uses Solana addresses directly, no normalization needed"""
        return symbol
