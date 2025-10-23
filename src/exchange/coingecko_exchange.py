"""
CoinGecko Exchange Implementation
Data-only source for market data and OHLCV (no trading)
Supports 10,000+ tokens via CoinGecko API
"""

# Standard library imports
import os
import time

# Third-party imports
import pandas as pd
import requests

# Standard library from imports
from typing import Dict, List, Optional

# Local from imports
from .base_exchange import BaseExchange


class CoinGeckoExchange(BaseExchange):
    """CoinGecko data source (no trading, data only)"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY', '')

        # Use FREE API if no key, PRO API if key provided
        if not self.api_key:
            self.base_url = "https://api.coingecko.com/api/v3"
            self.headers = {"Content-Type": "application/json"}
        else:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
            self.headers = {
                "x-cg-pro-api-key": self.api_key,
                "Content-Type": "application/json"
            }

        # Cache for coin IDs (symbol -> coingecko_id mapping)
        self._coin_list = None

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 429:
                # Rate limit hit, wait and retry
                time.sleep(60)
                return self._make_request(endpoint, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"CoinGecko API error: {e}")
            return {} if isinstance(e, requests.exceptions.JSONDecodeError) else []

    def _load_coin_list(self):
        """Load list of all coins (cached)"""
        if self._coin_list is None:
            data = self._make_request("coins/list")
            # Create mapping: symbol -> coin_id
            self._coin_list = {coin['symbol'].upper(): coin['id'] for coin in data}
        return self._coin_list

    def _symbol_to_coin_id(self, symbol: str) -> Optional[str]:
        """Convert trading symbol to CoinGecko coin ID"""
        # Remove quote currency if present (BTC/USD -> BTC)
        base_symbol = symbol.split('/')[0].upper()

        # Load coin list
        coin_list = self._load_coin_list()

        # Try direct lookup
        if base_symbol in coin_list:
            return coin_list[base_symbol]

        # Try lowercase lookup
        if base_symbol.lower() in [k.lower() for k in coin_list.keys()]:
            for k, v in coin_list.items():
                if k.lower() == base_symbol.lower():
                    return v

        return None

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Fetch OHLCV data from CoinGecko

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'BTC/USD', 'ETH')
            timeframe: Candle timeframe (Note: CoinGecko uses days, not granular timeframes)
            limit: Number of candles to fetch (approximated by days)

        Returns:
            List of OHLCV data: [[timestamp, open, high, low, close, volume], ...]
            Or empty list on error

        Note:
            CoinGecko's OHLC endpoint returns daily candles only.
            The 'days' parameter is calculated from limit and timeframe.
            For intraday data, CoinGecko's granularity is limited.
        """
        try:
            # Convert symbol to CoinGecko coin ID
            coin_id = self._symbol_to_coin_id(symbol)
            if not coin_id:
                print(f"CoinGecko: Could not find coin ID for {symbol}")
                return []

            # CoinGecko OHLC endpoint supports: 1, 7, 14, 30, 90, 180, 365, max days
            # Map timeframe/limit to appropriate days parameter
            if limit >= 365:
                days = 'max'
            elif limit >= 180:
                days = 365
            elif limit >= 90:
                days = 180
            elif limit >= 30:
                days = 90
            elif limit >= 14:
                days = 30
            elif limit >= 7:
                days = 14
            else:
                days = 7  # Minimum

            params = {
                'vs_currency': 'usd',
                'days': days
            }

            # Fetch OHLC data
            ohlc_data = self._make_request(f"coins/{coin_id}/ohlc", params)

            if not ohlc_data:
                return []

            # CoinGecko returns: [[timestamp_ms, open, high, low, close], ...]
            # We need to add volume (set to 0 as CoinGecko OHLC doesn't include it)
            formatted_data = []
            for candle in ohlc_data:
                if len(candle) >= 5:
                    # [timestamp, open, high, low, close, volume]
                    formatted_data.append([
                        candle[0],  # timestamp (ms)
                        candle[1],  # open
                        candle[2],  # high
                        candle[3],  # low
                        candle[4],  # close
                        0           # volume (not provided by CoinGecko OHLC)
                    ])

            return formatted_data[:limit]  # Limit to requested number

        except Exception as e:
            print(f"CoinGecko get_ohlcv error for {symbol}: {e}")
            return []

    def supports_symbol(self, symbol: str) -> bool:
        """Check if CoinGecko has data for this symbol"""
        coin_id = self._symbol_to_coin_id(symbol)
        return coin_id is not None

    # ============================================================================
    # DATA-ONLY SOURCE: Trading methods not supported
    # ============================================================================

    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")

    def get_balance(self, asset: str = None) -> Dict:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")

    def get_position(self, symbol: str) -> Dict:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")

    def get_all_positions(self) -> List[Dict]:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")

    def get_ticker(self, symbol: str) -> Dict:
        """Get current price data (supported)"""
        try:
            coin_id = self._symbol_to_coin_id(symbol)
            if not coin_id:
                return {}

            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }

            data = self._make_request("simple/price", params)

            if coin_id in data:
                coin_data = data[coin_id]
                return {
                    'symbol': symbol,
                    'last': coin_data.get('usd', 0),
                    'bid': coin_data.get('usd', 0),  # Approximation
                    'ask': coin_data.get('usd', 0),  # Approximation
                    'volume': coin_data.get('usd_24h_vol', 0),
                    'timestamp': int(time.time() * 1000)
                }

            return {}

        except Exception as e:
            print(f"CoinGecko get_ticker error: {e}")
            return {}

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        raise NotImplementedError("CoinGecko is a data-only source. Order book not available.")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        raise NotImplementedError("CoinGecko is a data-only source. Trading not supported.")
