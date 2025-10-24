"""
HyperLiquid Exchange Implementation
Perpetuals trading with leverage (1-50x)
Supports 100+ symbols with granular OHLCV data
"""

# Standard library imports
import os
import time

# Third-party imports
import eth_account
import pandas as pd
import requests

# Fix for eth_account API change (encode_typed_data -> encode_structured_data)
try:
    from eth_account.messages import encode_typed_data
except ImportError:
    from eth_account.messages import encode_structured_data
    import eth_account.messages
    eth_account.messages.encode_typed_data = encode_structured_data

# Standard library from imports
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Third-party from imports
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Local from imports
try:
    from .base_exchange import BaseExchange
except ImportError:
    from base_exchange import BaseExchange


class HyperLiquidExchange(BaseExchange):
    """HyperLiquid perpetuals exchange implementation"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Get private key from env if not provided
        self.api_key = api_key or os.getenv('HYPER_LIQUID_KEY', '')

        if not self.api_key:
            raise ValueError("HYPER_LIQUID_KEY required for HyperLiquid")

        # Initialize account from private key
        self.account = eth_account.Account.from_key(self.api_key)

        # Initialize HyperLiquid clients
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(self.account, constants.MAINNET_API_URL)

        # Base URL for data fetching
        self.base_url = 'https://api.hyperliquid.xyz/info'

        # Constants
        self.MAX_CANDLES = 5000  # HyperLiquid limit

        # Cache for meta data
        self._meta_cache = None
        self._meta_cache_time = None
        self.CACHE_DURATION = 300  # 5 minutes

    def _get_meta(self) -> Dict:
        """Get exchange metadata (cached)"""
        now = time.time()

        # Return cache if valid
        if self._meta_cache and self._meta_cache_time:
            if now - self._meta_cache_time < self.CACHE_DURATION:
                return self._meta_cache

        # Fetch fresh data
        try:
            response = requests.post(
                self.base_url,
                headers={'Content-Type': 'application/json'},
                json={'type': 'meta'},
                timeout=10
            )

            if response.status_code == 200:
                self._meta_cache = response.json()
                self._meta_cache_time = now
                return self._meta_cache

        except Exception as e:
            print(f"HyperLiquid meta fetch error: {e}")

        return {}

    def supports_symbol(self, symbol: str) -> bool:
        """Check if HyperLiquid supports this symbol"""
        meta = self._get_meta()
        if not meta:
            return False

        symbols = meta.get('universe', [])
        symbol_names = [s['name'] for s in symbols]

        # Remove /USD suffix if present
        clean_symbol = symbol.replace('/USD', '').replace('/USDT', '')

        return clean_symbol in symbol_names

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to HyperLiquid format (no slashes)"""
        return symbol.replace('/USD', '').replace('/USDT', '')

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Fetch OHLCV data from HyperLiquid

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH', 'SOL')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 5000)

        Returns:
            List of OHLCV data: [[timestamp, open, high, low, close, volume], ...]
            Or empty list on error
        """
        try:
            symbol = self.normalize_symbol(symbol)

            # Validate limit
            if limit > self.MAX_CANDLES:
                print(f"HyperLiquid: Limiting {limit} candles to max {self.MAX_CANDLES}")
                limit = self.MAX_CANDLES

            # Calculate time range
            now = datetime.utcnow()

            # Map timeframe to timedelta
            timeframe_map = {
                '1m': timedelta(minutes=limit),
                '5m': timedelta(minutes=limit * 5),
                '15m': timedelta(minutes=limit * 15),
                '1h': timedelta(hours=limit),
                '4h': timedelta(hours=limit * 4),
                '1d': timedelta(days=limit),
            }

            delta = timeframe_map.get(timeframe, timedelta(hours=limit))
            start_time = now - delta

            # Convert to milliseconds
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(now.timestamp() * 1000)

            # Make API request
            response = requests.post(
                self.base_url,
                headers={'Content-Type': 'application/json'},
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": timeframe,
                        "startTime": start_ts,
                        "endTime": end_ts,
                        "limit": limit
                    }
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"HyperLiquid API error: {response.status_code}")
                return []

            snapshot_data = response.json()

            if not snapshot_data:
                return []

            # Format data: HyperLiquid returns [{t, o, h, l, c, v}, ...]
            # We need: [timestamp_ms, open, high, low, close, volume]
            formatted_data = []
            for candle in snapshot_data:
                formatted_data.append([
                    candle['t'],  # timestamp (already in ms)
                    float(candle['o']),  # open
                    float(candle['h']),  # high
                    float(candle['l']),  # low
                    float(candle['c']),  # close
                    float(candle['v'])   # volume
                ])

            return formatted_data

        except Exception as e:
            print(f"HyperLiquid get_ohlcv error for {symbol}: {e}")
            return []

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data"""
        try:
            symbol = self.normalize_symbol(symbol)

            # Get L2 book for bid/ask
            response = requests.post(
                self.base_url,
                headers={'Content-Type': 'application/json'},
                json={
                    'type': 'l2Book',
                    'coin': symbol
                },
                timeout=10
            )

            if response.status_code != 200:
                return {}

            l2_data = response.json()
            levels = l2_data.get('levels', [])

            if len(levels) < 2:
                return {}

            bid = float(levels[0][0]['px']) if levels[0] else 0
            ask = float(levels[1][0]['px']) if levels[1] else 0
            last = (bid + ask) / 2

            return {
                'symbol': symbol,
                'last': last,
                'bid': bid,
                'ask': ask,
                'volume': 0,  # Not provided in L2 book
                'timestamp': int(time.time() * 1000)
            }

        except Exception as e:
            print(f"HyperLiquid get_ticker error: {e}")
            return {}

    def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        try:
            user_state = self.info.user_state(self.account.address)

            if asset:
                # Return specific asset balance
                # HyperLiquid shows all balances in USD
                return {
                    'USD': {
                        'free': float(user_state.get('withdrawable', 0)),
                        'locked': 0,
                        'total': float(user_state.get('marginSummary', {}).get('accountValue', 0))
                    }
                }
            else:
                # Return all balances
                margin_summary = user_state.get('marginSummary', {})
                return {
                    'USD': {
                        'free': float(user_state.get('withdrawable', 0)),
                        'locked': float(margin_summary.get('totalMarginUsed', 0)),
                        'total': float(margin_summary.get('accountValue', 0))
                    }
                }

        except Exception as e:
            print(f"HyperLiquid get_balance error: {e}")
            return {}

    def get_position(self, symbol: str) -> Dict:
        """Get current position for a symbol"""
        try:
            symbol = self.normalize_symbol(symbol)

            user_state = self.info.user_state(self.account.address)

            for position in user_state.get("assetPositions", []):
                pos = position.get("position", {})
                if pos.get("coin") == symbol and float(pos.get("szi", 0)) != 0:
                    pos_size = float(pos["szi"])
                    entry_price = float(pos["entryPx"])
                    pnl_percent = float(pos.get("returnOnEquity", 0)) * 100

                    # Get current price
                    ticker = self.get_ticker(symbol)
                    current_price = ticker.get('last', entry_price)

                    # Calculate PnL
                    pnl = (current_price - entry_price) * abs(pos_size)

                    return {
                        'symbol': symbol,
                        'quantity': pos_size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percent
                    }

            # No position found
            return {
                'symbol': symbol,
                'quantity': 0,
                'entry_price': 0,
                'current_price': 0,
                'pnl': 0,
                'pnl_percentage': 0
            }

        except Exception as e:
            print(f"HyperLiquid get_position error: {e}")
            return {}

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            user_state = self.info.user_state(self.account.address)

            positions = []
            for position in user_state.get("assetPositions", []):
                pos = position.get("position", {})
                if float(pos.get("szi", 0)) != 0:
                    symbol = pos["coin"]
                    pos_size = float(pos["szi"])
                    entry_price = float(pos["entryPx"])
                    pnl_percent = float(pos.get("returnOnEquity", 0)) * 100

                    # Get current price
                    ticker = self.get_ticker(symbol)
                    current_price = ticker.get('last', entry_price)

                    # Calculate PnL
                    pnl = (current_price - entry_price) * abs(pos_size)

                    positions.append({
                        'symbol': symbol,
                        'quantity': pos_size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percent
                    })

            return positions

        except Exception as e:
            print(f"HyperLiquid get_all_positions error: {e}")
            return []

    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        """Execute market buy order"""
        try:
            symbol = self.normalize_symbol(symbol)

            # Get current price
            ticker = self.get_ticker(symbol)
            price = ticker.get('ask', 0)

            if price == 0:
                return {'error': 'Could not get price'}

            # Calculate quantity
            quantity = usd_amount / price

            # Get size decimals
            meta = self._get_meta()
            symbols = meta.get('universe', [])
            symbol_info = next((s for s in symbols if s['name'] == symbol), None)

            if symbol_info:
                sz_decimals = symbol_info['szDecimals']
                quantity = round(quantity, sz_decimals)

            # Place market order (limit order at best ask)
            result = self.exchange.order(
                symbol,
                True,  # is_buy
                quantity,
                price,
                {"limit": {"tif": "Ioc"}}  # Immediate-or-cancel for market-like behavior
            )

            return {
                'order_id': str(result),
                'symbol': symbol,
                'side': 'buy',
                'amount': quantity,
                'price': price,
                'cost': usd_amount,
                'status': 'filled',
                'exchange': 'hyperliquid'
            }

        except Exception as e:
            print(f"HyperLiquid market_buy error: {e}")
            return {'error': str(e)}

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        """Execute market sell order"""
        try:
            symbol = self.normalize_symbol(symbol)

            # Get current price
            ticker = self.get_ticker(symbol)
            price = ticker.get('bid', 0)

            if price == 0:
                return {'error': 'Could not get price'}

            # Get size decimals
            meta = self._get_meta()
            symbols = meta.get('universe', [])
            symbol_info = next((s for s in symbols if s['name'] == symbol), None)

            if symbol_info:
                sz_decimals = symbol_info['szDecimals']
                quantity = round(quantity, sz_decimals)

            # Place market sell order
            result = self.exchange.order(
                symbol,
                False,  # is_buy = False for sell
                quantity,
                price,
                {"limit": {"tif": "Ioc"}}
            )

            return {
                'order_id': str(result),
                'symbol': symbol,
                'side': 'sell',
                'amount': quantity,
                'price': price,
                'cost': quantity * price,
                'status': 'filled',
                'exchange': 'hyperliquid'
            }

        except Exception as e:
            print(f"HyperLiquid market_sell error: {e}")
            return {'error': str(e)}

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """Get order book"""
        try:
            symbol = self.normalize_symbol(symbol)

            response = requests.post(
                self.base_url,
                headers={'Content-Type': 'application/json'},
                json={
                    'type': 'l2Book',
                    'coin': symbol
                },
                timeout=10
            )

            if response.status_code != 200:
                return {}

            l2_data = response.json()
            levels = l2_data.get('levels', [])

            if len(levels) < 2:
                return {}

            # Format: [[price, quantity], ...]
            bids = [[float(level['px']), float(level['sz'])] for level in levels[0][:limit]]
            asks = [[float(level['px']), float(level['sz'])] for level in levels[1][:limit]]

            return {
                'bids': bids,
                'asks': asks,
                'timestamp': int(time.time() * 1000)
            }

        except Exception as e:
            print(f"HyperLiquid get_order_book error: {e}")
            return {}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        try:
            symbol = self.normalize_symbol(symbol)
            self.exchange.cancel(symbol, int(order_id))
            return True
        except Exception as e:
            print(f"HyperLiquid cancel_order error: {e}")
            return False

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get status of a specific order"""
        try:
            symbol = self.normalize_symbol(symbol)

            # Get order history
            open_orders = self.info.open_orders(self.account.address)

            for order in open_orders:
                if str(order.get('oid')) == order_id and order.get('coin') == symbol:
                    return {
                        'order_id': order_id,
                        'symbol': symbol,
                        'status': 'open',
                        'side': 'buy' if order.get('side') == 'B' else 'sell',
                        'price': float(order.get('limitPx', 0)),
                        'quantity': float(order.get('sz', 0)),
                        'filled': float(order.get('sz', 0)) - float(order.get('szLeft', 0))
                    }

            return {'order_id': order_id, 'status': 'not_found'}

        except Exception as e:
            print(f"HyperLiquid get_order_status error: {e}")
            return {'error': str(e)}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (HyperLiquid specific)"""
        try:
            symbol = self.normalize_symbol(symbol)
            self.exchange.update_leverage(leverage, symbol, is_cross=True)
            print(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            print(f"HyperLiquid set_leverage error: {e}")
            return False
