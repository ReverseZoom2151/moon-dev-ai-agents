"""
Binance Exchange Implementation
Supports spot trading on Binance
"""

# Standard library imports
import os

# Third-party imports
import ccxt

# Standard library from imports
from typing import Dict, List, Optional

# Local from imports
try:
    from .base_exchange import BaseExchange
except ImportError:
    from base_exchange import BaseExchange


class BinanceExchange(BaseExchange):
    """Binance exchange implementation using CCXT"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Get credentials from env if not provided
        self.api_key = api_key or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')

        # Initialize CCXT Binance client
        self.client = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # spot, future, margin
            }
        })

        # Cache for market info
        self._markets = None

    def _load_markets(self):
        """Load market information (cached)"""
        if self._markets is None:
            self._markets = self.client.load_markets()
        return self._markets

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Binance format
        BTC/USDT, ETH/USDT, etc.
        """
        if '/' not in symbol:
            # Assume USDT pair if not specified
            symbol = f"{symbol}/USDT"
        return symbol.upper()

    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        """Execute market buy order"""
        try:
            symbol = self.normalize_symbol(symbol)

            # Get current price to calculate quantity
            ticker = self.client.fetch_ticker(symbol)
            current_price = ticker['last']
            quantity = usd_amount / current_price

            # Execute market order
            order = self.client.create_market_buy_order(symbol, quantity)

            return {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'buy',
                'amount': quantity,
                'filled': order.get('filled', quantity),
                'price': order.get('average', current_price),
                'cost': order.get('cost', usd_amount),
                'status': order['status'],
                'exchange': 'binance',
                'raw': order
            }
        except Exception as e:
            print(f"Binance market_buy error: {e}")
            return {
                'error': str(e),
                'exchange': 'binance',
                'symbol': symbol
            }

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        """Execute market sell order"""
        try:
            symbol = self.normalize_symbol(symbol)

            # Execute market order
            order = self.client.create_market_sell_order(symbol, quantity)

            return {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'sell',
                'amount': quantity,
                'filled': order.get('filled', quantity),
                'price': order.get('average', 0),
                'cost': order.get('cost', 0),
                'status': order['status'],
                'exchange': 'binance',
                'raw': order
            }
        except Exception as e:
            print(f"Binance market_sell error: {e}")
            return {
                'error': str(e),
                'exchange': 'binance',
                'symbol': symbol
            }

    def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        try:
            balance = self.client.fetch_balance()

            if asset:
                asset = asset.upper()
                if asset in balance:
                    return {
                        asset: {
                            'free': balance[asset]['free'],
                            'locked': balance[asset]['used'],
                            'total': balance[asset]['total']
                        }
                    }
                return {asset: {'free': 0, 'locked': 0, 'total': 0}}

            # Return all non-zero balances
            result = {}
            for currency, bal in balance.items():
                if isinstance(bal, dict) and bal.get('total', 0) > 0:
                    result[currency] = {
                        'free': bal['free'],
                        'locked': bal['used'],
                        'total': bal['total']
                    }
            return result
        except Exception as e:
            print(f"Binance get_balance error: {e}")
            return {}

    def get_position(self, symbol: str) -> Dict:
        """Get current position for a symbol"""
        try:
            symbol = self.normalize_symbol(symbol)
            base_currency = symbol.split('/')[0]

            # Get balance of base currency
            balance = self.get_balance(base_currency)
            quantity = balance.get(base_currency, {}).get('total', 0)

            if quantity == 0:
                return {
                    'symbol': symbol,
                    'quantity': 0,
                    'entry_price': 0,
                    'current_price': 0,
                    'pnl': 0,
                    'pnl_percentage': 0
                }

            # Get current price
            ticker = self.client.fetch_ticker(symbol)
            current_price = ticker['last']

            # Note: Binance spot doesn't track entry price, so we can't calculate true PnL
            return {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': None,  # Not available in spot
                'current_price': current_price,
                'pnl': None,
                'pnl_percentage': None,
                'value_usd': quantity * current_price
            }
        except Exception as e:
            print(f"Binance get_position error: {e}")
            return {}

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions (balances in spot trading)"""
        try:
            balances = self.get_balance()
            positions = []

            for asset, bal in balances.items():
                if bal['total'] > 0 and asset != 'USDT':
                    try:
                        symbol = f"{asset}/USDT"
                        ticker = self.client.fetch_ticker(symbol)
                        positions.append({
                            'symbol': symbol,
                            'quantity': bal['total'],
                            'current_price': ticker['last'],
                            'value_usd': bal['total'] * ticker['last']
                        })
                    except:
                        pass  # Skip if ticker not available

            return positions
        except Exception as e:
            print(f"Binance get_all_positions error: {e}")
            return []

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data"""
        try:
            symbol = self.normalize_symbol(symbol)
            ticker = self.client.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            print(f"Binance get_ticker error: {e}")
            return {}

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """Get order book"""
        try:
            symbol = self.normalize_symbol(symbol)
            orderbook = self.client.fetch_order_book(symbol, limit)

            return {
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'timestamp': orderbook['timestamp']
            }
        except Exception as e:
            print(f"Binance get_order_book error: {e}")
            return {'bids': [], 'asks': [], 'timestamp': 0}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        try:
            symbol = self.normalize_symbol(symbol)
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Binance cancel_order error: {e}")
            return False

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get status of a specific order"""
        try:
            symbol = self.normalize_symbol(symbol)
            order = self.client.fetch_order(order_id, symbol)

            return {
                'order_id': order['id'],
                'symbol': symbol,
                'status': order['status'],
                'side': order['side'],
                'amount': order['amount'],
                'filled': order['filled'],
                'price': order.get('average', 0),
                'cost': order.get('cost', 0)
            }
        except Exception as e:
            print(f"Binance get_order_status error: {e}")
            return {}

    def supports_symbol(self, symbol: str) -> bool:
        """Check if exchange supports this trading pair"""
        try:
            markets = self._load_markets()
            symbol = self.normalize_symbol(symbol)
            return symbol in markets
        except:
            return False

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Fetch OHLCV (candlestick) data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Number of candles to fetch (max 1000)

        Returns:
            List of OHLCV data: [[timestamp, open, high, low, close, volume], ...]
            Or empty list on error
        """
        try:
            symbol = self.normalize_symbol(symbol)

            # Fetch OHLCV data using CCXT
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            return ohlcv
        except Exception as e:
            print(f"Binance get_ohlcv error for {symbol}: {e}")
            return []
