"""
Bitfinex Exchange Implementation
Supports spot and margin trading on Bitfinex
"""

# Standard library imports
import os

# Third-party imports
import ccxt

# Standard library from imports
from typing import Dict, List, Optional

# Local from imports
from .base_exchange import BaseExchange


class BitfinexExchange(BaseExchange):
    """Bitfinex exchange implementation using CCXT"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Get credentials from env if not provided
        self.api_key = api_key or os.getenv('BITFINEX_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BITFINEX_API_SECRET', '')

        # Initialize CCXT Bitfinex client
        self.client = ccxt.bitfinex({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
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
        Normalize symbol to Bitfinex format
        BTC/USD, ETH/USD, etc. (Bitfinex uses USD not USDT)
        """
        if '/' not in symbol:
            # Assume USD pair if not specified
            symbol = f"{symbol}/USD"

        # Bitfinex uses USD instead of USDT
        symbol = symbol.replace('/USDT', '/USD')

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
                'exchange': 'bitfinex',
                'raw': order
            }
        except Exception as e:
            print(f"Bitfinex market_buy error: {e}")
            return {
                'error': str(e),
                'exchange': 'bitfinex',
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
                'exchange': 'bitfinex',
                'raw': order
            }
        except Exception as e:
            print(f"Bitfinex market_sell error: {e}")
            return {
                'error': str(e),
                'exchange': 'bitfinex',
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
            print(f"Bitfinex get_balance error: {e}")
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
            print(f"Bitfinex get_position error: {e}")
            return {}

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions (balances in spot trading)"""
        try:
            balances = self.get_balance()
            positions = []

            for asset, bal in balances.items():
                if bal['total'] > 0 and asset not in ['USD', 'USDT']:
                    try:
                        symbol = f"{asset}/USD"
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
            print(f"Bitfinex get_all_positions error: {e}")
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
            print(f"Bitfinex get_ticker error: {e}")
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
            print(f"Bitfinex get_order_book error: {e}")
            return {'bids': [], 'asks': [], 'timestamp': 0}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        try:
            symbol = self.normalize_symbol(symbol)
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Bitfinex cancel_order error: {e}")
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
            print(f"Bitfinex get_order_status error: {e}")
            return {}

    def supports_symbol(self, symbol: str) -> bool:
        """Check if exchange supports this trading pair"""
        try:
            markets = self._load_markets()
            symbol = self.normalize_symbol(symbol)
            return symbol in markets
        except:
            return False
