"""
Base Exchange Interface
Provides a unified abstraction for all trading exchanges
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseExchange(ABC):
    """Abstract base class for all exchange implementations"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = self.__class__.__name__.replace('Exchange', '')

    @abstractmethod
    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        """
        Execute market buy order

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'ETHUSDT')
            usd_amount: Amount in USD to buy

        Returns:
            Dict with order details:
            {
                'order_id': str,
                'symbol': str,
                'side': 'buy',
                'amount': float,
                'filled': float,
                'price': float,
                'cost': float,
                'status': str,
                'exchange': str
            }
        """
        pass

    @abstractmethod
    def market_sell(self, symbol: str, quantity: float) -> Dict:
        """
        Execute market sell order

        Args:
            symbol: Trading pair symbol
            quantity: Amount of asset to sell

        Returns:
            Dict with order details (same format as market_buy)
        """
        pass

    @abstractmethod
    def get_balance(self, asset: str = None) -> Dict:
        """
        Get account balance

        Args:
            asset: Specific asset to query (None = all assets)

        Returns:
            Dict: {'asset': {'free': float, 'locked': float, 'total': float}}
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict:
        """
        Get current position for a symbol

        Returns:
            Dict: {
                'symbol': str,
                'quantity': float,
                'entry_price': float,
                'current_price': float,
                'pnl': float,
                'pnl_percentage': float
            }
        """
        pass

    @abstractmethod
    def get_all_positions(self) -> List[Dict]:
        """Get all open positions across all symbols"""
        pass

    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker data

        Returns:
            Dict: {
                'symbol': str,
                'last': float,
                'bid': float,
                'ask': float,
                'volume': float,
                'timestamp': int
            }
        """
        pass

    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """
        Get order book

        Returns:
            Dict: {
                'bids': [(price, quantity), ...],
                'asks': [(price, quantity), ...],
                'timestamp': int
            }
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get status of a specific order"""
        pass

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        """Check if exchange supports this trading pair"""
        pass

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for this exchange
        Override if exchange has specific format requirements
        """
        return symbol

    def __str__(self):
        return f"{self.exchange_name}Exchange"

    def __repr__(self):
        return f"<{self.exchange_name}Exchange>"
