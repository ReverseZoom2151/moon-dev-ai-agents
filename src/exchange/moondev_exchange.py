"""
Moon Dev API Exchange Implementation
Data-only source for trading signals and market analytics
Specialized for liquidations, funding rates, open interest, and copy trading signals
"""

# Standard library imports
import os

# Third-party imports
import pandas as pd
import requests

# Standard library from imports
from typing import Dict, List, Optional

# Local from imports
from .base_exchange import BaseExchange


class MoonDevExchange(BaseExchange):
    """Moon Dev API data source (signals, liquidations, funding, OI)"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv('MOONDEV_API_KEY', '')

        self.base_url = "http://api.moondev.com:8000"
        self.headers = {'X-API-Key': self.api_key} if self.api_key else {}

        # Session for connection pooling
        self.session = requests.Session()

    def get_liquidation_data(self, limit: int = 10000) -> Optional[pd.DataFrame]:
        """
        Get historical liquidation data

        Args:
            limit: Number of most recent rows to fetch (default 10000)

        Returns:
            DataFrame with liquidation events or None on error
        """
        try:
            url = f'{self.base_url}/files/liq_data.csv'
            if limit:
                url += f'?limit={limit}'

            response = self.session.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                print(f"MoonDev API liquidation error: {response.status_code}")
                return None

            # Parse CSV from response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return df

        except Exception as e:
            print(f"MoonDev get_liquidation_data error: {e}")
            return None

    def get_funding_data(self) -> Optional[pd.DataFrame]:
        """
        Get current funding rate data

        Returns:
            DataFrame with funding rates across symbols or None on error
        """
        try:
            url = f'{self.base_url}/files/funding.csv'

            response = self.session.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                print(f"MoonDev API funding error: {response.status_code}")
                return None

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return df

        except Exception as e:
            print(f"MoonDev get_funding_data error: {e}")
            return None

    def get_open_interest_data(self) -> Optional[pd.DataFrame]:
        """
        Get detailed open interest data (per-token)

        Returns:
            DataFrame with OI data or None on error
        """
        try:
            url = f'{self.base_url}/files/oi_data.csv'

            response = self.session.get(url, headers=self.headers, timeout=30, stream=True)

            if response.status_code != 200:
                print(f"MoonDev API OI error: {response.status_code}")
                return None

            # Use streaming for large files
            from io import BytesIO
            content = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content.write(chunk)

            content.seek(0)
            df = pd.read_csv(content)

            return df

        except Exception as e:
            print(f"MoonDev get_open_interest_data error: {e}")
            return None

    def get_open_interest_total(self) -> Optional[pd.DataFrame]:
        """
        Get total open interest data (combined ETH & BTC)

        Returns:
            DataFrame with total OI or None on error
        """
        try:
            url = f'{self.base_url}/files/oi_total.csv'

            response = self.session.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                print(f"MoonDev API OI total error: {response.status_code}")
                return None

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return df

        except Exception as e:
            print(f"MoonDev get_open_interest_total error: {e}")
            return None

    def get_token_addresses(self) -> Optional[pd.DataFrame]:
        """
        Get new Solana token launches

        Returns:
            DataFrame with token addresses or None on error
        """
        try:
            url = f'{self.base_url}/files/token_addresses.csv'

            response = self.session.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                print(f"MoonDev API token addresses error: {response.status_code}")
                return None

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return df

        except Exception as e:
            print(f"MoonDev get_token_addresses error: {e}")
            return None

    def get_copybot_follow_list(self) -> Optional[pd.DataFrame]:
        """
        Get copy trading follow list (requires API key)

        Returns:
            DataFrame with follow list or None on error
        """
        try:
            if not self.api_key:
                print("MoonDev: API key required for copybot endpoints")
                return None

            url = f'{self.base_url}/copybot/data/follow_list'

            response = self.session.get(url, headers=self.headers, timeout=30)

            if response.status_code == 403:
                print("MoonDev: Invalid API key or insufficient permissions")
                return None

            if response.status_code != 200:
                print(f"MoonDev API follow list error: {response.status_code}")
                return None

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return df

        except Exception as e:
            print(f"MoonDev get_copybot_follow_list error: {e}")
            return None

    def get_copybot_recent_txs(self) -> Optional[pd.DataFrame]:
        """
        Get recent copy trading transactions (requires API key)

        Returns:
            DataFrame with recent transactions or None on error
        """
        try:
            if not self.api_key:
                print("MoonDev: API key required for copybot endpoints")
                return None

            url = f'{self.base_url}/copybot/data/recent_txs'

            response = self.session.get(url, headers=self.headers, timeout=30)

            if response.status_code == 403:
                print("MoonDev: Invalid API key or insufficient permissions")
                return None

            if response.status_code != 200:
                print(f"MoonDev API recent txs error: {response.status_code}")
                return None

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return df

        except Exception as e:
            print(f"MoonDev get_copybot_recent_txs error: {e}")
            return None

    def supports_symbol(self, symbol: str) -> bool:
        """MoonDev API doesn't have symbol-specific support checking"""
        return True  # All symbols potentially have liquidation/funding data

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """MoonDev API doesn't provide OHLCV data"""
        print("MoonDev: OHLCV data not supported (use for signals/analytics only)")
        return []

    def get_ticker(self, symbol: str) -> Dict:
        """MoonDev API doesn't provide ticker data"""
        print("MoonDev: Ticker data not supported (use for signals/analytics only)")
        return {}

    # ============================================================================
    # DATA-ONLY SOURCE: Trading methods not supported
    # ============================================================================

    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def get_balance(self, asset: str = None) -> Dict:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def get_position(self, symbol: str) -> Dict:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def get_all_positions(self) -> List[Dict]:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        raise NotImplementedError("MoonDev is a data-only source. Order book not available.")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        raise NotImplementedError("MoonDev is a data-only source. Trading not supported.")

    def normalize_symbol(self, symbol: str) -> str:
        """MoonDev uses standard symbols"""
        return symbol.upper()
