"""
Multi-Exchange Manager V3
Unified interface for trading across multiple exchanges simultaneously
Supports:
- Trading: Binance (spot), Bitfinex (spot), HyperLiquid (perps), Jupiter (Solana DEX), Solana (direct)
- Data: CoinGecko (10K+ tokens, FREE), Birdeye (Solana tokens, PAID), MoonDev (signals/analytics)
"""

# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd

# Standard library from imports
from typing import Dict, List, Optional, Union

# Third-party from imports
from dotenv import load_dotenv
from termcolor import colored, cprint

# Load environment variables
load_dotenv()


class MultiExchangeManager:
    """
    Advanced exchange manager supporting multiple exchanges simultaneously
    Smart order routing based on token/symbol availability
    """

    # Token/Symbol to Exchange mapping
    TOKEN_EXCHANGE_MAP = {
        # Solana tokens (token addresses)
        'solana': ['token_addresses'],  # Will be populated dynamically

        # HyperLiquid perpetuals
        'hyperliquid': ['BTC', 'ETH', 'SOL', 'DOGE', 'PEPE', 'WIF', 'BONK'],

        # Binance spot pairs
        'binance': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
                    'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT'],

        # Bitfinex spot pairs (uses USD not USDT)
        'bitfinex': ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOGE/USD',
                     'MATIC/USD', 'DOT/USD', 'AVAX/USD', 'LINK/USD']
    }

    def __init__(self, active_exchanges: List[str] = None, default_exchange: str = None):
        """
        Initialize multi-exchange manager

        Args:
            active_exchanges: List of exchanges to enable ['binance', 'bitfinex', 'solana', 'hyperliquid']
                             If None, will check environment for available credentials
            default_exchange: Preferred exchange for ambiguous orders (defaults to first available)
        """
        from src import config

        self.active_exchanges: Dict[str, any] = {}
        self.default_exchange = default_exchange or config.EXCHANGE

        # Auto-detect available exchanges if not specified
        if active_exchanges is None:
            active_exchanges = self._detect_available_exchanges()

        # Initialize requested exchanges
        self._initialize_exchanges(active_exchanges)

        if not self.active_exchanges:
            cprint("⚠️ No exchanges initialized! Check your API credentials.", "yellow")
        else:
            cprint(f"✅ Initialized {len(self.active_exchanges)} exchange(s): {', '.join(self.active_exchanges.keys())}", "green")

    def _detect_available_exchanges(self) -> List[str]:
        """Auto-detect which exchanges have credentials configured"""
        available = []

        # Check for Binance credentials
        if os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_API_SECRET'):
            available.append('binance')

        # Check for Bitfinex credentials
        if os.getenv('BITFINEX_API_KEY') and os.getenv('BITFINEX_API_SECRET'):
            available.append('bitfinex')

        # Check for Solana credentials
        if os.getenv('SOLANA_PRIVATE_KEY') or os.getenv('address'):
            available.append('solana')

        # Check for HyperLiquid credentials (perpetuals trading)
        if os.getenv('HYPER_LIQUID_KEY'):
            available.append('hyperliquid')

        # Check for Jupiter/Solana credentials (DEX swaps)
        if os.getenv('SOLANA_PRIVATE_KEY') or os.getenv('address'):
            available.append('jupiter')

        # CoinGecko always available (free tier works without API key)
        available.append('coingecko')

        # Check for Birdeye credentials (Solana token data)
        if os.getenv('BIRDEYE_API_KEY'):
            available.append('birdeye')

        # Check for MoonDev API (signals and analytics)
        if os.getenv('MOONDEV_API_KEY'):
            available.append('moondev')

        return available

    def _initialize_exchanges(self, exchange_list: List[str]):
        """Initialize all requested exchanges"""
        for exchange_name in exchange_list:
            exchange_name = exchange_name.lower()

            try:
                if exchange_name == 'binance':
                    from src.exchange.binance_exchange import BinanceExchange
                    self.active_exchanges['binance'] = BinanceExchange()

                elif exchange_name == 'bitfinex':
                    from src.exchange.bitfinex_exchange import BitfinexExchange
                    self.active_exchanges['bitfinex'] = BitfinexExchange()

                elif exchange_name == 'solana':
                    from src import nice_funcs as solana
                    self.active_exchanges['solana'] = solana

                elif exchange_name == 'hyperliquid':
                    # Use new exchange adapter (replaces old module approach)
                    from src.exchange.hyperliquid_exchange import HyperLiquidExchange
                    try:
                        self.active_exchanges['hyperliquid'] = HyperLiquidExchange()
                    except ValueError as e:
                        cprint(f"⚠️ HyperLiquid: {str(e)}", "yellow")

                elif exchange_name == 'jupiter':
                    from src.exchange.jupiter_exchange import JupiterExchange
                    self.active_exchanges['jupiter'] = JupiterExchange()

                elif exchange_name == 'coingecko':
                    from src.exchange.coingecko_exchange import CoinGeckoExchange
                    self.active_exchanges['coingecko'] = CoinGeckoExchange()

                elif exchange_name == 'birdeye':
                    from src.exchange.birdeye_exchange import BirdeyeExchange
                    self.active_exchanges['birdeye'] = BirdeyeExchange()

                elif exchange_name == 'moondev':
                    from src.exchange.moondev_exchange import MoonDevExchange
                    self.active_exchanges['moondev'] = MoonDevExchange()

            except Exception as e:
                cprint(f"⚠️ Failed to initialize {exchange_name}: {str(e)}", "yellow")

    def route_order(self, token_or_symbol: str, action: str, amount: float) -> Dict:
        """
        Smart order routing - automatically selects best exchange for the token/symbol

        Args:
            token_or_symbol: Token address (Solana) or trading symbol (CEX)
            action: 'buy' or 'sell'
            amount: USD amount to trade (for buy) or quantity/percentage (for sell)

        Returns:
            Order result with exchange information
        """
        # Determine which exchange supports this token/symbol
        target_exchange = self._determine_exchange(token_or_symbol)

        if not target_exchange:
            return {
                'error': f"No active exchange supports {token_or_symbol}",
                'token': token_or_symbol
            }

        cprint(f"📍 Routing {action.upper()} order for {token_or_symbol} to {target_exchange}", "cyan")

        # Execute on the appropriate exchange
        if action.lower() == 'buy':
            return self.market_buy(token_or_symbol, amount, exchange=target_exchange)
        elif action.lower() == 'sell':
            return self.market_sell(token_or_symbol, amount, exchange=target_exchange)
        else:
            return {'error': f"Unknown action: {action}"}

    def _determine_exchange(self, token_or_symbol: str) -> Optional[str]:
        """Determine which exchange to use for this token/symbol"""
        # Check if it's a Solana token address (long alphanumeric string)
        if len(token_or_symbol) > 32 and token_or_symbol.isalnum():
            if 'solana' in self.active_exchanges:
                return 'solana'

        # Check HyperLiquid symbols (uppercase, no /)
        if '/' not in token_or_symbol and token_or_symbol.upper() in self.TOKEN_EXCHANGE_MAP['hyperliquid']:
            if 'hyperliquid' in self.active_exchanges:
                return 'hyperliquid'

        # Check Binance pairs (contains /USDT)
        if '/USDT' in token_or_symbol.upper() or token_or_symbol.upper() in [s.split('/')[0] for s in self.TOKEN_EXCHANGE_MAP['binance']]:
            if 'binance' in self.active_exchanges:
                return 'binance'

        # Check Bitfinex pairs (contains /USD but not /USDT)
        if '/USD' in token_or_symbol.upper() and '/USDT' not in token_or_symbol.upper():
            if 'bitfinex' in self.active_exchanges:
                return 'bitfinex'

        # Fallback to default exchange
        if self.default_exchange in self.active_exchanges:
            return self.default_exchange

        # Last resort: return first available exchange
        if self.active_exchanges:
            return list(self.active_exchanges.keys())[0]

        return None

    def market_buy(self, token_or_symbol: str, usd_amount: float, exchange: str = None) -> Dict:
        """
        Execute market buy on specified or auto-selected exchange

        Args:
            token_or_symbol: Token address or trading symbol
            usd_amount: USD amount to buy
            exchange: Optional exchange override

        Returns:
            Order result
        """
        exchange = exchange or self._determine_exchange(token_or_symbol)

        if not exchange or exchange not in self.active_exchanges:
            return {'error': f"Exchange {exchange} not available"}

        try:
            if exchange in ['binance', 'bitfinex']:
                # Use BaseExchange interface
                return self.active_exchanges[exchange].market_buy(token_or_symbol, usd_amount)

            elif exchange == 'hyperliquid':
                hl_data = self.active_exchanges['hyperliquid']
                return hl_data['module'].market_buy(token_or_symbol, usd_amount, hl_data['account'])

            elif exchange == 'solana':
                return self.active_exchanges['solana'].market_buy(token_or_symbol, usd_amount)

        except Exception as e:
            return {'error': str(e), 'exchange': exchange}

    def market_sell(self, token_or_symbol: str, amount: float, exchange: str = None) -> Dict:
        """
        Execute market sell on specified or auto-selected exchange

        Args:
            token_or_symbol: Token address or trading symbol
            amount: Quantity to sell (or percentage for Solana)
            exchange: Optional exchange override

        Returns:
            Order result
        """
        exchange = exchange or self._determine_exchange(token_or_symbol)

        if not exchange or exchange not in self.active_exchanges:
            return {'error': f"Exchange {exchange} not available"}

        try:
            if exchange in ['binance', 'bitfinex']:
                # Use BaseExchange interface
                return self.active_exchanges[exchange].market_sell(token_or_symbol, amount)

            elif exchange == 'hyperliquid':
                hl_data = self.active_exchanges['hyperliquid']
                return hl_data['module'].market_sell(token_or_symbol, amount, hl_data['account'])

            elif exchange == 'solana':
                # Solana uses percentage (0-100)
                return self.active_exchanges['solana'].market_sell(token_or_symbol, amount)

        except Exception as e:
            return {'error': str(e), 'exchange': exchange}

    def get_aggregated_positions(self) -> pd.DataFrame:
        """
        Get all positions across all active exchanges

        Returns:
            DataFrame with columns: [exchange, symbol, quantity, current_price, value_usd, entry_price, pnl]
        """
        all_positions = []

        for exchange_name, exchange_obj in self.active_exchanges.items():
            try:
                if exchange_name in ['binance', 'bitfinex']:
                    positions = exchange_obj.get_all_positions()
                    for pos in positions:
                        all_positions.append({
                            'exchange': exchange_name,
                            'symbol': pos['symbol'],
                            'quantity': pos['quantity'],
                            'current_price': pos['current_price'],
                            'value_usd': pos['value_usd'],
                            'entry_price': None,
                            'pnl': None
                        })

                elif exchange_name == 'hyperliquid':
                    hl_data = exchange_obj
                    positions = hl_data['module'].get_all_positions(hl_data['account'])
                    for pos in positions:
                        all_positions.append({
                            'exchange': 'hyperliquid',
                            'symbol': pos.get('symbol', ''),
                            'quantity': pos.get('size', 0),
                            'current_price': pos.get('current_price', 0),
                            'value_usd': pos.get('value_usd', 0),
                            'entry_price': pos.get('entry_price', None),
                            'pnl': pos.get('pnl', None)
                        })

                elif exchange_name == 'solana':
                    from src.config import address, EXCLUDED_TOKENS
                    holdings = exchange_obj.fetch_wallet_holdings_og(address)
                    if holdings is not None and not holdings.empty:
                        for _, row in holdings.iterrows():
                            token = row.get('token_address', '')
                            if token not in EXCLUDED_TOKENS and row.get('value_usd', 0) > 0:
                                all_positions.append({
                                    'exchange': 'solana',
                                    'symbol': token,
                                    'quantity': row.get('amount', 0),
                                    'current_price': row.get('price', 0),
                                    'value_usd': row.get('value_usd', 0),
                                    'entry_price': None,
                                    'pnl': None
                                })

            except Exception as e:
                cprint(f"⚠️ Error fetching positions from {exchange_name}: {e}", "yellow")

        return pd.DataFrame(all_positions)

    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value across all exchanges"""
        positions_df = self.get_aggregated_positions()
        if positions_df.empty:
            return 0.0
        return positions_df['value_usd'].sum()

    def get_balance(self, exchange: str = None) -> Dict:
        """
        Get available balance(s)

        Args:
            exchange: Specific exchange, or None for all

        Returns:
            Dict of balances by exchange
        """
        if exchange:
            if exchange not in self.active_exchanges:
                return {exchange: 0}

            try:
                if exchange in ['binance', 'bitfinex']:
                    bal = self.active_exchanges[exchange].get_balance('USDT' if exchange == 'binance' else 'USD')
                    return {exchange: bal.get('USDT' if exchange == 'binance' else 'USD', {}).get('free', 0)}

                elif exchange == 'hyperliquid':
                    hl_data = self.active_exchanges['hyperliquid']
                    return {exchange: hl_data['module'].get_balance(hl_data['account'])}

                elif exchange == 'solana':
                    from src.config import USDC_ADDRESS
                    return {exchange: self.active_exchanges['solana'].get_token_balance_usd(USDC_ADDRESS)}
            except:
                return {exchange: 0}

        # Get all balances
        balances = {}
        for exch in self.active_exchanges.keys():
            balances.update(self.get_balance(exch))
        return balances

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Unified OHLCV data fetching with intelligent routing and fallback

        Args:
            symbol: Token address (Solana) or trading symbol (CEX)
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
            Or None if no data available
        """
        # Priority order: Binance → Bitfinex → HyperLiquid → CoinGecko → Birdeye

        # Try Binance first for USDT pairs (spot trading)
        if 'binance' in self.active_exchanges:
            try:
                test_symbol = symbol if '/' in symbol else f"{symbol}/USDT"
                binance_exchange = self.active_exchanges['binance']

                if binance_exchange.supports_symbol(test_symbol):
                    cprint(f"📊 Fetching {test_symbol} from Binance (FREE, spot)", "cyan")
                    ohlcv_data = binance_exchange.get_ohlcv(test_symbol, timeframe, limit)

                    if ohlcv_data:
                        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        cprint(f"✅ Got {len(df)} candles from Binance", "green")
                        return df
            except Exception as e:
                cprint(f"⚠️ Binance failed: {e}", "yellow")

        # Try Bitfinex for USD pairs (spot trading)
        if 'bitfinex' in self.active_exchanges:
            try:
                test_symbol = symbol if '/' in symbol else f"{symbol}/USD"
                bitfinex_exchange = self.active_exchanges['bitfinex']

                if bitfinex_exchange.supports_symbol(test_symbol):
                    cprint(f"📊 Fetching {test_symbol} from Bitfinex (FREE, spot)", "cyan")
                    ohlcv_data = bitfinex_exchange.get_ohlcv(test_symbol, timeframe, limit)

                    if ohlcv_data:
                        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        cprint(f"✅ Got {len(df)} candles from Bitfinex", "green")
                        return df
            except Exception as e:
                cprint(f"⚠️ Bitfinex failed: {e}", "yellow")

        # Try HyperLiquid for perpetuals (100+ symbols, high quality data)
        if 'hyperliquid' in self.active_exchanges:
            try:
                # HyperLiquid uses symbols without slashes (BTC, ETH, etc.)
                clean_symbol = symbol.replace('/USD', '').replace('/USDT', '')
                hyperliquid_exchange = self.active_exchanges['hyperliquid']

                if hyperliquid_exchange.supports_symbol(clean_symbol):
                    cprint(f"📊 Fetching {clean_symbol} from HyperLiquid (FREE, perps)", "cyan")
                    ohlcv_data = hyperliquid_exchange.get_ohlcv(clean_symbol, timeframe, limit)

                    if ohlcv_data:
                        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        cprint(f"✅ Got {len(df)} candles from HyperLiquid", "green")
                        return df
            except Exception as e:
                cprint(f"⚠️ HyperLiquid failed: {e}", "yellow")

        # Try CoinGecko for general token data (10,000+ tokens, FREE)
        if 'coingecko' in self.active_exchanges:
            try:
                coingecko_exchange = self.active_exchanges['coingecko']

                # CoinGecko works with trading symbols (e.g., 'BTC', 'ETH', 'SOL')
                # Skip if it looks like a Solana address (use Birdeye for those)
                if not (len(symbol) > 32 and symbol.isalnum()):
                    cprint(f"📊 Fetching {symbol} from CoinGecko (FREE)", "cyan")
                    ohlcv_data = coingecko_exchange.get_ohlcv(symbol, timeframe, limit)

                    if ohlcv_data:
                        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        cprint(f"✅ Got {len(df)} candles from CoinGecko", "green")
                        return df
            except Exception as e:
                cprint(f"⚠️ CoinGecko failed: {e}", "yellow")

        # Final fallback: Try Birdeye for Solana tokens (PAID)
        if 'birdeye' in self.active_exchanges:
            try:
                birdeye_exchange = self.active_exchanges['birdeye']

                # Birdeye is for Solana token addresses (long alphanumeric strings)
                if len(symbol) > 32 and symbol.isalnum():
                    cprint(f"📊 Fetching Solana token from Birdeye (PAID)", "cyan")
                    ohlcv_data = birdeye_exchange.get_ohlcv(symbol, timeframe, limit)

                    if ohlcv_data:
                        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        cprint(f"✅ Got {len(df)} candles from Birdeye", "green")
                        return df
            except Exception as e:
                cprint(f"⚠️ Birdeye failed: {e}", "yellow")

        cprint(f"❌ No data available for {symbol} from any source", "red")
        return None

    # ============================================================================
    # MOON DEV API - Specialized Signal/Analytics Methods
    # ============================================================================

    def get_liquidations(self, limit: int = 10000) -> Optional[pd.DataFrame]:
        """Get liquidation data from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_liquidation_data(limit)
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def get_funding_rates(self) -> Optional[pd.DataFrame]:
        """Get funding rate data from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_funding_data()
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def get_open_interest(self) -> Optional[pd.DataFrame]:
        """Get detailed open interest data from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_open_interest_data()
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def get_open_interest_total(self) -> Optional[pd.DataFrame]:
        """Get total open interest data from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_open_interest_total()
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def get_new_token_addresses(self) -> Optional[pd.DataFrame]:
        """Get new Solana token launches from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_token_addresses()
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def get_copybot_signals(self) -> Optional[pd.DataFrame]:
        """Get copy trading follow list from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_copybot_follow_list()
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def get_copybot_recent_trades(self) -> Optional[pd.DataFrame]:
        """Get recent copy trading transactions from Moon Dev API"""
        if 'moondev' in self.active_exchanges:
            return self.active_exchanges['moondev'].get_copybot_recent_txs()
        cprint("⚠️ MoonDev API not available (set MOONDEV_API_KEY)", "yellow")
        return None

    def __str__(self):
        return f"MultiExchangeManager(exchanges={list(self.active_exchanges.keys())})"

    def __repr__(self):
        return self.__str__()
