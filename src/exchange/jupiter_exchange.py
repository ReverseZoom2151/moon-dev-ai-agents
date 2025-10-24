"""
Jupiter Exchange Implementation
Solana DEX aggregator for best swap rates
Data-only source for quotes (trading requires Solana wallet integration)
"""

# Standard library imports
import base64
import json
import os

# Third-party imports
import requests

# Standard library from imports
from typing import Dict, List, Optional

# Third-party from imports
from solana.rpc.api import Client
from solana.rpc.types import TxOpts

# Handle different solana library versions
try:
    from solana.transaction import VersionedTransaction
except (ImportError, ModuleNotFoundError):
    try:
        from solders.transaction import VersionedTransaction  # type: ignore
    except (ImportError, ModuleNotFoundError):
        # Fallback - define a placeholder if neither works
        VersionedTransaction = None
        print("⚠️ Warning: VersionedTransaction not available. Install solana-py or solders.")

from solders.keypair import Keypair  # type: ignore

# Local from imports
try:
    from .base_exchange import BaseExchange
except ImportError:
    from base_exchange import BaseExchange


class JupiterExchange(BaseExchange):
    """Jupiter DEX aggregator for Solana token swaps"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key, api_secret)

        # Jupiter doesn't need an API key, but needs Solana wallet for execution
        self.base_url = 'https://lite-api.jup.ag/swap/v1'

        # USDC mint address (quote token)
        self.USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

        # Get Solana credentials from env
        self.private_key = os.getenv('SOLANA_PRIVATE_KEY', '')
        self.rpc_endpoint = os.getenv('RPC_ENDPOINT', 'https://api.mainnet-beta.solana.com/')

        # Initialize keypair and RPC client if credentials available
        self.keypair = None
        self.rpc_client = None

        if self.private_key:
            try:
                self.keypair = Keypair.from_base58_string(self.private_key)
                self.rpc_client = Client(self.rpc_endpoint)
            except Exception as e:
                print(f"Jupiter: Could not initialize Solana wallet: {e}")

    def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 50) -> Dict:
        """
        Get swap quote from Jupiter

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in token's smallest unit (e.g., lamports for SOL, units for USDC)
            slippage_bps: Slippage in basis points (50 = 0.5%, 500 = 5%)

        Returns:
            Quote response dict with swap details
        """
        try:
            url = f"{self.base_url}/quote"
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': amount,
                'slippageBps': slippage_bps
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"Jupiter quote error: {response.status_code}")
                return {}

            return response.json()

        except Exception as e:
            print(f"Jupiter get_quote error: {e}")
            return {}

    def get_swap_transaction(self, quote_response: Dict, user_public_key: str,
                           priority_fee: int = 'auto') -> Dict:
        """
        Get swap transaction from Jupiter

        Args:
            quote_response: Quote response from get_quote()
            user_public_key: User's Solana public key
            priority_fee: Priority fee in lamports or 'auto'

        Returns:
            Transaction response dict
        """
        try:
            url = f"{self.base_url}/swap"
            headers = {"Content-Type": "application/json"}

            data = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "prioritizationFeeLamports": priority_fee
            }

            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)

            if response.status_code != 200:
                print(f"Jupiter swap transaction error: {response.status_code}")
                return {}

            return response.json()

        except Exception as e:
            print(f"Jupiter get_swap_transaction error: {e}")
            return {}

    def supports_symbol(self, symbol: str) -> bool:
        """
        Check if Jupiter supports this token (any Solana token)
        Jupiter supports all SPL tokens, so we check if it looks like a Solana address
        """
        # Solana addresses are base58 encoded, typically 32-44 characters
        return len(symbol) >= 32 and symbol.isalnum() and '/' not in symbol

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Jupiter doesn't provide OHLCV data
        This is a data-only method (not supported)
        """
        print("Jupiter: OHLCV data not supported (DEX aggregator)")
        return []

    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current price via quote
        Uses a small amount to get price without executing
        """
        try:
            # Get quote for 1 USDC worth to determine price
            amount = 1_000_000  # 1 USDC (6 decimals)

            quote = self.get_quote(
                input_mint=self.USDC_MINT,
                output_mint=symbol,
                amount=amount,
                slippage_bps=50
            )

            if not quote:
                return {}

            # Extract price information
            in_amount = int(quote.get('inAmount', 0))
            out_amount = int(quote.get('outAmount', 0))

            if out_amount == 0:
                return {}

            # Calculate price (how many output tokens per 1 USDC)
            price = in_amount / out_amount

            return {
                'symbol': symbol,
                'last': price,
                'bid': price,  # Jupiter doesn't provide bid/ask, use price
                'ask': price,
                'volume': 0,  # Not provided
                'timestamp': 0  # Not provided
            }

        except Exception as e:
            print(f"Jupiter get_ticker error: {e}")
            return {}

    def market_buy(self, symbol: str, usd_amount: float) -> Dict:
        """
        Execute market buy via Jupiter swap

        Args:
            symbol: Token mint address to buy
            usd_amount: USD amount to spend

        Returns:
            Transaction result dict
        """
        try:
            if not self.keypair or not self.rpc_client:
                return {'error': 'Solana wallet not initialized (SOLANA_PRIVATE_KEY required)'}

            # Convert USD to USDC units (6 decimals)
            amount_in_units = int(usd_amount * 1_000_000)

            print(f"💰 Converting ${usd_amount} to {amount_in_units:,} USDC units")

            # Get quote
            quote = self.get_quote(
                input_mint=self.USDC_MINT,
                output_mint=symbol,
                amount=amount_in_units,
                slippage_bps=50  # 0.5% slippage
            )

            if not quote:
                return {'error': 'Could not get quote'}

            # Get swap transaction
            tx_response = self.get_swap_transaction(
                quote_response=quote,
                user_public_key=str(self.keypair.pubkey()),
                priority_fee='auto'
            )

            if not tx_response:
                return {'error': 'Could not get swap transaction'}

            # Decode and sign transaction
            swap_tx_b64 = tx_response.get('swapTransaction')
            if not swap_tx_b64:
                return {'error': 'No swap transaction in response'}

            swap_tx = base64.b64decode(swap_tx_b64)
            tx1 = VersionedTransaction.from_bytes(swap_tx)
            tx = VersionedTransaction(tx1.message, [self.keypair])

            # Send transaction
            tx_id = self.rpc_client.send_raw_transaction(
                bytes(tx),
                TxOpts(skip_preflight=True)
            ).value

            print(f"✅ Jupiter swap executed: https://solscan.io/tx/{str(tx_id)}")

            return {
                'order_id': str(tx_id),
                'symbol': symbol,
                'side': 'buy',
                'amount': int(quote.get('outAmount', 0)),
                'price': usd_amount / int(quote.get('outAmount', 1)),
                'cost': usd_amount,
                'status': 'submitted',
                'exchange': 'jupiter',
                'tx_url': f"https://solscan.io/tx/{str(tx_id)}"
            }

        except Exception as e:
            print(f"Jupiter market_buy error: {e}")
            return {'error': str(e)}

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        """
        Execute market sell via Jupiter swap

        Args:
            symbol: Token mint address to sell
            quantity: Quantity in token's native units

        Returns:
            Transaction result dict
        """
        try:
            if not self.keypair or not self.rpc_client:
                return {'error': 'Solana wallet not initialized (SOLANA_PRIVATE_KEY required)'}

            amount = int(quantity)

            # Get quote (selling token for USDC)
            quote = self.get_quote(
                input_mint=symbol,
                output_mint=self.USDC_MINT,
                amount=amount,
                slippage_bps=50  # 0.5% slippage
            )

            if not quote:
                return {'error': 'Could not get quote'}

            # Get swap transaction
            tx_response = self.get_swap_transaction(
                quote_response=quote,
                user_public_key=str(self.keypair.pubkey()),
                priority_fee='auto'
            )

            if not tx_response:
                return {'error': 'Could not get swap transaction'}

            # Decode and sign transaction
            swap_tx_b64 = tx_response.get('swapTransaction')
            if not swap_tx_b64:
                return {'error': 'No swap transaction in response'}

            swap_tx = base64.b64decode(swap_tx_b64)
            tx1 = VersionedTransaction.from_bytes(swap_tx)
            tx = VersionedTransaction(tx1.message, [self.keypair])

            # Send transaction
            tx_id = self.rpc_client.send_raw_transaction(
                bytes(tx),
                TxOpts(skip_preflight=True)
            ).value

            print(f"✅ Jupiter swap executed: https://solscan.io/tx/{str(tx_id)}")

            # Calculate USD value received
            usdc_received = int(quote.get('outAmount', 0)) / 1_000_000

            return {
                'order_id': str(tx_id),
                'symbol': symbol,
                'side': 'sell',
                'amount': quantity,
                'price': usdc_received / quantity if quantity > 0 else 0,
                'cost': usdc_received,
                'status': 'submitted',
                'exchange': 'jupiter',
                'tx_url': f"https://solscan.io/tx/{str(tx_id)}"
            }

        except Exception as e:
            print(f"Jupiter market_sell error: {e}")
            return {'error': str(e)}

    def get_balance(self, asset: str = None) -> Dict:
        """Get wallet balance (requires RPC integration)"""
        raise NotImplementedError("Use Solana RPC directly for balance queries")

    def get_position(self, symbol: str) -> Dict:
        """DEX doesn't have positions, only wallet balances"""
        raise NotImplementedError("Jupiter is a DEX (no positions, only wallet balances)")

    def get_all_positions(self) -> List[Dict]:
        """DEX doesn't have positions"""
        raise NotImplementedError("Jupiter is a DEX (no positions, only wallet balances)")

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """Jupiter aggregates liquidity, doesn't expose order books"""
        raise NotImplementedError("Jupiter is an aggregator (no order book)")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Swaps are atomic, cannot be cancelled once submitted"""
        raise NotImplementedError("Jupiter swaps are atomic (cannot cancel)")

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Check transaction status via Solana RPC"""
        try:
            if not self.rpc_client:
                return {'error': 'RPC client not initialized'}

            # Query transaction status
            result = self.rpc_client.get_transaction(order_id)

            if result.value:
                return {
                    'order_id': order_id,
                    'status': 'confirmed',
                    'tx_url': f"https://solscan.io/tx/{order_id}"
                }
            else:
                return {
                    'order_id': order_id,
                    'status': 'pending'
                }

        except Exception as e:
            print(f"Jupiter get_order_status error: {e}")
            return {'error': str(e)}

    def normalize_symbol(self, symbol: str) -> str:
        """Jupiter uses Solana mint addresses directly"""
        return symbol
