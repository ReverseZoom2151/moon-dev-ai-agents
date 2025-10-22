"""
Multi-Exchange Trading Demo
Demonstrates the new multi-exchange functionality

DEMO MODE - No real trades will be executed
"""
import sys
import codecs
from pathlib import Path

# Fix Windows UTF-8 encoding for emojis
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from termcolor import cprint
from src.multi_exchange_manager import MultiExchangeManager
import pandas as pd

def print_header(text):
    """Print a formatted header"""
    cprint("\n" + "="*70, "cyan")
    cprint(f"  {text}", "cyan", attrs=["bold"])
    cprint("="*70, "cyan")

def demo_multi_exchange():
    """Demonstrate multi-exchange capabilities"""

    print_header("🌙 Moon Dev's Multi-Exchange Demo")

    cprint("\n📋 This demo showcases:", "yellow")
    cprint("  1. Auto-detection of available exchanges based on API keys", "yellow")
    cprint("  2. Smart order routing (automatic exchange selection)", "yellow")
    cprint("  3. Aggregated positions across all exchanges", "yellow")
    cprint("  4. Unified balance checking", "yellow")

    # Initialize multi-exchange manager with demo mode
    print_header("1. Initializing Multi-Exchange Manager")

    cprint("\n⚠️ DEMO MODE: Will detect which exchanges you have credentials for", "yellow")
    cprint("Add API keys to .env to enable each exchange:", "cyan")
    cprint("  - BINANCE_API_KEY + BINANCE_API_SECRET", "cyan")
    cprint("  - BITFINEX_API_KEY + BITFINEX_API_SECRET", "cyan")
    cprint("  - SOLANA_PRIVATE_KEY (or address)", "cyan")
    cprint("  - HYPER_LIQUID_KEY", "cyan")

    try:
        # Auto-detect available exchanges
        mem = MultiExchangeManager()

        if not mem.active_exchanges:
            cprint("\n❌ No exchanges initialized!", "red")
            cprint("Please add API credentials to your .env file", "yellow")
            return

        # Show which exchanges are active
        print_header("2. Active Exchanges")
        for exchange_name in mem.active_exchanges.keys():
            cprint(f"  ✅ {exchange_name.upper()}", "green")

        # Demonstrate smart order routing
        print_header("3. Smart Order Routing Demo")

        test_symbols = [
            ("BTC/USDT", "Binance spot pair"),
            ("ETH/USD", "Bitfinex spot pair"),
            ("BTC", "HyperLiquid perpetual"),
            ("9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump", "Solana token"),
        ]

        cprint("\nDetermining which exchange handles each symbol:", "cyan")
        for symbol, description in test_symbols:
            exchange = mem._determine_exchange(symbol)
            if exchange:
                cprint(f"  {symbol:50s} -> {exchange.upper():12s} ({description})", "green")
            else:
                cprint(f"  {symbol:50s} -> NOT AVAILABLE  ({description})", "red")

        # Show aggregated positions
        print_header("4. Aggregated Positions")

        cprint("\nFetching positions across all active exchanges...", "cyan")
        try:
            positions_df = mem.get_aggregated_positions()

            if positions_df.empty:
                cprint("  No open positions found", "yellow")
            else:
                cprint(f"\n  Found {len(positions_df)} position(s):", "green")
                print(positions_df.to_string(index=False))

                total_value = mem.get_total_portfolio_value()
                cprint(f"\n  💰 Total Portfolio Value: ${total_value:,.2f}", "green", attrs=["bold"])
        except Exception as e:
            cprint(f"  ⚠️ Could not fetch positions: {str(e)}", "yellow")

        # Show balances
        print_header("5. Account Balances")

        cprint("\nChecking available balances...", "cyan")
        try:
            balances = mem.get_balance()

            if not balances:
                cprint("  No balances found (may need API permissions)", "yellow")
            else:
                for exchange, balance in balances.items():
                    if isinstance(balance, (int, float)) and balance > 0:
                        cprint(f"  {exchange.upper():12s}: ${balance:,.2f}", "green")
                    elif balance == 0:
                        cprint(f"  {exchange.upper():12s}: $0.00", "yellow")
        except Exception as e:
            cprint(f"  ⚠️ Could not fetch balances: {str(e)}", "yellow")

        # Example order routing (DEMO - not executed)
        print_header("6. Example Order Routing (Demo)")

        cprint("\n📝 Example: Routing a BUY order for BTC/USDT", "cyan")
        cprint("   mem.route_order('BTC/USDT', 'buy', 100.0)", "white")
        cprint("   -> Would automatically route to Binance", "green")

        cprint("\n📝 Example: Routing a BUY order for ETH/USD", "cyan")
        cprint("   mem.route_order('ETH/USD', 'buy', 100.0)", "white")
        cprint("   -> Would automatically route to Bitfinex", "green")

        cprint("\n📝 Example: Routing a BUY order for BTC perpetual", "cyan")
        cprint("   mem.route_order('BTC', 'buy', 100.0)", "white")
        cprint("   -> Would automatically route to HyperLiquid", "green")

        cprint("\n⚠️ No actual orders executed in demo mode!", "yellow", attrs=["bold"])

        # Summary
        print_header("✅ Demo Complete")

        cprint("\nTo enable multi-exchange trading in your agents:", "cyan")
        cprint("  1. Update .env with API credentials for desired exchanges", "cyan")
        cprint("  2. Edit src/config.py:", "cyan")
        cprint("     - Set USE_MULTI_EXCHANGE = True", "cyan")
        cprint("     - Set ACTIVE_EXCHANGES = ['binance', 'bitfinex', 'solana']", "cyan")
        cprint("  3. Agents using BaseAgent will automatically use MultiExchangeManager", "cyan")

    except Exception as e:
        cprint(f"\n❌ Demo failed: {str(e)}", "red")
        import traceback
        cprint(traceback.format_exc(), "red")

if __name__ == "__main__":
    demo_multi_exchange()
