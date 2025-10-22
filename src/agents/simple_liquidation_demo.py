"""
Simple demo using Moon Dev API (no other keys needed!)
Shows liquidation data without requiring BIRDEYE or other APIs
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.api import MoonDevAPI
import pandas as pd

print("=" * 60)
print("Moon Dev API Demo - Liquidation Analysis")
print("=" * 60)

# Initialize API (no key needed for most endpoints)
api = MoonDevAPI()

print("\n1. Fetching recent liquidation data...")
liq_data = api.get_liquidation_data(limit=1000)

if liq_data is not None:
    print(f"   Loaded {len(liq_data)} liquidations\n")

    # Basic analysis
    print("2. Analyzing liquidation patterns...\n")

    # Most liquidated coins
    if 'symbol' in liq_data.columns:
        top_liqs = liq_data['symbol'].value_counts().head(10)
        print("   Top 10 Most Liquidated Coins:")
        print("   " + "-" * 40)
        for symbol, count in top_liqs.items():
            print(f"   {symbol:15} {count:5} liquidations")

    # Total liquidation value
    if 'value' in liq_data.columns:
        total_value = liq_data['value'].sum()
        avg_value = liq_data['value'].mean()
        max_value = liq_data['value'].max()

        print(f"\n   Total Liquidation Value: ${total_value:,.2f}")
        print(f"   Average Liquidation: ${avg_value:,.2f}")
        print(f"   Largest Liquidation: ${max_value:,.2f}")

    # Buy vs Sell liquidations
    if 'side' in liq_data.columns:
        side_counts = liq_data['side'].value_counts()
        print(f"\n   Buy Liquidations: {side_counts.get('BUY', 0)}")
        print(f"   Sell Liquidations: {side_counts.get('SELL', 0)}")

print("\n" + "=" * 60)
print("3. Fetching funding rate data...")
funding_data = api.get_funding_data()

if funding_data is not None:
    print(f"   Loaded {len(funding_data)} funding rates\n")

    print("   Current Funding Rates:")
    print("   " + "-" * 40)
    for _, row in funding_data.iterrows():
        symbol = row.get('symbol', 'Unknown')
        rate = row.get('yearly_funding_rate', 0)
        sign = "+" if rate > 0 else ""
        print(f"   {symbol:15} {sign}{rate:>8.2f}% annually")

print("\n" + "=" * 60)
print("4. Checking open interest...")
oi_total = api.get_oi_total()

if oi_total is not None:
    print(f"   Loaded {len(oi_total)} OI data points")

    if len(oi_total) > 0:
        # Get recent OI trend
        recent = oi_total.tail(10)

        if len(oi_total.columns) > 1:
            latest_oi = oi_total.iloc[-1, 1]
            prev_oi = oi_total.iloc[-10, 1] if len(oi_total) > 10 else latest_oi

            change = ((latest_oi - prev_oi) / prev_oi * 100) if prev_oi != 0 else 0

            print(f"\n   Latest Total OI: ${latest_oi:,.0f}")
            print(f"   10-period change: {change:+.2f}%")

            if change > 5:
                print("   📈 OI increasing - More capital entering")
            elif change < -5:
                print("   📉 OI decreasing - Capital leaving")
            else:
                print("   ➡️  OI stable")

print("\n" + "=" * 60)
print("✨ Demo Complete!")
print("=" * 60)
print("\nThis data is available WITHOUT any API keys!")
print("To run full agents, add these free API keys to .env:")
print("  - GROQ_API_KEY (console.groq.com)")
print("  - ANTHROPIC_KEY (console.anthropic.com)")
print("  - BIRDEYE_API_KEY (birdeye.so)")
print("  - RPC_ENDPOINT (helius.dev)")
