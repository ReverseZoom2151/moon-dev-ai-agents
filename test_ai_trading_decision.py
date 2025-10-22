"""
Test AI Trading Decision with Model Priority
Demonstrates how to use the priority system for trading decisions
"""
import sys
import codecs
from pathlib import Path

# Fix Windows UTF-8 encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from termcolor import cprint
from src.models.model_priority import ModelPriority, model_priority_queue

def test_trading_decision():
    """Test AI for a trading decision"""

    cprint("\n" + "="*70, "cyan")
    cprint("  🌙 Testing AI Trading Decision with Model Priority", "cyan", attrs=["bold"])
    cprint("="*70, "cyan")

    # Simulate market data
    market_scenario = """
    Current Market Conditions:
    - BTC Price: $65,000
    - 24h Change: +5.2%
    - Market Sentiment: Bullish
    - RSI: 68 (approaching overbought)
    - Volume: High (above average)
    - News: Positive institutional adoption headlines
    """

    # Test CRITICAL priority (best model for real money)
    cprint("\n📊 CRITICAL Priority - Trading Decision (Uses Best Model)", "yellow", attrs=["bold"])
    cprint("-" * 70, "yellow")

    cprint("\n🔍 Question: Should I buy BTC now?", "cyan")
    cprint(market_scenario, "white")

    cprint("\n⏳ Requesting AI analysis...", "cyan")

    response, provider, model = model_priority_queue.get_model(
        priority=ModelPriority.CRITICAL,
        system_prompt="""You are an expert crypto trader with 10 years of experience.
Analyze the market conditions and provide a clear BUY, HOLD, or SELL recommendation.
Keep your response concise (2-3 sentences) and actionable.""",
        user_content=f"""Given these market conditions, should I buy BTC now?

{market_scenario}

Provide:
1. Your recommendation (BUY/HOLD/SELL)
2. Brief reasoning (1-2 sentences)
3. Risk level (LOW/MEDIUM/HIGH)""",
        temperature=0.5,  # Lower temperature for more consistent decisions
        max_tokens=300
    )

    if response:
        cprint(f"\n✅ AI Response (from {provider.upper()}:{model}):", "green", attrs=["bold"])
        cprint("-" * 70, "green")
        cprint(response.content, "white")
        cprint("-" * 70, "green")

        # Show which model was used
        if provider == 'claude':
            cprint(f"\n💡 Used: Claude Sonnet (Primary - Best reasoning)", "cyan")
        elif provider == 'gemini':
            cprint(f"\n💡 Used: Gemini Pro (Fallback - Claude unavailable)", "yellow")

    else:
        cprint("\n❌ All models failed! Check API keys and connectivity.", "red")

    # Test MEDIUM priority (general analysis)
    cprint("\n\n📈 MEDIUM Priority - Quick Analysis (Uses Fast Model)", "yellow", attrs=["bold"])
    cprint("-" * 70, "yellow")

    cprint("\n🔍 Question: Summarize BTC sentiment", "cyan")

    response2, provider2, model2 = model_priority_queue.get_model(
        priority=ModelPriority.MEDIUM,
        system_prompt="You are a crypto analyst. Be concise.",
        user_content=f"Summarize the market sentiment for BTC in one sentence:\n{market_scenario}",
        temperature=0.7,
        max_tokens=100
    )

    if response2:
        cprint(f"\n✅ AI Response (from {provider2.upper()}:{model2}):", "green", attrs=["bold"])
        cprint("-" * 70, "green")
        cprint(response2.content, "white")
        cprint("-" * 70, "green")

        if provider2 == 'claude':
            cprint(f"\n💡 Used: Claude Haiku (Primary - Fast & cheap)", "cyan")
        elif provider2 == 'gemini':
            cprint(f"\n💡 Used: Gemini Flash (Fallback)", "yellow")

    # Show cost comparison
    cprint("\n\n💰 Cost Comparison:", "cyan", attrs=["bold"])
    cprint("-" * 70, "cyan")
    cprint("CRITICAL (Claude Sonnet): ~$15 per 1M tokens - Worth it for real money!", "yellow")
    cprint("MEDIUM (Claude Haiku):    ~$1 per 1M tokens  - Perfect for general tasks", "yellow")
    cprint("MEDIUM (Gemini Flash):    FREE (generous tier) - Great for testing!", "green")

    cprint("\n\n✅ Demo Complete!", "green", attrs=["bold"])
    cprint("\nYour AI trading assistant is ready to use! 🚀", "cyan")

if __name__ == "__main__":
    test_trading_decision()
