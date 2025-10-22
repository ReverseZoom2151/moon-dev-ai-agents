"""
🌙 Moon Dev's LLM Trading Agent 🌙

DUAL-MODE AI TRADING SYSTEM:

🤖 SINGLE MODEL MODE (Fast - ~10 seconds per token):
   - Uses one AI model for quick trading decisions
   - Best for: Fast execution, high-frequency strategies
   - Configure model in config.py: AI_MODEL_TYPE and AI_MODEL_NAME

🌊 SWARM MODE (Consensus - ~45-60 seconds per token):
   - Queries 6 AI models simultaneously for consensus voting
   - Models vote: "Buy", "Sell", or "Do Nothing"
   - Majority decision wins with confidence percentage
   - Best for: Higher confidence trades, 15-minute+ timeframes

   Active Swarm Models:
   1. Claude Sonnet 4.5 - Anthropic's latest reasoning model
   2. GPT-5 - OpenAI's most advanced model
   3. Gemini 2.5 Flash - Google's fast multimodal model
   4. Grok-4 Fast Reasoning - xAI's 2M context model
   5. DeepSeek Chat - DeepSeek API reasoning model
   6. DeepSeek-R1 Local - Local reasoning model (free!)

   Trading Actions:
   - "Buy" = Open/maintain position at target size ($25)
   - "Sell" = Close entire position (exit to cash)
   - "Do Nothing" = Hold current position unchanged (no action)

CONFIGURATION:
   ⚙️ ALL settings are configured at the top of THIS file (lines 66-120)

   🌊 AI Mode (line 71):
   - USE_SWARM_MODE: True = 6-model consensus, False = single model

   📈 Trading Mode (line 75):
   - LONG_ONLY: True = Long positions only (Solana/on-chain)
   - LONG_ONLY: False = Long & Short (HyperLiquid perpetuals)
   - When LONG_ONLY: SELL closes position, can't open shorts
   - When LONG/SHORT: SELL can close long OR open short

   💰 Position Sizing (lines 93-96):
   - usd_size: $25 target position
   - max_usd_order_size: $3 order chunks
   - MAX_POSITION_PERCENTAGE: 30% max per position
   - CASH_PERCENTAGE: 20% min cash buffer

   📊 Market Data (lines 99-103):
   - DAYSBACK_4_DATA: 3 days history
   - DATA_TIMEFRAME: '1H' bars (~72 bars)
     Change to '15m' for 15-minute bars (~288 bars)
     Options: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M

   🎯 Tokens (lines 115-118):
   - MONITORED_TOKENS: List of tokens to analyze and trade
   - ⚠️ ALL tokens in this list will be analyzed (one at a time)
   - Comment out tokens you don't want with # to disable them

   Portfolio Allocation:
   - Automatically runs if swarm recommends BUY signals
   - Skipped if all signals are SELL or DO NOTHING

   Each swarm query receives:
   - Full OHLCV dataframe (Open, High, Low, Close, Volume)
   - Strategy signals (if available)
   - Token metadata

Built with love by Moon Dev 🚀
"""

# ============================================================================
# 🔧 TRADING AGENT CONFIGURATION - ALL SETTINGS IN ONE PLACE
# ============================================================================

# 🌊 AI MODE SELECTION
USE_SWARM_MODE = True  # True = 6-model swarm consensus (~45-60s per token)
                        # False = Single model fast execution (~10s per token)

# 📈 TRADING MODE SETTINGS
LONG_ONLY = True  # True = Long positions only (Solana on-chain, no shorting available)
                  # False = Long & Short positions (HyperLiquid perpetuals)
                  #
                  # When LONG_ONLY = True:
                  #   - "Buy" = Opens/maintains long position
                  #   - "Sell" = Closes long position (exit to cash)
                  #   - Can NOT open short positions
                  #
                  # When LONG_ONLY = False:
                  #   - "Buy" = Opens/maintains long position
                  #   - "Sell" = Closes long OR opens short position
                  #   - Full long/short capability (for HyperLiquid)

# 🤖 SINGLE MODEL SETTINGS (only used when USE_SWARM_MODE = False)
AI_MODEL_TYPE = 'xai'  # Options: 'groq', 'openai', 'claude', 'deepseek', 'xai', 'ollama'
AI_MODEL_NAME = None   # None = use default, or specify: 'grok-4-fast-reasoning', 'claude-3-5-sonnet-latest', etc.
AI_TEMPERATURE = 0.7   # Creativity vs precision (0-1)
AI_MAX_TOKENS = 1024   # Max tokens for AI response

# 💰 POSITION SIZING & RISK MANAGEMENT
usd_size = 25                    # Target position size in USD
max_usd_order_size = 3           # Maximum order chunk size in USD
MAX_POSITION_PERCENTAGE = 30     # Max % of portfolio per position (0-100)
CASH_PERCENTAGE = 20             # Minimum % to keep in USDC cash buffer (0-100)

# 📊 MARKET DATA COLLECTION
DAYSBACK_4_DATA = 3              # Days of historical data to fetch
DATA_TIMEFRAME = '1H'            # Bar timeframe: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M
                                 # Current: 3 days @ 1H = ~72 bars
                                 # For 15-min: '15m' = ~288 bars
SAVE_OHLCV_DATA = False          # True = save data permanently, False = temp data only

# ⚡ TRADING EXECUTION SETTINGS
slippage = 199                   # Slippage tolerance (199 = ~2%)
SLEEP_BETWEEN_RUNS_MINUTES = 15  # Minutes between trading cycles

# 🎯 TOKEN CONFIGURATION
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Never trade
SOL_ADDRESS = "So11111111111111111111111111111111111111111"   # Never trade
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

# ⚠️ IMPORTANT: The swarm will analyze ALL tokens in this list (one at a time)
# Each token takes ~45-60 seconds in swarm mode
# Comment out tokens you don't want to trade (add # at start of line)
MONITORED_TOKENS = [
    '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',    # 🌬️ FART (DISABLED)
    #'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump',   # 🏠 housecoin (ACTIVE)
]

# Example: To trade multiple tokens, uncomment the ones you want:
# MONITORED_TOKENS = [
#     '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',    # FART
#     'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump',   # housecoin
#     'YourTokenAddressHere',                              # Your token
# ]

# ============================================================================
# END CONFIGURATION - CODE BELOW
# ============================================================================

# Keep only these prompts
TRADING_PROMPT = """
You are Moon Dev's AI Trading Assistant 🌙

Analyze the provided market data and strategy signals (if available) to make a trading decision.

Market Data Criteria:
1. Price action relative to MA20 and MA40
2. RSI levels and trend
3. Volume patterns
4. Recent price movements

{strategy_context}

Respond in this exact format:
1. First line must be one of: BUY, SELL, or NOTHING (in caps)
2. Then explain your reasoning, including:
   - Technical analysis
   - Strategy signals analysis (if available)
   - Risk factors
   - Market conditions
   - Confidence level (as a percentage, e.g. 75%)

Remember: 
- Moon Dev always prioritizes risk management! 🛡️
- Never trade USDC or SOL directly
- Consider both technical and strategy signals
"""

ALLOCATION_PROMPT = """
You are Moon Dev's Portfolio Allocation Assistant 🌙

Given the total portfolio size and trading recommendations, allocate capital efficiently.
Consider:
1. Position sizing based on confidence levels
2. Risk distribution
3. Keep cash buffer as specified
4. Maximum allocation per position

Format your response as a Python dictionary:
{
    "token_address": allocated_amount,  # In USD
    ...
    "USDC_ADDRESS": remaining_cash  # Always use USDC_ADDRESS for cash
}

Remember:
- Total allocations must not exceed total_size
- Higher confidence should get larger allocations
- Never allocate more than {MAX_POSITION_PERCENTAGE}% to a single position
- Keep at least {CASH_PERCENTAGE}% in USDC as safety buffer
- Only allocate to BUY recommendations
- Cash must be stored as USDC using USDC_ADDRESS: {USDC_ADDRESS}
"""

SWARM_TRADING_PROMPT = """You are an expert cryptocurrency trading AI analyzing market data.

CRITICAL RULES:
1. Your response MUST be EXACTLY one of these three words: Buy, Sell, or Do Nothing
2. Do NOT provide any explanation, reasoning, or additional text
3. Respond with ONLY the action word

Analyze the market data below and decide:

- "Buy" = Strong bullish signals, recommend opening/holding position
- "Sell" = Bearish signals or major weakness, recommend closing position entirely
- "Do Nothing" = Unclear/neutral signals, recommend holding current state unchanged

IMPORTANT: "Do Nothing" means maintain current position (if we have one, keep it; if we don't, stay out)

RESPOND WITH ONLY ONE WORD: Buy, Sell, or Do Nothing"""

# Standard library imports
import json
import os
import sys
import time

# Third-party imports
import pandas as pd

# Standard library from imports
from datetime import datetime, timedelta
from pathlib import Path

# Third-party from imports
from dotenv import load_dotenv
from termcolor import cprint

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Local imports - trading_agent.py is now fully self-contained!
# No config.py imports needed - all settings are at the top of this file (lines 55-111)
# Local from imports
from src import nice_funcs as n
from src.agents.swarm_agent import SwarmAgent
from src.data.ohlcv_collector import collect_all_tokens
from src.models.model_factory import model_factory

# Load environment variables
load_dotenv()

class TradingAgent:
    def __init__(self):
        # Check if using swarm mode or single model
        if USE_SWARM_MODE:
            cprint(f"\n🌊 Initializing Trading Agent in SWARM MODE (6 AI consensus)...", "cyan", attrs=['bold'])
            self.swarm = SwarmAgent()
            cprint("✅ Swarm mode initialized with 6 AI models!", "green")

            # Still need a lightweight model for portfolio allocation (not trading decisions)
            cprint("💼 Initializing fast model for portfolio calculations...", "cyan")
            self.model = model_factory.get_model(AI_MODEL_TYPE, AI_MODEL_NAME)
            if self.model:
                cprint(f"✅ Allocation model ready: {self.model.model_name}", "green")
        else:
            # Initialize AI model via model factory (original behavior)
            cprint(f"\n🤖 Initializing Trading Agent with {AI_MODEL_TYPE} model...", "cyan")
            self.model = model_factory.get_model(AI_MODEL_TYPE, AI_MODEL_NAME)
            self.swarm = None  # Not used in single model mode

            if not self.model:
                cprint(f"❌ Failed to initialize {AI_MODEL_TYPE} model!", "red")
                cprint("Available models:", "yellow")
                for model_type in model_factory._models.keys():
                    cprint(f"  - {model_type}", "yellow")
                sys.exit(1)

            cprint(f"✅ Using model: {self.model.model_name}", "green")

        self.recommendations_df = pd.DataFrame(columns=['token', 'action', 'confidence', 'reasoning'])

        # Show which tokens will be analyzed
        cprint("\n🎯 Active Tokens for Trading:", "yellow", attrs=['bold'])
        for i, token in enumerate(MONITORED_TOKENS, 1):
            token_display = token[:8] + "..." if len(token) > 8 else token
            cprint(f"   {i}. {token_display}", "cyan")
        cprint(f"\n⏱️  Estimated swarm analysis time: ~{len(MONITORED_TOKENS) * 60} seconds ({len(MONITORED_TOKENS)} tokens × 60s)\n", "yellow")

        # Show trading mode
        cprint("📈 Trading Mode:", "yellow", attrs=['bold'])
        if LONG_ONLY:
            cprint("   📊 LONG ONLY - Solana on-chain (no shorting)", "cyan")
            cprint("   💡 SELL signals close positions, can't open shorts", "white")
        else:
            cprint("   ⚡ LONG/SHORT - HyperLiquid perpetuals", "green")
            cprint("   💡 SELL signals can close longs OR open shorts", "white")

        cprint("\n🤖 Moon Dev's LLM Trading Agent initialized!", "green")

    def chat_with_ai(self, system_prompt, user_content):
        """Send prompt to AI model via model factory"""
        try:
            response = self.model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=AI_TEMPERATURE,
                max_tokens=AI_MAX_TOKENS
            )

            # Handle response format
            if hasattr(response, 'content'):
                return response.content
            return str(response)

        except Exception as e:
            cprint(f"❌ AI model error: {e}", "red")
            return None

    def _format_market_data_for_swarm(self, token, market_data):
        """Format market data into a clean, readable format for swarm analysis"""
        try:
            # Print market data visibility for confirmation
            cprint(f"\n📊 MARKET DATA RECEIVED FOR {token[:8]}...", "cyan", attrs=['bold'])

            # Check if market_data is a DataFrame
            if isinstance(market_data, pd.DataFrame):
                cprint(f"✅ DataFrame received: {len(market_data)} bars", "green")
                cprint(f"📅 Date range: {market_data.index[0]} to {market_data.index[-1]}", "yellow")
                cprint(f"🕐 Timeframe: {DATA_TIMEFRAME}", "yellow")

                # Show the first 5 bars
                cprint("\n📈 First 5 Bars (OHLCV):", "cyan")
                print(market_data.head().to_string())

                # Show the last 3 bars
                cprint("\n📉 Last 3 Bars (Most Recent):", "cyan")
                print(market_data.tail(3).to_string())

                # Format for swarm
                formatted = f"""
TOKEN: {token}
TIMEFRAME: {DATA_TIMEFRAME} bars
TOTAL BARS: {len(market_data)}
DATE RANGE: {market_data.index[0]} to {market_data.index[-1]}

RECENT PRICE ACTION (Last 10 bars):
{market_data.tail(10).to_string()}

FULL DATASET:
{market_data.to_string()}
"""
            else:
                # If it's not a DataFrame, show what we got
                cprint(f"⚠️ Market data is not a DataFrame: {type(market_data)}", "yellow")
                formatted = f"TOKEN: {token}\nMARKET DATA:\n{str(market_data)}"

            # Add strategy signals if available
            if isinstance(market_data, dict) and 'strategy_signals' in market_data:
                formatted += f"\n\nSTRATEGY SIGNALS:\n{json.dumps(market_data['strategy_signals'], indent=2)}"

            cprint("\n✅ Market data formatted and ready for swarm!\n", "green")
            return formatted

        except Exception as e:
            cprint(f"❌ Error formatting market data: {e}", "red")
            return str(market_data)

    def _calculate_swarm_consensus(self, swarm_result):
        """
        Calculate consensus from individual swarm responses

        Args:
            swarm_result: Result dict from swarm.query() containing individual responses

        Returns:
            tuple: (action, confidence, reasoning_summary)
                - action: "BUY", "SELL", or "NOTHING"
                - confidence: percentage based on vote distribution
                - reasoning_summary: Summary of all model votes
        """
        try:
            votes = {"BUY": 0, "SELL": 0, "NOTHING": 0}
            model_votes = []

            # Count votes from each model's response
            for provider, data in swarm_result["responses"].items():
                if not data["success"]:
                    continue

                response_text = data["response"].strip().upper()

                # Parse the response - look for Buy, Sell, or Do Nothing
                if "BUY" in response_text:
                    votes["BUY"] += 1
                    model_votes.append(f"{provider}: Buy")
                elif "SELL" in response_text:
                    votes["SELL"] += 1
                    model_votes.append(f"{provider}: Sell")
                else:
                    votes["NOTHING"] += 1
                    model_votes.append(f"{provider}: Do Nothing")

            # Calculate total votes
            total_votes = sum(votes.values())
            if total_votes == 0:
                return "NOTHING", 0, "No valid responses from swarm"

            # Find majority vote
            majority_action = max(votes, key=votes.get)
            majority_count = votes[majority_action]

            # Calculate confidence as percentage of votes for majority action
            confidence = int((majority_count / total_votes) * 100)

            # Create reasoning summary
            reasoning = f"Swarm Consensus ({total_votes} models voted):\n"
            reasoning += f"  Buy: {votes['BUY']} votes\n"
            reasoning += f"  Sell: {votes['SELL']} votes\n"
            reasoning += f"  Do Nothing: {votes['NOTHING']} votes\n\n"
            reasoning += "Individual votes:\n"
            reasoning += "\n".join(f"  - {vote}" for vote in model_votes)
            reasoning += f"\n\nMajority decision: {majority_action} ({confidence}% consensus)"

            cprint(f"\n🌊 Swarm Consensus: {majority_action} with {confidence}% agreement", "cyan", attrs=['bold'])

            return majority_action, confidence, reasoning

        except Exception as e:
            cprint(f"❌ Error calculating swarm consensus: {e}", "red")
            return "NOTHING", 0, f"Error calculating consensus: {str(e)}"

    def analyze_market_data(self, token, market_data):
        """Analyze market data using AI model (single or swarm mode)"""
        try:
            # Skip analysis for excluded tokens
            if token in EXCLUDED_TOKENS:
                print(f"⚠️ Skipping analysis for excluded token: {token}")
                return None

            # ============= SWARM MODE =============
            if USE_SWARM_MODE:
                cprint(f"\n🌊 Analyzing {token[:8]}... with SWARM (6 AI models voting)", "cyan", attrs=['bold'])

                # Format market data for swarm
                formatted_data = self._format_market_data_for_swarm(token, market_data)

                # Query the swarm (takes ~45-60 seconds)
                swarm_result = self.swarm.query(
                    prompt=formatted_data,
                    system_prompt=SWARM_TRADING_PROMPT
                )

                if not swarm_result:
                    cprint(f"❌ No response from swarm for {token}", "red")
                    return None

                # Calculate consensus from individual model votes
                action, confidence, reasoning = self._calculate_swarm_consensus(swarm_result)

                # Add to recommendations DataFrame
                self.recommendations_df = pd.concat([
                    self.recommendations_df,
                    pd.DataFrame([{
                        'token': token,
                        'action': action,
                        'confidence': confidence,
                        'reasoning': reasoning
                    }])
                ], ignore_index=True)

                cprint(f"✅ Swarm analysis complete for {token[:8]}!", "green")
                return swarm_result

            # ============= SINGLE MODEL MODE (Original) =============
            else:
                # Prepare strategy context
                strategy_context = ""
                if 'strategy_signals' in market_data:
                    strategy_context = f"""
Strategy Signals Available:
{json.dumps(market_data['strategy_signals'], indent=2)}
                    """
                else:
                    strategy_context = "No strategy signals available."

                # Call AI model via model factory
                response = self.chat_with_ai(
                    TRADING_PROMPT.format(strategy_context=strategy_context),
                    f"Market Data to Analyze:\n{market_data}"
                )

                if not response:
                    cprint(f"❌ No response from AI for {token}", "red")
                    return None

                # Parse the response
                lines = response.split('\n')
                action = lines[0].strip() if lines else "NOTHING"

                # Extract confidence from the response (assuming it's mentioned as a percentage)
                confidence = 0
                for line in lines:
                    if 'confidence' in line.lower():
                        # Extract number from string like "Confidence: 75%"
                        try:
                            confidence = int(''.join(filter(str.isdigit, line)))
                        except:
                            confidence = 50  # Default if not found

                # Add to recommendations DataFrame with proper reasoning
                reasoning = '\n'.join(lines[1:]) if len(lines) > 1 else "No detailed reasoning provided"
                self.recommendations_df = pd.concat([
                    self.recommendations_df,
                    pd.DataFrame([{
                        'token': token,
                        'action': action,
                        'confidence': confidence,
                        'reasoning': reasoning
                    }])
                ], ignore_index=True)

                print(f"🎯 Moon Dev's AI Analysis Complete for {token[:4]}!")
                return response

        except Exception as e:
            print(f"❌ Error in AI analysis: {str(e)}")
            # Still add to DataFrame even on error, but mark as NOTHING with 0 confidence
            self.recommendations_df = pd.concat([
                self.recommendations_df,
                pd.DataFrame([{
                    'token': token,
                    'action': "NOTHING",
                    'confidence': 0,
                    'reasoning': f"Error during analysis: {str(e)}"
                }])
            ], ignore_index=True)
            return None
    
    def allocate_portfolio(self):
        """Get AI-recommended portfolio allocation"""
        try:
            cprint("\n💰 Calculating optimal portfolio allocation...", "cyan")
            max_position_size = usd_size * (MAX_POSITION_PERCENTAGE / 100)
            cprint(f"🎯 Maximum position size: ${max_position_size:.2f} ({MAX_POSITION_PERCENTAGE}% of ${usd_size:.2f})", "cyan")

            # Get allocation from AI via model factory
            allocation_prompt = f"""You are Moon Dev's Portfolio Allocation AI 🌙

Given:
- Total portfolio size: ${usd_size}
- Maximum position size: ${max_position_size} ({MAX_POSITION_PERCENTAGE}% of total)
- Minimum cash (USDC) buffer: {CASH_PERCENTAGE}%
- Available tokens: {MONITORED_TOKENS}
- USDC Address: {USDC_ADDRESS}

Provide a portfolio allocation that:
1. Never exceeds max position size per token
2. Maintains minimum cash buffer
3. Returns allocation as a JSON object with token addresses as keys and USD amounts as values
4. Uses exact USDC address: {USDC_ADDRESS} for cash allocation

Example format:
{{
    "token_address": amount_in_usd,
    "{USDC_ADDRESS}": remaining_cash_amount  # Use exact USDC address
}}"""

            response = self.chat_with_ai("", allocation_prompt)

            if not response:
                cprint("❌ No response from AI for portfolio allocation", "red")
                return None

            # Parse the response
            allocations = self.parse_allocation_response(response)
            if not allocations:
                return None
                
            # Fix USDC address if needed
            if "USDC_ADDRESS" in allocations:
                amount = allocations.pop("USDC_ADDRESS")
                allocations[USDC_ADDRESS] = amount
                
            # Validate allocation totals
            total_allocated = sum(allocations.values())
            if total_allocated > usd_size:
                cprint(f"❌ Total allocation ${total_allocated:.2f} exceeds portfolio size ${usd_size:.2f}", "red")
                return None
                
            # Print allocations
            cprint("\n📊 Portfolio Allocation:", "green")
            for token, amount in allocations.items():
                token_display = "USDC" if token == USDC_ADDRESS else token
                cprint(f"  • {token_display}: ${amount:.2f}", "green")
                
            return allocations
            
        except Exception as e:
            cprint(f"❌ Error in portfolio allocation: {str(e)}", "red")
            return None

    def execute_allocations(self, allocation_dict):
        """Execute the allocations using AI entry for each position"""
        try:
            print("\n🚀 Moon Dev executing portfolio allocations...")
            
            for token, amount in allocation_dict.items():
                # Skip USDC and other excluded tokens
                if token in EXCLUDED_TOKENS:
                    print(f"💵 Keeping ${amount:.2f} in {token}")
                    continue
                    
                print(f"\n🎯 Processing allocation for {token}...")
                
                try:
                    # Get current position value
                    current_position = n.get_token_balance_usd(token)
                    target_allocation = amount
                    
                    print(f"🎯 Target allocation: ${target_allocation:.2f} USD")
                    print(f"📊 Current position: ${current_position:.2f} USD")
                    
                    if current_position < target_allocation:
                        print(f"✨ Executing entry for {token}")
                        n.ai_entry(token, amount)
                        print(f"✅ Entry complete for {token}")
                    else:
                        print(f"⏸️ Position already at target size for {token}")
                    
                except Exception as e:
                    print(f"❌ Error executing entry for {token}: {str(e)}")
                
                time.sleep(2)  # Small delay between entries
                
        except Exception as e:
            print(f"❌ Error executing allocations: {str(e)}")
            print("🔧 Moon Dev suggests checking the logs and trying again!")

    def handle_exits(self):
        """Check and exit positions based on SELL recommendations"""
        cprint("\n🔄 Checking for positions to exit...", "white", "on_blue")

        for _, row in self.recommendations_df.iterrows():
            token = row['token']
            token_short = token[:8] + "..." if len(token) > 8 else token

            # Skip excluded tokens (USDC and SOL)
            if token in EXCLUDED_TOKENS:
                continue

            action = row['action']

            # Check if we have a position
            current_position = n.get_token_balance_usd(token)

            cprint(f"\n{'='*60}", "cyan")
            cprint(f"🎯 Token: {token_short}", "cyan", attrs=['bold'])
            cprint(f"🤖 Swarm Signal: {action} ({row['confidence']}% confidence)", "yellow", attrs=['bold'])
            cprint(f"💼 Current Position: ${current_position:.2f}", "white")
            cprint(f"{'='*60}", "cyan")

            if current_position > 0:
                # We have a position - take action based on signal
                if action == "SELL":
                    cprint(f"🚨 SELL signal with position - CLOSING POSITION", "white", "on_red")
                    try:
                        cprint(f"📉 Executing chunk_kill (${max_usd_order_size} chunks)...", "yellow")
                        n.chunk_kill(token, max_usd_order_size, slippage)
                        cprint(f"✅ Position closed successfully!", "white", "on_green")
                    except Exception as e:
                        cprint(f"❌ Error closing position: {str(e)}", "white", "on_red")
                elif action == "NOTHING":
                    cprint(f"⏸️  DO NOTHING signal - HOLDING POSITION", "white", "on_blue")
                    cprint(f"💎 Maintaining ${current_position:.2f} position", "cyan")
                else:  # BUY
                    cprint(f"✅ BUY signal - KEEPING POSITION", "white", "on_green")
                    cprint(f"💎 Maintaining ${current_position:.2f} position", "cyan")
            else:
                # No position - explain what this means
                if action == "SELL":
                    if LONG_ONLY:
                        cprint(f"⏭️  SELL signal but NO POSITION to close", "white", "on_blue")
                        cprint(f"📊 LONG ONLY mode: Can't open short, doing nothing", "cyan")
                    else:
                        cprint(f"📉 SELL signal with no position - Could open SHORT", "white", "on_yellow")
                        cprint(f"⚡ HyperLiquid mode: Shorting available (not implemented yet)", "cyan")
                elif action == "NOTHING":
                    cprint(f"⏸️  DO NOTHING signal with no position", "white", "on_blue")
                    cprint(f"⏭️  Staying out of market", "cyan")
                else:  # BUY
                    cprint(f"📊 BUY signal with no position - Portfolio allocation will handle entry", "white", "on_cyan")
                    cprint(f"💼 Target size: ${usd_size}", "yellow")

    def parse_allocation_response(self, response):
        """Parse the AI's allocation response and handle both string and TextBlock formats"""
        try:
            # Handle TextBlock format from Claude 3
            if isinstance(response, list):
                response = response[0].text if hasattr(response[0], 'text') else str(response[0])
            
            print("🔍 Raw response received:")
            print(response)
            
            # Find the JSON block between curly braces
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start:end]
            
            # More aggressive JSON cleaning
            json_str = (json_str
                .replace('\n', '')          # Remove newlines
                .replace('    ', '')        # Remove indentation
                .replace('\t', '')          # Remove tabs
                .replace('\\n', '')         # Remove escaped newlines
                .replace(' ', '')           # Remove all spaces
                .strip())                   # Remove leading/trailing whitespace
            
            print("\n🧹 Cleaned JSON string:")
            print(json_str)
            
            # Parse the cleaned JSON
            allocations = json.loads(json_str)
            
            print("\n📊 Parsed allocations:")
            for token, amount in allocations.items():
                print(f"  • {token}: ${amount}")
            
            # Validate amounts are numbers
            for token, amount in allocations.items():
                if not isinstance(amount, (int, float)):
                    raise ValueError(f"Invalid amount type for {token}: {type(amount)}")
                if amount < 0:
                    raise ValueError(f"Negative allocation for {token}: {amount}")
            
            return allocations
            
        except Exception as e:
            print(f"❌ Error parsing allocation response: {str(e)}")
            print("🔍 Raw response:")
            print(response)
            return None

    def parse_portfolio_allocation(self, allocation_text):
        """Parse portfolio allocation from text response"""
        try:
            # Clean up the response text
            cleaned_text = allocation_text.strip()
            if "```json" in cleaned_text:
                # Extract JSON from code block if present
                json_str = cleaned_text.split("```json")[1].split("```")[0]
            else:
                # Find the JSON object between curly braces
                start = cleaned_text.find('{')
                end = cleaned_text.rfind('}') + 1
                json_str = cleaned_text[start:end]
            
            # Parse the JSON
            allocations = json.loads(json_str)
            
            print("📊 Parsed allocations:")
            for token, amount in allocations.items():
                print(f"  • {token}: ${amount}")
            
            return allocations
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing allocation JSON: {e}")
            print(f"🔍 Raw text received:\n{allocation_text}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error parsing allocations: {e}")
            return None

    def run(self):
        """Run the trading agent (implements BaseAgent interface)"""
        self.run_trading_cycle()

    def run_trading_cycle(self, strategy_signals=None):
        """Run one complete trading cycle"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cprint(f"\n⏰ AI Agent Run Starting at {current_time}", "white", "on_green")
            
            # Collect OHLCV data for all tokens using this agent's config
            cprint("📊 Collecting market data...", "white", "on_blue")
            cprint(f"🎯 Tokens to collect: {MONITORED_TOKENS}", "yellow")
            cprint(f"📅 Settings: {DAYSBACK_4_DATA} days @ {DATA_TIMEFRAME}", "yellow")

            market_data = collect_all_tokens(
                tokens=MONITORED_TOKENS,
                days_back=DAYSBACK_4_DATA,
                timeframe=DATA_TIMEFRAME
            )

            cprint(f"📦 Market data received for {len(market_data)} tokens", "green")
            if len(market_data) == 0:
                cprint("⚠️ WARNING: No market data collected! Check MONITORED_TOKENS list.", "red")
                cprint(f"🔍 MONITORED_TOKENS = {MONITORED_TOKENS}", "red")
            
            # Analyze each token's data
            for token, data in market_data.items():
                cprint(f"\n🤖 AI Agent Analyzing Token: {token}", "white", "on_green")
                
                # Include strategy signals in analysis if available
                if strategy_signals and token in strategy_signals:
                    cprint(f"📊 Including {len(strategy_signals[token])} strategy signals in analysis", "cyan")
                    data['strategy_signals'] = strategy_signals[token]
                
                analysis = self.analyze_market_data(token, data)
                print(f"\n📈 Analysis for contract: {token}")
                print(analysis)
                print("\n" + "="*50 + "\n")
            
            # Show recommendations summary
            cprint("\n📊 Moon Dev's Trading Recommendations:", "white", "on_blue")
            summary_df = self.recommendations_df[['token', 'action', 'confidence']].copy()
            print(summary_df.to_string(index=False))

            # Handle exits first (always runs - manages SELL recommendations)
            self.handle_exits()

            # Portfolio allocation (only run if there are BUY recommendations)
            buy_recommendations = self.recommendations_df[self.recommendations_df['action'] == 'BUY']

            if len(buy_recommendations) > 0:
                cprint(f"\n💰 Found {len(buy_recommendations)} BUY signal(s) - Calculating portfolio allocation...", "white", "on_green")
                allocation = self.allocate_portfolio()

                if allocation:
                    cprint("\n💼 Moon Dev's Portfolio Allocation:", "white", "on_blue")
                    print(json.dumps(allocation, indent=4))

                    cprint("\n🎯 Executing allocations...", "white", "on_blue")
                    self.execute_allocations(allocation)
                    cprint("\n✨ All allocations executed!", "white", "on_blue")
                else:
                    cprint("\n⚠️ No allocations to execute!", "white", "on_yellow")
            else:
                cprint("\n⏭️  No BUY signals - Skipping portfolio allocation", "white", "on_blue")
                cprint("📊 All signals were SELL or DO NOTHING", "cyan")
            
            # Clean up temp data
            cprint("\n🧹 Cleaning up temporary data...", "white", "on_blue")
            try:
                for file in os.listdir('temp_data'):
                    if file.endswith('_latest.csv'):
                        os.remove(os.path.join('temp_data', file))
                cprint("✨ Temp data cleaned successfully!", "white", "on_green")
            except Exception as e:
                cprint(f"⚠️ Error cleaning temp data: {str(e)}", "white", "on_yellow")
            
        except Exception as e:
            cprint(f"\n❌ Error in trading cycle: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")

def main():
    """Main function to run the trading agent every 15 minutes"""
    cprint("🌙 Moon Dev AI Trading System Starting Up! 🚀", "white", "on_blue")
    
    agent = TradingAgent()
    INTERVAL = SLEEP_BETWEEN_RUNS_MINUTES * 60  # Convert minutes to seconds
    
    while True:
        try:
            agent.run_trading_cycle()
            
            next_run = datetime.now() + timedelta(minutes=SLEEP_BETWEEN_RUNS_MINUTES)
            cprint(f"\n⏳ AI Agent run complete. Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "white", "on_green")
            
            # Sleep until next interval
            time.sleep(INTERVAL)
                
        except KeyboardInterrupt:
            cprint("\n👋 Moon Dev AI Agent shutting down gracefully...", "white", "on_blue")
            break
        except Exception as e:
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")
            # Still sleep and continue on error
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main() 