"""
🌙 Moon Dev's OHLCV Data Collector
Collects Open-High-Low-Close-Volume data for specified tokens
Built with love by Moon Dev 🚀
"""

# Standard library imports
import os
import time

# Third-party imports
import pandas as pd

# Standard library from imports
from datetime import datetime

# Third-party from imports
from termcolor import colored, cprint

# Local from imports
from src import nice_funcs as n
from src.config import *

def collect_token_data(token, days_back=DAYSBACK_4_DATA, timeframe=DATA_TIMEFRAME):
    """Collect OHLCV data for a single token"""
    cprint(f"\n🤖 Moon Dev's AI Agent fetching data for {token}...", "white", "on_blue")
    
    try:
        # Get data from Birdeye
        data = n.get_data(token, days_back, timeframe)
        
        if data is None or data.empty:
            cprint(f"❌ Moon Dev's AI Agent couldn't fetch data for {token}", "white", "on_red")
            return None
            
        cprint(f"📊 Moon Dev's AI Agent processed {len(data)} candles for analysis", "white", "on_blue")
        
        # Save data if configured
        if SAVE_OHLCV_DATA:
            save_path = f"data/{token}_latest.csv"
        else:
            save_path = f"temp_data/{token}_latest.csv"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        data.to_csv(save_path)
        cprint(f"💾 Moon Dev's AI Agent cached data for {token[:4]}", "white", "on_green")
        
        return data
        
    except Exception as e:
        cprint(f"❌ Moon Dev's AI Agent encountered an error: {str(e)}", "white", "on_red")
        return None

def collect_all_tokens(tokens=None, days_back=None, timeframe=None):
    """
    Collect OHLCV data for all monitored tokens

    Args:
        tokens: List of token addresses (defaults to MONITORED_TOKENS from config)
        days_back: Days of historical data (defaults to DAYSBACK_4_DATA from config)
        timeframe: Bar timeframe (defaults to DATA_TIMEFRAME from config)
    """
    market_data = {}

    # Use defaults from config if not provided
    if tokens is None:
        tokens = MONITORED_TOKENS
    if days_back is None:
        days_back = DAYSBACK_4_DATA
    if timeframe is None:
        timeframe = DATA_TIMEFRAME

    cprint("\n🔍 Moon Dev's AI Agent starting market data collection...", "white", "on_blue")
    cprint(f"📊 Settings: {days_back} days @ {timeframe} timeframe", "cyan")

    for token in tokens:
        data = collect_token_data(token, days_back, timeframe)
        if data is not None:
            market_data[token] = data

    cprint("\n✨ Moon Dev's AI Agent completed market data collection!", "white", "on_green")

    return market_data

if __name__ == "__main__":
    try:
        collect_all_tokens()
    except KeyboardInterrupt:
        print("\n👋 Moon Dev OHLCV Collector shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("🔧 Moon Dev suggests checking the logs and trying again!") 