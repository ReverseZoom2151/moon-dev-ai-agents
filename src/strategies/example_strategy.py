"""
🌙 Moon Dev's Example Strategy
Simple Moving Average Crossover Strategy
"""

# Standard library imports
import sys

# Standard library from imports
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import pandas as pd

# Third-party from imports
from termcolor import cprint

# Local from imports
from src import nice_funcs as n
from src.config import MONITORED_TOKENS

# Relative import for base_strategy
try:
    from .base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy

class SimpleMAStrategy(BaseStrategy):
    def __init__(self):
        """Initialize the strategy"""
        super().__init__("Simple MA Crossover")
        self.fast_ma = 20  # 20-period MA
        self.slow_ma = 50  # 50-period MA
        
    def generate_signals(self) -> dict:
        """Generate trading signals based on MA crossover"""
        try:
            for token in MONITORED_TOKENS:
                # Get market data using nice_funcs
                data = n.get_data(token, days_back=3, timeframe='15m')  
                if data is None or data.empty:
                    continue
                    
                # Calculate moving averages
                fast_ma = data['close'].rolling(self.fast_ma).mean()
                slow_ma = data['close'].rolling(self.slow_ma).mean()
                
                # Get latest values
                current_fast = fast_ma.iloc[-1]
                current_slow = slow_ma.iloc[-1]
                prev_fast = fast_ma.iloc[-2]
                prev_slow = slow_ma.iloc[-2]
                
                # Check for crossover
                signal = {
                    'token': token,
                    'signal': 0,
                    'direction': 'NEUTRAL',
                    'metadata': {
                        'strategy_type': 'ma_crossover',
                        'fast_ma': float(current_fast),
                        'slow_ma': float(current_slow),
                        'current_price': float(data['close'].iloc[-1])
                    }
                }
                
                # Bullish crossover (fast crosses above slow)
                if prev_fast <= prev_slow and current_fast > current_slow:
                    signal.update({
                        'signal': 1.0,
                        'direction': 'BUY'
                    })
                
                # Bearish crossover (fast crosses below slow)
                elif prev_fast >= prev_slow and current_fast < current_slow:
                    signal.update({
                        'signal': 1.0,
                        'direction': 'SELL'
                    })
                
                # Validate and format signal
                if self.validate_signal(signal):
                    signal['metadata'] = self.format_metadata(signal['metadata'])
                    return signal
                    
            return None
            
        except Exception as e:
            cprint(f"❌ Error generating signals: {str(e)}", "red")
            return None 