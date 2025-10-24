"""
Wrapper to run chartanalysis_agent with correct Python path
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import and run the agent
from src.agents.chartanalysis_agent import ChartAnalysisAgent, SYMBOLS

if __name__ == "__main__":
    # Create and run the agent
    print("\n🌙 Moon Dev's Chart Analysis Agent Starting Up...")
    print("👋 Hey! I'm Chuck, your friendly chart analysis agent! 📊")
    print(f"🎯 Monitoring {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    agent = ChartAnalysisAgent()

    # Run the continuous monitoring cycle
    agent.run()
