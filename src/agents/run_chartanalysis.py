"""
Wrapper to run chartanalysis_agent with correct Python path
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import and run the agent
from src.agents import chartanalysis_agent
