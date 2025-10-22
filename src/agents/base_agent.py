"""
🌙 Moon Dev's Base Agent
Parent class for all trading agents with unified exchange support
Supports both single exchange and multi-exchange modes
"""

# Standard library from imports
from datetime import datetime

# Third-party from imports
from termcolor import cprint

class BaseAgent:
    def __init__(self, agent_type, use_exchange_manager=False, use_model_priority=False):
        """
        Initialize base agent with type and optional exchange manager

        Args:
            agent_type: Type of agent (e.g., 'trading', 'risk', 'strategy')
            use_exchange_manager: If True, initialize ExchangeManager for unified trading
            use_model_priority: If True, use priority-based model selection with fallback
        """
        self.type = agent_type
        self.start_time = datetime.now()
        self.em = None  # Exchange manager instance
        self.model_priority = None  # Model priority queue instance

        # Initialize model priority system if requested
        if use_model_priority:
            try:
                from src.config import USE_MODEL_PRIORITY, OPENAI_ANTHROPIC_ONLY

                if USE_MODEL_PRIORITY:
                    from src.models.model_priority import model_priority_queue
                    self.model_priority = model_priority_queue

                    # Configure for OpenAI/Anthropic only if requested
                    if OPENAI_ANTHROPIC_ONLY:
                        self.model_priority.set_openai_anthropic_only(True)
                        cprint(f"🎯 {agent_type.capitalize()} agent using OpenAI & Anthropic ONLY", "cyan")
                    else:
                        cprint(f"🎯 {agent_type.capitalize()} agent using model priority with fallback", "cyan")
                else:
                    cprint(f"ℹ️ Model priority disabled in config", "blue")
            except Exception as e:
                cprint(f"⚠️ Could not initialize model priority: {str(e)}", "yellow")

        # Initialize exchange manager if requested
        if use_exchange_manager:
            try:
                from src.config import USE_MULTI_EXCHANGE, ACTIVE_EXCHANGES, EXCHANGE

                if USE_MULTI_EXCHANGE:
                    # Use multi-exchange manager
                    from src.multi_exchange_manager import MultiExchangeManager
                    self.em = MultiExchangeManager(active_exchanges=ACTIVE_EXCHANGES)
                    cprint(f"✅ {agent_type.capitalize()} agent initialized with MULTI-EXCHANGE mode", "green")
                    self.exchange = 'multi'
                else:
                    # Use single exchange manager
                    from src.exchange_manager import ExchangeManager
                    self.em = ExchangeManager()
                    cprint(f"✅ {agent_type.capitalize()} agent initialized with {EXCHANGE} exchange", "green")
                    self.exchange = EXCHANGE

            except Exception as e:
                cprint(f"⚠️ Could not initialize ExchangeManager: {str(e)}", "yellow")
                cprint("   Falling back to direct nice_funcs imports", "yellow")

                # Fallback to direct imports
                from src import nice_funcs as n
                self.n = n
                self.exchange = 'solana'  # Default fallback

    def get_active_tokens(self):
        """Get the appropriate token/symbol list based on active exchange"""
        try:
            from src.config import get_active_tokens
            return get_active_tokens()
        except:
            from src.config import MONITORED_TOKENS
            return MONITORED_TOKENS

    def run(self):
        """Default run method - should be overridden by child classes"""
        raise NotImplementedError("Each agent must implement its own run method") 