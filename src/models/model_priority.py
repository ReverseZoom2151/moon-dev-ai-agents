"""
Model Priority System with Automatic Fallback
Prioritizes OpenAI and Anthropic, falls back to others if unavailable
"""
from typing import Optional, List, Tuple, Dict
from enum import Enum
from termcolor import cprint
from .model_factory import model_factory


class ModelPriority(Enum):
    """Priority levels for different use cases"""
    CRITICAL = 1      # Production trading decisions (OpenAI GPT-4, Claude Opus)
    HIGH = 2          # Important analysis (OpenAI GPT-4o, Claude Sonnet)
    MEDIUM = 3        # General tasks (Anthropic Haiku, Groq, Gemini)
    LOW = 4           # Quick tasks, content generation (xAI, DeepSeek, Ollama)


class ModelPriorityQueue:
    """
    Manages model priority chains with automatic fallback

    Example:
        CRITICAL priority chain: OpenAI GPT-4 -> Claude Opus -> xAI Grok
        If GPT-4 fails (API down, rate limit), automatically try Claude
        If Claude fails, try xAI Grok as last resort
    """

    # Default priority chains (provider, model_name)
    DEFAULT_PRIORITY_CHAINS = {
        ModelPriority.CRITICAL: [
            ("openai", "gpt-4o"),                    # Primary: GPT-4 Optimized
            ("claude", "claude-3-5-sonnet-latest"),  # Secondary: Claude Sonnet
            ("xai", "grok-2-fast-reasoning"),        # Tertiary: Grok reasoning
            ("gemini", "gemini-2.5-pro"),            # Backup: Gemini Pro
        ],

        ModelPriority.HIGH: [
            ("openai", "gpt-4o"),                    # Primary: GPT-4o
            ("claude", "claude-3-5-sonnet-latest"),  # Secondary: Claude Sonnet
            ("gemini", "gemini-2.5-pro"),            # Tertiary: Gemini Pro
            ("groq", "llama-3.3-70b-versatile"),     # Backup: Fast Groq
        ],

        ModelPriority.MEDIUM: [
            ("claude", "claude-3-5-haiku-latest"),   # Primary: Fast Claude
            ("openai", "gpt-4o-mini"),               # Secondary: Mini GPT-4
            ("groq", "mixtral-8x7b-32768"),          # Tertiary: Fast Mixtral
            ("gemini", "gemini-2.5-flash"),          # Backup: Fast Gemini
        ],

        ModelPriority.LOW: [
            ("groq", "mixtral-8x7b-32768"),          # Primary: Fast and cheap
            ("xai", "grok-4-fast-reasoning"),        # Secondary: xAI fast
            ("deepseek", "deepseek-reasoner"),       # Tertiary: DeepSeek
            ("ollama", "llama3.2"),                  # Backup: Local Ollama
        ]
    }

    def __init__(self):
        """Initialize priority queue with default chains"""
        self.priority_chains = self.DEFAULT_PRIORITY_CHAINS.copy()
        self.fallback_history: Dict[str, int] = {}  # Track fallback frequency

        cprint("\n🎯 Model Priority System Initialized", "cyan")
        self._print_available_chains()

    def _print_available_chains(self):
        """Print which priority chains have available models"""
        cprint("═" * 60, "cyan")
        for priority in ModelPriority:
            chain = self.priority_chains[priority]
            available = []
            for provider, model_name in chain:
                if model_factory.is_model_available(provider):
                    available.append(f"{provider}:{model_name}")

            if available:
                cprint(f"  ✅ {priority.name}: {' -> '.join(available[:3])}", "green")
            else:
                cprint(f"  ⚠️ {priority.name}: No models available!", "yellow")
        cprint("═" * 60, "cyan")

    def get_model(self, priority: ModelPriority, system_prompt: str = "",
                  user_content: str = "", temperature: float = 0.7,
                  max_tokens: Optional[int] = None):
        """
        Get response from highest priority available model with automatic fallback

        Args:
            priority: Priority level determining which chain to use
            system_prompt: System prompt for the model
            user_content: User message content
            temperature: Model temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response, provider_used, model_name_used)
        """
        chain = self.priority_chains[priority]

        for idx, (provider, model_name) in enumerate(chain):
            # Check if this provider is available
            if not model_factory.is_model_available(provider):
                cprint(f"  ⏭️ Skipping {provider} (not available)", "yellow")
                continue

            try:
                # Get model instance
                model = model_factory.get_model(provider, model_name)
                if not model:
                    continue

                # Indicate which model is being used
                if idx == 0:
                    cprint(f"  🎯 Using PRIMARY: {provider}:{model_name}", "green")
                else:
                    cprint(f"  🔄 FALLBACK #{idx}: {provider}:{model_name}", "yellow")
                    self._record_fallback(f"{provider}:{model_name}")

                # Generate response
                response = model.generate_response(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                if response:
                    return (response, provider, model_name)
                else:
                    cprint(f"  ❌ {provider} returned empty response, trying next...", "red")

            except Exception as e:
                cprint(f"  ❌ {provider} failed: {str(e)[:100]}", "red")
                cprint(f"  🔄 Falling back to next model in chain...", "yellow")
                continue

        # All models in chain failed
        cprint(f"  ❌ ALL models failed for {priority.name} priority!", "red")
        return (None, None, None)

    def _record_fallback(self, model_key: str):
        """Record fallback usage for monitoring"""
        if model_key not in self.fallback_history:
            self.fallback_history[model_key] = 0
        self.fallback_history[model_key] += 1

    def add_priority_chain(self, priority: ModelPriority, models: List[Tuple[str, str]]):
        """
        Add or update a custom priority chain

        Args:
            priority: Priority level
            models: List of (provider, model_name) tuples

        Example:
            queue.add_priority_chain(
                ModelPriority.CRITICAL,
                [("openai", "gpt-4"), ("claude", "opus-3"), ("groq", "mixtral")]
            )
        """
        self.priority_chains[priority] = models
        cprint(f"✅ Updated {priority.name} priority chain", "green")

    def get_fallback_stats(self) -> Dict[str, int]:
        """Get statistics on fallback usage"""
        return self.fallback_history.copy()

    def get_available_models_for_priority(self, priority: ModelPriority) -> List[Tuple[str, str]]:
        """
        Get list of available models for a given priority level

        Returns:
            List of (provider, model_name) tuples that are currently available
        """
        chain = self.priority_chains[priority]
        available = []

        for provider, model_name in chain:
            if model_factory.is_model_available(provider):
                available.append((provider, model_name))

        return available

    def set_openai_anthropic_only(self, enable: bool = True):
        """
        Configure to use ONLY OpenAI and Anthropic (user preference)

        Args:
            enable: If True, filter out all non-OpenAI/Anthropic models
        """
        if enable:
            cprint("\n🎯 Configuring OpenAI & Anthropic ONLY mode", "cyan")

            for priority in ModelPriority:
                original_chain = self.priority_chains[priority]
                filtered_chain = [
                    (provider, model) for provider, model in original_chain
                    if provider in ['openai', 'claude']
                ]

                if filtered_chain:
                    self.priority_chains[priority] = filtered_chain
                    cprint(f"  ✅ {priority.name}: {len(filtered_chain)} models", "green")
                else:
                    cprint(f"  ⚠️ {priority.name}: No OpenAI/Anthropic models available!", "yellow")
        else:
            # Restore defaults
            self.priority_chains = self.DEFAULT_PRIORITY_CHAINS.copy()
            cprint("\n✅ Restored default priority chains (all providers)", "green")

    def __str__(self):
        return f"ModelPriorityQueue(chains={len(self.priority_chains)})"

    def __repr__(self):
        return self.__str__()


# Create singleton instance
model_priority_queue = ModelPriorityQueue()
