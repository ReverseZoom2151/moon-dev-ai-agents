"""
Model Priority System Demo
Demonstrates automatic model fallback and priority chains

DEMO MODE - Tests model availability without making API calls
"""

# Standard library imports
import codecs
import sys

# Standard library from imports
from pathlib import Path

# Fix Windows UTF-8 encoding for emojis
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Third-party from imports
from termcolor import cprint

# Local from imports
from src.models.model_factory import model_factory
from src.models.model_priority import ModelPriority, model_priority_queue

def print_header(text):
    """Print a formatted header"""
    cprint("\n" + "="*70, "cyan")
    cprint(f"  {text}", "cyan", attrs=["bold"])
    cprint("="*70, "cyan")

def demo_model_priority():
    """Demonstrate model priority system"""

    print_header("🌙 Moon Dev's Model Priority System Demo")

    cprint("\n📋 This demo showcases:", "yellow")
    cprint("  1. Priority-based model selection (CRITICAL, HIGH, MEDIUM, LOW)", "yellow")
    cprint("  2. Automatic fallback when primary model fails", "yellow")
    cprint("  3. OpenAI/Anthropic-only mode", "yellow")
    cprint("  4. Fallback statistics tracking", "yellow")

    # Show available models
    print_header("1. Available AI Models")

    cprint("\nChecking which AI models you have configured...", "cyan")
    available_count = 0
    for model_type in ['openai', 'claude', 'groq', 'gemini', 'xai', 'deepseek', 'ollama']:
        if model_factory.is_model_available(model_type):
            cprint(f"  ✅ {model_type.upper()}", "green")
            available_count += 1
        else:
            cprint(f"  ❌ {model_type.upper()} (no API key or not available)", "red")

    if available_count == 0:
        cprint("\n⚠️ No AI models available!", "red")
        cprint("Add API keys to .env file:", "yellow")
        cprint("  - OPENAI_KEY", "cyan")
        cprint("  - ANTHROPIC_KEY", "cyan")
        cprint("  - GROQ_API_KEY", "cyan")
        cprint("  - GEMINI_KEY", "cyan")
        cprint("  - GROK_API_KEY", "cyan")
        cprint("  - DEEPSEEK_KEY", "cyan")
        return

    # Show priority chains
    print_header("2. Priority Chains")

    cprint("\nDefault priority chains (primary -> fallback):", "cyan")

    for priority in ModelPriority:
        chain = model_priority_queue.priority_chains[priority]
        available_models = []
        unavailable_models = []

        for provider, model_name in chain:
            if model_factory.is_model_available(provider):
                available_models.append(f"{provider}:{model_name}")
            else:
                unavailable_models.append(f"{provider}:{model_name}")

        cprint(f"\n  {priority.name}:", "yellow", attrs=["bold"])
        if available_models:
            cprint(f"    Available: {' -> '.join(available_models)}", "green")
        if unavailable_models:
            cprint(f"    Unavailable: {' -> '.join(unavailable_models)}", "red")

    # Demonstrate fallback mechanism
    print_header("3. Fallback Mechanism Demo")

    cprint("\n📝 Simulating model priority selection...", "cyan")

    test_cases = [
        (ModelPriority.CRITICAL, "Trading decision (needs most reliable model)"),
        (ModelPriority.HIGH, "Chart analysis (needs good reasoning)"),
        (ModelPriority.MEDIUM, "Token overview (general task)"),
        (ModelPriority.LOW, "Content generation (quick task)"),
    ]

    for priority, description in test_cases:
        cprint(f"\n  Testing {priority.name} priority: {description}", "cyan")
        available = model_priority_queue.get_available_models_for_priority(priority)

        if available:
            primary = available[0]
            cprint(f"    🎯 PRIMARY: {primary[0]}:{primary[1]}", "green")

            if len(available) > 1:
                cprint(f"    🔄 FALLBACKS: {len(available)-1} available", "yellow")
                for idx, (provider, model) in enumerate(available[1:], 1):
                    cprint(f"       #{idx}: {provider}:{model}", "yellow")
        else:
            cprint(f"    ❌ No models available for {priority.name}!", "red")

    # Test OpenAI/Anthropic only mode
    print_header("4. OpenAI & Anthropic ONLY Mode")

    has_openai_or_anthropic = (
        model_factory.is_model_available('openai') or
        model_factory.is_model_available('claude')
    )

    if has_openai_or_anthropic:
        cprint("\n🎯 Enabling OpenAI/Anthropic ONLY mode...", "cyan")
        model_priority_queue.set_openai_anthropic_only(True)

        cprint("\nFiltered priority chains:", "cyan")
        for priority in ModelPriority:
            available = model_priority_queue.get_available_models_for_priority(priority)
            if available:
                models_str = ' -> '.join([f"{p}:{m}" for p, m in available])
                cprint(f"  {priority.name}: {models_str}", "green")
            else:
                cprint(f"  {priority.name}: No OpenAI/Anthropic models available", "red")

        # Restore defaults
        cprint("\n🔄 Restoring default chains...", "cyan")
        model_priority_queue.set_openai_anthropic_only(False)
        cprint("  ✅ All providers restored", "green")
    else:
        cprint("\n⚠️ Neither OpenAI nor Anthropic is available", "yellow")
        cprint("Add OPENAI_KEY or ANTHROPIC_KEY to test this feature", "yellow")

    # Example usage
    print_header("5. Example Usage in Agents")

    cprint("\n📝 Using model priority in your agent code:", "cyan")
    cprint("""
    from src.models.model_priority import ModelPriority, model_priority_queue

    # For critical trading decisions (uses best available model)
    response, provider, model = model_priority_queue.get_model(
        priority=ModelPriority.CRITICAL,
        system_prompt="You are a trading expert...",
        user_content="Should I buy BTC now?",
        temperature=0.7
    )

    # Automatically falls back if primary model fails!
    """, "white")

    cprint("📝 In BaseAgent (already integrated):", "cyan")
    cprint("""
    class MyAgent(BaseAgent):
        def __init__(self):
            # Enable model priority in base agent
            super().__init__(
                agent_type='trading',
                use_model_priority=True
            )

        def run(self):
            if self.model_priority:
                # Use priority system
                response, provider, model = self.model_priority.get_model(
                    priority=ModelPriority.HIGH,
                    system_prompt="...",
                    user_content="..."
                )
    """, "white")

    # Summary
    print_header("✅ Demo Complete")

    cprint("\nTo enable model priority in your agents:", "cyan")
    cprint("  1. Update src/config.py:", "cyan")
    cprint("     - Set USE_MODEL_PRIORITY = True", "cyan")
    cprint("     - Set OPENAI_ANTHROPIC_ONLY = True (optional)", "cyan")
    cprint("  2. Agents using BaseAgent will automatically get model_priority", "cyan")
    cprint("  3. Use model_priority_queue.get_model() for AI requests", "cyan")

    fallback_stats = model_priority_queue.get_fallback_stats()
    if fallback_stats:
        cprint(f"\n📊 Fallback Statistics:", "cyan")
        for model_key, count in fallback_stats.items():
            cprint(f"  {model_key}: Used {count} time(s) as fallback", "yellow")

if __name__ == "__main__":
    demo_model_priority()
