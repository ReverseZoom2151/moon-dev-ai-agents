# AI Model Priority System Guide

## Overview

Moon Dev's AI agents now support **intelligent model selection with automatic fallback**:
- **Priority Levels**: CRITICAL, HIGH, MEDIUM, LOW
- **Automatic Fallback**: If primary model fails, tries backup models
- **Provider Flexibility**: Use OpenAI, Anthropic, Groq, Gemini, xAI, DeepSeek, or Ollama
- **Cost Optimization**: Match model capability to task importance

This guide shows you how to use the model priority system.

---

## Quick Start

### 1. Add API Keys

Edit your `.env` file with your preferred providers:

```bash
# Primary choices (recommended)
ANTHROPIC_KEY=sk-ant-xxx        # Claude models
OPENAI_KEY=sk-xxx               # GPT-4 models

# Fast/cheap alternatives
GROQ_API_KEY=gsk_xxx            # Mixtral, Llama (very fast)
GEMINI_KEY=xxx                  # Google's Gemini

# Additional options
GROK_API_KEY=xxx                # xAI's Grok (large context)
DEEPSEEK_KEY=xxx                # DeepSeek (cheapest reasoning)
```

### 2. Enable Model Priority

Edit `src/config.py`:

```python
# Enable priority system
USE_MODEL_PRIORITY = True

# Optional: Use ONLY OpenAI and Anthropic
OPENAI_ANTHROPIC_ONLY = False  # Set to True to filter out other providers
```

### 3. Test with Demo

```bash
python demo_model_priority.py
```

This will show:
- Which AI models you have configured
- Priority chains for each level
- Available fallback paths

---

## Priority Levels

### CRITICAL (Level 1)
**Use for**: Production trading decisions, real money on the line

**Default Chain**:
1. OpenAI GPT-4o (primary)
2. Claude Sonnet (backup)
3. xAI Grok-2 (tertiary)
4. Gemini Pro (last resort)

**Example**: Should I buy BTC now? Close this position?

### HIGH (Level 2)
**Use for**: Important analysis, strategy generation

**Default Chain**:
1. OpenAI GPT-4o (primary)
2. Claude Sonnet (backup)
3. Gemini Pro (tertiary)
4. Groq Llama (last resort)

**Example**: Analyze this chart, generate backtest strategy

### MEDIUM (Level 3)
**Use for**: General tasks, data analysis

**Default Chain**:
1. Claude Haiku (primary - fast and efficient)
2. OpenAI GPT-4o-mini (backup)
3. Groq Mixtral (tertiary)
4. Gemini Flash (last resort)

**Example**: Summarize news, format data

### LOW (Level 4)
**Use for**: Quick tasks, content generation, non-critical

**Default Chain**:
1. Groq Mixtral (primary - fastest)
2. xAI Grok-4-fast (backup)
3. DeepSeek (tertiary)
4. Ollama (last resort - local)

**Example**: Generate tweet, quick summary

---

## Using in Code

### Basic Usage

```python
from src.models.model_priority import ModelPriority, model_priority_queue

# For critical trading decision
response, provider, model_name = model_priority_queue.get_model(
    priority=ModelPriority.CRITICAL,
    system_prompt="You are an expert crypto trader.",
    user_content="Should I buy BTC at $65,000? Market sentiment is bullish.",
    temperature=0.7,
    max_tokens=500
)

if response:
    print(f"Used: {provider}:{model_name}")
    print(f"Answer: {response.content}")
else:
    print("All models failed!")
```

### With BaseAgent (Automatic)

Agents that extend `BaseAgent` get automatic integration:

```python
from src.agents.base_agent import BaseAgent
from src.models.model_priority import ModelPriority

class MyTradingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type='trading',
            use_model_priority=True  # Enable priority system
        )

    def run(self):
        if self.model_priority:
            # Use CRITICAL for real trading decisions
            response, provider, model = self.model_priority.get_model(
                priority=ModelPriority.CRITICAL,
                system_prompt="You are a trading expert...",
                user_content="Should I enter this position?"
            )

            # Use MEDIUM for general analysis
            analysis, _, _ = self.model_priority.get_model(
                priority=ModelPriority.MEDIUM,
                system_prompt="Analyze this data...",
                user_content=str(market_data)
            )
```

---

## Automatic Fallback

The system **automatically** tries backup models if primary fails:

```python
# Example: OpenAI is down
response, provider, model = model_priority_queue.get_model(
    priority=ModelPriority.CRITICAL,
    system_prompt="...",
    user_content="..."
)

# Console output:
# 🎯 Using PRIMARY: openai:gpt-4o
# ❌ openai failed: Rate limit exceeded
# 🔄 FALLBACK #1: claude:claude-3-5-sonnet-latest
# ✅ Successfully got response

print(f"Used fallback: {provider}:{model}")
# Output: Used fallback: claude:claude-3-5-sonnet-latest
```

### Fallback Statistics

Track how often fallbacks are used:

```python
stats = model_priority_queue.get_fallback_stats()
print(stats)

# Output:
# {
#   'claude:claude-3-5-sonnet-latest': 5,  # Used 5 times as fallback
#   'groq:mixtral-8x7b-32768': 2           # Used 2 times as fallback
# }
```

---

## OpenAI & Anthropic Only Mode

If you only want to use OpenAI and Anthropic (ignore other providers):

### Option 1: In Config

```python
# src/config.py
OPENAI_ANTHROPIC_ONLY = True
```

### Option 2: Programmatically

```python
from src.models.model_priority import model_priority_queue

# Enable filtering
model_priority_queue.set_openai_anthropic_only(True)

# Now only OpenAI and Anthropic models will be used
response, provider, model = model_priority_queue.get_model(
    priority=ModelPriority.CRITICAL,
    system_prompt="...",
    user_content="..."
)

# provider will be either 'openai' or 'claude'

# Restore all providers
model_priority_queue.set_openai_anthropic_only(False)
```

---

## Custom Priority Chains

Override default chains for your specific needs:

```python
from src.models.model_priority import ModelPriority, model_priority_queue

# Custom CRITICAL chain: Only use your preferred models
model_priority_queue.add_priority_chain(
    ModelPriority.CRITICAL,
    [
        ("openai", "gpt-4o"),           # Primary
        ("claude", "claude-3-opus"),    # Backup
        ("groq", "mixtral-8x7b-32768")  # Emergency
    ]
)

# Custom LOW chain: Fastest models only
model_priority_queue.add_priority_chain(
    ModelPriority.LOW,
    [
        ("groq", "llama-3.3-70b-versatile"),
        ("gemini", "gemini-2.5-flash")
    ]
)
```

---

## Cost Optimization

### Match Priority to Budget

| Priority | Use Case | Cost/1M Tokens | Speed |
|----------|----------|----------------|-------|
| CRITICAL | Real money trades | $15-30 | Medium |
| HIGH | Analysis | $10-20 | Medium |
| MEDIUM | General tasks | $1-5 | Fast |
| LOW | Content gen | $0.10-1 | Very fast |

### Example: Minimize Costs

```python
# Use FREE Groq for most tasks
model_priority_queue.add_priority_chain(
    ModelPriority.MEDIUM,
    [
        ("groq", "mixtral-8x7b-32768"),     # FREE, fast
        ("gemini", "gemini-2.5-flash"),     # FREE
        ("ollama", "llama3.2")              # FREE, local
    ]
)

# Only use paid models for critical decisions
model_priority_queue.add_priority_chain(
    ModelPriority.CRITICAL,
    [
        ("openai", "gpt-4o"),
        ("claude", "claude-3-5-sonnet-latest")
    ]
)
```

---

## Configuration Options

In `src/config.py`:

```python
# Enable/disable priority system
USE_MODEL_PRIORITY = True

# Filter to only OpenAI and Anthropic
OPENAI_ANTHROPIC_ONLY = False

# Temperature and tokens (can still override per-call)
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 1024
```

---

## Available Models

### OpenAI
- `gpt-4o` - Latest GPT-4 optimized (recommended)
- `gpt-4o-mini` - Smaller, faster, cheaper
- `gpt-4-turbo` - Previous generation

### Anthropic (Claude)
- `claude-3-5-sonnet-latest` - Best balanced model
- `claude-3-5-haiku-latest` - Fastest Claude
- `claude-3-opus-latest` - Most capable (expensive)

### Groq (Fast inference)
- `llama-3.3-70b-versatile` - Best Llama model
- `mixtral-8x7b-32768` - Fast, good quality

### Google Gemini
- `gemini-2.5-pro` - Best Gemini model
- `gemini-2.5-flash` - Fastest Gemini

### xAI Grok
- `grok-2-fast-reasoning` - Fast reasoning
- `grok-4-fast-reasoning` - Latest with 2M context

### DeepSeek
- `deepseek-reasoner` - Enhanced reasoning (cheapest)

### Ollama (Local)
- `llama3.2` - Run locally, no API needed

---

## Best Practices

### 1. Cost-Conscious Development

```python
# During development: Use free models
model_priority_queue.add_priority_chain(
    ModelPriority.HIGH,
    [
        ("groq", "mixtral-8x7b-32768"),  # FREE
        ("gemini", "gemini-2.5-pro")     # FREE
    ]
)

# In production: Use best models
model_priority_queue.add_priority_chain(
    ModelPriority.HIGH,
    [
        ("openai", "gpt-4o"),
        ("claude", "claude-3-5-sonnet-latest")
    ]
)
```

### 2. Latency-Sensitive Tasks

```python
# Use Groq for real-time responses
response, _, _ = model_priority_queue.get_model(
    priority=ModelPriority.LOW,  # Groq is primary for LOW
    system_prompt="Quick response needed",
    user_content="Summarize this in 2 sentences",
    temperature=0.5,
    max_tokens=100
)
```

### 3. High-Stakes Decisions

```python
# Use multiple models and compare (swarm consensus)
models_to_try = [
    ("openai", "gpt-4o"),
    ("claude", "claude-3-5-sonnet-latest"),
    ("gemini", "gemini-2.5-pro")
]

responses = []
for provider, model_name in models_to_try:
    model = model_factory.get_model(provider, model_name)
    response = model.generate_response(
        system_prompt="Should I buy BTC?",
        user_content=market_data
    )
    responses.append(response)

# Majority vote or weighted consensus
final_decision = analyze_consensus(responses)
```

### 4. Error Handling

```python
response, provider, model = model_priority_queue.get_model(
    priority=ModelPriority.CRITICAL,
    system_prompt="...",
    user_content="..."
)

if response is None:
    # All models failed
    cprint("❌ All AI models unavailable!", "red")
    # Fallback to rule-based logic
    decision = rule_based_decision(data)
else:
    # Success
    cprint(f"✅ Used: {provider}:{model}", "green")
    decision = parse_response(response)
```

---

## Troubleshooting

### "No models available"
- Check `.env` file has API keys
- Run `python demo_model_priority.py` to see status
- Ensure keys are valid (test on provider websites)

### "All models failed"
- Check API rate limits
- Verify network connectivity
- Check provider status pages
- Review error messages in console

### Fallback not working
- Ensure `USE_MODEL_PRIORITY = True` in config
- Check that backup models have valid API keys
- Review fallback stats: `model_priority_queue.get_fallback_stats()`

### "OPENAI_ANTHROPIC_ONLY not filtering"
- Make sure you called `set_openai_anthropic_only(True)`
- Or set `OPENAI_ANTHROPIC_ONLY = True` in config
- Check that at least one OpenAI or Anthropic key is valid

---

## Advanced Usage

### Context Caching (Claude)

```python
# Claude supports context caching for repeated prompts
from src.models.model_factory import model_factory

model = model_factory.get_model('claude', 'claude-3-5-sonnet-latest')

# First call: Caches the system prompt
response1 = model.generate_response(
    system_prompt=long_system_prompt,  # Will be cached
    user_content="Question 1"
)

# Second call: Reuses cached system prompt (faster, cheaper)
response2 = model.generate_response(
    system_prompt=long_system_prompt,  # Cache hit!
    user_content="Question 2"
)
```

### Streaming Responses

```python
# For real-time output (if model supports it)
model = model_factory.get_model('openai', 'gpt-4o')

for chunk in model.stream_response(
    system_prompt="...",
    user_content="..."
):
    print(chunk, end='', flush=True)
```

### Custom Temperature per Priority

```python
# High temperature for creative tasks
creative_response, _, _ = model_priority_queue.get_model(
    priority=ModelPriority.LOW,
    system_prompt="Generate a creative tweet",
    user_content="About Bitcoin hitting $65k",
    temperature=0.9  # More creative
)

# Low temperature for analytical tasks
analytical_response, _, _ = model_priority_queue.get_model(
    priority=ModelPriority.CRITICAL,
    system_prompt="Should I buy?",
    user_content=market_data,
    temperature=0.3  # More deterministic
)
```

---

## Files Reference

### New Files Created
- `src/models/model_priority.py` - Priority queue system
- `demo_model_priority.py` - Demo script

### Modified Files
- `src/config.py` - Added `USE_MODEL_PRIORITY` and `OPENAI_ANTHROPIC_ONLY`
- `src/agents/base_agent.py` - Integrated model priority support

### Existing Files Used
- `src/models/model_factory.py` - Manages individual models
- `src/models/base_model.py` - Base interface

---

## Next Steps

1. ✅ Run `python demo_model_priority.py`
2. ✅ Add API keys to `.env`
3. ✅ Enable in `config.py`
4. ✅ Test with different priority levels
5. ✅ Monitor fallback statistics
6. ✅ Customize chains for your needs

---

## Support

- **GitHub Issues**: [Report bugs](https://github.com/moondevonyt/moon-dev-ai-agents/issues)
- **Documentation**: See other guides in this repo
- **Demo**: Run `python demo_model_priority.py`

Built with 🌙 by Moon Dev
