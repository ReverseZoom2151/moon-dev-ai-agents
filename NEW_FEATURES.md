# New Features - Multi-Exchange & Model Priority

## What's New?

Two major features have been added to Moon Dev's AI Agents:

1. **Multi-Exchange Trading** - Trade across Solana, HyperLiquid, Binance, and Bitfinex simultaneously
2. **AI Model Priority System** - Intelligent model selection with automatic fallback

---

## 1. Multi-Exchange Trading

### Features
✅ **4 Exchanges Supported**: Solana (DEX), HyperLiquid (Perps), Binance (Spot), Bitfinex (Spot)
✅ **Smart Order Routing**: Automatically selects the right exchange for each symbol
✅ **Aggregated Positions**: View all positions across all exchanges in one place
✅ **Unified Balance**: Check total portfolio value with one command
✅ **No Breaking Changes**: Existing code continues to work

### Quick Enable

1. Add API keys to `.env`:
```bash
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BITFINEX_API_KEY=your_key
BITFINEX_API_SECRET=your_secret
```

2. Enable in `src/config.py`:
```python
USE_MULTI_EXCHANGE = True
ACTIVE_EXCHANGES = ['solana', 'binance', 'bitfinex']
```

3. Test:
```bash
python demo_multi_exchange.py
```

### Documentation
📖 See [MULTI_EXCHANGE_GUIDE.md](MULTI_EXCHANGE_GUIDE.md) for full guide

---

## 2. AI Model Priority System

### Features
✅ **4 Priority Levels**: CRITICAL, HIGH, MEDIUM, LOW
✅ **Automatic Fallback**: If primary model fails, tries backups
✅ **7 Providers Supported**: OpenAI, Anthropic, Groq, Gemini, xAI, DeepSeek, Ollama
✅ **Cost Optimization**: Match model capability to task importance
✅ **OpenAI/Anthropic Only Mode**: Filter out other providers if desired

### Quick Enable

1. Add API keys to `.env`:
```bash
OPENAI_KEY=sk-xxx
ANTHROPIC_KEY=sk-ant-xxx
GROQ_API_KEY=gsk-xxx  # Optional: Fast + FREE
```

2. Enable in `src/config.py`:
```python
USE_MODEL_PRIORITY = True
OPENAI_ANTHROPIC_ONLY = False  # Set to True for OpenAI/Anthropic only
```

3. Test:
```bash
python demo_model_priority.py
```

### Documentation
📖 See [MODEL_PRIORITY_GUIDE.md](MODEL_PRIORITY_GUIDE.md) for full guide

---

## How They Work Together

### Example: Multi-Exchange Trading Agent with Model Priority

```python
from src.agents.base_agent import BaseAgent
from src.models.model_priority import ModelPriority

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type='trading',
            use_exchange_manager=True,    # Enable multi-exchange
            use_model_priority=True       # Enable model priority
        )

    def run(self):
        # Get AI analysis (uses priority system with fallback)
        analysis, provider, model = self.model_priority.get_model(
            priority=ModelPriority.HIGH,
            system_prompt="Analyze this opportunity",
            user_content=f"BTC at $65k, bullish sentiment"
        )

        # Execute on best exchange (automatic routing)
        if "buy" in analysis.content.lower():
            result = self.em.route_order('BTC/USDT', 'buy', 100.0)
            # Automatically routed to Binance
```

---

## Configuration Reference

### src/config.py

```python
# Multi-Exchange Settings
USE_MULTI_EXCHANGE = False  # Set to True to enable
ACTIVE_EXCHANGES = ['solana']  # Add: 'binance', 'bitfinex', 'hyperliquid'

# Model Priority Settings
USE_MODEL_PRIORITY = False  # Set to True to enable
OPENAI_ANTHROPIC_ONLY = False  # Set to True to filter to OpenAI/Anthropic only
```

### .env

```bash
# Exchange APIs (NEW)
BINANCE_API_KEY=
BINANCE_API_SECRET=
BITFINEX_API_KEY=
BITFINEX_API_SECRET=

# AI Models (existing, enhanced)
OPENAI_KEY=
ANTHROPIC_KEY=
GROQ_API_KEY=
GEMINI_KEY=
GROK_API_KEY=
DEEPSEEK_KEY=
```

---

## Demo Scripts

Test each feature independently:

```bash
# Test multi-exchange (shows which exchanges you have configured)
python demo_multi_exchange.py

# Test model priority (shows which AI models are available)
python demo_model_priority.py
```

---

## Files Added

### Multi-Exchange
- `src/exchange/base_exchange.py` - Exchange interface
- `src/exchange/binance_exchange.py` - Binance implementation
- `src/exchange/bitfinex_exchange.py` - Bitfinex implementation
- `src/multi_exchange_manager.py` - Multi-exchange orchestration
- `demo_multi_exchange.py` - Demo script
- `MULTI_EXCHANGE_GUIDE.md` - Full documentation

### Model Priority
- `src/models/model_priority.py` - Priority queue with fallback
- `demo_model_priority.py` - Demo script
- `MODEL_PRIORITY_GUIDE.md` - Full documentation

### Modified Files
- `src/config.py` - Added settings for both features
- `src/agents/base_agent.py` - Integrated both features
- `.env` - Added API key placeholders

---

## Priority Levels Explained

| Level | Use Case | Default Models | Cost |
|-------|----------|---------------|------|
| CRITICAL | Real trading decisions | GPT-4o → Claude Sonnet | High |
| HIGH | Analysis, strategies | GPT-4o → Claude Sonnet | Medium |
| MEDIUM | General tasks | Claude Haiku → GPT-4o-mini | Low |
| LOW | Quick tasks, content | Groq Mixtral → xAI Grok | Very Low |

---

## Exchange Routing Rules

| Symbol Format | Example | Routes To |
|---------------|---------|-----------|
| `/USDT` | `BTC/USDT` | Binance |
| `/USD` (not USDT) | `ETH/USD` | Bitfinex |
| Uppercase, no slash | `BTC`, `SOL` | HyperLiquid |
| Long address | `9BB6NFE...` | Solana |

---

## Backward Compatibility

✅ **Existing agents continue to work unchanged**
✅ **Single exchange mode is default**
✅ **Single model mode is default**
✅ **Enable features via config when ready**

---

## Testing Checklist

### Multi-Exchange
- [ ] Run `python demo_multi_exchange.py`
- [ ] Add API keys to `.env` (read-only first)
- [ ] Set `USE_MULTI_EXCHANGE = True` in config
- [ ] Test with small positions ($10-20)
- [ ] Verify order routing
- [ ] Check aggregated positions

### Model Priority
- [ ] Run `python demo_model_priority.py`
- [ ] Add at least 2 AI provider keys to `.env`
- [ ] Set `USE_MODEL_PRIORITY = True` in config
- [ ] Test different priority levels
- [ ] Verify fallback mechanism
- [ ] Monitor fallback statistics

---

## Security Notes

### Multi-Exchange
⚠️ **API Key Permissions**:
- Binance: Enable "Spot Trading" + "Read Info" only
- Bitfinex: Enable "Trading" + "Read Wallet" only
- Disable withdrawals on both exchanges

⚠️ **Test First**:
- Use read-only keys initially
- Test with small positions
- Verify fills before scaling

### Model Priority
⚠️ **API Keys**:
- Don't commit `.env` to git
- Rotate keys regularly
- Monitor usage/costs
- Use separate keys for testing

---

## Cost Optimization Tips

### Multi-Exchange
- Start with one exchange at a time
- Compare fees before routing
- Use limit orders when possible
- Monitor slippage

### Model Priority
- Use FREE Groq for development
- Use CRITICAL only for real money decisions
- Use MEDIUM/LOW for testing
- Monitor model costs in provider dashboards

---

## Support

📖 **Documentation**:
- [MULTI_EXCHANGE_GUIDE.md](MULTI_EXCHANGE_GUIDE.md) - Detailed exchange guide
- [MODEL_PRIORITY_GUIDE.md](MODEL_PRIORITY_GUIDE.md) - Detailed model guide

🐛 **Issues**: [GitHub Issues](https://github.com/moondevonyt/moon-dev-ai-agents/issues)

🎥 **Video Guides**: [Moon Dev YouTube](https://youtube.com/@moondevonyt)

---

## What's Next?

After testing these features, consider:

1. **Combine Both**: Use AI model priority to make decisions, execute on multiple exchanges
2. **Cross-Exchange Arbitrage**: Monitor price differences, execute on best exchange
3. **Portfolio Rebalancing**: AI analyzes entire portfolio, rebalances across exchanges
4. **Swarm Trading**: Multiple AI models vote on decisions, execute on best exchange

---

Built with 🌙 by Moon Dev

**Version**: 2.0 (Multi-Exchange + Model Priority)
**Date**: 2025-10-22
**Branch**: adriandev
