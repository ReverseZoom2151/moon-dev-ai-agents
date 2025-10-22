# 🤖 Agent API Key Requirements

Quick reference for which agents need which API keys.

## 📊 Legend
- ✅ = Required (agent won't run without it)
- 🟡 = Optional but recommended
- ⚪ = Not needed

---

## 🆓 Agents That Work WITHOUT API Keys

### 1. **api.py** - Moon Dev API Client
**Keys needed:** None (most endpoints work without MOONDEV_API_KEY)
```bash
python src/agents/api.py
```
**What it does:** Fetches liquidation data, funding rates, OI, new tokens

### 2. **backtest_runner.py** - Backtest Executor
**Keys needed:** None
```bash
python src/agents/backtest_runner.py
```
**What it does:** Runs backtest files and captures results

---

## 🔑 Agents Requiring AI Keys ONLY

These agents only need 1-2 AI model keys (Groq, Claude, or OpenAI):

### 3. **rbi_agent.py** - Research → Backtest → Implement
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY or DEEPSEEK_KEY
```bash
python src/agents/rbi_agent.py
```
**What it does:** Generates trading strategies from research

### 4. **tweet_agent.py** - Tweet Generator
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY or DEEPSEEK_KEY
```bash
python src/agents/tweet_agent.py
```
**What it does:** Creates tweets using AI

### 5. **research_agent.py** - Strategy Researcher
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
```bash
python src/agents/research_agent.py
```
**What it does:** Researches trading ideas

---

## 📈 Agents Requiring Market Data APIs

These need both AI keys AND market data APIs:

### 6. **chartanalysis_agent.py** - Chart Analyzer
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY (for AI vision)
- ✅ BIRDEYE_API_KEY (for OHLCV data)
- ✅ RPC_ENDPOINT (for blockchain data)
```bash
cd moon-dev-ai-agents
python -m src.agents.chartanalysis_agent
```
**What it does:** Generates charts, analyzes with AI vision, provides trade signals

### 7. **trading_agent.py** - AI Trading System (Swarm Mode)
**Keys needed:**
- ✅ Multiple AI keys for swarm consensus:
  - ANTHROPIC_KEY (Claude)
  - OPENAI_KEY (GPT-4)
  - GROQ_API_KEY (Fast inference)
  - GEMINI_KEY (Google)
  - GROK_API_KEY (xAI)
  - DEEPSEEK_KEY (Reasoning)
- ✅ BIRDEYE_API_KEY
- ✅ RPC_ENDPOINT
- 🟡 SOLANA_PRIVATE_KEY (only for live trading)
```bash
cd moon-dev-ai-agents
python -m src.agents.trading_agent
```
**What it does:** 6-model AI consensus for trading decisions

### 8. **sentiment_agent.py** - Twitter Sentiment Analysis
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ TWITTER_USERNAME, TWITTER_EMAIL, TWITTER_PASSWORD
- 🟡 ELEVENLABS_API_KEY (for voice alerts)
```bash
python src/agents/sentiment_agent.py
```
**What it does:** Analyzes Twitter sentiment for tokens

### 9. **whale_agent.py** - Whale Tracker
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ RPC_ENDPOINT
- 🟡 ELEVENLABS_API_KEY (for voice alerts)
```bash
python src/agents/whale_agent.py
```
**What it does:** Monitors whale wallet activity

### 10. **liquidation_agent.py** - Liquidation Monitor
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- 🟡 MOONDEV_API_KEY (works without, but limited)
- 🟡 ELEVENLABS_API_KEY (for voice alerts)
```bash
python src/agents/liquidation_agent.py
```
**What it does:** Tracks liquidation spikes with AI analysis

### 11. **funding_agent.py** - Funding Rate Monitor
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- 🟡 MOONDEV_API_KEY (works without)
- 🟡 ELEVENLABS_API_KEY (for voice alerts)
```bash
python src/agents/funding_agent.py
```
**What it does:** Monitors funding rates for arbitrage opportunities

### 12. **fundingarb_agent.py** - Funding Arbitrage
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ RPC_ENDPOINT
- ✅ BIRDEYE_API_KEY
- 🟡 SOLANA_PRIVATE_KEY (for live trading)
```bash
python src/agents/fundingarb_agent.py
```
**What it does:** Finds funding rate arbitrage between HyperLiquid and Solana

### 13. **listingarb_agent.py** - Listing Arbitrage
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ COINGECKO_API_KEY
- ✅ BIRDEYE_API_KEY
```bash
python src/agents/listingarb_agent.py
```
**What it does:** Identifies tokens before major exchange listings

### 14. **sniper_agent.py** - Token Sniper
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ RPC_ENDPOINT
- ✅ BIRDEYE_API_KEY
- 🟡 SOLANA_PRIVATE_KEY (for auto-sniping)
```bash
python src/agents/sniper_agent.py
```
**What it does:** Analyzes and snipes new Solana token launches

---

## 🎥 Content Creation Agents

### 15. **video_agent.py** - Video Creator
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ ELEVENLABS_API_KEY
```bash
python src/agents/video_agent.py
```
**What it does:** Creates videos from text

### 16. **clips_agent.py** - Video Clipper
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
```bash
python src/agents/clips_agent.py
```
**What it does:** Clips long videos into shorter segments

### 17. **chat_agent.py** - YouTube Chat Monitor
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ YOUTUBE_API_KEY
- ✅ RESTREAM_CLIENT_ID, RESTREAM_CLIENT_SECRET, RESTREAM_EMBED_TOKEN
```bash
python src/agents/chat_agent.py
```
**What it does:** Monitors and responds to YouTube live chat

### 18. **phone_agent.py** - Phone Call Handler
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
```bash
python src/agents/phone_agent.py
```
**What it does:** Handles phone calls with AI

---

## 📦 Special Agents

### 19. **swarm_agent.py** - Multi-Model Consensus
**Keys needed:**
- ✅ ANTHROPIC_KEY (Claude)
- ✅ OPENAI_KEY (GPT-4)
- ✅ GEMINI_KEY (Google)
- ✅ GROK_API_KEY (xAI)
- ✅ GROQ_API_KEY (Groq)
- ✅ DEEPSEEK_KEY (DeepSeek)
- 🟡 Ollama running locally (DeepSeek-R1)
```bash
python src/agents/swarm_agent.py
```
**What it does:** Queries 6 AI models for consensus decision-making

### 20. **focus_agent.py** - Productivity Monitor
**Keys needed:**
- ✅ ANTHROPIC_KEY or OPENAI_KEY
- ✅ GOOGLE_APPLICATION_CREDENTIALS
```bash
python src/agents/focus_agent.py
```
**What it does:** Monitors coding sessions for productivity

---

## 🎯 Quick Start Paths

### Path 1: Testing AI Agents (FREE)
**Keys needed:**
1. GROQ_API_KEY (free, generous)
2. ANTHROPIC_KEY ($5 credit)

**What you can run:**
- ✅ RBI agent (strategy generation)
- ✅ Tweet agent
- ✅ Research agent
- ✅ Backtest runner
- ✅ Moon Dev API client

### Path 2: Market Analysis (FREE)
**Keys needed:**
1. GROQ_API_KEY (free)
2. ANTHROPIC_KEY ($5 credit)
3. BIRDEYE_API_KEY (free tier)
4. RPC_ENDPOINT (free from Helius)
5. COINGECKO_API_KEY (free tier)

**What you can run:**
- ✅ Everything from Path 1
- ✅ Chart analysis agent
- ✅ Liquidation agent
- ✅ Funding agent
- ✅ Whale agent
- ✅ Sentiment agent

### Path 3: Live Trading (PAID - Use Caution!)
**Keys needed:**
- All from Path 2, plus:
- SOLANA_PRIVATE_KEY (your wallet)
- HYPER_LIQUID_KEY (your wallet)

**What you can run:**
- ✅ Everything from Paths 1 & 2
- ✅ Trading agent (live execution)
- ✅ Sniper agent (auto-sniping)
- ✅ Funding arbitrage (live trades)

---

## 💡 Pro Tips

1. **Start with just 2 keys:** GROQ_API_KEY + ANTHROPIC_KEY
2. **Test without trading keys first** - All agents work in analysis mode
3. **Birdeye + Helius are free** - Generous free tiers for market data
4. **Don't need all 6 AI models** - Swarm works with 2-3 models too
5. **Private keys only when ready** - Start with paper trading

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
**Solution:** Run from project root:
```bash
cd moon-dev-ai-agents
python -m src.agents.agent_name
```

### "BIRDEYE_API_KEY not found"
**Solution:** Add key to `.env` file or comment out the check in `nice_funcs.py`

### "API rate limit exceeded"
**Solution:** Most free tiers are generous, but wait a bit or upgrade

---

**Last Updated:** 2025-10-22
**Repository:** moon-dev-ai-agents
