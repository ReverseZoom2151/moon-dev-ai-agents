# Exchange Orchestrator Integration Plan

## Current Status

### ✅ Already Integrated (4 sources)
1. **Binance** - Spot trading, OHLCV data
2. **Bitfinex** - Spot trading, OHLCV data
3. **CoinGecko** - Price data (10K+ tokens, FREE)
4. **Birdeye** - Solana token data (PAID)

### ❌ Missing from Orchestrator (4 sources)
1. **HyperLiquid** - Perpetuals exchange (100+ symbols, leverage trading)
2. **Jupiter Protocol** - Solana DEX aggregator (best swap rates)
3. **Moon Dev API** - Custom signals (liquidations, funding, copy trades)
4. **Solana RPC** - Blockchain access (wrapped in nice_funcs.py)

---

## Implementation Priority

### Phase 1: High-Value Trading Sources
**Target:** Add major trading platforms

#### 1.1 HyperLiquid Exchange Adapter
- **Purpose:** Perpetuals trading with leverage (1-50x)
- **Existing Code:** `src/nice_funcs_hl.py`, `src/nice_funcs_hyperliquid.py`
- **Symbols:** BTC, ETH, SOL, DOGE, PEPE, WIF, BONK + 100 more
- **Features:**
  - OHLCV data (up to 5000 candles)
  - Perpetual positions
  - Leverage management
  - Funding rates
  - L2 order book
- **API Key:** `HYPER_LIQUID_KEY`

#### 1.2 Jupiter Exchange Adapter (Solana DEX)
- **Purpose:** Token swaps on Solana with best rates
- **Existing Code:** `src/nice_funcs.py` (jupiter functions)
- **Features:**
  - Swap quotes with slippage
  - Execute swaps
  - Price impact calculations
  - Multi-route aggregation
- **API Key:** None required (public API)
- **Base URL:** `https://lite-api.jup.ag/swap/v1`

### Phase 2: Data & Signal Sources
**Target:** Add custom data feeds

#### 2.1 Moon Dev API Adapter
- **Purpose:** Custom trading signals and market data
- **Existing Code:** `src/agents/api.py`
- **Data Available:**
  - Liquidation data (`liq_data.csv`)
  - Funding rates (`funding.csv`)
  - Open interest (`oi_data.csv`, `oi_total.csv`)
  - New token launches (`token_addresses.csv`)
  - Copy trading signals (`copybot_follow_list.csv`)
  - Recent transactions (`copybot_recent_txs.csv`)
- **API Key:** `MOONDEV_API_KEY`
- **Base URL:** `http://api.moondev.com:8000`
- **Rate Limit:** 100 requests/minute

#### 2.2 Solana RPC Integration
- **Purpose:** Direct blockchain access
- **Existing Code:** Already wrapped in `src/nice_funcs.py`
- **Decision:** Keep as-is (specialized functions), no adapter needed
- **Functions:**
  - `fetch_wallet_holdings_og()` - Get all token balances
  - `fetch_wallet_token_single()` - Get single token balance
  - Transaction submission

---

## Smart Routing Enhancement

### Current Priority Chain (Data Fetching)
```
1. Binance (FREE, 200+ tokens)
2. Bitfinex (FREE, 100+ tokens)
3. CoinGecko (FREE, 10K+ tokens)
4. Birdeye (PAID, Solana tokens)
```

### Enhanced Priority Chain (After Integration)
```
DATA FETCHING:
1. Binance (FREE, spot, 200+ tokens)
2. Bitfinex (FREE, spot, 100+ tokens)
3. HyperLiquid (FREE*, perps, 100+ tokens) *requires account
4. CoinGecko (FREE, prices only, 10K+ tokens)
5. Birdeye (PAID, Solana OHLCV)

TRADING EXECUTION:
- Spot: Binance, Bitfinex
- Perps: HyperLiquid (leverage trading)
- Solana: Jupiter (DEX swaps)

SIGNALS & ANALYTICS:
- Moon Dev API (liquidations, funding, copy trades)
```

---

## Agents to Refactor

### High Priority (Direct API Usage)
1. **trading_agent.py** - Uses OHLCV data → Use orchestrator
2. **housecoin_agent.py** - Uses Birdeye directly → Use orchestrator
3. **funding_agent.py** - Uses Moon Dev API → Use orchestrator
4. **liquidation_agent.py** - Uses Moon Dev API → Use orchestrator
5. **whale_agent.py** - Uses Moon Dev API → Use orchestrator
6. **listingarb_agent.py** - Uses CoinGecko directly → Use orchestrator
7. **new_or_top_agent.py** - Uses CoinGecko directly → Use orchestrator

### Medium Priority (Mixed Usage)
8. **rbi_agent.py** / **rbi_agent_v2.py** / **rbi_agent_v3.py** - Backtesting with OHLCV
9. **solana_agent.py** - Uses RPC + Birdeye
10. **sentiment_agent.py** - Market data queries
11. **fundingarb_agent.py** - Funding rate arbitrage

### Low Priority (Specialized)
12. **sniper_agent.py** - Real-time trading (may need direct access)
13. **stream_agent.py** - Streaming data (specialized)
14. **risk_agent.py** - Risk calculations (portfolio-focused)

---

## Implementation Steps

### Step 1: Create Exchange Adapters (3 files)
- [ ] `src/exchange/hyperliquid_exchange.py`
- [ ] `src/exchange/jupiter_exchange.py`
- [ ] `src/exchange/moondev_exchange.py`

### Step 2: Update MultiExchangeManager
- [ ] Add HyperLiquid detection and initialization
- [ ] Add Jupiter detection and initialization
- [ ] Add Moon Dev API detection and initialization
- [ ] Update `get_ohlcv()` routing to include HyperLiquid
- [ ] Add specialized methods for perps, swaps, signals

### Step 3: Refactor High-Priority Agents
- [ ] Create helper function: `get_market_data(symbol, timeframe, limit)` in orchestrator
- [ ] Refactor trading_agent.py
- [ ] Refactor housecoin_agent.py
- [ ] Refactor funding_agent.py
- [ ] Refactor liquidation_agent.py
- [ ] Test each refactored agent

### Step 4: Documentation & Testing
- [ ] Update README with orchestrator usage
- [ ] Create test suite for all data sources
- [ ] Performance benchmarks (latency, reliability)
- [ ] Error handling validation

---

## Expected Benefits

### 1. Cost Optimization
- **Before:** Multiple paid APIs (Birdeye, HyperLiquid, Moon Dev)
- **After:** Smart routing minimizes paid API calls
- **Savings:** 80-90% reduction in API costs

### 2. Code Simplification
- **Before:** Each agent implements own data fetching
- **After:** Single orchestrator handles all data sources
- **Lines Saved:** ~500-1000 lines of redundant code

### 3. Reliability
- **Before:** Single point of failure per agent
- **After:** Automatic fallback across all sources
- **Uptime:** 99.9% (multi-source redundancy)

### 4. Maintainability
- **Before:** API changes require updates in multiple files
- **After:** API changes only affect exchange adapters
- **Complexity:** 70% reduction in maintenance burden

### 5. Performance
- **Before:** Sequential API calls in agents
- **After:** Orchestrator can batch/parallel fetch
- **Speed:** 30-50% faster data acquisition

---

## Technical Architecture

### Exchange Adapter Interface
```python
class BaseExchange(ABC):
    @abstractmethod
    def get_ohlcv(symbol, timeframe, limit) -> List[List]

    @abstractmethod
    def get_ticker(symbol) -> Dict

    @abstractmethod
    def market_buy(symbol, amount) -> Dict

    @abstractmethod
    def market_sell(symbol, quantity) -> Dict

    # ... other standard methods
```

### Orchestrator Usage
```python
from src.multi_exchange_manager import MultiExchangeManager

# Initialize (auto-detects available exchanges)
manager = MultiExchangeManager()

# Fetch data (automatic smart routing)
df = manager.get_ohlcv('BTC', timeframe='1h', limit=100)
# Tries: Binance → Bitfinex → HyperLiquid → CoinGecko → Birdeye

# Trading (symbol-based routing)
result = manager.market_buy('BTC/USDT', usd_amount=1000)
# Routes to: Binance (spot) or HyperLiquid (perps) based on config

# Signals (specialized data)
liquidations = manager.get_liquidations('BTC')  # Moon Dev API
funding = manager.get_funding_rates('ETH')      # Moon Dev API
```

---

## Risk Mitigation

### API Rate Limits
- Implement request throttling per source
- Cache frequently accessed data
- Respect rate limit headers

### Error Handling
- Graceful degradation on API failures
- Clear error messages for missing credentials
- Fallback to alternative sources

### Testing Strategy
- Unit tests for each adapter
- Integration tests for orchestrator
- Mock API responses for CI/CD
- Live API testing in staging

---

## Timeline Estimate

- **Phase 1 (Adapters):** 2-3 hours
  - HyperLiquid adapter: 1 hour
  - Jupiter adapter: 1 hour
  - Moon Dev API adapter: 1 hour

- **Phase 2 (Integration):** 1-2 hours
  - Update orchestrator: 1 hour
  - Testing: 1 hour

- **Phase 3 (Refactoring):** 3-4 hours
  - High-priority agents: 2 hours
  - Medium-priority agents: 2 hours

- **Total:** 6-9 hours

---

## Success Metrics

1. **Code Coverage:** All agents use orchestrator for data fetching
2. **API Cost:** 80-90% reduction in paid API usage
3. **Reliability:** 99.9% uptime with multi-source fallback
4. **Performance:** <500ms average data fetch time
5. **Maintainability:** Single point of integration for all data sources

---

## Next Steps

1. ✅ Complete comprehensive survey (DONE)
2. ⏳ Create HyperLiquid exchange adapter (IN PROGRESS)
3. ⏳ Create Jupiter exchange adapter
4. ⏳ Create Moon Dev API adapter
5. ⏳ Update orchestrator smart routing
6. ⏳ Refactor high-priority agents
7. ⏳ Testing and validation
8. ⏳ Documentation and deployment
