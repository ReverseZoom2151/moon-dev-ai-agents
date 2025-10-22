# Multi-Exchange Trading Guide

## Overview

Moon Dev's AI agents now support trading across **multiple exchanges simultaneously**:
- **Solana** (DEX via Jupiter)
- **HyperLiquid** (Perpetual futures)
- **Binance** (Spot trading)
- **Bitfinex** (Spot trading)

This guide shows you how to configure and use multi-exchange trading.

---

## Quick Start

### 1. Add API Credentials

Edit your `.env` file:

```bash
# For Binance
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# For Bitfinex
BITFINEX_API_KEY=your_key_here
BITFINEX_API_SECRET=your_secret_here

# Existing (Solana/HyperLiquid remain the same)
SOLANA_PRIVATE_KEY=your_key_here
HYPER_LIQUID_KEY=your_key_here
```

### 2. Enable Multi-Exchange Mode

Edit `src/config.py`:

```python
# Change this:
USE_MULTI_EXCHANGE = False

# To this:
USE_MULTI_EXCHANGE = True

# Specify which exchanges to use:
ACTIVE_EXCHANGES = ['binance', 'bitfinex', 'solana']
```

### 3. Test with Demo

```bash
python demo_multi_exchange.py
```

This will show:
- Which exchanges you have configured
- Smart order routing examples
- Aggregated positions across all exchanges
- Combined balances

---

## Features

### 🎯 Smart Order Routing

The system **automatically** routes orders to the correct exchange:

```python
from src.multi_exchange_manager import MultiExchangeManager

mem = MultiExchangeManager()

# Automatically routed to Binance (BTC/USDT pair)
mem.route_order('BTC/USDT', 'buy', 100.0)

# Automatically routed to Bitfinex (ETH/USD pair)
mem.route_order('ETH/USD', 'buy', 100.0)

# Automatically routed to HyperLiquid (BTC perpetual)
mem.route_order('BTC', 'buy', 100.0)

# Automatically routed to Solana (token address)
mem.route_order('9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump', 'buy', 100.0)
```

### 📊 Aggregated Positions

View all positions across all exchanges in one DataFrame:

```python
positions = mem.get_aggregated_positions()
print(positions)

# Output:
#   exchange   symbol      quantity  current_price  value_usd
#   binance    BTC/USDT    0.005     65000          325.00
#   bitfinex   ETH/USD     0.5       3500           1750.00
#   solana     TOKEN...    1000      0.50           500.00
```

### 💰 Unified Balance Checking

```python
balances = mem.get_balance()
print(balances)

# Output:
# {
#   'binance': 1500.00,
#   'bitfinex': 2000.00,
#   'solana': 500.00
# }

total = mem.get_total_portfolio_value()
print(f"Total: ${total:,.2f}")
```

### 🔀 Manual Exchange Override

Force a specific exchange if needed:

```python
# Force Binance even for ambiguous symbol
mem.market_buy('BTC', 100.0, exchange='binance')

# Force Bitfinex
mem.market_sell('ETH', 0.5, exchange='bitfinex')
```

---

## Exchange-Specific Behavior

### Binance
- **Type**: Spot trading
- **Symbol Format**: `BTC/USDT`, `ETH/USDT`
- **Quote Currency**: USDT (Tether)
- **Features**: Largest liquidity, widest token selection

### Bitfinex
- **Type**: Spot trading
- **Symbol Format**: `BTC/USD`, `ETH/USD`
- **Quote Currency**: USD (not USDT)
- **Features**: Professional trading, margin available

### HyperLiquid
- **Type**: Perpetual futures
- **Symbol Format**: `BTC`, `ETH`, `SOL` (no slash)
- **Features**: Up to 50x leverage, on-chain settlement

### Solana
- **Type**: DEX (Jupiter aggregator)
- **Symbol Format**: Token addresses (long strings)
- **Features**: Decentralized, MEV protection, fast

---

## Using in Agents

### With BaseAgent (Automatic)

Agents that extend `BaseAgent` automatically get multi-exchange support:

```python
from src.agents.base_agent import BaseAgent

class MyTradingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type='trading',
            use_exchange_manager=True  # This enables exchange manager
        )

    def run(self):
        # If USE_MULTI_EXCHANGE = True in config:
        # self.em is now a MultiExchangeManager

        # Smart routing works automatically
        self.em.route_order('BTC/USDT', 'buy', 100.0)

        # Get all positions
        positions = self.em.get_aggregated_positions()
        print(positions)
```

### Direct Usage (Manual)

```python
from src.multi_exchange_manager import MultiExchangeManager

# Initialize with specific exchanges
mem = MultiExchangeManager(active_exchanges=['binance', 'bitfinex'])

# Or auto-detect from .env
mem = MultiExchangeManager()

# Market orders
buy_result = mem.market_buy('BTC/USDT', 100.0)
sell_result = mem.market_sell('ETH/USDT', 0.5)

# Smart routing
mem.route_order('BTC/USDT', 'buy', 100.0)

# Check positions
positions = mem.get_aggregated_positions()
total_value = mem.get_total_portfolio_value()
```

---

## Symbol/Token Mapping

The system uses these rules to determine which exchange:

| Pattern | Example | Exchange |
|---------|---------|----------|
| Contains `/USDT` | `BTC/USDT` | Binance |
| Contains `/USD` (not `/USDT`) | `BTC/USD` | Bitfinex |
| Uppercase, no `/` | `BTC`, `ETH` | HyperLiquid |
| Long alphanumeric | `9BB6NFE...` | Solana |

You can customize this in `MultiExchangeManager.TOKEN_EXCHANGE_MAP`.

---

## Configuration Options

In `src/config.py`:

```python
# Enable/disable multi-exchange mode
USE_MULTI_EXCHANGE = True

# Which exchanges to activate
ACTIVE_EXCHANGES = ['solana', 'binance', 'bitfinex']

# Default exchange for ambiguous orders
EXCHANGE = 'binance'
```

---

## Testing Without Real Money

### 1. Demo Mode
```bash
python demo_multi_exchange.py
```

### 2. Paper Trading
- Don't add private keys
- Only read-only API keys
- Test order routing logic without execution

### 3. Small Positions
- Start with $10-20 positions
- Test one exchange at a time
- Verify fills before scaling up

---

## Security Best Practices

### API Key Permissions

**Binance:**
- ✅ Enable: Spot Trading, Read Info
- ❌ Disable: Withdraw, Universal Transfer

**Bitfinex:**
- ✅ Enable: Trading, Read Wallet
- ❌ Disable: Withdraw, Transfer

**General:**
- Use IP whitelist when possible
- Rotate keys regularly
- Never commit `.env` to git
- Use separate keys for testing

### Risk Management

```python
# In your agent:
def safe_trade(self, symbol, amount):
    # Check balance first
    balances = self.em.get_balance()
    if balances.get('binance', 0) < amount:
        return "Insufficient balance"

    # Small test order first
    test_result = self.em.market_buy(symbol, 10.0)

    if test_result.get('error'):
        return f"Test failed: {test_result['error']}"

    # Execute full order
    return self.em.market_buy(symbol, amount)
```

---

## Troubleshooting

### "No exchanges initialized"
- Check `.env` file has API keys
- Verify key format (no spaces, quotes)
- Test keys on exchange website first

### "Exchange X not available"
- Ensure `ACTIVE_EXCHANGES` includes the exchange
- Check API key permissions
- Verify network connectivity

### "Symbol not supported"
- Check symbol format for that exchange
- Use `/USDT` for Binance, `/USD` for Bitfinex
- Verify token exists on that exchange

### Order fails on specific exchange
- Check account balance
- Verify trading permissions
- Check minimum order size
- Review exchange-specific errors

---

## Advanced Usage

### Custom Exchange Priority

```python
mem = MultiExchangeManager(
    active_exchanges=['binance', 'bitfinex'],
    default_exchange='binance'
)

# Ambiguous orders go to default
mem.market_buy('BTC', 100.0)  # Goes to Binance
```

### Exchange-Specific Configuration

```python
# Get specific exchange object
binance = mem.active_exchanges['binance']

# Use exchange-specific methods
ticker = binance.get_ticker('BTC/USDT')
orderbook = binance.get_order_book('BTC/USDT', limit=20)
```

### Cross-Exchange Arbitrage

```python
# Get prices from multiple exchanges
binance_price = mem.active_exchanges['binance'].get_ticker('BTC/USDT')['last']
bitfinex_price = mem.active_exchanges['bitfinex'].get_ticker('BTC/USD')['last']

if binance_price < bitfinex_price * 0.99:  # 1% spread
    mem.market_buy('BTC/USDT', 100.0, exchange='binance')
    mem.market_sell('BTC/USD', amount, exchange='bitfinex')
```

---

## Files Reference

### New Files Created
- `src/exchange/base_exchange.py` - Exchange interface
- `src/exchange/binance_exchange.py` - Binance implementation
- `src/exchange/bitfinex_exchange.py` - Bitfinex implementation
- `src/multi_exchange_manager.py` - Multi-exchange manager
- `demo_multi_exchange.py` - Demo script

### Modified Files
- `src/config.py` - Added `USE_MULTI_EXCHANGE` and `ACTIVE_EXCHANGES`
- `src/agents/base_agent.py` - Integrated multi-exchange support
- `.env` - Added Binance/Bitfinex API key placeholders

---

## Next Steps

1. ✅ Run `python demo_multi_exchange.py`
2. ✅ Add API keys to `.env` (read-only first)
3. ✅ Enable in `config.py`
4. ✅ Test with small positions
5. ✅ Monitor aggregated positions
6. ✅ Scale up gradually

---

## Support

- **GitHub Issues**: [Report bugs](https://github.com/moondevonyt/moon-dev-ai-agents/issues)
- **Documentation**: See other guides in this repo
- **Demo**: Run `python demo_multi_exchange.py`

Built with 🌙 by Moon Dev
