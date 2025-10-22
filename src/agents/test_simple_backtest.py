"""
Simple test backtest to verify backtest_runner.py works
Uses local data from moon-dev-trading-bots
"""
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

print("Moon Dev's Simple SMA Strategy Test")

class SimpleSMA(Strategy):
    """Simple SMA crossover strategy for testing"""
    fast_sma = 10
    slow_sma = 20

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(talib.SMA, close, self.fast_sma)
        self.sma_slow = self.I(talib.SMA, close, self.slow_sma)
        print(f"Strategy initialized: SMA({self.fast_sma}) x SMA({self.slow_sma})")

    def next(self):
        # Buy when fast SMA crosses above slow SMA
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy()
                print(f"BUY at {self.data.Close[-1]:.2f}")

        # Sell when fast SMA crosses below slow SMA
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()
                print(f"SELL at {self.data.Close[-1]:.2f}")

if __name__ == "__main__":
    print("\nLoading data...")

    # Use local AAPL daily data
    data_path = r"C:\Users\adria\Downloads\personal-wall-street\moon-dev-trading-bots\data\yahoo_daily\AAPL_stock_1d.csv"
    data = pd.read_csv(data_path)

    # Standardize column names (handle both formats)
    data.columns = [col.capitalize() for col in data.columns]

    # Set date as index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

    print(f"Loaded {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")

    # Run backtest
    print("\nStarting backtest...")
    bt = Backtest(data, SimpleSMA, cash=10000, commission=0.002)
    stats = bt.run()

    print("\nRESULTS:")
    print("=" * 60)
    print(f"Return:              {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio:        {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:        {stats['Max. Drawdown [%]']:.2f}%")
    print(f"# Trades:            {stats['# Trades']}")
    print(f"Win Rate:            {stats['Win Rate [%]']:.2f}%")
    print("=" * 60)

    print("\nBacktest Complete!")
    print("This proves the backtest runner can execute and capture results!")
