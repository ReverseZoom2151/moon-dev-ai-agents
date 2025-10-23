import pandas as pd
import talib
from backtesting import Backtest, Strategy
from pathlib import Path

class TestStrategy(Strategy):
    def init(self):
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)

    def next(self):
        if self.data.Close[-1] > self.sma[-1]:
            self.buy()

# Load data using relative path (works on Windows and Unix)
data_path = Path(__file__).parent.parent.parent / 'data' / 'rbi' / 'BTC-USD-15m.csv'
data = pd.read_csv(data_path)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
# Drop any empty columns (from trailing commas)
data = data.dropna(axis=1, how='all')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

bt = Backtest(data, TestStrategy, cash=1000000)
stats = bt.run()
print(stats)