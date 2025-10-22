import pandas as pd
import talib
from backtesting import Backtest, Strategy

class TestStrategy(Strategy):
    def init(self):
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)
    
    def next(self):
        if self.data.Close[-1] > self.sma[-1]:
            self.buy()

# Load data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

bt = Backtest(data, TestStrategy, cash=1000000)
stats = bt.run()
print(stats)