import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy


class VegaDivergence(Strategy):
    vix_roc_threshold = 0.15
    z_threshold = 1.0
    risk_pct = 0.01
    risk_atr_mult = 1.5
    tp_mult = 2.0
    dte = 45
    max_days_in_trade = 15
    min_dte_exit = 10
    vix_reduce_level = 35.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Indicators
        self.sma10 = self.I(talib.SMA, close, 10)
        self.atr = self.I(talib.ATR, high, low, close, 14)

        # SPX 20-day log return
        rocr_spx_20 = self.I(talib.ROCR, close, 20)
        self.r_spx_20 = self.I(np.log, rocr_spx_20)

        # VIX proxy: NATR as vol proxy, then 20-day log ROC
        natr = self.I(talib.NATR, high, low, close, 14)
        rocr_vix_20 = self.I(talib.ROCR, natr, 20)
        self.r_vix_20 = self.I(np.log, rocr_vix_20)

        # Z-score of r_VIXF_20 over 252 sessions
        mean_rvix_252 = self.I(talib.SMA, self.r_vix_20, 252)
        std_rvix_252 = self.I(talib.STDDEV, self.r_vix_20, 252)
        self.z_r_vix_20 = self.I(lambda r, m, s: (r - m) / (s + 1e-12), self.r_vix_20, mean_rvix_252, std_rvix_252)

        # VIX regime proxy level (scaled NATR)
        self.vix_proxy_level = self.I(lambda x: x * 2.0, natr)

        # State
        self.entry_bar = None
        self.entry_side_sign = None
        self.entry_close = None
        self.entry_dte = None

    def next(self):
        i = len(self.data.Close) - 1
        close = self.data.Close[-1]
        sma10 = self.sma10[-1]
        atr = self.atr[-1]
        r_spx_20 = self.r_spx_20[-1]
        r_vix_20 = self.r_vix_20[-1]
        z_vix_20 = self.z_r_vix_20[-1]
        vix_level = self.vix_proxy_level[-1]

        # Entry logic: Bearish divergence
        divergence = (r_spx_20 >= 0) and ((r_vix_20 >= self.vix_roc_threshold) or (z_vix_20 >= self.z_threshold))

        # Debug prints
        print(f"🌙 [Moon Dev] Bar {i} | Close={close:.2f} SMA10={sma10:.2f} ATR={atr:.2f} | r_SPX20={r_spx_20:.4f} r_VIX20={r_vix_20:.4f} zVIX={z_vix_20:.2f} | VIXproxy={vix_level:.2f}")

        # Manage existing position
        if self.position:
            days_in_trade = i - (self.entry_bar if self.entry_bar is not None else i)
            dte_left = (self.entry_dte if self.entry_dte is not None else self.dte) - days_in_trade

            # SMA10 breach exit
            if self.entry_side_sign is not None:
                if self.entry_side_sign >= 0 and close < sma10:
                    print("🚀 [Moon Dev Exit] SMA10 breach from above detected. Exiting short put-proxy! ✨")
                    self.position.close()
                    self._reset_state()
                    return
                if self.entry_side_sign < 0 and close > sma10:
                    print("🚀 [Moon Dev Exit] SMA10 breach from below detected. Exiting short put-proxy! ✨")
                    self.position.close()
                    self._reset_state()
                    return

            # Time-based exits
            if days_in_trade >= self.max_days_in_trade:
                print(f"⏳ [Moon Dev Exit] Max days in trade {days_in_trade} reached. Exiting! 🌙")
                self.position.close()
                self._reset_state()
                return

            if dte_left < self.min_dte_exit:
                print(f"⏳ [Moon Dev Exit] DTE < {self.min_dte_exit} (left {dte_left}). Exiting! 🌙")
                self.position.close()
                self._reset_state()
                return

            return

        # If no position, check entry
        if divergence and not self.position:
            # Risk sizing via ATR stop distance
            stop_dist = max(atr * self.risk_atr_mult, 1e-8)
            sl_price = float(close + stop_dist)  # short stop
            tp_price = float(close - stop_dist * self.tp_mult)  # profit target

            equity = float(self.equity)
            risk_amount = equity * self.risk_pct
            size = risk_amount / stop_dist

            # Vol regime sizing
            if vix_level > self.vix_reduce_level:
                size *= 0.5
                print(f"🛰️ [Moon Dev] High vol regime detected (proxy {vix_level:.2f} > {self.vix_reduce_level}). Reducing size by 50%.")

            # 2x notional cap using underlying as delta 1 proxy
            max_notional = 2.0 * equity
            current_notional = abs(self.position.size) * close if self.position else 0.0
            available_notional = max(0.0, max_notional - current_notional)
            cap_size = available_notional / max(close, 1e-8)
            size = min(size, cap_size)

            size = int(round(size))
            if size <= 0:
                print("🛰️ [Moon Dev] Position size computed as 0 under leverage/premium constraints. Skipping entry.")
                return

            # Enter short (proxy for buying puts)
            print(f"🌑 [Moon Dev Entry] Vega Divergence detected! Shorting {size} units with SL={sl_price:.2f}, TP={tp_price:.2f} 🚀")
            self.sell(size=size, sl=sl_price, tp=tp_price)

            # Record state for SMA10 exit and DTE tracking
            self.entry_bar = i
            self.entry_side_sign = np.sign(close - sma10)
            self.entry_close = close
            self.entry_dte = self.dte

    def _reset_state(self):
        self.entry_bar = None
        self.entry_side_sign = None
        self.entry_close = None
        self.entry_dte = None


# Data loading and preparation
path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
df = pd.read_csv(path, parse_dates=['datetime'])

# Clean columns
df.columns = df.columns.str.strip().str.lower()
df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')

# Ensure columns are present
required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.set_index('datetime').sort_index()

# Resample to daily to align with strategy logic
daily = pd.DataFrame()
daily['Open'] = df['open'].resample('1D').first()
daily['High'] = df['high'].resample('1D').max()
daily['Low'] = df['low'].resample('1D').min()
daily['Close'] = df['close'].resample('1D').last()
daily['Volume'] = df['volume'].resample('1D').sum()
daily = daily.dropna().copy()

print("🌙 [Moon Dev] Starting backtest for Vega Divergence on daily-resampled data... 🚀")

bt = Backtest(
    daily,
    VegaDivergence,
    cash=1_000_000,
    commission=0.0005,
    hedging=False,
    exclusive_orders=True,
    trade_on_close=False,
    margin=1.0
)

stats = bt.run()
print(stats)
print(stats._strategy)