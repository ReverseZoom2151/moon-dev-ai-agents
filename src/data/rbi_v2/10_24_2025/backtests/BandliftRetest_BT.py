import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy


class BandliftRetest(Strategy):
    # Parameters
    n_bb = 10
    k = 2.0
    sma_period = 20
    n_atr = 14
    M = 3  # regime must hold for M bars
    recent_signal_ignore_bars = 5  # ignore signals within N bars of prior entry
    cooldown_bars = 10  # cooldown after exit
    reentries_allowed = 2  # allowed re-entries after cooldown
    time_stop = 50  # bars
    risk_per_trade = 0.005  # 0.5% default
    aggressive_entry = True  # intrabar tag + reclaim
    max_units = 1_000_000  # cap size to 1,000,000 as requested

    def init(self):
        # Indicators using TA-Lib via self.I wrapper
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS,
            self.data.Close,
            self.n_bb,
            self.k,
            self.k,
            0
        )
        self.baseline = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        # Slope of baseline using LINEARREG_SLOPE on the SMA(20)
        self.baseline_slope = self.I(talib.LINEARREG_SLOPE, self.baseline, 5)
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.n_atr)

        # State
        self.entry_index = None
        self.entry_price = None
        self.highest_since_entry = None
        self.trail_stop = None
        self.last_entry_bar = -10**9
        self.last_exit_bar = -10**9
        self.reentries_used = 0

    def _regime_ok(self):
        if len(self.data.Close) < max(self.sma_period, self.n_bb, self.n_atr) + self.M:
            return False
        lb = self.bb_lower
        bl = self.baseline
        if np.any(np.isnan(lb[-self.M:])) or np.any(np.isnan(bl[-self.M:])):
            return False
        # Regime: BB lower band above SMA20 for M bars, and SMA20 slope > 0
        cond1 = np.all(lb[-self.M:] > bl[-self.M:])
        slope_ok = False
        if not np.isnan(self.baseline_slope[-1]):
            slope_ok = self.baseline_slope[-1] > 0
        return cond1 and slope_ok

    def _trigger_ok(self):
        lb_now = self.bb_lower[-1]
        c = self.data.Close[-1]
        l = self.data.Low[-1]
        if self.aggressive_entry:
            # Intrabar tag with reclaim
            return (l <= lb_now) and (c >= lb_now)
        else:
            # Conservative close below/equal then mean-revert
            return c <= lb_now

    def _cooldown_ok(self):
        return (self.bar_index - self.last_exit_bar) >= self.cooldown_bars

    def _enforce_multi_touch_filter(self):
        return (self.bar_index - self.last_entry_bar) >= self.recent_signal_ignore_bars

    def _reset_reentries_if_regime_breaks(self):
        # If regime is off now, reset the reentry count to allow fresh sequence
        if not self._regime_ok():
            self.reentries_used = 0

    def next(self):
        self._reset_reentries_if_regime_breaks()

        # Update trailing stop if in position
        if self.position:
            # Initialize on the first bar after entry
            if self.entry_index is None:
                self.entry_index = self.bar_index
                self.entry_price = float(self.position.price)
                atr_now = float(self.atr[-1]) if not np.isnan(self.atr[-1]) else None
                if atr_now is None:
                    # Fallback minimal ATR to avoid NaN
                    atr_now = max(1e-8, float(np.nanmean(self.atr[-10:])))
                self.highest_since_entry = float(self.data.High[-1])
                self.trail_stop = self.entry_price - 2.0 * atr_now
                print(f"[🌙 Entry Filled] Bar {self.bar_index} | Price: {self.entry_price:.2f} | Initial Trail: {self.trail_stop:.2f} | ATR: {atr_now:.2f} 🚀")
            else:
                # Update highest high since entry and trail stop
                self.highest_since_entry = max(self.highest_since_entry, float(self.data.High[-1]))
                atr_now = float(self.atr[-1]) if not np.isnan(self.atr[-1]) else max(1e-8, float(np.nanmean(self.atr[-10:])))
                new_trail = self.highest_since_entry - 2.0 * atr_now
                if self.trail_stop is None:
                    self.trail_stop = new_trail
                else:
                    self.trail_stop = max(self.trail_stop, new_trail)

                # Moon Dev trail update print
                print(f"[✨ Trail Update] Bar {self.bar_index} | Highest: {self.highest_since_entry:.2f} | ATR: {atr_now:.2f} | Trail: {self.trail_stop:.2f}")

                # Exit if Low crosses below trail stop (intra-bar), or time stop
                bars_held = self.bar_index - self.entry_index
                if float(self.data.Low[-1]) <= self.trail_stop:
                    print(f"[🌘 Exit - Trail Hit] Bar {self.bar_index} | Close: {float(self.data.Close[-1]):.2f} | Trail: {self.trail_stop:.2f} | Held: {bars_held} bars 🛡️")
                    self.position.close()
                    self.last_exit_bar = self.bar_index
                    self.entry_index = None
                    self.entry_price = None
                    self.highest_since_entry = None
                    self.trail_stop = None
                elif bars_held >= self.time_stop:
                    print(f"[🛰️ Exit - Time Stop] Bar {self.bar_index} | Close: {float(self.data.Close[-1]):.2f} | Held: {bars_held} bars ⏳")
                    self.position.close()
                    self.last_exit_bar = self.bar_index
                    self.entry_index = None
                    self.entry_price = None
                    self.highest_since_entry = None
                    self.trail_stop = None

            return  # manage position only this bar

        # No open position: Check entries
        if not self._regime_ok():
            return

        # Multi-touch and cooldown gating
        if not self._enforce_multi_touch_filter():
            return
        if not self._cooldown_ok() and self.reentries_used >= self.reentries_allowed:
            return
        if not self._trigger_ok():
            return

        # Risk-based position sizing using 2x ATR stop distance
        atr_now = float(self.atr[-1]) if not np.isnan(self.atr[-1]) else None
        if atr_now is None or atr_now <= 0:
            atr_now = max(1e-8, float(np.nanmean(self.atr[-10:])))
        risk_per_unit = 2.0 * atr_now
        equity = float(self.equity)
        risk_amount = self.risk_per_trade * equity
        if risk_per_unit <= 0:
            return
        raw_size = risk_amount / risk_per_unit
        size_units = int(max(1, min(self.max_units, round(raw_size))))

        # Place order
        self.buy(size=size_units)
        self.last_entry_bar = self.bar_index
        # If this is a re-entry after a recent exit, increment usage
        if (self.bar_index - self.last_exit_bar) < (10**9):
            if (self.bar_index - self.last_exit_bar) >= self.cooldown_bars:
                self.reentries_used += 1

        print(f"[🚀 Long Signal] Bar {self.bar_index} | Size: {size_units} | Equity: {equity:.2f} | Risk/Trade: {self.risk_per_trade*100:.2f}% | ATR: {atr_now:.2f} | Regime OK 🌙")


def load_data(path):
    df = pd.read_csv(path, parse_dates=['datetime'])
    # Data hygiene
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')
    # Rename to required columns with proper case
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=rename_map)
    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Sort by datetime and set index
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').set_index('datetime')
    # Coerce numeric
    df[required] = df[required].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=required)
    return df


if __name__ == "__main__":
    data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    data = load_data(data_path)

    bt = Backtest(
        data,
        BandliftRetest,
        cash=100000.0,
        commission=0.0005,
        slippage=0.0,
        exclusive_orders=False
    )

    stats = bt.run()
    print(stats)
    print(stats._strategy)