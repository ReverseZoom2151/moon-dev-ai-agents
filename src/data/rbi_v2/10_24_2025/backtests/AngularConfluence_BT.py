import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy


def compute_gann_angles(high, low, time_index):
    n = len(high)
    r1x1 = np.full(n, np.nan, dtype=float)
    r1x2 = np.full(n, np.nan, dtype=float)
    r2x1 = np.full(n, np.nan, dtype=float)
    f1x1 = np.full(n, np.nan, dtype=float)
    f1x2 = np.full(n, np.nan, dtype=float)
    f2x1 = np.full(n, np.nan, dtype=float)

    # Ensure time_index is datetime64[ns]
    if not np.issubdtype(time_index.dtype, np.datetime64):
        time_index = pd.to_datetime(time_index)

    dates = pd.DatetimeIndex(time_index).date
    dates = np.array(dates)

    unique_days = pd.unique(dates)
    for d in unique_days:
        idxs = np.where(dates == d)[0]
        if len(idxs) == 0:
            continue
        anchor_i = idxs[0]
        anchor_low = low[anchor_i]
        anchor_high = high[anchor_i]
        steps = np.arange(len(idxs), dtype=float)

        # Gann scaling: 1 price unit per bar for 1x1; 0.5 for 1x2; 2 for 2x1
        r1x1[idxs] = anchor_low + steps * 1.0
        r1x2[idxs] = anchor_low + steps * 0.5
        r2x1[idxs] = anchor_low + steps * 2.0

        f1x1[idxs] = anchor_high - steps * 1.0
        f1x2[idxs] = anchor_high - steps * 0.5
        f2x1[idxs] = anchor_high - steps * 2.0

    return r1x1, r1x2, r2x1, f1x1, f1x2, f2x1


class AngularConfluence(Strategy):
    # Parameters
    risk_pct = 0.005  # 0.5% risk per trade
    min_rr = 1.5
    time_stop_bars = 30
    size_cap = 1_000_000  # enforce size to 1,000,000 as requested
    tick_size = 0.01  # tolerance tick base

    def init(self):
        o, h, l, c = self.data.Open, self.data.High, self.data.Low, self.data.Close

        # Indicators using self.I and TA-Lib
        self.sma5 = self.I(talib.SMA, c, timeperiod=5)
        self.atr14 = self.I(talib.ATR, h, l, c, timeperiod=14)
        self.atr_sma200 = self.I(talib.SMA, self.atr14, timeperiod=200)

        # Candlestick patterns for triggers
        self.bull_engulf = self.I(talib.CDLENGULFING, o, h, l, c)
        self.bear_engulf = self.I(talib.CDLENGULFING, o, h, l, c)  # same function; sign determines direction

        # Gann Angles (session anchored)
        self.r1x1, self.r1x2, self.r2x1, self.f1x1, self.f1x2, self.f2x1 = self.I(
            compute_gann_angles, h, l, self.data.index
        )

        # For state management
        self.entry_price = None
        self.entry_bar = None
        self.initial_stop = None
        self.r_multiple = None
        self.active_angle_side = None  # 'long' uses rising, 'short' uses falling

    def _nearest_rising_angle(self, i, price):
        candidates = [self.r1x1[i], self.r1x2[i], self.r2x1[i]]
        diffs = [abs(price - v) if np.isfinite(v) else np.inf for v in candidates]
        idx = int(np.argmin(diffs))
        return candidates[idx], ['r1x1', 'r1x2', 'r2x1'][idx]

    def _nearest_falling_angle(self, i, price):
        candidates = [self.f1x1[i], self.f1x2[i], self.f2x1[i]]
        diffs = [abs(price - v) if np.isfinite(v) else np.inf for v in candidates]
        idx = int(np.argmin(diffs))
        return candidates[idx], ['f1x1', 'f1x2', 'f2x1'][idx]

    def _intersection_tolerance(self, i):
        atr = float(self.atr14[i])
        return max(0.1 * atr, 2 * self.tick_size)

    def _atr_ok(self, i):
        atr = float(self.atr14[i])
        base = float(self.atr_sma200[i]) if np.isfinite(self.atr_sma200[i]) else atr
        return atr >= 0.6 * base

    def _sma_slope_up(self, i):
        if i < 4:
            return False
        s = self.sma5
        if not all(np.isfinite([s[i], s[i-1], s[i-2], s[i-3]])):
            return False
        return (s[i] - s[i-3]) > 0

    def _sma_slope_down(self, i):
        if i < 4:
            return False
        s = self.sma5
        if not all(np.isfinite([s[i], s[i-1], s[i-2], s[i-3]])):
            return False
        return (s[i] - s[i-3]) < 0

    def _reclaim_angle_long(self, i):
        # Close moved from below to above rising 1x1 in last 3 bars
        if i < 3 or not np.isfinite(self.r1x1[i]):
            return False
        c = self.data.Close
        r = self.r1x1
        conds = []
        for k in range(1, 4):
            conds.append((c[i-k] <= r[i-k]) and (c[i] > r[i]))
        return any(conds)

    def _reclaim_angle_short(self, i):
        # Close moved from above to below falling 1x1 in last 3 bars
        if i < 3 or not np.isfinite(self.f1x1[i]):
            return False
        c = self.data.Close
        f = self.f1x1
        conds = []
        for k in range(1, 4):
            conds.append((c[i-k] >= f[i-k]) and (c[i] < f[i]))
        return any(conds)

    def next(self):
        i = len(self.data.Close) - 1
        c = float(self.data.Close[-1])
        o = float(self.data.Open[-1])
        h = float(self.data.High[-1])
        l = float(self.data.Low[-1])

        sma = float(self.sma5[-1]) if np.isfinite(self.sma5[-1]) else np.nan
        atr = float(self.atr14[-1]) if np.isfinite(self.atr14[-1]) else np.nan
        tol = self._intersection_tolerance(i)

        # Moon Dev Debugging
        print(f"[🌙 AngularConfluence] Bar {i} | Price: {c:.2f} | SMA5: {sma:.2f} | ATR14: {atr:.2f} | Tol: {tol:.4f} 🚀")

        # Manage open position first
        if self.position:
            if self.position.is_long:
                # Intersection-based target: SMA close enough to rising 1x1
                if np.isfinite(self.r1x1[-1]) and abs(self.sma5[-1] - self.r1x1[-1]) <= tol:
                    print("✨ [Moon Dev] SMA met rising 1x1 confluence. Locking lunar profits on LONG! 🌕")
                    self.position.close()
                    self.entry_price = self.entry_bar = self.initial_stop = self.r_multiple = None
                    self.active_angle_side = None
                    return

                # Trail after +1R
                if self.entry_price is not None and self.initial_stop is not None:
                    one_r = self.entry_price - self.initial_stop
                    if (c - self.entry_price) >= one_r:
                        trail_angle = (self.r1x1[-1] if np.isfinite(self.r1x1[-1]) else c) - 0.3 * atr
                        trail_sma = sma - 0.2 * atr if np.isfinite(sma) else c
                        trail = max(trail_angle, trail_sma)
                        if c <= trail:
                            print("🛡️ [Moon Dev] Trailing stop hit on LONG. Banking stardust gains! 🌠")
                            self.position.close()
                            self.entry_price = self.entry_bar = self.initial_stop = self.r_multiple = None
                            self.active_angle_side = None
                            return

                # Time stop
                if self.entry_bar is not None and (i - self.entry_bar) >= self.time_stop_bars:
                    print("⏳ [Moon Dev] Time stop: Exiting LONG due to time decay in orbit. 🛰️")
                    self.position.close()
                    self.entry_price = self.entry_bar = self.initial_stop = self.r_multiple = None
                    self.active_angle_side = None
                    return

            elif self.position.is_short:
                # Intersection-based target: SMA close enough to falling 1x1
                if np.isfinite(self.f1x1[-1]) and abs(self.sma5[-1] - self.f1x1[-1]) <= tol:
                    print("✨ [Moon Dev] SMA met falling 1x1 confluence. Locking lunar profits on SHORT! 🌘")
                    self.position.close()
                    self.entry_price = self.entry_bar = self.initial_stop = self.r_multiple = None
                    self.active_angle_side = None
                    return

                # Trail after +1R
                if self.entry_price is not None and self.initial_stop is not None:
                    one_r = self.initial_stop - self.entry_price
                    if (self.entry_price - c) >= one_r:
                        trail_angle = (self.f1x1[-1] if np.isfinite(self.f1x1[-1]) else c) + 0.3 * atr
                        trail_sma = sma + 0.2 * atr if np.isfinite(sma) else c
                        trail = min(trail_angle, trail_sma)
                        if c >= trail:
                            print("🛡️ [Moon Dev] Trailing stop hit on SHORT. Banking stardust gains! 🌠")
                            self.position.close()
                            self.entry_price = self.entry_bar = self.initial_stop = self.r_multiple = None
                            self.active_angle_side = None
                            return

                # Time stop
                if self.entry_bar is not None and (i - self.entry_bar) >= self.time_stop_bars:
                    print("⏳ [Moon Dev] Time stop: Exiting SHORT due to time decay in orbit. 🛰️")
                    self.position.close()
                    self.entry_price = self.entry_bar = self.initial_stop = self.r_multiple = None
                    self.active_angle_side = None
                    return

            # If managed, skip new entries till next bar
            return

        # No position: Evaluate entries
        if not np.isfinite(sma) or not np.isfinite(atr):
            return

        atr_ok = self._atr_ok(i)

        # Long conditions
        rising_near_val, rising_tag = self._nearest_rising_angle(i, c)
        confluence_long = np.isfinite(rising_near_val) and (abs(c - rising_near_val) <= 0.5 * atr or abs(c - sma) <= 0.5 * atr)
        trend_up = self._sma_slope_up(i) and c > sma and (np.isfinite(self.r1x1[-1]) and c >= self.r1x1[-1] or self._reclaim_angle_long(i))
        trigger_long = (self.bull_engulf[-1] > 0) or (np.isfinite(self.r1x2[-1]) and c > self.r1x2[-1] and self.sma5[-1] < c and self.sma5[-2] < self.data.Close[-2])

        if trend_up and confluence_long and atr_ok and trigger_long:
            # Compute stop
            stop_confluence = min(sma, rising_near_val) - 0.5 * atr
            stop_struct = (self.data.Low[0] if len(self.data.Low) > 0 else l) - 0.5 * atr  # fallback; session anchor low unknown here
            stop_long = max(stop_confluence, stop_struct)  # choose tighter but reasonable
            if stop_long >= c:
                stop_long = c - 0.5 * atr

            # Reward projection to first intersection (approx current rising 1x1 value)
            target_price = self.r1x1[-1] if np.isfinite(self.r1x1[-1]) else c + atr
            reward = max(0.0, target_price - c)
            risk = max(1e-8, c - stop_long)
            rr = reward / risk if risk > 0 else 0

            if rr >= self.min_rr:
                # Position size based on risk percentage (then enforce 1,000,000 units as cap)
                account_risk = self.equity * self.risk_pct
                raw_size = account_risk / risk
                size = int(round(min(max(1, raw_size), self.size_cap)))  # ensure integer, respect cap 1,000,000

                print(f"🚀 [Moon Dev LONG] Trend up + confluence at {rising_tag}. RR={rr:.2f} | Size={size} | SL={stop_long:.2f} 🌙")
                self.buy(size=size, sl=stop_long)
                self.entry_price = c
                self.initial_stop = stop_long
                self.entry_bar = i
                self.r_multiple = rr
                self.active_angle_side = 'long'
                return
            else:
                print(f"🌓 [Moon Dev] LONG rejected: RR {rr:.2f} below {self.min_rr}. Waiting for better lunar alignment.")

        # Short conditions
        falling_near_val, falling_tag = self._nearest_falling_angle(i, c)
        confluence_short = np.isfinite(falling_near_val) and (abs(c - falling_near_val) <= 0.5 * atr or abs(c - sma) <= 0.5 * atr)
        trend_down = self._sma_slope_down(i) and c < sma and (np.isfinite(self.f1x1[-1]) and c <= self.f1x1[-1] or self._reclaim_angle_short(i))
        trigger_short = (self.bear_engulf[-1] < 0) or (np.isfinite(self.f1x2[-1]) and c < self.f1x2[-1] and self.sma5[-1] > c and self.sma5[-2] > self.data.Close[-2])

        if trend_down and confluence_short and atr_ok and trigger_short:
            # Compute stop
            stop_confluence = max(sma, falling_near_val) + 0.5 * atr
            stop_struct = (self.data.High[0] if len(self.data.High) > 0 else h) + 0.5 * atr  # fallback
            stop_short = min(stop_confluence, stop_struct)
            if stop_short <= c:
                stop_short = c + 0.5 * atr

            # Reward projection to first intersection (approx current falling 1x1 value)
            target_price = self.f1x1[-1] if np.isfinite(self.f1x1[-1]) else c - atr
            reward = max(0.0, c - target_price)
            risk = max(1e-8, stop_short - c)
            rr = reward / risk if risk > 0 else 0

            if rr >= self.min_rr:
                # Position size based on risk percentage (then enforce 1,000,000 units as cap)
                account_risk = self.equity * self.risk_pct
                raw_size = account_risk / risk
                size = int(round(min(max(1, raw_size), self.size_cap)))  # ensure integer, respect cap 1,000,000

                print(f"🚀 [Moon Dev SHORT] Trend down + confluence at {falling_tag}. RR={rr:.2f} | Size={size} | SL={stop_short:.2f} 🌑")
                self.sell(size=size, sl=stop_short)
                self.entry_price = c
                self.initial_stop = stop_short
                self.entry_bar = i
                self.r_multiple = rr
                self.active_angle_side = 'short'
                return
            else:
                print(f"🌗 [Moon Dev] SHORT rejected: RR {rr:.2f} below {self.min_rr}. Awaiting better eclipse.")

        # Otherwise no trade
        print("🛰️ [Moon Dev] Standing by... No confluence or RR not met this bar.")


def load_data(path):
    df = pd.read_csv(path)
    # Clean columns
    df.columns = df.columns.str.strip().str.lower()
    # Drop unnamed columns
    df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')

    # Rename to required columns with proper case
    mapping = {}
    if 'datetime' in df.columns:
        mapping['datetime'] = 'Datetime'
    if 'open' in df.columns:
        mapping['open'] = 'Open'
    if 'high' in df.columns:
        mapping['high'] = 'High'
    if 'low' in df.columns:
        mapping['low'] = 'Low'
    if 'close' in df.columns:
        mapping['close'] = 'Close'
    if 'volume' in df.columns:
        mapping['volume'] = 'Volume'
    df = df.rename(columns=mapping)

    # Ensure datetime
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

    # Ensure correct column order and types
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df[required].astype(float)
    return df


if __name__ == "__main__":
    data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    data = load_data(data_path)

    bt = Backtest(
        data,
        AngularConfluence,
        cash=10**14,            # Large cash to support size of 1,000,000 units if needed
        commission=0.0005,
        exclusive_orders=True
    )

    stats = bt.run()
    print(stats)
    print(stats._strategy)