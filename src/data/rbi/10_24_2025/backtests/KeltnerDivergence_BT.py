import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy


def pivot_low_series(lows, left=3, right=3):
    arr = np.asarray(lows, dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    for i in range(left, n - right):
        if np.all(arr[i] < arr[i - left:i]) and np.all(arr[i] < arr[i + 1:i + right + 1]):
            out[i] = 1.0
    return out


def rolling_percentile(x, window=100, q=60):
    x = np.asarray(x, dtype=float)
    n = len(x)
    res = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        vals = x[start:i + 1]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 5:
            res[i] = np.percentile(vals, q)
    return res


def diff1(arr):
    arr = np.asarray(arr, dtype=float)
    d = np.full_like(arr, np.nan)
    d[1:] = arr[1:] - arr[:-1]
    return d


def bb_upper_func(close, period=20, nbdev=2):
    up, mid, low = talib.BBANDS(close, timeperiod=int(period), nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return up


def bb_middle_func(close, period=20, nbdev=2):
    up, mid, low = talib.BBANDS(close, timeperiod=int(period), nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return mid


def bb_lower_func(close, period=20, nbdev=2):
    up, mid, low = talib.BBANDS(close, timeperiod=int(period), nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return low


def add_mul(a, b, mult=1.5):
    return np.asarray(a, dtype=float) + float(mult) * np.asarray(b, dtype=float)


def sub_mul(a, b, mult=1.5):
    return np.asarray(a, dtype=float) - float(mult) * np.asarray(b, dtype=float)


def arr_sub(a, b):
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


class KeltnerDivergence(Strategy):
    kc_mult = 1.5
    ema_len = 20
    atr_len = 20
    bb_period = 20
    bb_std = 2
    rsi_period = 14
    ema200_period = 200
    atr14_period = 14
    rsi_div_threshold = 3.0
    percentile_window = 100
    percentile_threshold = 60
    width_sma_period = 20
    div_pivot_left = 3
    div_pivot_right = 3
    div_lookback_bars = 50
    div_valid_bars = 20
    risk_pct = 0.01  # 1% risk per trade
    time_stop_bars = 40

    def init(self):
        # Indicators using self.I and TA-Lib
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low

        # EMA and ATR for Keltner
        self.kc_mid = self.I(talib.EMA, self.close, self.ema_len)
        self.atr20 = self.I(talib.ATR, self.high, self.low, self.close, self.atr_len)
        self.kc_upper = self.I(add_mul, self.kc_mid, self.atr20, self.kc_mult)
        self.kc_lower = self.I(sub_mul, self.kc_mid, self.atr20, self.kc_mult)

        # Bollinger Bands and width
        self.bb_upper = self.I(bb_upper_func, self.close, self.bb_period, self.bb_std)
        self.bb_mid = self.I(bb_middle_func, self.close, self.bb_period, self.bb_std)
        self.bb_lower = self.I(bb_lower_func, self.close, self.bb_period, self.bb_std)
        self.bb_width = self.I(arr_sub, self.bb_upper, self.bb_lower)
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, self.width_sma_period)
        self.bb_width_p60 = self.I(rolling_percentile, self.bb_width, self.percentile_window, self.percentile_threshold)

        # RSI for divergence
        self.rsi = self.I(talib.RSI, self.close, self.rsi_period)

        # EMA200 and slope
        self.ema200 = self.I(talib.EMA, self.close, self.ema200_period)
        self.ema200_slope = self.I(diff1, self.ema200)

        # ATR14 for position sizing and stop buffers
        self.atr14 = self.I(talib.ATR, self.high, self.low, self.close, self.atr14_period)

        # Pivot Lows for divergence detection
        self.pivot_lows = self.I(pivot_low_series, self.low, self.div_pivot_left, self.div_pivot_right)

        # Internal state
        self.last_divergence_idx = -1
        self.setup_active = False
        self.divergence_expiry_idx = -1
        self.l2_low_value = None
        self.l2_index = None

        self.bars_since_entry = None
        self.entry_price = None
        self.initial_stop = None
        self.initial_risk_per_unit = None
        self.moved_to_half_atr_trail = False
        self.moved_to_kc_lower_trail = False
        self.scaled_out_once = False
        self.width_below_sma_count = 0

    def next(self):
        i = len(self.close) - 1
        c = float(self.close[-1])
        h_prev = float(self.high[-2]) if i >= 1 else float(self.high[-1])

        # Update width below SMA count for optional scale-out
        width_now = float(self.bb_width[-1]) if not np.isnan(self.bb_width[-1]) else np.nan
        width_sma_now = float(self.bb_width_sma[-1]) if not np.isnan(self.bb_width_sma[-1]) else np.nan
        if np.isnan(width_now) or np.isnan(width_sma_now):
            self.width_below_sma_count = 0
        else:
            if width_now < width_sma_now:
                self.width_below_sma_count += 1
            else:
                self.width_below_sma_count = 0

        # Detect new bullish divergence signal within last 50 bars
        pivots = np.where(np.asarray(self.pivot_lows)[:i + 1] == 1)[0]
        if len(pivots) >= 2 and not self.position:
            i2 = int(pivots[-1])
            i1 = int(pivots[-2])
            if i2 != self.last_divergence_idx and (i - i2) <= self.div_lookback_bars:
                l1 = float(self.low[i1])
                l2 = float(self.low[i2])
                r1 = float(self.rsi[i1]) if not np.isnan(self.rsi[i1]) else np.nan
                r2 = float(self.rsi[i2]) if not np.isnan(self.rsi[i2]) else np.nan
                kc_mid_l2 = float(self.kc_mid[i2]) if not np.isnan(self.kc_mid[i2]) else np.nan

                if not np.isnan(r1) and not np.isnan(r2) and not np.isnan(kc_mid_l2):
                    price_condition = l2 <= l1
                    rsi_condition = (r2 - r1) >= self.rsi_div_threshold
                    hygiene_condition = l2 <= kc_mid_l2  # prefer divergence near/below KC mid

                    if price_condition and rsi_condition and hygiene_condition:
                        self.setup_active = True
                        self.last_divergence_idx = i2
                        self.divergence_expiry_idx = i2 + self.div_valid_bars
                        self.l2_low_value = l2
                        self.l2_index = i2
                        print(f"🌙 Divergence detected at idx {i2} -> valid until {self.divergence_expiry_idx}. L1={l1:.2f}, L2={l2:.2f}, RSI Δ={r2 - r1:.2f} ✨")

        # Deactivate setup if expired or position opened
        if self.setup_active and (i > self.divergence_expiry_idx or self.position):
            if i > self.divergence_expiry_idx and not self.position:
                print(f"🌙 Divergence setup expired at idx {i}. No breakout trigger. 💤")
            self.setup_active = False

        # Preconditions for entry
        ema200_now = float(self.ema200[-1]) if not np.isnan(self.ema200[-1]) else np.nan
        ema200_slope_now = float(self.ema200_slope[-1]) if not np.isnan(self.ema200_slope[-1]) else np.nan
        kc_upper_now = float(self.kc_upper[-1]) if not np.isnan(self.kc_upper[-1]) else np.nan
        kc_mid_now = float(self.kc_mid[-1]) if not np.isnan(self.kc_mid[-1]) else np.nan
        kc_lower_now = float(self.kc_lower[-1]) if not np.isnan(self.kc_lower[-1]) else np.nan
        atr14_now = float(self.atr14[-1]) if not np.isnan(self.atr14[-1]) else np.nan
        bb60_now = float(self.bb_width_p60[-1]) if not np.isnan(self.bb_width_p60[-1]) else np.nan
        bb_sma_now = width_sma_now

        trend_ok = (not np.isnan(ema200_now) and not np.isnan(ema200_slope_now) and c > ema200_now and ema200_slope_now > 0)
        vol_ok = (not np.isnan(width_now) and not np.isnan(bb_sma_now) and not np.isnan(bb60_now) and width_now > bb_sma_now * 1.10 and width_now > bb60_now)
        breakout_ok = (not np.isnan(kc_upper_now) and c > kc_upper_now)

        # Entry logic: long-only
        if self.setup_active and (i <= self.divergence_expiry_idx) and trend_ok and vol_ok and breakout_ok and not self.position:
            if np.isnan(atr14_now) or np.isnan(kc_mid_now):
                pass
            else:
                # Initial stop as per rules
                s1 = self.l2_low_value - 0.5 * atr14_now if self.l2_low_value is not None else c - 1.0 * atr14_now
                s2 = kc_mid_now - 1.0 * atr14_now
                initial_stop = min(s1, s2)
                entry_price = c  # Next bar market open; approximate with current close for sizing
                risk_per_unit = entry_price - initial_stop
                if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                    print("🌙🚫 Invalid risk per unit, skipping entry.")
                else:
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    pos_size = int(round(risk_amount / risk_per_unit))
                    pos_size = max(pos_size, 1)
                    # Place market order for next bar
                    self.buy(size=pos_size, sl=initial_stop)
                    self.setup_active = False
                    print(f"🚀 Moon Entry Signal! idx={i} size={pos_size} planned_SL={initial_stop:.2f} risk_per_unit={risk_per_unit:.2f} ✨")
                    print(f"   TrendOK={trend_ok}, VolOK={vol_ok}, BreakoutOK={breakout_ok}, BBWidth={width_now:.2f} > SMA*1.10={bb_sma_now*1.10:.2f} and > P60={bb60_now:.2f}")

        # Post-entry management
        if self.position:
            # Initialize after entry
            if self.bars_since_entry is None:
                self.bars_since_entry = 0
                self.entry_price = float(self.position.price)
                # Use current stop if available; store our initial stop estimate
                self.initial_stop = self.position.sl if hasattr(self.position, 'sl') and self.position.sl else (kc_mid_now - 1.0 * atr14_now)
                if self.initial_stop is None or np.isnan(self.initial_stop):
                    self.initial_stop = kc_mid_now - 1.0 * atr14_now
                self.initial_risk_per_unit = max(1e-6, self.entry_price - self.initial_stop)
                self.moved_to_half_atr_trail = False
                self.moved_to_kc_lower_trail = False
                self.scaled_out_once = False
                print(f"🌙 Entered long at {self.entry_price:.2f} with SL {self.initial_stop:.2f}. Initial R/unit={self.initial_risk_per_unit:.2f} ✨")

            # Increment bar counter
            self.bars_since_entry += 1

            # Current R multiple
            r_mult = (c - self.entry_price) / self.initial_risk_per_unit if self.initial_risk_per_unit else 0.0

            # Trailing logic
            if r_mult >= 1.0 and not self.moved_to_half_atr_trail and not np.isnan(kc_mid_now) and not np.isnan(atr14_now):
                new_sl = kc_mid_now - 0.5 * atr14_now
                self.position.set_sl(new_sl)
                self.moved_to_half_atr_trail = True
                print(f"🌙🔒 +1R reached. Moving SL to KC_mid - 0.5*ATR = {new_sl:.2f} ✨")

            if r_mult >= 2.0 and not self.moved_to_kc_lower_trail and not np.isnan(kc_lower_now):
                new_sl2 = kc_lower_now
                self.position.set_sl(new_sl2)
                self.moved_to_kc_lower_trail = True
                print(f"🌙🛡️ +2R reached. Trailing SL to KC_lower = {new_sl2:.2f} 🚀")

            # Primary exit: close below KC lower
            if not np.isnan(kc_lower_now) and c < kc_lower_now:
                print(f"🌙 Exit: Close below KC_lower at idx {i}. Closing full position. ❌")
                self.position.close()
                self._reset_trade_state()

            # Protective early exit within 5 bars: close below KC mid
            elif self.bars_since_entry is not None and self.bars_since_entry <= 5 and not np.isnan(kc_mid_now) and c < kc_mid_now:
                print(f"🌙 Protective Exit: Within 5 bars price < KC_mid at idx {i}. ❌")
                self.position.close()
                self._reset_trade_state()

            # Volatility contraction scale-out (optional)
            elif self.width_below_sma_count >= 3 and c < kc_upper_now and not self.scaled_out_once and self.position.size > 1:
                reduce_size = max(1, int(round(self.position.size * 0.5)))
                self.position.close(size=reduce_size)
                self.scaled_out_once = True
                print(f"🌙 Scale-Out: Vol contraction detected for 3 bars. Reduced by 50%, size={reduce_size}. 🌗")

            # Time stop: after 40 bars if unrealized R < +1
            elif self.bars_since_entry is not None and self.bars_since_entry >= self.time_stop_bars and r_mult < 1.0:
                print(f"🌙 Time Stop: {self.time_stop_bars} bars elapsed with R={r_mult:.2f} < 1. Exiting. ⏳")
                self.position.close()
                self._reset_trade_state()
        else:
            # Reset trade-related counters when flat
            self.bars_since_entry = None

    def _reset_trade_state(self):
        self.bars_since_entry = None
        self.entry_price = None
        self.initial_stop = None
        self.initial_risk_per_unit = None
        self.moved_to_half_atr_trail = False
        self.moved_to_kc_lower_trail = False
        self.scaled_out_once = False
        # Keep divergence state reset handled by logic above


if __name__ == "__main__":
    # Load and preprocess data
    path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.lower()
    df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')

    # Ensure proper mapping
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    df = df.rename(columns=rename_map)

    # Backtest
    bt = Backtest(
        df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(),
        KeltnerDivergence,
        cash=1_000_000,
        commission=0.0005,
        trade_on_close=False,
        exclusive_orders=True
    )

    stats = bt.run()
    print(stats)
    print(stats._strategy)