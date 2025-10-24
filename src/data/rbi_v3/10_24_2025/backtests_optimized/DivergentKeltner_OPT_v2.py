import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
import sys
import builtins

# ========== Moon Dev Safe Print (handles Windows console encoding) ==========
def _moon_safe_print(*args, **kwargs):
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', '\n')
    s = sep.join(str(a) for a in args)
    enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        s = s.encode(enc, errors='replace').decode(enc)
    except Exception:
        pass
    builtins.print(s, end=end)

print = _moon_safe_print

# ========== Utility Indicator Functions (all wrapped via self.I in Strategy) ==========

def kc_upper_func(high, low, close, ema_len=20, atr_len=20, mult=1.5):
    ema = talib.EMA(close, timeperiod=int(ema_len))
    atr = talib.ATR(high, low, close, timeperiod=int(atr_len))
    return ema + mult * atr

def kc_mid_func(high, low, close, ema_len=20, atr_len=20, mult=1.5):
    ema = talib.EMA(close, timeperiod=int(ema_len))
    return ema

def kc_lower_func(high, low, close, ema_len=20, atr_len=20, mult=1.5):
    ema = talib.EMA(close, timeperiod=int(ema_len))
    atr = talib.ATR(high, low, close, timeperiod=int(atr_len))
    return ema - mult * atr

def bb_bandwidth(close, period=20, std=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=int(period), nbdevup=float(std), nbdevdn=float(std), matype=0)
    sma = talib.SMA(close, timeperiod=int(period))
    bw = (upper - lower) / sma
    return bw

def bb_upper(close, period=20, std=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=int(period), nbdevup=float(std), nbdevdn=float(std), matype=0)
    return upper

def bb_middle(close, period=20, std=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=int(period), nbdevup=float(std), nbdevdn=float(std), matype=0)
    return middle

def bb_lower(close, period=20, std=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=int(period), nbdevup=float(std), nbdevdn=float(std), matype=0)
    return lower

def rolling_percentile(arr, window=100, q=75):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if i + 1 >= int(window):
            segment = arr[i + 1 - int(window):i + 1]
            segment = segment[np.isfinite(segment)]
            if len(segment) > 0:
                out[i] = np.percentile(segment, float(q))
    return out

def smamult(arr, timeperiod=50, mult=1.25):
    sma = talib.SMA(arr, timeperiod=int(timeperiod))
    return sma * float(mult)

def chandelier_exit(high, low, close, atr_len=14, lookback=22, mult=3.0):
    atr = talib.ATR(high, low, close, timeperiod=int(atr_len))
    highest_close = talib.MAX(close, timeperiod=int(lookback))
    return highest_close - float(mult) * atr

def pivot_lows(low, left=2, right=2):
    # True where low[i] is a pivot low: strictly lower than the preceding 'left' lows and following 'right' lows
    low = np.asarray(low, dtype=float)
    n = len(low)
    piv = np.zeros(n, dtype=bool)
    L = int(left)
    R = int(right)
    for i in range(L, n - R):
        left_min = np.min(low[i - L:i]) if L > 0 else np.inf
        right_min = np.min(low[i + 1:i + 1 + R]) if R > 0 else np.inf
        if np.isfinite(low[i]) and low[i] < left_min and low[i] < right_min:
            piv[i] = True
    return piv

def sma_mult_of_volume(volume, timeperiod=20, mult=1.2):
    sma_v = talib.SMA(volume, timeperiod=int(timeperiod))
    return sma_v * float(mult)

# ====== Moon Dev Added Indicators for Optimization ======
def macd_hist_func(close, fast=12, slow=26, signal=9):
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=int(fast), slowperiod=int(slow), signalperiod=int(signal))
    return macdhist

def adx_func(high, low, close, period=14):
    return talib.ADX(high, low, close, timeperiod=int(period))

def mtf_ema(close, step=4, ema_len=200):
    # Build a higher-timeframe EMA by condensing every 'step' bars, computing EMA, and forward-filling
    c = np.asarray(close, dtype=float)
    n = len(c)
    if n == 0:
        return np.array([], dtype=float)
    st = max(1, int(step))
    idx = np.arange(0, n, st, dtype=int)
    condensed = c[idx]
    ema_c = talib.EMA(condensed, timeperiod=int(ema_len))
    out = np.full(n, np.nan, dtype=float)
    for k, start in enumerate(idx):
        val = ema_c[k]
        end = idx[k + 1] if k + 1 < len(idx) else n
        out[start:end] = val
    return out

def ema_slope(arr, period=50, slope_len=5):
    ema = talib.EMA(arr, timeperiod=int(period))
    slope = talib.LINEARREG_SLOPE(ema, timeperiod=int(slope_len))
    return slope

# ========== Strategy Implementation ==========

class DivergentKeltner(Strategy):
    kc_len = 20
    kc_mult = 1.5
    kc_atr_len = 20

    bb_period = 20
    bb_std = 2

    # Moon Dev: Increased base risk to 1.5% with dynamic volatility adjustments below
    risk_pct = 0.015  # 1.5% per trade (adjusted dynamically based on ATR%)
    max_hold_bars = 40  # Moon Dev: allow strong trends more time to develop
    div_lookback = 60
    div_sep_min = 3
    div_sep_max = 20

    # Internal state variables
    def init(self):
        # Core indicators (ALL via self.I)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200)
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.ema50_slope5 = self.I(ema_slope, self.data.Close, 50, 5)  # Moon Dev: trend slope filter

        # Keltner Channel
        self.kc_upper = self.I(kc_upper_func, self.data.High, self.data.Low, self.data.Close,
                               self.kc_len, self.kc_atr_len, self.kc_mult)
        self.kc_mid = self.I(kc_mid_func, self.data.High, self.data.Low, self.data.Close,
                             self.kc_len, self.kc_atr_len, self.kc_mult)
        self.kc_lower = self.I(kc_lower_func, self.data.High, self.data.Low, self.data.Close,
                               self.kc_len, self.kc_atr_len, self.kc_mult)

        # Bollinger and Bandwidth
        self.bb_u = self.I(bb_upper, self.data.Close, self.bb_period, self.bb_std)
        self.bb_m = self.I(bb_middle, self.data.Close, self.bb_period, self.bb_std)
        self.bb_l = self.I(bb_lower, self.data.Close, self.bb_period, self.bb_std)
        self.bw = self.I(bb_bandwidth, self.data.Close, self.bb_period, self.bb_std)
        self.bw_slope_3 = self.I(talib.LINEARREG_SLOPE, self.bw, timeperiod=3)
        self.bw_p75_100 = self.I(rolling_percentile, self.bw, 100, 75)
        self.bw_sma50_125x = self.I(smamult, self.bw, 50, 1.25)

        # RSI and pivots for divergence
        self.rsi14 = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.piv_low = self.I(pivot_lows, self.data.Low, 2, 2)

        # ATRs
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)

        # Chandelier Exit
        self.chand = self.I(chandelier_exit, self.data.High, self.data.Low, self.data.Close, 14, 22, 3.0)

        # Volume filter
        self.vol_sma20_1p2x = self.I(sma_mult_of_volume, self.data.Volume, 20, 1.2)

        # Additional helpers
        self.max10_close = self.I(talib.MAX, self.data.Close, timeperiod=10)

        # Moon Dev: Momentum and regime filters
        self.macd_hist = self.I(macd_hist_func, self.data.Close, 12, 26, 9)
        self.adx14 = self.I(adx_func, self.data.High, self.data.Low, self.data.Close, 14)
        # Moon Dev MTF confirmation (15m base -> 60m HTF via step=4, using 200 EMA)
        self.htf_ema200 = self.I(mtf_ema, self.data.Close, 4, 200)

        # State
        self.last_div_index = None
        self.div_swing_low_price = None
        self.initial_stop = None
        self.entry_price = None
        self.risk_per_unit = None
        self.bars_since_entry = 0
        self.moved_to_be = False
        self.partial_scaled = False  # legacy flag, maintained for structure
        self.partial1_done = False   # Moon Dev: first scale out at +1R
        self.partial2_done = False   # Moon Dev: second scale out at +2.5R
        self.highest_close_since_entry = -np.inf
        self.reversion_below_upper_count = 0
        self.reduced_by_contraction = False

        # Moon Dev: Risk control via max drawdown and cooldown
        self.equity_peak = None
        self.cooldown_bars = 0

    def find_recent_bullish_divergence(self, i):
        # Find two most recent pivot lows within lookback, separated by [div_sep_min, div_sep_max]
        look_start = max(0, i - int(self.div_lookback))
        piv_indices = np.where(self.piv_low[look_start:i + 1])[0] + look_start  # absolute indices
        if len(piv_indices) < 2:
            return None

        # Most recent pivot (second low)
        for second in piv_indices[::-1]:
            # Divergence pivot must be confirmed within last 5 bars
            if second < i - 5:
                continue
            # Find a first pivot before 'second' with required separation
            min_first_index = second - int(self.div_sep_max)
            max_first_index = second - int(self.div_sep_min)
            candidates = piv_indices[(piv_indices >= min_first_index) & (piv_indices <= max_first_index)]
            if len(candidates) == 0:
                continue
            first = candidates[-1]  # the latest that meets sep

            low_first = float(self.data.Low[first])
            low_second = float(self.data.Low[second])
            rsi_first = float(self.rsi14[first]) if np.isfinite(self.rsi14[first]) else np.nan
            rsi_second = float(self.rsi14[second]) if np.isfinite(self.rsi14[second]) else np.nan

            if not (np.isfinite(low_first) and np.isfinite(low_second) and np.isfinite(rsi_first) and np.isfinite(rsi_second)):
                continue

            # Bullish divergence: price LL, RSI HL, also require RSI at second low less than 45
            if low_second < low_first and rsi_second > rsi_first and rsi_second < 45:
                return second, low_second
        return None

    def next(self):
        i = len(self.data.Close) - 1
        close = float(self.data.Close[-1])
        high = float(self.data.High[-1])
        low = float(self.data.Low[-1])

        # FIX: Safe Volume access without relying on non-existent _fields
        vol_series = getattr(self.data, 'Volume', None)
        if vol_series is not None and len(vol_series) > 0 and np.isfinite(vol_series[-1]):
            volume = float(vol_series[-1])
        else:
            volume = np.nan

        ema200 = float(self.ema200[-1]) if np.isfinite(self.ema200[-1]) else np.nan
        ema50v = float(self.ema50[-1]) if np.isfinite(self.ema50[-1]) else np.nan
        ema50_slope = float(self.ema50_slope5[-1]) if np.isfinite(self.ema50_slope5[-1]) else np.nan

        kc_u = float(self.kc_upper[-1]) if np.isfinite(self.kc_upper[-1]) else np.nan
        kc_m = float(self.kc_mid[-1]) if np.isfinite(self.kc_mid[-1]) else np.nan
        kc_l = float(self.kc_lower[-1]) if np.isfinite(self.kc_lower[-1]) else np.nan

        bw = float(self.bw[-1]) if np.isfinite(self.bw[-1]) else np.nan
        bw_p75 = float(self.bw_p75_100[-1]) if np.isfinite(self.bw_p75_100[-1]) else np.nan
        bw_slope = float(self.bw_slope_3[-1]) if np.isfinite(self.bw_slope_3[-1]) else np.nan
        bw_alt = float(self.bw_sma50_125x[-1]) if np.isfinite(self.bw_sma50_125x[-1]) else np.nan

        atr14 = float(self.atr14[-1]) if np.isfinite(self.atr14[-1]) else np.nan
        chand = float(self.chand[-1]) if np.isfinite(self.chand[-1]) else np.nan
        vol_thr = float(self.vol_sma20_1p2x[-1]) if np.isfinite(self.vol_sma20_1p2x[-1]) else np.nan
        max10 = float(self.max10_close[-1]) if np.isfinite(self.max10_close[-1]) else np.nan

        macdh = float(self.macd_hist[-1]) if np.isfinite(self.macd_hist[-1]) else np.nan
        macdh_prev = float(self.macd_hist[-2]) if len(self.macd_hist) > 1 and np.isfinite(self.macd_hist[-2]) else np.nan
        adx14 = float(self.adx14[-1]) if np.isfinite(self.adx14[-1]) else np.nan
        htf200 = float(self.htf_ema200[-1]) if np.isfinite(self.htf_ema200[-1]) else np.nan
        rsi14 = float(self.rsi14[-1]) if np.isfinite(self.rsi14[-1]) else np.nan

        # Moon Dev: Track equity peak and manage cooldown on drawdown
        eq = float(self.equity)
        if self.equity_peak is None or eq > self.equity_peak:
            self.equity_peak = eq
        dd = (self.equity_peak - eq) / self.equity_peak if self.equity_peak > 0 else 0.0
        if dd > 0.10 and self.cooldown_bars == 0:
            self.cooldown_bars = 40
            print(f"🌙⚠️ Max DD exceeded (>{10}%) -> Cooling down for {self.cooldown_bars} bars | DD={dd:.2%}")
        if self.cooldown_bars > 0:
            self.cooldown_bars -= 1

        # Trend filter (Moon Dev: stronger trend confirmation with HTF and slope)
        trend_ok = (
            np.isfinite(ema200) and close > ema200 and
            np.isfinite(ema50v) and ema50v > ema200 and
            np.isfinite(ema50_slope) and ema50_slope > 0 and
            np.isfinite(htf200) and close > htf200
        )

        # Divergence detection - returns most recent valid pivot within last 5 bars
        div_info = self.find_recent_bullish_divergence(i)
        divergence_ok = False
        div_pivot_price = None
        if div_info is not None:
            self.last_div_index, div_pivot_price = div_info
            divergence_ok = True
        else:
            self.last_div_index = None
            div_pivot_price = None

        # Breakout trigger (Moon Dev: require minor recent close breakout for momentum)
        prev1 = float(self.data.Close[-2]) if np.isfinite(self.data.Close[-2]) else -np.inf
        prev2 = float(self.data.Close[-3]) if np.isfinite(self.data.Close[-3]) else -np.inf
        mini_break = close > max(prev1, prev2)
        breakout_ok = np.isfinite(kc_u) and close > kc_u and mini_break

        # Volatility expansion confirmation (unchanged)
        vol_expand_a = np.isfinite(bw) and np.isfinite(bw_p75) and np.isfinite(bw_slope) and (bw > bw_p75) and (bw_slope > 0)
        vol_expand_b = np.isfinite(bw) and np.isfinite(bw_alt) and (bw > bw_alt)
        vol_expand_ok = vol_expand_a or vol_expand_b

        # Optional volume filter
        volume_ok = True
        if np.isfinite(volume) and np.isfinite(vol_thr):
            volume_ok = volume > vol_thr

        # Moon Dev: Momentum filter to avoid choppy conditions
        momentum_ok = np.isfinite(macdh) and np.isfinite(macdh_prev) and (macdh > 0) and (macdh > macdh_prev) and np.isfinite(adx14) and (adx14 > 18)

        # Moon Dev: Recent pullback-to-KC-mid confirmation to avoid late breakouts
        recent_pull = False
        if np.isfinite(kc_m) and np.isfinite(atr14) and atr14 > 0:
            for j in range(1, 7):
                c_j = float(self.data.Close[-j]) if np.isfinite(self.data.Close[-j]) else np.nan
                kc_m_j = float(self.kc_mid[-j]) if np.isfinite(self.kc_mid[-j]) else np.nan
                atr_j = float(self.atr14[-j]) if np.isfinite(self.atr14[-j]) else np.nan
                if np.isfinite(c_j) and np.isfinite(kc_m_j) and np.isfinite(atr_j) and atr_j > 0:
                    if abs(c_j - kc_m_j) / atr_j <= 0.8:
                        recent_pull = True
                        break

        # Overbought guard (avoid blow-off top entries)
        overbought_ok = np.isfinite(rsi14) and rsi14 < 80

        # Entry logic (Long only)
        if not self.position:
            # Reset internal state when flat
            self.bars_since_entry = 0
            self.moved_to_be = False
            self.partial_scaled = False
            self.partial1_done = False
            self.partial2_done = False
            self.highest_close_since_entry = -np.inf
            self.reversion_below_upper_count = 0
            self.reduced_by_contraction = False

            # Moon Dev: skip new entries during cooldown after large drawdown
            if self.cooldown_bars > 0:
                return

            # Moon Dev: tightened entry - require trend+momo+divergence+pullback+breakout+vol expansion+volume
            if trend_ok and momentum_ok and divergence_ok and recent_pull and breakout_ok and vol_expand_ok and volume_ok and overbought_ok:
                # Initial stop using balanced KC mid and divergence swing with ATR buffer
                if np.isfinite(kc_m) and np.isfinite(kc_l) and np.isfinite(atr14) and (div_pivot_price is not None):
                    candidate1 = div_pivot_price - 0.8 * atr14
                    candidate2 = kc_m - 1.2 * atr14
                    floor = kc_l - 0.2 * atr14
                    prelim_stop = max(candidate1, candidate2, floor)
                    # Moon Dev: ensure minimum risk per unit to avoid oversizing
                    min_risk = 0.6 * atr14
                    if close - prelim_stop < min_risk:
                        prelim_stop = close - min_risk
                else:
                    prelim_stop = np.nan

                if np.isfinite(prelim_stop) and close > prelim_stop:
                    risk_per_unit = close - prelim_stop
                    equity = self.equity
                    # Moon Dev: dynamic position sizing by volatility regime (ATR as % of price)
                    atr_pct = (atr14 / close) if (np.isfinite(atr14) and close > 0) else np.nan
                    risk_amount = equity * float(self.risk_pct)

                    # Adjust risk for volatility extremes
                    vol_mult = 1.0
                    if np.isfinite(atr_pct):
                        if atr_pct < 0.008:
                            vol_mult *= 1.25  # calmer regime -> a bit more size
                        elif atr_pct > 0.050:
                            vol_mult *= 0.50  # very high vol -> cut risk
                        elif atr_pct > 0.035:
                            vol_mult *= 0.70  # elevated vol
                        elif atr_pct > 0.030:
                            vol_mult *= 0.85
                    risk_amount *= vol_mult

                    pos_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                    # Additional volatility cap (legacy kept)
                    if np.isfinite(atr14) and close > 0 and atr14 / close > 0.03:
                        pos_size *= 0.5

                    pos_size = int(round(pos_size))
                    if pos_size > 0:
                        print(f"🌙 DivergentKeltner ENTRY signal 🚀 | Bar {i} | Close={close:.2f} | KC_U={kc_u:.2f} | BW={bw:.4f} | Size={pos_size}")
                        print(f"✨ Filters -> Trend={trend_ok}, Momentum={momentum_ok}, Divergence={divergence_ok}, Pullback={recent_pull}, Breakout={breakout_ok}, VolExpand={vol_expand_ok}, VolOK={volume_ok}")
                        print(f"🛡️ Initial Stop={prelim_stop:.2f} | Risk/Unit={risk_per_unit:.2f} | RiskAmt={risk_amount:.2f} | ATR%={(atr_pct*100 if np.isfinite(atr_pct) else np.nan):.2f}%")
                        self.buy(size=pos_size)
                        # Store state for management
                        self.initial_stop = prelim_stop
                        self.risk_per_unit = risk_per_unit
                        self.entry_price = close  # approximate; will adjust after order fills if needed
                        self.div_swing_low_price = div_pivot_price
                    else:
                        print(f"🌙❗ ENTRY skipped due to zero size | Risk per unit too high. Bar {i}")

        else:
            # Manage open long position
            self.bars_since_entry += 1
            self.highest_close_since_entry = max(self.highest_close_since_entry, close)

            # Adjust entry price to actual filled entry using trades list
            if len(self.trades) > 0:
                last_trade = self.trades[-1]
                try:
                    self.entry_price = float(last_trade.entry_price)
                except Exception:
                    # Keep previous value if any; otherwise fallback to current close
                    self.entry_price = self.entry_price if self.entry_price is not None else close

            rr = (close - self.entry_price) / self.risk_per_unit if (self.risk_per_unit and self.risk_per_unit > 0) else 0.0

            # Reversion warning: count consecutive closes back inside channel (below KC upper)
            if np.isfinite(kc_u) and close < kc_u:
                self.reversion_below_upper_count += 1
            else:
                self.reversion_below_upper_count = 0

            # Moon Dev: Move to breakeven earlier at +0.8R, then raise BE at +1.5R
            if (not self.moved_to_be) and rr >= 0.8:
                self.moved_to_be = True
                print(f"🌙🔒 Move stop to breakeven (early) at +0.8R | Bar {i} | Close={close:.2f}")

            # Partial scale-outs: 1) +1R reduce 1/3, 2) +2.5R reduce 1/3
            if (not self.partial1_done) and rr >= 1.0:
                reduce_units = int(round(self.position.size * (1/3)))
                if reduce_units > 0:
                    self.sell(size=reduce_units)
                    self.partial1_done = True
                    print(f"🌙✨ Partial TP1 executed (+1.0R) | Reduced {reduce_units} units | Bar {i} | Close={close:.2f}")

            if (not self.partial2_done) and rr >= 2.5:
                reduce_units = int(round(self.position.size * 0.5))  # reduce half of remaining (≈1/3 original if TP1 done)
                if reduce_units > 0:
                    self.sell(size=reduce_units)
                    self.partial2_done = True
                    print(f"🌙🌕 Partial TP2 executed (+2.5R) | Reduced {reduce_units} units | Bar {i} | Close={close:.2f}")

            # Volatility contraction: BW < SMA50(BW) and no new 10-bar high within 5 bars -> reduce 50% once
            bw_sma50 = self.bw_sma50_125x / 1.25  # since bw_sma50_125x = 1.25 * SMA50(BW)
            bw_sma50_val = float(bw_sma50[-1]) if np.isfinite(bw_sma50[-1]) else np.nan
            no_new_high = close < max10 if np.isfinite(max10) else False
            if (not self.reduced_by_contraction) and np.isfinite(bw) and np.isfinite(bw_sma50_val):
                if bw < bw_sma50_val and no_new_high:
                    cnt = getattr(self, "_no_new_high_count", 0) + 1
                    self._no_new_high_count = cnt
                    if cnt >= 5:
                        reduce_units = int(round(self.position.size * 0.5))
                        if reduce_units > 0:
                            self.sell(size=reduce_units)
                            self.reduced_by_contraction = True
                            print(f"🌙📉 Volatility contraction detected, reducing risk by 50% | Bar {i} | Close={close:.2f}")
                else:
                    self._no_new_high_count = 0

            # Trailing stop: use combo of KC lower, Chandelier, and dynamic ATR trail tightened with performance
            trail1 = max(kc_l, chand) if np.isfinite(kc_l) and np.isfinite(chand) else (kc_l if np.isfinite(kc_l) else chand)
            # Moon Dev: progressive ATR trailing based on RR
            if np.isfinite(atr14):
                if rr >= 2.0:
                    alt_trail = self.highest_close_since_entry - 1.7 * atr14
                elif rr >= 1.0:
                    alt_trail = self.highest_close_since_entry - 2.0 * atr14
                else:
                    alt_trail = self.highest_close_since_entry - 2.2 * atr14
            else:
                alt_trail = np.nan
            trail_candidates = [t for t in [trail1, alt_trail] if np.isfinite(t)]
            trailing_stop = max(trail_candidates) if len(trail_candidates) > 0 else None

            # Tighten stop if reversion condition (2 consecutive closes back inside channel)
            if self.reversion_below_upper_count >= 2:
                tighten_to = max(kc_m, chand) if np.isfinite(kc_m) and np.isfinite(chand) else (kc_m if np.isfinite(kc_m) else chand)
                if trailing_stop is not None and np.isfinite(tighten_to):
                    trailing_stop = max(trailing_stop, tighten_to)
                elif trailing_stop is None:
                    trailing_stop = tighten_to
                if trailing_stop is not None and np.isfinite(trailing_stop):
                    print(f"🌙🧲 Reversion warning: tightening stop | Bar {i} | New Stop≈{trailing_stop:.2f}")

            # Apply breakeven floor if applicable; at +1.5R, raise to entry + 0.3R
            if self.moved_to_be and trailing_stop is not None:
                be_floor = self.entry_price
                if rr >= 1.5 and (self.risk_per_unit and self.risk_per_unit > 0):
                    be_floor = max(be_floor, self.entry_price + 0.3 * self.risk_per_unit)
                trailing_stop = max(trailing_stop, be_floor)

            # Hard exit: close below KC lower
            if np.isfinite(kc_l) and close < kc_l:
                print(f"🌙🛑 Hard exit: Close below KC lower | Bar {i} | Close={close:.2f} < KC_L={kc_l:.2f}")
                self.position.close()
                return

            # Initial protective stop check (keep early guard within 5 bars)
            if self.initial_stop is not None and close <= self.initial_stop and self.bars_since_entry <= 5:
                print(f"🌙🛡️ Initial stop hit | Bar {i} | Close={close:.2f} <= Stop={self.initial_stop:.2f}")
                self.position.close()
                return

            # Momentum/regime deterioration exit to protect gains
            bw_sma50_val2 = float(bw_sma50[-1]) if np.isfinite(bw_sma50[-1]) else np.nan
            if rr >= 1.0 and np.isfinite(adx14) and adx14 < 15 and np.isfinite(bw) and np.isfinite(bw_sma50_val2) and bw < bw_sma50_val2:
                print(f"🌙🥶 Momentum fade exit | Bar {i} | ADX={adx14:.1f} | BW<BW_SMA50 | RR={rr:.2f}")
                self.position.close()
                return

            # Trailing stop trigger
            if trailing_stop is not None and np.isfinite(trailing_stop) and close <= trailing_stop:
                print(f"🌙🔄 Trailing stop exit | Bar {i} | Close={close:.2f} <= Trail={trailing_stop:.2f}")
                self.position.close()
                return

            # Time-based exit: if not +2R within max_hold_bars, exit
            if self.bars_since_entry >= int(self.max_hold_bars) and rr < 2.0:
                print(f"🌙⏳ Time-based exit | Bar {i} | Held {self.bars_since_entry} bars | RR={rr:.2f}")
                self.position.close()
                return

# ========== Robust Data Loading ==========

def find_data_file(preferred_path):
    # Try preferred path first
    candidates = []
    if preferred_path:
        candidates.append(Path(preferred_path).expanduser())

    # Environment override
    env_path = os.environ.get("MD_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # Common relative candidates
    cwd = Path(os.getcwd())
    here = Path(__file__).resolve().parent if '__file__' in globals() else cwd
    repo_root = here
    for _ in range(5):
        if (repo_root / "src").exists():
            break
        repo_root = repo_root.parent

    rel_candidates = [
        cwd / "BTC-USD-15m.csv",
        here / "BTC-USD-15m.csv",
        repo_root / "src" / "data" / "rbi" / "BTC-USD-15m.csv",
        repo_root / "src" / "data" / "rbi_v3" / "BTC-USD-15m.csv",
        repo_root / "src" / "data" / "rbi" / "BTC-USD-15m-15m.csv",
    ]
    candidates.extend(rel_candidates)

    # Globs for rescue
    glob_patterns = [
        str(repo_root / "src" / "data" / "**" / "BTC-USD-15m*.csv"),
        str(cwd / "**" / "BTC-USD-15m*.csv"),
    ]
    for pattern in glob_patterns:
        for p in glob.glob(pattern, recursive=True):
            candidates.append(Path(p))

    # Return first that exists
    seen = set()
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            print(f"🌙✨ Data file located: {p}")
            return str(p)

    return None

def load_price_data(data_path=None):
    # Locate file
    if data_path is None:
        data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    resolved = find_data_file(data_path)
    if not resolved:
        raise FileNotFoundError(
            "🌙❌ Could not find BTC-USD-15m.csv. Set MD_DATA_PATH env var or place the file under src/data/rbi/. "
            "Searched multiple common locations."
        )

    print(f"🌙 Loading CSV from: {resolved}")
    data = pd.read_csv(resolved)

    # Clean columns (Critical requirement)
    data.columns = data.columns.str.strip().str.lower()

    # Drop unnamed columns (Critical requirement)
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()], errors='ignore')

    # Standardize column names (Critical requirement)
    rename_map = {}
    if 'open' in data.columns: rename_map['open'] = 'Open'
    if 'high' in data.columns: rename_map['high'] = 'High'
    if 'low' in data.columns: rename_map['low'] = 'Low'
    if 'close' in data.columns: rename_map['close'] = 'Close'
    if 'volume' in data.columns: rename_map['volume'] = 'Volume'
    data = data.rename(columns=rename_map)

    # Ensure datetime exists; common fallbacks if needed
    if 'datetime' not in data.columns:
        for alt in ['date', 'time', 'timestamp']:
            if alt in data.columns:
                data['datetime'] = data[alt]
                break
    if 'datetime' not in data.columns:
        raise KeyError("🌙❌ 'datetime' column not found in CSV after cleaning.")

    # Set datetime as index (Critical requirement)
    data = data.set_index(pd.to_datetime(data['datetime']))

    # Optional: keep a capitalized Datetime column for reference
    data['Datetime'] = data.index

    # Ensure required OHLCV columns exist and order them
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise KeyError(f"🌙❌ Missing required columns after processing: {missing}")

    cols_to_return = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Datetime' in data.columns:
        cols_to_return = ['Datetime'] + cols_to_return
    data = data[cols_to_return].copy()

    print(f"🌙✅ Data loaded | Rows={len(data)} | Columns={list(data.columns)} | Start={data.index.min()} | End={data.index.max()}")
    return data

# ========== Data Loading and Backtest Run ==========

if __name__ == "__main__":
    try:
        df = load_price_data()
    except Exception as e:
        # Surface clear error for debugging
        raise

    bt = Backtest(
        df,
        DivergentKeltner,
        cash=1_000_000,
        commission=0.0005,
        exclusive_orders=True
    )

    print("🌙 Running backtest... ✨")
    stats = bt.run()
    print("🌙 Backtest complete. 📊")
    print(stats)
    # Print strategy instance summary safely if available
    strat = getattr(stats, "_strategy", None)
    if strat is not None:
        print("🌙 Strategy state snapshot ✨:", strat)