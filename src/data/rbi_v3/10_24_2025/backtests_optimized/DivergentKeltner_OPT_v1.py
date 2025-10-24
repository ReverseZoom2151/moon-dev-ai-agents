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

def kc_upper_func(high, low, close, ema_len=24, atr_len=18, mult=1.8):
    # 🌙 Tuned KC params for stronger breakouts: ema_len 24, atr_len 18, mult 1.8
    ema = talib.EMA(close, timeperiod=int(ema_len))
    atr = talib.ATR(high, low, close, timeperiod=int(atr_len))
    return ema + mult * atr

def kc_mid_func(high, low, close, ema_len=24, atr_len=18, mult=1.8):
    ema = talib.EMA(close, timeperiod=int(ema_len))
    return ema

def kc_lower_func(high, low, close, ema_len=24, atr_len=18, mult=1.8):
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

def macd_hist_func(close, fast=12, slow=26, signal=9):
    # 🌙 MACD histogram (momentum confirmation)
    macd, macdsig, macdhist = talib.MACD(close, fastperiod=int(fast), slowperiod=int(slow), signalperiod=int(signal))
    return macdhist

# ========== Strategy Implementation ==========

class DivergentKeltner(Strategy):
    # 🌙 Tuned Keltner params for stronger trend detection and fewer false breakouts
    kc_len = 24
    kc_mult = 1.8
    kc_atr_len = 18

    bb_period = 20
    bb_std = 2

    # 🌙 Risk management: moderate increase with dynamic scaling via ADX (see next)
    risk_pct = 0.0125  # 1.25% base risk per trade, dynamically adjusted by trend strength
    max_hold_bars = 60  # allow winners more time to trend
    div_lookback = 70
    div_sep_min = 3
    div_sep_max = 25

    # 🌙 Momentum/Regime config
    adx_len = 14
    adx_entry_min = 17
    adx_boost = 25

    # 🌙 Staged entry config
    use_early_entry = True
    early_entry_frac = 0.6  # 60% on early entry near KC mid, add remainder on breakout above KC upper

    # Internal state variables
    def init(self):
        # Core trend indicators
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200)
        self.ema400 = self.I(talib.EMA, self.data.Close, timeperiod=400)
        self.ema50_slope5 = self.I(talib.LINEARREG_SLOPE, self.ema50, timeperiod=5)  # short-term slope

        # Keltner Channel (tuned)
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

        # Momentum and divergence helpers
        self.rsi14 = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.macd_hist = self.I(macd_hist_func, self.data.Close, 12, 26, 9)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_len)
        self.piv_low = self.I(pivot_lows, self.data.Low, 2, 2)

        # ATRs
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)

        # Chandelier Exit
        self.chand = self.I(chandelier_exit, self.data.High, self.data.Low, self.data.Close, 14, 22, 3.0)

        # Volume filter
        self.vol_sma20_1p2x = self.I(sma_mult_of_volume, self.data.Volume, 20, 1.2)

        # Additional helpers
        self.max10_close = self.I(talib.MAX, self.data.Close, timeperiod=10)
        self.hh20 = self.I(talib.MAX, self.data.High, timeperiod=20)  # recent structure high for added context

        # State
        self.last_div_index = None
        self.div_swing_low_price = None
        self.initial_stop = None
        self.entry_price = None
        self.risk_per_unit = None
        self.bars_since_entry = 0
        self.moved_to_be = False
        self.partial_scaled = False
        self.second_scaled = False
        self.highest_close_since_entry = -np.inf
        self.reversion_below_upper_count = 0
        self.reduced_by_contraction = False

        # 🌙 Staged entry state
        self.awaiting_breakout_add = False
        self.planned_total_size = 0

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

            # Require second pivot low formed below KC mid at that time to favor mean-reversion setups that can trend
            kc_m_at_second = float(self.kc_mid[second]) if np.isfinite(self.kc_mid[second]) else np.nan
            if not np.isfinite(kc_m_at_second) or not (low_second < kc_m_at_second):
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

        ema50 = float(self.ema50[-1]) if np.isfinite(self.ema50[-1]) else np.nan
        ema200 = float(self.ema200[-1]) if np.isfinite(self.ema200[-1]) else np.nan
        ema400 = float(self.ema400[-1]) if np.isfinite(self.ema400[-1]) else np.nan
        ema50_slope = float(self.ema50_slope5[-1]) if np.isfinite(self.ema50_slope5[-1]) else np.nan

        kc_u = float(self.kc_upper[-1]) if np.isfinite(self.kc_upper[-1]) else np.nan
        kc_m = float(self.kc_mid[-1]) if np.isfinite(self.kc_mid[-1]) else np.nan
        kc_l = float(self.kc_lower[-1]) if np.isfinite(self.kc_lower[-1]) else np.nan

        bb_u = float(self.bb_u[-1]) if np.isfinite(self.bb_u[-1]) else np.nan

        bw = float(self.bw[-1]) if np.isfinite(self.bw[-1]) else np.nan
        bw_p75 = float(self.bw_p75_100[-1]) if np.isfinite(self.bw_p75_100[-1]) else np.nan
        bw_slope = float(self.bw_slope_3[-1]) if np.isfinite(self.bw_slope_3[-1]) else np.nan
        bw_alt = float(self.bw_sma50_125x[-1]) if np.isfinite(self.bw_sma50_125x[-1]) else np.nan

        atr14 = float(self.atr14[-1]) if np.isfinite(self.atr14[-1]) else np.nan
        chand = float(self.chand[-1]) if np.isfinite(self.chand[-1]) else np.nan
        vol_thr = float(self.vol_sma20_1p2x[-1]) if np.isfinite(self.vol_sma20_1p2x[-1]) else np.nan
        max10 = float(self.max10_close[-1]) if np.isfinite(self.max10_close[-1]) else np.nan

        macd_h = float(self.macd_hist[-1]) if np.isfinite(self.macd_hist[-1]) else np.nan
        adx_val = float(self.adx[-1]) if np.isfinite(self.adx[-1]) else np.nan

        # Trend filter (multi-EMA confirmation and slope)
        trend_ok = (np.isfinite(ema200) and np.isfinite(ema50) and np.isfinite(ema400) and
                    close > ema200 and ema50 > ema200 and ema200 > ema400 and
                    np.isfinite(ema50_slope) and ema50_slope > 0)

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

        # Breakout trigger refinements
        breakout_ok = (np.isfinite(kc_u) and close > kc_u) and (not np.isfinite(bb_u) or close > bb_u)

        # Volatility expansion confirmation
        vol_expand_a = np.isfinite(bw) and np.isfinite(bw_p75) and np.isfinite(bw_slope) and (bw > bw_p75) and (bw_slope > 0)
        vol_expand_b = np.isfinite(bw) and np.isfinite(bw_alt) and (bw > bw_alt)
        vol_expand_ok = vol_expand_a or vol_expand_b

        # Momentum confirmation
        momentum_ok = (np.isfinite(macd_h) and macd_h > 0) and (np.isfinite(self.rsi14[-1]) and float(self.rsi14[-1]) > 52)

        # ADX regime filter
        adx_ok = np.isfinite(adx_val) and (adx_val >= float(self.adx_entry_min))

        # Optional volume filter
        volume_ok = True
        if np.isfinite(volume) and np.isfinite(vol_thr):
            volume_ok = volume > vol_thr

        # Dynamic risk multiplier by regime strength
        risk_mult = 1.0
        if adx_ok and adx_val >= float(self.adx_boost) and bw_slope > 0:
            risk_mult = 1.3  # 🌙 boost risk slightly in strong trends
        elif np.isfinite(adx_val) and adx_val < (self.adx_entry_min - 2):
            risk_mult = 0.6  # 🌙 cut risk in choppier markets

        # Cap position when volatility extreme
        vol_cap_mult = 1.0
        if np.isfinite(atr14) and close > 0 and (atr14 / close) > 0.05:
            vol_cap_mult = 0.5

        # Early entry option: enter near KC mid with momentum and strong regime
        early_entry_ok = (
            self.use_early_entry and divergence_ok and trend_ok and volume_ok and vol_expand_ok and momentum_ok and adx_ok
            and np.isfinite(kc_m) and close > kc_m
        )

        # Entry logic (Long only)
        if not self.position:
            # Reset internal state when flat
            self.bars_since_entry = 0
            self.moved_to_be = False
            self.partial_scaled = False
            self.second_scaled = False
            self.highest_close_since_entry = -np.inf
            self.reversion_below_upper_count = 0
            self.reduced_by_contraction = False
            self.awaiting_breakout_add = False
            self.planned_total_size = 0

            # Only trade when regime is favorable and we have divergence context
            if trend_ok and divergence_ok and volume_ok and vol_expand_ok and momentum_ok and adx_ok:
                # Initial stop using KC lower and recent swing pivot; buffer with ATR
                if np.isfinite(kc_l) and np.isfinite(atr14) and (div_pivot_price is not None):
                    prelim_stop = min(kc_l, div_pivot_price - 0.5 * atr14) - 0.1 * atr14
                else:
                    prelim_stop = np.nan

                if np.isfinite(prelim_stop) and close > prelim_stop:
                    risk_per_unit = close - prelim_stop
                    equity = self.equity
                    risk_amount = equity * float(self.risk_pct) * float(risk_mult) * float(vol_cap_mult)

                    # Guard against tiny/huge risk per unit
                    if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
                        print(f"🌙❗ ENTRY skipped due to invalid risk per unit | Bar {i}")
                    else:
                        pos_size_total = risk_amount / risk_per_unit
                        pos_size_total = int(round(pos_size_total))
                        if pos_size_total <= 0:
                            print(f"🌙❗ ENTRY skipped due to zero size after risk scaling | Bar {i}")
                        else:
                            # Breakout path: full size, Early-entry path: partial size then plan add-on
                            if breakout_ok:
                                print(f"🌙 DivergentKeltner ENTRY (Breakout) 🚀 | Bar {i} | Close={close:.2f} | KC_U={kc_u:.2f} | BW={bw:.4f} | Size={pos_size_total}")
                                print(f"✨ Trend OK={trend_ok}, Divergence OK={divergence_ok}, Breakout OK={breakout_ok}, VolExpand OK={vol_expand_ok}, Vol OK={volume_ok}, ADX={adx_val:.2f}, Risk x={risk_mult:.2f}")
                                print(f"🛡️ Initial Stop={prelim_stop:.2f} | Risk/Unit={risk_per_unit:.2f} | Risk Amt={risk_amount:.2f}")
                                self.buy(size=pos_size_total)
                                self.initial_stop = prelim_stop
                                self.risk_per_unit = risk_per_unit
                                self.entry_price = close
                                self.div_swing_low_price = div_pivot_price
                                self.planned_total_size = pos_size_total
                                self.awaiting_breakout_add = False
                            elif early_entry_ok:
                                early_size = max(1, int(round(pos_size_total * float(self.early_entry_frac))))
                                print(f"🌙 DivergentKeltner ENTRY (Early) 🌗 | Bar {i} | Close={close:.2f} | KC_M={kc_m:.2f} | Size={early_size}/{pos_size_total}")
                                print(f"✨ Trend OK={trend_ok}, Div OK={divergence_ok}, Early OK={early_entry_ok}, VolExpand OK={vol_expand_ok}, ADX={adx_val:.2f}, Risk x={risk_mult:.2f}")
                                print(f"🛡️ Initial Stop={prelim_stop:.2f} | Risk/Unit={risk_per_unit:.2f} | Risk Amt={risk_amount:.2f}")
                                self.buy(size=early_size)
                                self.initial_stop = prelim_stop
                                self.risk_per_unit = risk_per_unit
                                self.entry_price = close
                                self.div_swing_low_price = div_pivot_price
                                self.planned_total_size = pos_size_total
                                self.awaiting_breakout_add = True  # plan to add on KC upper breakout
                            else:
                                # Conditions met but neither breakout nor early entry alignment -> wait
                                pass

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
                    self.entry_price = self.entry_price if self.entry_price is not None else close

            rr = (close - self.entry_price) / self.risk_per_unit if (self.risk_per_unit and self.risk_per_unit > 0) else 0.0

            # 🌙 Add-on on breakout if we used early entry
            if self.awaiting_breakout_add and (not breakout_ok is False) and np.isfinite(kc_u) and close > kc_u and vol_expand_ok and momentum_ok:
                remaining = int(self.planned_total_size - abs(int(self.position.size)))
                if remaining > 0:
                    self.buy(size=remaining)
                    self.awaiting_breakout_add = False
                    print(f"🌙🌓 Add-on filled (Breakout confirm) | Bar {i} | Added {remaining} units | Close={close:.2f}")

            # Reversion warning: count consecutive closes back inside channel (below KC upper)
            if np.isfinite(kc_u) and close < kc_u:
                self.reversion_below_upper_count += 1
            else:
                self.reversion_below_upper_count = 0

            # Move to breakeven at +1R
            if (not self.moved_to_be) and rr >= 1.0:
                self.moved_to_be = True
                print(f"🌙🔒 Move stop to breakeven triggered at +1R | Bar {i} | Close={close:.2f}")

            # Partial scale at +1.5R
            if (not self.partial_scaled) and rr >= 1.5:
                reduce_units = int(round(self.position.size * 0.5))
                if reduce_units > 0:
                    self.sell(size=reduce_units)
                    self.partial_scaled = True
                    print(f"🌙✨ Partial take-profit executed (+1.5R) | Reduced {reduce_units} units | Bar {i} | Close={close:.2f}")

            # Second scale at +3R to bank gains further
            if self.partial_scaled and (not self.second_scaled) and rr >= 3.0:
                reduce_units = int(round(self.position.size * 0.3))
                if reduce_units > 0:
                    self.sell(size=reduce_units)
                    self.second_scaled = True
                    print(f"🌙💎 Second take-profit executed (+3R) | Reduced {reduce_units} units | Bar {i} | Close={close:.2f}")

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

            # Trailing stop: blend of KC lower, Chandelier, and ATR-based highest-close trail
            trail1 = max(kc_l, chand) if np.isfinite(kc_l) and np.isfinite(chand) else (kc_l if np.isfinite(kc_l) else chand)
            # Adaptive ATR trail: tighter after +2R
            atr_mult = 2.0 if rr < 2.0 else 1.5
            alt_trail = self.highest_close_since_entry - atr_mult * atr14 if np.isfinite(atr14) else np.nan
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

            # Apply breakeven floor if applicable
            if self.moved_to_be and trailing_stop is not None:
                trailing_stop = max(trailing_stop, self.entry_price)

            # Hard exit: close below KC lower (structure broken)
            if np.isfinite(kc_l) and close < kc_l:
                print(f"🌙🛑 Hard exit: Close below KC lower | Bar {i} | Close={close:.2f} < KC_L={kc_l:.2f}")
                self.position.close()
                return

            # Initial protective stop check (early failure protection)
            if self.initial_stop is not None and close <= self.initial_stop and self.bars_since_entry <= 5:
                print(f"🌙🛡️ Initial stop hit | Bar {i} | Close={close:.2f} <= Stop={self.initial_stop:.2f}")
                self.position.close()
                return

            # Trailing stop trigger
            if trailing_stop is not None and np.isfinite(trailing_stop) and close <= trailing_stop:
                print(f"🌙🔄 Trailing stop exit | Bar {i} | Close={close:.2f} <= Trail={trailing_stop:.2f}")
                self.position.close()
                return

            # Time-based exit: if not +2R within max_hold_bars, exit; if not +0.5R within 15 bars, cut half risk
            if self.bars_since_entry >= int(self.max_hold_bars) and rr < 2.0:
                print(f"🌙⏳ Time-based exit | Bar {i} | Held {self.bars_since_entry} bars | RR={rr:.2f}")
                self.position.close()
                return
            if self.bars_since_entry == 15 and rr < 0.5:
                reduce_units = int(round(self.position.size * 0.5))
                if reduce_units > 0:
                    self.sell(size=reduce_units)
                    print(f"🌙⏱️ Momentum check: cutting 50% at 15 bars (RR<{0.5}) | Bar {i} | Close={close:.2f}")

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