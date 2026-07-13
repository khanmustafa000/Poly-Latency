"""Standalone technical-indicator signals used by the convergence filter and
the skew-as-signal probability adjustment. Each directional signal is
normalized to [-1, 1] so they can be compared/counted on equal footing,
following the same convention as the momentum z-score elsewhere in the bot.
"""
from __future__ import annotations

from typing import List, Optional


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Standard Wilder RSI from a sequence of closes (oldest -> newest)."""
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for prev, cur in zip(closes[-(period + 1):], closes[-period:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_signal(rsi_value: Optional[float]) -> Optional[float]:
    """Mean-reversion bias from RSI: oversold -> bullish bias, overbought ->
    bearish bias, mild pull-back-to-mean bias in between."""
    if rsi_value is None:
        return None
    if rsi_value < 30:
        s = 0.5 + (30 - rsi_value) / 30
    elif rsi_value > 70:
        s = -0.5 - (rsi_value - 70) / 30
    elif rsi_value < 45:
        s = (45 - rsi_value) / 30
    elif rsi_value > 55:
        s = -(rsi_value - 55) / 30
    else:
        s = 0.0
    return max(-1.0, min(1.0, s))


def sma_crossover_signal(closes: List[float], short_n: int, long_n: int, norm_pct: float = 0.03) -> Optional[float]:
    """(short SMA - long SMA) / long SMA, normalized by norm_pct and clamped."""
    if len(closes) < long_n:
        return None
    short_sma = sum(closes[-short_n:]) / short_n
    long_sma = sum(closes[-long_n:]) / long_n
    if long_sma <= 0:
        return None
    pct = (short_sma - long_sma) / long_sma * 100
    return max(-1.0, min(1.0, pct / norm_pct))


def vwap_deviation_signal(last_price: float, vwap_price: Optional[float], norm_pct: float = 0.05) -> Optional[float]:
    """(last - vwap) / vwap, normalized by norm_pct and clamped."""
    if not vwap_price or vwap_price <= 0 or not last_price:
        return None
    pct = (last_price - vwap_price) / vwap_price * 100
    return max(-1.0, min(1.0, pct / norm_pct))


def skew_signal(market_up_price: Optional[float]) -> Optional[float]:
    """Contrarian signal from how far the market's own Up price already sits
    from 50/50: strongly negative once Up is priced well above 0.5 (already
    priced in, fade it), strongly positive once priced well below 0.5."""
    if market_up_price is None:
        return None
    return max(-1.0, min(1.0, -4.0 * (market_up_price - 0.5)))
