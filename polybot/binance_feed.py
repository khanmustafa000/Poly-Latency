"""Binance WebSocket price feed with a rolling buffer and momentum calc."""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, List, Optional, Tuple

import websockets

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/{symbol}@trade"


@dataclass
class MomentumReading:
    ok: bool
    pct_change: float = 0.0
    window_sec: float = 0.0
    ref_price: float = 0.0
    last_price: float = 0.0


class BinanceFeed:
    """Maintains a rolling (timestamp, price) buffer from Binance's trade stream.

    Thread-safe: `run()` drives the websocket in an asyncio loop (usually on a
    background thread); `get_momentum()` / `latest_price()` are called from
    Streamlit's main thread.
    """

    def __init__(self, symbol: str, buffer_sec: int = 600):
        self.symbol = symbol.lower()
        self.buffer_sec = buffer_sec
        self._buf: Deque[Tuple[float, float]] = deque()
        self._lock = Lock()
        self._connected = False
        self._last_error: Optional[str] = None
        self._stop = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def stop(self) -> None:
        self._stop = True

    def latest_price(self) -> Optional[float]:
        with self._lock:
            if not self._buf:
                return None
            return self._buf[-1][1]

    def get_momentum(self, window_sec: int) -> MomentumReading:
        """% change from the oldest price within `window_sec` of now to the latest price."""
        now = time.time()
        with self._lock:
            if not self._buf:
                return MomentumReading(ok=False)
            last_ts, last_price = self._buf[-1]
            ref_price = None
            ref_ts = None
            for ts, price in self._buf:
                if now - ts <= window_sec:
                    ref_price = price
                    ref_ts = ts
                    break
            if ref_price is None:
                ref_price, ref_ts = self._buf[0]
        if ref_price <= 0 or last_price <= 0:
            return MomentumReading(ok=False)
        if ref_ts <= 0 or (now - ref_ts) > self.buffer_sec * 2:
            return MomentumReading(ok=False)  # corrupt/outlier timestamp — never trust it
        pct = (last_price - ref_price) / ref_price * 100
        return MomentumReading(
            ok=True,
            pct_change=pct,
            window_sec=now - ref_ts,
            ref_price=ref_price,
            last_price=last_price,
        )

    def best_momentum(self, windows: List[int]) -> MomentumReading:
        """Scans several lookback windows and returns whichever shows the strongest
        |% move| — the optimal lookback varies with volatility regime, so a single
        fixed window misses signals a shorter/longer window would have caught.
        """
        best: Optional[MomentumReading] = None
        for w in windows:
            r = self.get_momentum(w)
            if not r.ok:
                continue
            if best is None or abs(r.pct_change) > abs(best.pct_change):
                best = r
        return best or MomentumReading(ok=False)

    def price_at(self, ts: float) -> Optional[float]:
        """Price at (or immediately before) a given timestamp, e.g. a round's start —
        used as the reference/open price for the late-round directional strategy."""
        with self._lock:
            if not self._buf:
                return None
            candidates = [p for t, p in self._buf if t <= ts]
            if candidates:
                return candidates[-1]
            return self._buf[0][1]

    def realized_vol_per_sec(self, lookback_sec: int) -> Optional[float]:
        """Per-second realized volatility of log-returns, estimated from actual recent
        ticks (irregular spacing handled by scaling each interval's squared return by
        1/dt, per standard Brownian-motion diffusion scaling). Used to size how much
        a given price move 'should' matter given how choppy the market currently is.
        """
        now = time.time()
        with self._lock:
            pts = [(t, p) for t, p in self._buf if now - t <= lookback_sec]
        if len(pts) < 5:
            return None
        variance_terms = []
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            dt = t1 - t0
            if dt <= 0 or p0 <= 0 or p1 <= 0:
                continue
            r = math.log(p1 / p0)
            variance_terms.append((r * r) / dt)
        if not variance_terms:
            return None
        mean_var = sum(variance_terms) / len(variance_terms)
        return math.sqrt(mean_var) if mean_var > 0 else None

    def _push(self, ts: float, price: float) -> None:
        with self._lock:
            if price <= 0 or ts <= 0:
                return  # malformed tick — never let it into the buffer
            if self._buf:
                last_ts, last_price = self._buf[-1]
                if last_price > 0 and abs(price - last_price) / last_price > 0.15:
                    return  # >15% single-tick jump is not a real trade — drop the outlier
            self._buf.append((ts, price))
            cutoff = ts - self.buffer_sec
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()

    async def run(self) -> None:
        """Reconnect-forever loop. Call via asyncio.run() on a background thread."""
        url = BINANCE_WS_URL.format(symbol=self.symbol)
        backoff = 1
        while not self._stop:
            try:
                async with websockets.connect(url, ping_interval=15, ping_timeout=10) as ws:
                    self._connected = True
                    self._last_error = None
                    backoff = 1
                    while not self._stop:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        price = float(msg["p"])
                        ts = msg["T"] / 1000.0  # trade time, ms -> sec
                        self._push(ts, price)
            except Exception as e:  # noqa: BLE001
                self._connected = False
                self._last_error = str(e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
