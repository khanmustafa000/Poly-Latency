"""Multi-lane engine: runs a single BTC Binance feed, drives the BTC-5m and
BTC-15m lanes off it, and shares one portfolio-aware Broker across both. Runs
on a background thread so Streamlit's main thread stays responsive.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .binance_feed import BinanceFeed, MomentumReading
from .broker import Broker
from .config import BotConfig, lane_parts
from .market_finder import ActiveMarket, MarketFinder


@dataclass
class EngineEvent:
    ts: float
    text: str
    kind: str = "info"  # info | signal | trade | error
    lane: Optional[str] = None


class MultiEngine:
    def __init__(self, config: BotConfig):
        self.config = config
        self.broker = Broker(config)

        self.feeds: Dict[str, BinanceFeed] = {}
        self.finders: Dict[str, MarketFinder] = {}
        self._build_lanes()

        self._lock = threading.Lock()
        self.events: Deque[EngineEvent] = deque(maxlen=800)
        # per-lane rolling (price, momentum) history for charts
        self.history: Dict[str, Deque[dict]] = {lane: deque(maxlen=1800) for lane in config.lanes}

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_entry_ts: Dict[str, float] = {}
        self._last_market_slug: Dict[str, str] = {}
        self._last_loss_ts: Dict[str, float] = {}
        self._seen_closed_ids: set = set()
        self._start_ts: float = 0.0
        self._round_open_price: Dict[str, float] = {}

    def _build_lanes(self) -> None:
        for lane in self.config.lanes:
            symbol, duration = lane_parts(lane)
            if symbol not in self.feeds:
                self.feeds[symbol] = BinanceFeed(symbol)
            self.finders[lane] = MarketFinder(symbol, duration)

    # ---- lifecycle ----

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_ts = time.time()
        for feed in self.feeds.values():
            feed._stop = False
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._log(None, f"Engine started. Lanes: {', '.join(self.config.lanes)}", "info")

    def stop(self) -> None:
        self._running = False
        for feed in self.feeds.values():
            feed.stop()
        self._log(None, "Engine stopped.", "info")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_async())

    async def _run_async(self) -> None:
        feed_tasks = [asyncio.create_task(feed.run()) for feed in self.feeds.values()]
        try:
            while self._running:
                try:
                    self._tick()
                except Exception as e:  # noqa: BLE001
                    self._log(None, f"Tick error: {e}", "error")
                await asyncio.sleep(self.config.poll_interval_sec)
        finally:
            for feed in self.feeds.values():
                feed.stop()
            for t in feed_tasks:
                t.cancel()

    # ---- core loop step ----

    def _tick(self) -> None:
        self.broker.check_exits()
        self._sync_loss_cooldowns()

        for lane in self.config.lanes:
            symbol, duration = lane_parts(lane)
            feed = self.feeds[symbol]
            cfg = self.config
            reading = feed.best_momentum(cfg.scan_windows_sec) if cfg.multi_window_scan else feed.get_momentum(cfg.momentum_window_sec)
            price = feed.latest_price()

            with self._lock:
                self.history[lane].append(
                    {
                        "ts": time.time(),
                        "price": price,
                        "pct_change": reading.pct_change if reading.ok else None,
                    }
                )

            market = self.finders[lane].get_active_market()
            if market is None:
                continue
            if self._last_market_slug.get(lane) != market.slug:
                self._log(lane, f"Active market: {market.slug} (resolves in {market.seconds_left:.0f}s)", "info")
                self._last_market_slug[lane] = market.slug
                self._round_open_price.pop(lane, None)  # new round -> forget the old open reference

            if reading.ok:
                self._maybe_enter(lane, symbol, duration, market, reading)

    def _sync_loss_cooldowns(self) -> None:
        for p in self.broker.positions:
            if p.status != "closed" or p.id in self._seen_closed_ids:
                continue
            self._seen_closed_ids.add(p.id)
            if (p.pnl_usd or 0.0) < 0:
                self._last_loss_ts[p.lane] = p.exit_ts or time.time()

    def _maybe_enter(
        self, lane: str, symbol: str, duration: str, market: ActiveMarket, reading: MomentumReading
    ) -> None:
        cfg = self.config
        now = time.time()

        if now - self._start_ts < cfg.warmup_sec:
            return
        if abs(reading.pct_change) > 20 or reading.window_sec > self.feeds[symbol].buffer_sec * 2:
            self._log(lane, f"Ignored implausible reading (pct={reading.pct_change:.2f}%, window={reading.window_sec:.0f}s) — likely a bad tick.", "error")
            return
        if abs(reading.pct_change) < cfg.momentum_threshold_pct:
            return
        if now - self._last_entry_ts.get(lane, 0.0) < cfg.cooldown_sec:
            return
        if now - self._last_loss_ts.get(lane, 0.0) < cfg.cooldown_after_loss_sec:
            return
        if self.broker.has_open_position_on(market.condition_id):
            return
        if market.seconds_elapsed > cfg.max_seconds_into_window:
            return
        if market.seconds_left < cfg.min_seconds_left:
            return

        confidence = None
        if cfg.use_confidence_gate:
            confidence = self.signal_confidence(lane, symbol, market)
            if confidence is None or confidence < cfg.confidence_threshold:
                return

        ok, reason = self.broker.can_enter(symbol, duration)
        if not ok:
            return  # silently skip; portfolio limits are expected to bind sometimes

        side = "Up" if reading.pct_change > 0 else "Down"
        conf_str = f", confidence={confidence * 100:.1f}%" if confidence is not None else ""
        self._log(
            lane,
            f"SIGNAL: {symbol} moved {reading.pct_change:+.3f}% in {reading.window_sec:.0f}s{conf_str} "
            f"-> entering {side} on {market.slug}",
            "signal",
        )
        # apply the cooldown for this attempt regardless of outcome — otherwise a
        # persistently-failing entry (e.g. already priced in) retries every tick
        self._last_entry_ts[lane] = now
        try:
            pos, fail_reason = self.broker.enter(symbol, duration, market, side)
        except Exception as e:  # noqa: BLE001
            self._log(lane, f"Entry failed: {e}", "error")
            return
        if pos is None:
            self._log(lane, f"Entry skipped: {fail_reason}", "error")
            return
        self._log(
            lane,
            f"TRADE OPEN: {pos.side} {pos.market_slug} @ {pos.entry_price:.3f} "
            f"(${pos.size_usd:.2f}, {'PAPER' if cfg.paper_trading else 'LIVE'})",
            "trade",
        )

    def signal_confidence(self, lane: str, symbol: str, market: ActiveMarket) -> Optional[float]:
        """Extra confirmation on top of the momentum trigger: models P(price stays
        on the just-triggered side until expiry), treating price as a driftless
        random walk. confidence = Phi(|z|), z = ln(current/round_open) /
        (sigma_per_sec * sqrt(seconds_left)) — the same closed-form math behind a
        zero-drift binary option price, driven by actual distance-from-open,
        time-left, and live realized volatility (not a guess).
        """
        cfg = self.config
        feed = self.feeds.get(symbol)
        if feed is None:
            return None

        open_price = self._round_open_price.get(lane)
        if open_price is None:
            open_price = feed.price_at(market.start_ts)
            if open_price:
                self._round_open_price[lane] = open_price
        cur_price = feed.latest_price()
        seconds_left = max(1.0, market.seconds_left)
        sigma = feed.realized_vol_per_sec(cfg.confidence_vol_lookback_sec)

        if not open_price or not cur_price or not sigma or sigma <= 0:
            return None

        z = math.log(cur_price / open_price) / (sigma * math.sqrt(seconds_left))
        return 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))

    def _log(self, lane: Optional[str], text: str, kind: str = "info") -> None:
        with self._lock:
            self.events.append(EngineEvent(ts=time.time(), text=text, kind=kind, lane=lane))

    # ---- read-only snapshots for the UI ----

    def snapshot_momentum(self, lane: str) -> List[dict]:
        with self._lock:
            return list(self.history.get(lane, []))

    def snapshot_events(self, lane: Optional[str] = None) -> List[EngineEvent]:
        with self._lock:
            events = list(self.events)
        if lane:
            events = [e for e in events if e.lane == lane or e.lane is None]
        return events

    def latest_price(self, symbol: str) -> Optional[float]:
        feed = self.feeds.get(symbol)
        return feed.latest_price() if feed else None

    def momentum(self, lane: str) -> MomentumReading:
        symbol, _ = lane_parts(lane)
        feed = self.feeds.get(symbol)
        if feed is None:
            return MomentumReading(ok=False)
        cfg = self.config
        return feed.best_momentum(cfg.scan_windows_sec) if cfg.multi_window_scan else feed.get_momentum(cfg.momentum_window_sec)

    def active_market(self, lane: str) -> Optional[ActiveMarket]:
        finder = self.finders.get(lane)
        return finder.get_active_market() if finder else None
