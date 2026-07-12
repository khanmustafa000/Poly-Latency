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

from . import store
from .binance_feed import BinanceFeed, MomentumReading
from .broker import Broker
from .config import BotConfig, lane_parts
from .market_finder import ActiveMarket, MarketFinder
from .polymarket_client import get_book_top


@dataclass
class EngineEvent:
    ts: float
    text: str
    kind: str = "info"  # info | signal | trade | error
    lane: Optional[str] = None


class MultiEngine:
    def __init__(self, config: BotConfig):
        store.init_db()
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
        cfg = self.config
        btc_prices = {symbol: feed.latest_price() for symbol, feed in self.feeds.items()}
        btc_sigmas = {
            symbol: feed.realized_vol_per_sec(cfg.confidence_vol_lookback_sec) for symbol, feed in self.feeds.items()
        }
        self.broker.check_exits(btc_prices, btc_sigmas)
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

            store.save_price_point(lane, time.time(), price, reading.pct_change if reading.ok else None)

            market = self.finders[lane].get_active_market()
            store.save_live_state(
                lane, price, reading.pct_change if reading.ok else None,
                reading.window_sec if reading.ok else None,
                market.slug if market else None,
                market.seconds_left if market else None,
                market.seconds_elapsed if market else None,
            )
            if market is None:
                continue
            if self._last_market_slug.get(lane) != market.slug:
                self._log(lane, f"Active market: {market.slug} (resolves in {market.seconds_left:.0f}s)", "info")
                self._last_market_slug[lane] = market.slug
                self._round_open_price.pop(lane, None)  # new round -> forget the old open reference

            if reading.ok:
                self._maybe_enter(lane, symbol, duration, market, reading)

    def _sync_loss_cooldowns(self) -> None:
        """Logs every newly-closed position with full PnL detail (so 'why was this
        a loss' is answerable from the log alone: exit_reason tells you whether it
        was a stop-out or simply resolved the wrong way), and updates the per-lane
        post-loss cooldown clock.
        """
        for p in self.broker.positions:
            if p.status != "closed" or p.id in self._seen_closed_ids:
                continue
            self._seen_closed_ids.add(p.id)
            pnl_usd = p.pnl_usd or 0.0
            if pnl_usd < 0:
                self._last_loss_ts[p.lane] = p.exit_ts or time.time()
            outcome = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "FLAT")
            self._log(
                p.lane,
                f"TRADE CLOSED [{outcome}]: {p.side} {p.market_slug} entry={p.entry_price:.3f} "
                f"exit={p.exit_price:.3f} pnl=${pnl_usd:+.2f} ({p.pnl_pct:+.1f}%) reason={p.exit_reason}",
                "trade" if pnl_usd >= 0 else "error",
            )

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
        threshold_pct = cfg.momentum_threshold_pct
        threshold_note = ""
        if cfg.dynamic_threshold:
            sigma = self.feeds[symbol].realized_vol_per_sec(cfg.confidence_vol_lookback_sec)
            if sigma:
                threshold_pct = cfg.dynamic_threshold_z * sigma * math.sqrt(reading.window_sec) * 100
                threshold_note = f" (dynamic threshold={threshold_pct:.3f}% @ {cfg.dynamic_threshold_z}σ)"
            else:
                return  # not enough volatility data yet to size a dynamic threshold — wait rather than guess

        if abs(reading.pct_change) < threshold_pct:
            return

        # Momentum threshold cleared from here on — every evaluation gets logged
        # with its outcome (entered or the specific reason it was skipped), throttled
        # to once per cooldown_sec so a persistent signal doesn't spam every tick.
        if now - self._last_entry_ts.get(lane, 0.0) < cfg.cooldown_sec:
            return
        self._last_entry_ts[lane] = now

        side = "Up" if reading.pct_change > 0 else "Down"
        skip_reason = None

        if now - self._last_loss_ts.get(lane, 0.0) < cfg.cooldown_after_loss_sec:
            remaining = cfg.cooldown_after_loss_sec - (now - self._last_loss_ts.get(lane, 0.0))
            skip_reason = f"post-loss cooldown active ({remaining:.0f}s left)"
        elif self.broker.has_open_position_on(market.condition_id):
            skip_reason = "already have an open position on this round"
        elif market.seconds_elapsed > cfg.max_seconds_into_window:
            skip_reason = f"too late into round ({market.seconds_elapsed:.0f}s elapsed > max {cfg.max_seconds_into_window}s)"
        elif market.seconds_left < cfg.min_seconds_left:
            skip_reason = f"too close to round end ({market.seconds_left:.0f}s left < min {cfg.min_seconds_left}s)"

        confidence = None
        if skip_reason is None and cfg.use_confidence_gate:
            confidence = self.signal_confidence(lane, symbol, market)
            if confidence is None:
                skip_reason = "confidence gate: not enough volatility data yet to compute"
            elif confidence < cfg.confidence_threshold:
                skip_reason = f"confidence gate: modeled win prob {confidence * 100:.1f}% < required {cfg.confidence_threshold * 100:.0f}%"

        edge_pct = None
        current_price = None
        if skip_reason is None and cfg.use_edge_gate:
            if confidence is None:
                confidence = self.signal_confidence(lane, symbol, market)
            token_id = market.up_token_id if side == "Up" else market.down_token_id
            book = get_book_top(token_id)
            current_price = book.best_ask
            if confidence is None:
                skip_reason = "edge gate: not enough volatility data yet to model fair value"
            elif not current_price:
                skip_reason = "edge gate: no liquidity (empty order book) to price the edge"
            else:
                edge_pct = (confidence - current_price) / current_price * 100
                if edge_pct < cfg.min_edge_pct:
                    skip_reason = (
                        f"edge gate: modeled fair value {confidence:.3f} vs current price {current_price:.3f} "
                        f"= {edge_pct:+.1f}% edge < required {cfg.min_edge_pct:.0f}%"
                    )

        if skip_reason is None:
            ok, reason = self.broker.can_enter(symbol, duration)
            if not ok:
                skip_reason = f"portfolio limit: {reason}"

        conf_str = f", confidence={confidence * 100:.1f}%" if confidence is not None else ""
        edge_str = f", price={current_price:.3f}, edge={edge_pct:+.1f}%" if edge_pct is not None else ""
        base_msg = f"SIGNAL: {symbol} moved {reading.pct_change:+.3f}% in {reading.window_sec:.0f}s{threshold_note}{conf_str}{edge_str} on {market.slug}"

        if skip_reason is not None:
            self._log(lane, f"{base_msg} -> SKIPPED: {skip_reason}", "signal")
            return

        self._log(lane, f"{base_msg} -> entering {side}", "signal")
        round_open_price = self._get_round_open_price(lane, symbol, market)
        try:
            pos, fail_reason = self.broker.enter(symbol, duration, market, side, round_open_price)
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

    def _get_round_open_price(self, lane: str, symbol: str, market: ActiveMarket) -> Optional[float]:
        open_price = self._round_open_price.get(lane)
        if open_price is None:
            feed = self.feeds.get(symbol)
            open_price = feed.price_at(market.start_ts) if feed else None
            if open_price:
                self._round_open_price[lane] = open_price
        return open_price

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

        open_price = self._get_round_open_price(lane, symbol, market)
        cur_price = feed.latest_price()
        seconds_left = max(1.0, market.seconds_left)
        sigma = feed.realized_vol_per_sec(cfg.confidence_vol_lookback_sec)

        if not open_price or not cur_price or not sigma or sigma <= 0:
            return None

        z = math.log(cur_price / open_price) / (sigma * math.sqrt(seconds_left))
        return 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))

    def _log(self, lane: Optional[str], text: str, kind: str = "info") -> None:
        ts = time.time()
        with self._lock:
            self.events.append(EngineEvent(ts=ts, text=text, kind=kind, lane=lane))
        store.save_event(ts, lane, kind, text)

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
