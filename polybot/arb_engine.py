"""Temporal Arbitrage paper-trading engine.

Strategy: a matched Up+Down pair always pays exactly $1.00 at settlement, no
matter which side wins. Buy the oversold side cheap after a BTC spike
(leg 1), then buy the other side cheap after BTC retraces (leg 2) — if the
pair costs under $1.00, the difference is locked-in, risk-free profit.

Between leg 1 and leg 2 the position is a NAKED directional bet on BTC
actually retracing. That assumption is unproven at the level of real,
crossed-the-ask fill prices (see polybot/arb_config.py's module docstring for
the full caveat) — this engine's job is to trade paper-only and log every
decision (including rejections and stop-loss counterfactuals) so that
assumption can eventually be validated for real.

Never places real orders. Simulates fills by crossing the live top-of-book
ask (paying it, plus slippage/fees) — never assumes a fill at the bid, which
is the single most common way a backtest lies about being profitable.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from . import arb_store, store
from .arb_config import ArbConfig
from .binance_feed import BinanceFeed
from .config import lane_parts
from .market_finder import ActiveMarket, MarketFinder
from .polymarket_client import get_book_top


@dataclass
class ArbPosition:
    id: str
    symbol: str
    duration: str
    market_slug: str
    condition_id: str
    round_open_price: float
    trigger_price: float
    trigger_move_pct: float
    leg1_side: str  # "Up" or "Down"
    leg1_token_id: str
    leg1_price: float
    leg1_shares: float
    leg1_cost: float
    leg1_ts: float
    market_end_ts: float
    up_token_id: str
    down_token_id: str
    leg2_side: Optional[str] = None
    leg2_token_id: Optional[str] = None
    leg2_price: Optional[float] = None
    leg2_shares: Optional[float] = None
    leg2_cost: Optional[float] = None
    leg2_ts: Optional[float] = None
    leg2_fill_kind: Optional[str] = None  # normal | chase | panic
    status: str = "naked"  # naked | paired | dumped | expired_naked
    pair_cost: Optional[float] = None
    locked_profit: Optional[float] = None
    dump_price: Optional[float] = None
    dump_ts: Optional[float] = None
    dump_reason: Optional[str] = None
    dump_pnl: Optional[float] = None
    would_have_pnl: Optional[float] = None
    would_have_note: Optional[str] = None
    realized_pnl: Optional[float] = None

    @property
    def lane(self) -> str:
        return f"{self.symbol}-{self.duration}"


@dataclass
class ArbEvent:
    ts: float
    text: str
    kind: str = "info"
    lane: Optional[str] = None


class ArbEngine:
    def __init__(self, config: ArbConfig):
        arb_store.init_arb_db()
        self.config = config

        self.feeds: Dict[str, BinanceFeed] = {}
        self.finders: Dict[str, MarketFinder] = {}
        self._build_lanes()

        self.positions: List[ArbPosition] = [ArbPosition(**row) for row in arb_store.load_arb_positions()]

        self._lock = threading.Lock()
        self.events: Deque[ArbEvent] = deque(maxlen=800)

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._round_open_price: Dict[str, float] = {}
        self._last_market_slug: Dict[str, str] = {}
        self._blocked_markets: Dict[str, set] = {}

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
        for feed in self.feeds.values():
            feed._stop = False
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._log(None, f"Arb engine started. Lanes: {', '.join(self.config.lanes)} (PAPER ONLY)", "info")

    def stop(self) -> None:
        self._running = False
        for feed in self.feeds.values():
            feed.stop()
        self._log(None, "Arb engine stopped.", "info")

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

    # ---- core loop ----

    def _tick(self) -> None:
        cfg = self.config
        btc_prices = {symbol: feed.latest_price() for symbol, feed in self.feeds.items()}

        # Stop-loss / leg-2 completion runs BEFORE any new entry — never open a
        # new leg while risk from an existing one hasn't been reassessed this tick.
        self._process_exits(btc_prices)
        self._sync_resolutions()
        self._sync_would_have_pnl()

        for lane in cfg.lanes:
            symbol, duration = lane_parts(lane)
            feed = self.feeds[symbol]
            price = feed.latest_price()
            if price is not None:
                store.save_price_point(lane, time.time(), price, None)

            market = self.finders[lane].get_active_market()
            if market is None or price is None:
                continue
            if self._last_market_slug.get(lane) != market.slug:
                self._round_open_price.pop(lane, None)
                self._last_market_slug[lane] = market.slug

            self._maybe_enter_leg1(lane, symbol, duration, market, price)

    def _get_round_open_price(self, lane: str, symbol: str, market: ActiveMarket) -> Optional[float]:
        open_price = self._round_open_price.get(lane)
        if open_price is None:
            feed = self.feeds.get(symbol)
            open_price = feed.price_at(market.start_ts) if feed else None
            if open_price:
                self._round_open_price[lane] = open_price
        return open_price

    def _has_active_position(self, lane: str) -> bool:
        return any(p.lane == lane and p.status == "naked" for p in self.positions)

    def _total_naked_exposure(self) -> float:
        return sum(p.leg1_cost for p in self.positions if p.status == "naked")

    def _realized_pnl_today(self) -> float:
        today = (time.localtime().tm_yday, time.localtime().tm_year)
        total = 0.0
        for p in self.positions:
            if p.realized_pnl is None:
                continue
            ref_ts = p.dump_ts or p.leg2_ts or p.market_end_ts
            t = time.localtime(ref_ts)
            if (t.tm_yday, t.tm_year) == today:
                total += p.realized_pnl
        return total

    # ---- entry ----

    def _maybe_enter_leg1(self, lane: str, symbol: str, duration: str, market: ActiveMarket, price: float) -> None:
        cfg = self.config

        if self._has_active_position(lane):
            return  # never open a leg while another on this lane is naked

        if market.slug in self._blocked_markets.get(lane, set()):
            return  # max_blocks_per_market == 1: one attempt per market, no stacking

        if market.seconds_left < cfg.min_seconds_left:
            return

        if cfg.daily_loss_limit > 0 and self._realized_pnl_today() <= -abs(cfg.daily_loss_limit):
            return

        open_price = self._get_round_open_price(lane, symbol, market)
        if not open_price:
            return

        move_pct = (price - open_price) / open_price * 100
        if abs(move_pct) < cfg.leg1_trigger_pct:
            return

        spike_up = move_pct > 0
        leg1_side = "Down" if spike_up else "Up"
        token_id = market.down_token_id if leg1_side == "Down" else market.up_token_id
        book = get_book_top(token_id)
        ask = book.best_ask

        if ask is None:
            self._log(lane, f"REJECT leg1 on {market.slug}: no ask liquidity for {leg1_side}", "reject")
            self._block(lane, market.slug)
            return
        if not (cfg.min_leg1_price <= ask <= cfg.max_leg1_price):
            self._log(
                lane,
                f"REJECT leg1 on {market.slug}: {leg1_side} ask={ask:.3f} outside "
                f"[{cfg.min_leg1_price:.2f}, {cfg.max_leg1_price:.2f}] (move={move_pct:+.3f}%)",
                "reject",
            )
            self._block(lane, market.slug)
            return

        current_naked = self._total_naked_exposure()
        room = cfg.max_total_naked_dollars - current_naked
        if room <= 0:
            self._log(
                lane,
                f"REJECT leg1 on {market.slug}: combined naked exposure cap reached "
                f"(${current_naked:.2f}/${cfg.max_total_naked_dollars:.2f} across both lanes)",
                "reject",
            )
            return  # don't block the market permanently — cap may free up before it's too late
        size_usd = min(cfg.max_leg1_dollars, room)
        if size_usd < 0.50:
            self._log(lane, f"REJECT leg1 on {market.slug}: remaining exposure room too small (${size_usd:.2f})", "reject")
            return

        fill_price = ask * (1 + cfg.slippage)
        shares = size_usd / fill_price
        cost = shares * fill_price * (1 + cfg.fee_rate)

        pos = ArbPosition(
            id=str(uuid.uuid4())[:8], symbol=symbol, duration=duration,
            market_slug=market.slug, condition_id=market.condition_id,
            round_open_price=open_price, trigger_price=price, trigger_move_pct=move_pct,
            leg1_side=leg1_side, leg1_token_id=token_id, leg1_price=fill_price,
            leg1_shares=shares, leg1_cost=cost, leg1_ts=time.time(),
            market_end_ts=market.end_ts,
            up_token_id=market.up_token_id, down_token_id=market.down_token_id,
        )
        self.positions.append(pos)
        arb_store.save_arb_position(pos)
        self._block(lane, market.slug)
        self._log(
            lane,
            f"LEG1 OPEN: {pos.id} {leg1_side} {market.slug} ask={ask:.3f} fill={fill_price:.3f} "
            f"cost=${cost:.2f} move={move_pct:+.3f}% (naked exposure now ${current_naked + cost:.2f}/${cfg.max_total_naked_dollars:.2f})",
            "trade",
        )

    def _block(self, lane: str, slug: str) -> None:
        self._blocked_markets.setdefault(lane, set()).add(slug)

    # ---- exits: four-layer stop-loss + happy-path leg-2 completion ----

    def _process_exits(self, btc_prices: Dict[str, Optional[float]]) -> None:
        cfg = self.config
        now = time.time()
        for pos in [p for p in self.positions if p.status == "naked"]:
            seconds_left = pos.market_end_ts - now
            if seconds_left <= 0:
                continue  # handled defensively by _sync_resolutions

            leg2_side = "Down" if pos.leg1_side == "Up" else "Up"
            leg2_token_id = pos.down_token_id if leg2_side == "Down" else pos.up_token_id
            leg2_book = get_book_top(leg2_token_id)
            leg2_ask = leg2_book.best_ask

            leg1_book = get_book_top(pos.leg1_token_id)
            leg1_bid = leg1_book.best_bid

            cur_btc = btc_prices.get(pos.symbol)

            # --- layer 1: BTC ran further against us since trigger -> abandon ---
            if cur_btc and pos.round_open_price:
                trigger_move = pos.trigger_price - pos.round_open_price
                move_since_open = cur_btc - pos.round_open_price
                if trigger_move != 0 and (move_since_open * trigger_move > 0):
                    extra_frac = (abs(move_since_open) - abs(trigger_move)) / abs(trigger_move)
                    if extra_frac >= cfg.adverse_move_pct:
                        self._dump(pos, leg1_bid, f"adverse_move: BTC ran {extra_frac*100:.0f}% further against leg1 since trigger")
                        continue

            # --- layer 2: leg1's own value has decayed too far -> cut losses ---
            if leg1_bid is not None and pos.leg1_price:
                naked_dd_pct = (pos.leg1_price - leg1_bid) / pos.leg1_price * 100
                if naked_dd_pct >= cfg.max_naked_loss_pct * 100:
                    self._dump(pos, leg1_bid, f"max_naked_loss: leg1 value down {naked_dd_pct:.0f}% from entry")
                    continue

            # --- happy path: fill leg 2 whenever required margin is met ---
            if leg2_ask is not None:
                projected_fill = leg2_ask * (1 + cfg.slippage)
                projected_pair_cost = pos.leg1_price + projected_fill
                if (1.0 - projected_pair_cost) >= cfg.required_margin:
                    self._complete_leg2(pos, leg2_side, leg2_token_id, leg2_ask, "normal")
                    continue

            # --- layers 3/4: time-based deadline ladder ---
            if seconds_left <= cfg.panic_deadline_sec:
                if leg2_ask is not None and leg2_ask <= cfg.max_chase_price:
                    self._complete_leg2(pos, leg2_side, leg2_token_id, leg2_ask, "panic")
                else:
                    self._dump(
                        pos, leg1_bid,
                        f"panic_deadline: {seconds_left:.0f}s left, leg2 ask="
                        f"{leg2_ask if leg2_ask is not None else 'n/a'} unavailable/too expensive",
                    )
                continue
            elif seconds_left <= cfg.hard_deadline_sec:
                if leg2_ask is not None:
                    projected_fill = leg2_ask * (1 + cfg.slippage)
                    if (pos.leg1_price + projected_fill) <= cfg.max_pair_cost:
                        self._complete_leg2(pos, leg2_side, leg2_token_id, leg2_ask, "chase")
                        continue
                self._log(
                    pos.lane,
                    f"CHASE: {pos.id} {seconds_left:.0f}s left, leg2 ask="
                    f"{leg2_ask if leg2_ask is not None else 'n/a'} still above max_pair_cost",
                    "chase",
                )

    def _complete_leg2(self, pos: ArbPosition, leg2_side: str, leg2_token_id: str, ask: float, kind: str) -> None:
        cfg = self.config
        fill_price = ask * (1 + cfg.slippage)
        shares = pos.leg1_shares  # leg 2 must match leg 1 exactly, 1:1 — that's what makes it a hedge
        cost = shares * fill_price * (1 + cfg.fee_rate)

        pos.leg2_side = leg2_side
        pos.leg2_token_id = leg2_token_id
        pos.leg2_price = fill_price
        pos.leg2_shares = shares
        pos.leg2_cost = cost
        pos.leg2_ts = time.time()
        pos.leg2_fill_kind = kind
        pos.pair_cost = pos.leg1_price + fill_price
        pos.locked_profit = 1.0 - pos.pair_cost
        pos.status = "paired"
        pos.realized_pnl = shares * pos.locked_profit

        arb_store.save_arb_position(pos)
        self._log(
            pos.lane,
            f"LEG2 FILL [{kind}]: {pos.id} {leg2_side} ask={ask:.3f} fill={fill_price:.3f} "
            f"pair_cost={pos.pair_cost:.3f} locked_profit=${pos.realized_pnl:+.2f}",
            "trade",
        )

    def _dump(self, pos: ArbPosition, bid_price: Optional[float], reason: str) -> None:
        cfg = self.config
        effective_bid = bid_price if bid_price is not None else 0.0
        fill_price = effective_bid * (1 - cfg.slippage)
        proceeds = pos.leg1_shares * fill_price * (1 - cfg.fee_rate)
        dump_pnl = proceeds - pos.leg1_cost

        pos.status = "dumped"
        pos.dump_price = fill_price
        pos.dump_ts = time.time()
        pos.dump_reason = reason
        pos.dump_pnl = dump_pnl
        pos.realized_pnl = dump_pnl

        arb_store.save_arb_position(pos)
        self._log(
            pos.lane,
            f"DUMP: {pos.id} {pos.leg1_side} reason=[{reason}] bid={effective_bid:.3f} pnl=${dump_pnl:+.2f}",
            "stop",
        )

    # ---- post-resolution bookkeeping ----

    def _sync_resolutions(self) -> None:
        """Safety net: the deadline ladder should always force a chase/panic
        decision before a round ends, but if it's ever missed (e.g. an outage),
        force-settle against the real outcome rather than leaving the position
        in limbo forever."""
        now = time.time()
        for pos in self.positions:
            if pos.status != "naked" or now < pos.market_end_ts:
                continue
            close_price = self._round_close_price(pos)
            if close_price is None:
                continue
            settled_up = close_price > pos.round_open_price
            leg1_won = (settled_up and pos.leg1_side == "Up") or (not settled_up and pos.leg1_side == "Down")
            payoff = pos.leg1_shares * (1.0 if leg1_won else 0.0)
            pos.status = "expired_naked"
            pos.realized_pnl = payoff - pos.leg1_cost
            arb_store.save_arb_position(pos)
            self._log(
                pos.lane,
                f"EXPIRED NAKED (deadline ladder missed!): {pos.id} settled "
                f"{'WIN' if leg1_won else 'LOSS'} pnl=${pos.realized_pnl:+.2f}",
                "error",
            )

    def _sync_would_have_pnl(self) -> None:
        """For every dumped position, once its round has actually resolved,
        log the counterfactual: what would holding leg1 to settlement instead
        of dumping have paid out? That's how you learn whether the stop-loss
        helps or hurts, instead of assuming."""
        now = time.time()
        for pos in self.positions:
            if pos.status != "dumped" or pos.would_have_pnl is not None:
                continue
            if now < pos.market_end_ts + 3:
                continue
            close_price = self._round_close_price(pos)
            if close_price is None:
                continue
            settled_up = close_price > pos.round_open_price
            leg1_won = (settled_up and pos.leg1_side == "Up") or (not settled_up and pos.leg1_side == "Down")
            payoff = pos.leg1_shares * (1.0 if leg1_won else 0.0)
            would_have_pnl = payoff - pos.leg1_cost
            pos.would_have_pnl = would_have_pnl
            pos.would_have_note = (
                f"round settled {'Up' if settled_up else 'Down'}; holding leg1 to resolution would have "
                f"{'WON' if leg1_won else 'LOST'} (${would_have_pnl:+.2f} vs actual dump ${pos.dump_pnl:+.2f})"
            )
            arb_store.save_arb_position(pos)
            verdict = "STOP-LOSS HELPED" if would_have_pnl < pos.dump_pnl else "STOP-LOSS HURT"
            self._log(pos.lane, f"COUNTERFACTUAL: {pos.id} {pos.would_have_note} -> {verdict}", "counterfactual")

    def _round_close_price(self, pos: ArbPosition) -> Optional[float]:
        hist = store.load_price_history(pos.lane, since_ts=pos.market_end_ts - 30)
        close_price = None
        for h in hist:
            if h["price"] is not None and h["ts"] <= pos.market_end_ts:
                close_price = h["price"]
        return close_price

    # ---- logging ----

    def _log(self, lane: Optional[str], text: str, kind: str = "info") -> None:
        ts = time.time()
        with self._lock:
            self.events.append(ArbEvent(ts=ts, text=text, kind=kind, lane=lane))
        arb_store.save_arb_event(ts, lane, kind, text)

    # ---- read-only snapshots ----

    def snapshot_events(self, lane: Optional[str] = None) -> List[ArbEvent]:
        with self._lock:
            events = list(self.events)
        if lane:
            events = [e for e in events if e.lane == lane or e.lane is None]
        return events
