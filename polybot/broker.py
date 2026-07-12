"""Position tracking + exit logic (stop-loss / take-profit / hold-to-resolution).
One shared Broker instance backs the whole portfolio (all lanes), so cash and
concurrency limits are enforced across BTC/ETH x 5m/15m together.

Paper mode fills against the real live order book but moves no real money.
Live mode routes through LiveClobClient and places real orders.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from . import store
from .config import BotConfig
from .market_finder import ActiveMarket
from .polymarket_client import LiveClobClient, get_book_top

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


@dataclass
class Position:
    id: str
    symbol: str
    duration: str
    market_slug: str
    condition_id: str
    token_id: str
    side: str  # "Up" or "Down"
    entry_price: float
    size_usd: float
    shares: float
    entry_ts: float
    market_end_ts: float
    round_open_price: Optional[float] = None  # BTC reference price the round resolves against
    status: str = "open"  # open | closed
    exit_price: Optional[float] = None
    exit_ts: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None

    @property
    def lane(self) -> str:
        return f"{self.symbol}-{self.duration}"

    def unrealized(self, current_price: float) -> tuple[float, float]:
        pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        pnl_usd = self.size_usd * pnl_pct / 100
        return pnl_usd, pnl_pct


class Broker:
    def __init__(self, config: BotConfig):
        self.config = config
        self.positions: List[Position] = []
        self._live_client: Optional[LiveClobClient] = None

    def _get_live_client(self) -> LiveClobClient:
        if self._live_client is None:
            self._live_client = LiveClobClient()
        return self._live_client

    # ---- portfolio-level views ----

    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.status == "open"]

    def open_positions_for_lane(self, symbol: str, duration: str) -> List[Position]:
        return [p for p in self.open_positions() if p.symbol == symbol and p.duration == duration]

    def exposure_usd(self) -> float:
        return sum(p.size_usd for p in self.open_positions())

    def available_cash(self) -> float:
        return self.config.bankroll_usd - self.exposure_usd()

    def realized_pnl_usd(self) -> float:
        return sum(p.pnl_usd or 0.0 for p in self.positions if p.status == "closed")

    def can_enter(self, symbol: str, duration: str) -> tuple[bool, str]:
        cfg = self.config
        if len(self.open_positions()) >= cfg.max_concurrent_positions:
            return False, "max concurrent positions (portfolio) reached"
        if len(self.open_positions_for_lane(symbol, duration)) >= cfg.max_concurrent_per_lane:
            return False, "max concurrent positions for this lane reached"
        if self.available_cash() < cfg.per_trade_usd:
            return False, "insufficient available cash"
        if cfg.daily_loss_limit_usd > 0 and self.realized_pnl_today_usd() <= -abs(cfg.daily_loss_limit_usd):
            return False, "daily loss limit reached"
        return True, ""

    def has_open_position_on(self, condition_id: str) -> bool:
        return any(p.condition_id == condition_id for p in self.open_positions())

    def realized_pnl_today_usd(self) -> float:
        today = (time.localtime().tm_yday, time.localtime().tm_year)
        total = 0.0
        for p in self.positions:
            if p.status != "closed" or p.exit_ts is None:
                continue
            t = time.localtime(p.exit_ts)
            if (t.tm_yday, t.tm_year) == today:
                total += p.pnl_usd or 0.0
        return total

    def realized_pnl_last_24h_usd(self) -> tuple[float, int]:
        """Rolling 24h window (not calendar day) — returns (total pnl, trade count)."""
        cutoff = time.time() - 86400
        closed = [p for p in self.positions if p.status == "closed" and (p.exit_ts or 0) >= cutoff]
        return sum(p.pnl_usd or 0.0 for p in closed), len(closed)

    def _lane_kelly_stake(self, symbol: str, duration: str) -> float:
        """Quarter-Kelly by default, scaled down during portfolio drawdown and
        after loss streaks, scaled up slightly on win streaks, and penalized for
        small sample size. Falls back to the flat per_trade_usd stake until a
        lane has enough closed trades to estimate win rate / payoff ratio.
        """
        cfg = self.config
        closed = sorted(
            (p for p in self.positions if p.status == "closed" and p.symbol == symbol and p.duration == duration),
            key=lambda p: p.exit_ts or 0,
        )
        if len(closed) < cfg.kelly_min_trades:
            return cfg.per_trade_usd

        wins = [p for p in closed if (p.pnl_usd or 0.0) > 0]
        losses = [p for p in closed if (p.pnl_usd or 0.0) <= 0]
        win_rate = len(wins) / len(closed)
        avg_win = (sum(p.pnl_usd for p in wins) / len(wins)) if wins else cfg.per_trade_usd * 0.5
        avg_loss = (abs(sum(p.pnl_usd for p in losses)) / len(losses)) if losses else cfg.per_trade_usd * 0.5
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5

        kelly_f = max(0.0, win_rate - (1 - win_rate) / payoff_ratio) * cfg.kelly_fraction

        # win/loss streak (walk backward from most recent trade)
        streak = 0
        for p in reversed(closed):
            won = (p.pnl_usd or 0.0) > 0
            if streak == 0:
                streak = 1 if won else -1
            elif (streak > 0) == won:
                streak += 1 if won else -1
            else:
                break
        if streak >= 3:
            kelly_f *= 1.25
        elif streak <= -1:
            kelly_f *= max(0.5, 1 - abs(streak) * 0.1)

        # portfolio drawdown reduction
        drawdown_pct = self._portfolio_drawdown_pct()
        if drawdown_pct >= 15:
            kelly_f *= 0.5
        elif drawdown_pct > 5:
            kelly_f *= 1 - (drawdown_pct - 5) / 10 * 0.5

        kelly_f = min(kelly_f, cfg.kelly_max_pct)
        return max(1.0, cfg.bankroll_usd * kelly_f)

    def _portfolio_drawdown_pct(self) -> float:
        closed = sorted(
            (p for p in self.positions if p.status == "closed" and p.exit_ts is not None),
            key=lambda p: p.exit_ts,
        )
        if not closed or not self.config.bankroll_usd:
            return 0.0
        cum = 0.0
        peak = 0.0
        for p in closed:
            cum += p.pnl_usd or 0.0
            peak = max(peak, cum)
        return max(0.0, (peak - cum) / self.config.bankroll_usd * 100)

    # ---- trading ----

    def enter(
        self, symbol: str, duration: str, market: ActiveMarket, side: str, round_open_price: Optional[float] = None
    ) -> tuple[Optional[Position], str]:
        ok, reason = self.can_enter(symbol, duration)
        if not ok:
            return None, reason

        token_id = market.up_token_id if side == "Up" else market.down_token_id
        book = get_book_top(token_id)
        if book.best_ask is None:
            return None, "no liquidity (empty order book)"
        entry_price = book.best_ask
        if entry_price > self.config.max_entry_price:
            # move already priced in; our edge is on the lag, not chasing
            return None, f"already priced in (ask={entry_price:.3f} > max {self.config.max_entry_price:.2f})"
        size_usd = self._lane_kelly_stake(symbol, duration) if self.config.use_kelly_sizing else self.config.per_trade_usd
        size_usd = min(size_usd, self.available_cash())

        if not self.config.paper_trading:
            try:
                client = self._get_live_client()
                client.market_buy(token_id, size_usd, entry_price)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Live order failed: {e}") from e

        shares = size_usd / entry_price
        pos = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            duration=duration,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=token_id,
            side=side,
            entry_price=entry_price,
            size_usd=size_usd,
            shares=shares,
            entry_ts=time.time(),
            market_end_ts=market.end_ts,
            round_open_price=round_open_price,
        )
        self.positions.append(pos)
        store.save_position(pos)
        return pos, ""

    def _close(self, pos: Position, exit_price: float, reason: str) -> None:
        if not self.config.paper_trading:
            try:
                client = self._get_live_client()
                client.market_sell(pos.token_id, pos.shares, exit_price)
            except Exception:
                pass  # best-effort; position still marked closed for bookkeeping
        pos.status = "closed"
        pos.exit_price = exit_price
        pos.exit_ts = time.time()
        pos.exit_reason = reason
        pos.pnl_usd, pos.pnl_pct = pos.unrealized(exit_price)
        store.save_position(pos)

    def check_exits(
        self, btc_prices: Optional[Dict[str, float]] = None, btc_sigmas: Optional[Dict[str, float]] = None
    ) -> None:
        btc_prices = btc_prices or {}
        btc_sigmas = btc_sigmas or {}
        cfg = self.config
        for pos in self.open_positions():
            now = time.time()

            if now >= pos.market_end_ts:
                final_price = self._resolve_outcome(pos)
                if final_price is not None:
                    self._close(pos, final_price, "resolution")
                continue

            if cfg.use_btc_reversal_stop and self._btc_reversal_triggered(pos, now, btc_prices, btc_sigmas):
                book = get_book_top(pos.token_id)
                exit_price = book.best_bid if book.best_bid is not None else pos.entry_price
                self._close(pos, exit_price, "btc_reversal")
                continue

            book = get_book_top(pos.token_id)
            if book.best_bid is None:
                continue
            _, pnl_pct = pos.unrealized(book.best_bid)

            if pnl_pct <= -abs(cfg.stop_loss_pct):
                self._close(pos, book.best_bid, "stop_loss")
                continue

            if (
                not cfg.hold_to_resolution
                and cfg.take_profit_pct > 0
                and pnl_pct >= cfg.take_profit_pct
            ):
                self._close(pos, book.best_bid, "take_profit")

    def _btc_reversal_triggered(
        self, pos: Position, now: float, btc_prices: Dict[str, float], btc_sigmas: Dict[str, float]
    ) -> bool:
        """True once BTC itself has moved against the position by more than
        btc_reversal_z sigma from the round's open/reference price — i.e. the
        directional thesis the trade was based on is actually broken, as opposed
        to the option's own price just being noisy.
        """
        cfg = self.config
        elapsed = now - pos.entry_ts
        if elapsed < cfg.btc_reversal_min_elapsed_sec:
            return False
        if not pos.round_open_price:
            return False
        cur_price = btc_prices.get(pos.symbol)
        sigma = btc_sigmas.get(pos.symbol)
        if not cur_price or not sigma or sigma <= 0:
            return False

        z = math.log(cur_price / pos.round_open_price) / (sigma * math.sqrt(elapsed))
        if pos.side == "Up":
            return z <= -abs(cfg.btc_reversal_z)
        return z >= abs(cfg.btc_reversal_z)

    def _resolve_outcome(self, pos: Position) -> Optional[float]:
        """After the round ends, read the market's final settlement price (1.0 or 0.0).
        Looked up by event slug — the gamma /markets?condition_ids= filter doesn't
        actually filter server-side.
        """
        try:
            resp = requests.get(
                GAMMA_EVENTS_URL,
                params={"slug": pos.market_slug},
                timeout=8,
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None
            markets = events[0].get("markets") or []
            if not markets or not markets[0].get("closed"):
                return None
            m = markets[0]
            import json as _json
            outcomes = _json.loads(m.get("outcomes", "[]"))
            prices = _json.loads(m.get("outcomePrices", "[]"))
            idx = outcomes.index(pos.side) if pos.side in outcomes else 0
            return float(prices[idx])
        except Exception:
            return None
