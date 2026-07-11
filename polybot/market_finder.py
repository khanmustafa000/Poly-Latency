"""Finds the currently-open Polymarket crypto 'Up or Down' round for a given
duration (5m / 15m) so the engine has something to trade into.

These markets roll continuously: a new one opens the instant the previous one
resolves. Slugs look like `btc-updown-5m-<unix_start_ts>` / `eth-updown-15m-<...>`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

GAMMA_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"

SYMBOL_ASSET = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
}


@dataclass
class ActiveMarket:
    slug: str
    condition_id: str
    up_token_id: str
    down_token_id: str
    start_ts: int
    end_ts: int

    @property
    def seconds_elapsed(self) -> float:
        return time.time() - self.start_ts

    @property
    def seconds_left(self) -> float:
        return self.end_ts - time.time()


class MarketFinder:
    def __init__(self, symbol: str, duration: str, cache_sec: float = 5.0):
        self.symbol = symbol
        self.duration = duration  # "5m" or "15m"
        self.cache_sec = cache_sec
        self._cached: Optional[ActiveMarket] = None
        self._cached_at = 0.0

    def _asset_name(self) -> str:
        return SYMBOL_ASSET.get(self.symbol, "bitcoin")

    def get_active_market(self) -> Optional[ActiveMarket]:
        now = time.time()
        if self._cached and now < self._cached.end_ts and (now - self._cached_at) < self.cache_sec:
            return self._cached
        market = self._fetch()
        if market:
            self._cached = market
            self._cached_at = now
        return market

    def _fetch(self) -> Optional[ActiveMarket]:
        """Rounds align to clean UTC boundaries (duration_sec divides evenly into
        the unix clock), so the live round's slug can be computed directly instead
        of relying on the search endpoint's relevance ranking (which favors high
        volume/older rounds over the freshest one).
        """
        asset = self._asset_name()
        prefix = f"{'btc' if asset == 'bitcoin' else 'eth'}-updown-{self.duration}-"
        dur = self._duration_sec()
        now = time.time()
        current_start = int(now // dur) * dur

        # try current round, then the next one (in case of a brief gap at rollover),
        # then the previous one (in case the new round hasn't been created yet)
        for start_ts in (current_start, current_start + dur, current_start - dur):
            slug = f"{prefix}{start_ts}"
            market = self._fetch_by_slug(slug, start_ts, dur)
            if market and market.start_ts <= now <= market.end_ts + 5:
                return market

        # fall back to whichever of those is soonest upcoming
        for start_ts in (current_start + dur, current_start):
            slug = f"{prefix}{start_ts}"
            market = self._fetch_by_slug(slug, start_ts, dur)
            if market:
                return market
        return None

    @staticmethod
    def _fetch_by_slug(slug: str, start_ts: int, dur: int) -> Optional["ActiveMarket"]:
        try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=8,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception:
            return None
        if not events:
            return None
        markets = events[0].get("markets") or []
        if not markets:
            return None
        m = markets[0]
        import json as _json
        try:
            tokens = _json.loads(m.get("clobTokenIds", "[]"))
        except Exception:
            return None
        if len(tokens) != 2:
            return None
        return ActiveMarket(
            slug=slug,
            condition_id=m["conditionId"],
            up_token_id=tokens[0],
            down_token_id=tokens[1],
            start_ts=start_ts,
            end_ts=start_ts + dur,
        )

    def _duration_sec(self) -> int:
        return {"5m": 300, "15m": 900}.get(self.duration, 300)
