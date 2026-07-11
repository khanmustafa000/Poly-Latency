"""Thin wrapper around Polymarket's public CLOB REST (for order-book reads, used
by both paper and live modes) and py_clob_client (for real order placement,
live mode only).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

CLOB_BASE = "https://clob.polymarket.com"


@dataclass
class BookTop:
    best_bid: Optional[float]
    best_ask: Optional[float]

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask


def get_book_top(token_id: str, timeout: float = 5.0) -> BookTop:
    """Best bid/ask for a token, read straight off the public order-book endpoint.
    Used to simulate realistic paper fills and to price live orders.
    """
    try:
        resp = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        best_bid = max((float(b["price"]) for b in bids), default=None)
        best_ask = min((float(a["price"]) for a in asks), default=None)
        return BookTop(best_bid=best_bid, best_ask=best_ask)
    except Exception:
        return BookTop(best_bid=None, best_ask=None)


class LiveClobClient:
    """Real-money order placement. Only constructed when paper_trading=False and
    credentials are present. Requires the py-clob-client package and a funded
    Polymarket-linked wallet.
    """

    def __init__(self):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL
        except ImportError as e:
            raise RuntimeError(
                "py-clob-client is not installed. Run: pip install py-clob-client"
            ) from e

        private_key = os.environ.get("POLY_PRIVATE_KEY")
        funder = os.environ.get("POLY_FUNDER_ADDRESS")
        if not private_key:
            raise RuntimeError(
                "POLY_PRIVATE_KEY env var is required for live trading. "
                "Set it (and POLY_FUNDER_ADDRESS if using a proxy wallet) before enabling live mode."
            )

        self._OrderArgs = OrderArgs
        self._BUY = BUY
        self._SELL = SELL
        self.client = ClobClient(
            CLOB_BASE,
            key=private_key,
            chain_id=137,
            funder=funder,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def market_buy(self, token_id: str, usd_size: float, limit_price: float):
        order_args = self._OrderArgs(
            token_id=token_id,
            price=limit_price,
            size=round(usd_size / limit_price, 2),
            side=self._BUY,
        )
        signed = self.client.create_order(order_args)
        return self.client.post_order(signed)

    def market_sell(self, token_id: str, size: float, limit_price: float):
        order_args = self._OrderArgs(
            token_id=token_id,
            price=limit_price,
            size=size,
            side=self._SELL,
        )
        signed = self.client.create_order(order_args)
        return self.client.post_order(signed)
