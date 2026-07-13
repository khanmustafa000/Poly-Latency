"""Persistence for the Temporal Arbitrage paper bot. Separate tables from the
momentum bot's positions/events (same polybot.db file, different tables) so
the two strategies never collide, but the dashboard/DB tooling stays unified.
"""
from __future__ import annotations

import time
from typing import Dict, List

from .store import _conn

ARB_EVENTS_RETENTION = 6000


def init_arb_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS arb_positions (
                id TEXT PRIMARY KEY,
                symbol TEXT, duration TEXT, market_slug TEXT, condition_id TEXT,
                round_open_price REAL, trigger_price REAL, trigger_move_pct REAL,
                up_token_id TEXT, down_token_id TEXT,
                leg1_side TEXT, leg1_token_id TEXT, leg1_price REAL, leg1_shares REAL,
                leg1_cost REAL, leg1_ts REAL,
                leg2_side TEXT, leg2_token_id TEXT, leg2_price REAL, leg2_shares REAL,
                leg2_cost REAL, leg2_ts REAL, leg2_fill_kind TEXT,
                status TEXT, market_end_ts REAL,
                pair_cost REAL, locked_profit REAL,
                dump_price REAL, dump_ts REAL, dump_reason TEXT, dump_pnl REAL,
                would_have_pnl REAL, would_have_note TEXT,
                realized_pnl REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS arb_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, lane TEXT, kind TEXT, text TEXT
            )
            """
        )
        conn.commit()


def save_arb_position(pos) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO arb_positions (
                id, symbol, duration, market_slug, condition_id,
                round_open_price, trigger_price, trigger_move_pct,
                up_token_id, down_token_id,
                leg1_side, leg1_token_id, leg1_price, leg1_shares, leg1_cost, leg1_ts,
                leg2_side, leg2_token_id, leg2_price, leg2_shares, leg2_cost, leg2_ts, leg2_fill_kind,
                status, market_end_ts, pair_cost, locked_profit,
                dump_price, dump_ts, dump_reason, dump_pnl,
                would_have_pnl, would_have_note, realized_pnl
            ) VALUES (?,?,?,?,?, ?,?,?, ?,?, ?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                leg2_side=excluded.leg2_side, leg2_token_id=excluded.leg2_token_id,
                leg2_price=excluded.leg2_price, leg2_shares=excluded.leg2_shares,
                leg2_cost=excluded.leg2_cost, leg2_ts=excluded.leg2_ts, leg2_fill_kind=excluded.leg2_fill_kind,
                status=excluded.status, pair_cost=excluded.pair_cost, locked_profit=excluded.locked_profit,
                dump_price=excluded.dump_price, dump_ts=excluded.dump_ts,
                dump_reason=excluded.dump_reason, dump_pnl=excluded.dump_pnl,
                would_have_pnl=excluded.would_have_pnl, would_have_note=excluded.would_have_note,
                realized_pnl=excluded.realized_pnl
            """,
            (
                pos.id, pos.symbol, pos.duration, pos.market_slug, pos.condition_id,
                pos.round_open_price, pos.trigger_price, pos.trigger_move_pct,
                pos.up_token_id, pos.down_token_id,
                pos.leg1_side, pos.leg1_token_id, pos.leg1_price, pos.leg1_shares, pos.leg1_cost, pos.leg1_ts,
                pos.leg2_side, pos.leg2_token_id, pos.leg2_price, pos.leg2_shares, pos.leg2_cost, pos.leg2_ts, pos.leg2_fill_kind,
                pos.status, pos.market_end_ts, pos.pair_cost, pos.locked_profit,
                pos.dump_price, pos.dump_ts, pos.dump_reason, pos.dump_pnl,
                pos.would_have_pnl, pos.would_have_note, pos.realized_pnl,
            ),
        )
        conn.commit()


def load_arb_positions() -> List[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM arb_positions ORDER BY leg1_ts DESC").fetchall()
        return [dict(r) for r in rows]


def save_arb_event(ts: float, lane: str | None, kind: str, text: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT INTO arb_events (ts,lane,kind,text) VALUES (?,?,?,?)", (ts, lane, kind, text))
        conn.execute(
            "DELETE FROM arb_events WHERE id NOT IN (SELECT id FROM arb_events ORDER BY id DESC LIMIT ?)",
            (ARB_EVENTS_RETENTION,),
        )
        conn.commit()


def load_arb_events(limit: int = 300) -> List[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM arb_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
