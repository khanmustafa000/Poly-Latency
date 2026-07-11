"""SQLite persistence so the trading engine and the Streamlit dashboard can be
separate processes: the engine (run_bot.py) always runs server-side regardless
of whether anyone has the dashboard open; the dashboard just reads whatever
the engine has written, so it can be closed, reopened, or never opened at all
without affecting trading.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List

DB_PATH = Path(__file__).resolve().parent.parent / "polybot.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT, duration TEXT, market_slug TEXT, condition_id TEXT,
                token_id TEXT, side TEXT, entry_price REAL, size_usd REAL,
                shares REAL, entry_ts REAL, market_end_ts REAL, status TEXT,
                exit_price REAL, exit_ts REAL, exit_reason TEXT,
                pnl_usd REAL, pnl_pct REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, lane TEXT, kind TEXT, text TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_state (
                lane TEXT PRIMARY KEY,
                price REAL, pct_change REAL, window_sec REAL,
                market_slug TEXT, seconds_left REAL, seconds_elapsed REAL,
                updated_ts REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lane TEXT, ts REAL, price REAL, pct_change REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_price_history_lane_ts ON price_history(lane, ts)")
        conn.commit()


PRICE_HISTORY_RETENTION_SEC = 3 * 86400  # keep 3 days of raw ticks; closed trades (the long-term record) are kept forever


def save_price_point(lane: str, ts: float, price, pct_change) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO price_history (lane,ts,price,pct_change) VALUES (?,?,?,?)",
            (lane, ts, price, pct_change),
        )
        conn.execute("DELETE FROM price_history WHERE ts < ?", (time.time() - PRICE_HISTORY_RETENTION_SEC,))
        conn.commit()


def load_price_history(lane: str, since_ts: float = 0.0) -> List[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, price, pct_change FROM price_history WHERE lane=? AND ts>=? ORDER BY ts",
            (lane, since_ts),
        ).fetchall()
        return [dict(r) for r in rows]


def save_position(pos) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO positions (id,symbol,duration,market_slug,condition_id,token_id,side,
                entry_price,size_usd,shares,entry_ts,market_end_ts,status,exit_price,exit_ts,
                exit_reason,pnl_usd,pnl_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, exit_price=excluded.exit_price, exit_ts=excluded.exit_ts,
                exit_reason=excluded.exit_reason, pnl_usd=excluded.pnl_usd, pnl_pct=excluded.pnl_pct
            """,
            (
                pos.id, pos.symbol, pos.duration, pos.market_slug, pos.condition_id, pos.token_id,
                pos.side, pos.entry_price, pos.size_usd, pos.shares, pos.entry_ts, pos.market_end_ts,
                pos.status, pos.exit_price, pos.exit_ts, pos.exit_reason, pos.pnl_usd, pos.pnl_pct,
            ),
        )
        conn.commit()


def load_positions() -> List[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY entry_ts DESC").fetchall()
        return [dict(r) for r in rows]


def save_event(ts: float, lane: str | None, kind: str, text: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT INTO events (ts,lane,kind,text) VALUES (?,?,?,?)", (ts, lane, kind, text))
        conn.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 800)")
        conn.commit()


def load_events(limit: int = 300) -> List[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def save_live_state(
    lane: str, price, pct_change, window_sec, market_slug, seconds_left, seconds_elapsed
) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO live_state (lane,price,pct_change,window_sec,market_slug,seconds_left,seconds_elapsed,updated_ts)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(lane) DO UPDATE SET price=excluded.price, pct_change=excluded.pct_change,
                window_sec=excluded.window_sec, market_slug=excluded.market_slug,
                seconds_left=excluded.seconds_left, seconds_elapsed=excluded.seconds_elapsed,
                updated_ts=excluded.updated_ts
            """,
            (lane, price, pct_change, window_sec, market_slug, seconds_left, seconds_elapsed, time.time()),
        )
        conn.commit()


def load_live_state() -> Dict[str, dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM live_state").fetchall()
        return {r["lane"]: dict(r) for r in rows}
