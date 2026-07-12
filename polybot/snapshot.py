"""Rolling 6h snapshots: summarizes signals, skips, and trades over the trailing
window and writes them to snapshots/ so the raw picture survives even after the
live DB tables get pruned. Run standalone (via cron) every 6 hours; nothing else
depends on it, so a missed or failed run never affects trading.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from . import store
from .config import ALL_LANES

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"
WINDOW_SEC = 6 * 3600


def build_snapshot(window_sec: int = WINDOW_SEC, end_ts: Optional[float] = None) -> dict:
    end_ts = end_ts or time.time()
    start_ts = end_ts - window_sec

    events = store.load_events_since(start_ts)
    positions = store.load_positions()
    opened = [p for p in positions if p.get("entry_ts") and start_ts <= p["entry_ts"] <= end_ts]
    closed = [p for p in positions if p.get("exit_ts") and start_ts <= p["exit_ts"] <= end_ts]
    wins = [p for p in closed if (p.get("pnl_usd") or 0) > 0]
    losses = [p for p in closed if (p.get("pnl_usd") or 0) < 0]
    total_pnl = sum(p.get("pnl_usd") or 0 for p in closed)

    lanes = {}
    for lane in ALL_LANES:
        hist = store.load_price_history(lane, since_ts=start_ts)
        prices = [h["price"] for h in hist if h.get("price")]
        lane_events = [e for e in events if e["lane"] == lane]
        signal_events = [e for e in lane_events if e["kind"] == "signal"]
        entered = [e for e in signal_events if "-> entering" in e["text"]]
        skipped = [e for e in signal_events if "-> SKIPPED" in e["text"]]
        skip_reasons: dict = {}
        for e in skipped:
            reason = e["text"].split("SKIPPED:", 1)[-1].strip().split(":", 1)[0].strip()
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        lane_closed = [p for p in closed if f"{p['symbol']}-{p['duration']}" == lane]
        lane_wins = [p for p in lane_closed if (p.get("pnl_usd") or 0) > 0]
        lanes[lane] = {
            "price_low": min(prices) if prices else None,
            "price_high": max(prices) if prices else None,
            "price_range_pct": ((max(prices) - min(prices)) / min(prices) * 100) if prices else None,
            "signals_fired": len(signal_events),
            "signals_entered": len(entered),
            "signals_skipped": len(skipped),
            "skip_reasons": skip_reasons,
            "trades_closed": len(lane_closed),
            "trades_won": len(lane_wins),
            "win_rate_pct": (len(lane_wins) / len(lane_closed) * 100) if lane_closed else None,
            "pnl_usd": sum(p.get("pnl_usd") or 0 for p in lane_closed),
        }

    return {
        "window_start": start_ts,
        "window_end": end_ts,
        "window_start_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_ts)),
        "window_end_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(end_ts)),
        "trades_opened": len(opened),
        "trades_closed": len(closed),
        "trades_won": len(wins),
        "trades_lost": len(losses),
        "win_rate_pct": (len(wins) / len(closed) * 100) if closed else None,
        "total_pnl_usd": total_pnl,
        "best_trade_usd": max((p.get("pnl_usd") or 0 for p in closed), default=None),
        "worst_trade_usd": min((p.get("pnl_usd") or 0 for p in closed), default=None),
        "lanes": lanes,
    }


def save_snapshot(snap: dict) -> Path:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    fname = time.strftime("%Y%m%d_%H%M%S", time.gmtime(snap["window_end"])) + ".json"
    path = SNAPSHOT_DIR / fname
    path.write_text(json.dumps(snap, indent=2))
    return path


def main() -> None:
    snap = build_snapshot()
    path = save_snapshot(snap)
    print(f"Saved snapshot: {path}")


if __name__ == "__main__":
    main()
