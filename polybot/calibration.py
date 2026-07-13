"""Calibration tracking: compares each trade's modeled entry confidence against
its actual outcome, so threshold tuning is driven by measured win rates
instead of guesswork or one-off forensic post-mortems.
"""
from __future__ import annotations

from typing import List, Optional


def _scored(closed_positions: List[dict]) -> List[dict]:
    return [
        p for p in closed_positions
        if p.get("entry_confidence") is not None and p.get("pnl_usd") is not None
    ]


def brier_score(closed_positions: List[dict]) -> Optional[float]:
    """Mean squared error between stated win probability and actual outcome
    (0/1). 0 = perfect, 0.25 = no better than a coin flip, higher = worse."""
    scored = _scored(closed_positions)
    if not scored:
        return None
    total = 0.0
    for p in scored:
        outcome = 1.0 if (p["pnl_usd"] or 0.0) > 0 else 0.0
        total += (p["entry_confidence"] - outcome) ** 2
    return total / len(scored)


def calibration_buckets(closed_positions: List[dict], bucket_width: float = 0.1) -> List[dict]:
    """Buckets closed trades by stated entry_confidence and compares the
    average stated confidence in each bucket to the realized win rate — the
    two should track closely if the model is well-calibrated."""
    scored = _scored(closed_positions)
    buckets: dict = {}
    for p in scored:
        conf = min(0.999, max(0.0, p["entry_confidence"]))
        lo = int(conf / bucket_width) * bucket_width
        buckets.setdefault(lo, []).append(p)

    rows = []
    for lo in sorted(buckets):
        trades = buckets[lo]
        wins = sum(1 for p in trades if (p["pnl_usd"] or 0.0) > 0)
        avg_conf = sum(p["entry_confidence"] for p in trades) / len(trades)
        win_rate = wins / len(trades)
        rows.append({
            "bucket": f"{lo * 100:.0f}-{(lo + bucket_width) * 100:.0f}%",
            "n": len(trades),
            "avg_stated_confidence_%": round(avg_conf * 100, 1),
            "realized_win_rate_%": round(win_rate * 100, 1),
            "gap_pp": round((win_rate - avg_conf) * 100, 1),
        })
    return rows
