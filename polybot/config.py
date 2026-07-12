"""Central, editable bot configuration. Everything the strategy needs is a knob here.

BTC only, two round durations (5m, 15m) = 2 "lanes" that can run in parallel.
Strategy/risk params are shared across lanes; the portfolio section governs how
much capital each trade and the book as a whole can use.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

CONFIG_PATH = Path(__file__).resolve().parent.parent / "bot_config.json"

ALL_LANES = ["BTCUSDT-5m", "BTCUSDT-15m"]


def lane_key(symbol: str, duration: str) -> str:
    return f"{symbol}-{duration}"


def lane_parts(lane: str) -> tuple[str, str]:
    symbol, duration = lane.rsplit("-", 1)
    return symbol, duration


def lane_label(lane: str) -> str:
    symbol, duration = lane_parts(lane)
    coin = symbol.replace("USDT", "")
    return f"{coin} {duration}"


@dataclass
class BotConfig:
    # --- which lanes are active ---
    lanes: List[str] = field(default_factory=lambda: list(ALL_LANES))

    # --- signal (shared across lanes) ---
    momentum_threshold_pct: float = 0.3      # trigger if |move| >= this % ... (used when dynamic_threshold is off)
    momentum_window_sec: int = 60            # ... within this many seconds

    # --- dynamic threshold: instead of a fixed %, trigger when the move is
    # statistically unusual relative to BTC's own recent volatility (z-score of
    # sigma * sqrt(window)). Adapts automatically to calm vs. volatile regimes. ---
    dynamic_threshold: bool = False
    dynamic_threshold_z: float = 1.5         # trigger at |move| >= z * sigma_per_sec * sqrt(window_sec)

    # --- multi-window momentum scan: instead of one fixed lookback, check several and
    # take whichever shows the strongest move (only used if enabled) ---
    multi_window_scan: bool = False
    scan_windows_sec: List[int] = field(default_factory=lambda: [5, 10, 15, 30, 60, 90, 120])

    # --- entry quality gates ---
    max_entry_price: float = 0.85    # don't buy a side already priced above this (move likely priced in)
    warmup_sec: int = 15             # ignore signals for this many seconds after the engine (re)starts

    # --- loss-aware cooldown / circuit breaker ---
    cooldown_after_loss_sec: int = 60   # extra per-lane cooldown right after that lane takes a loss
    daily_loss_limit_usd: float = 0.0   # 0 = disabled; stop opening new trades once today's realized loss exceeds this

    # --- position sizing ---
    use_kelly_sizing: bool = False
    kelly_fraction: float = 0.25     # quarter-Kelly
    kelly_max_pct: float = 0.25      # cap stake at this fraction of bankroll
    kelly_min_trades: int = 5        # need at least this many closed trades in a lane before Kelly kicks in

    # --- confidence gate: an extra check applied on top of the momentum trigger.
    # Models P(price stays on the triggered side until expiry) as a driftless random
    # walk: confidence = Phi(|z|), z = ln(current/round_open) / (sigma*sqrt(seconds_left)).
    # Only enters if that modeled win probability clears the threshold. ---
    use_confidence_gate: bool = False
    confidence_threshold: float = 0.65        # min modeled win probability (0-1) to enter
    confidence_vol_lookback_sec: int = 300    # window used to estimate live volatility for the model

    # --- edge gate: compares that same modeled fair value against the side's
    # CURRENT market price (the ask we'd actually pay). Only enters if the model
    # says fair value sits at least min_edge_pct above the current price — i.e.
    # the BTC move should be worth a real repricing, not one already priced in. ---
    use_edge_gate: bool = True
    min_edge_pct: float = 20.0                # required (modeled_fair_value - current_price) / current_price * 100

    # --- entry timing (shared) ---
    max_seconds_into_window: int = 240       # don't enter after this many seconds of the round have elapsed
    min_seconds_left: int = 20               # don't enter if less than this many seconds remain
    cooldown_sec: int = 30                   # min gap between two entries on the SAME lane

    # --- risk / exit (shared) ---
    # Price-based stop is now a loose safety-net floor, not the primary trigger: binary
    # option prices are highly convex near expiry and were whipsawing 30-40% within
    # seconds on pure noise (BTC hadn't even moved) — 44% of stop-losses in the first
    # trading batch were actually correct-direction calls shaken out this way. The
    # BTC-reversal stop below is now the primary exit signal; this just catches
    # genuine liquidity/mispricing blowouts.
    stop_loss_pct: float = 45.0              # exit if position value drops this % from entry
    hold_to_resolution: bool = True          # if False, will also take profit at take_profit_pct
    take_profit_pct: float = 0.0             # 0 = disabled

    # --- BTC-reversal stop: exits based on whether the underlying BTC price has
    # actually moved against the position relative to the round's open/reference
    # price, using the same z-score-of-volatility math as the dynamic entry
    # threshold. This targets the real failure mode (the directional thesis is
    # wrong) instead of reacting to the option's own price noise. ---
    use_btc_reversal_stop: bool = True
    btc_reversal_z: float = 1.0              # exit once the adverse move is this many sigma from the round's open price
    btc_reversal_min_elapsed_sec: int = 15   # ignore reversal checks for this long after entry to avoid noise right after opening

    # --- portfolio / capital management ---
    bankroll_usd: float = 1000.0             # total capital the bot is allowed to risk
    per_trade_usd: float = 10.0              # stake per individual trade
    max_concurrent_positions: int = 4        # across ALL lanes combined
    max_concurrent_per_lane: int = 1         # per (symbol, duration) lane

    # --- execution mode ---
    paper_trading: bool = True               # True = simulate fills against live order book, no real orders
    poll_interval_sec: float = 1.0           # loop cadence for order-book polling / position checks

    def save(self, path: Path = CONFIG_PATH) -> None:
        # write + fsync to a temp file then atomically replace, so a crash/hard
        # reboot right after a save can't leave bot_config.json half-written or
        # silently revert to a stale in-page-cache-only version.
        import os
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            f.write(json.dumps(asdict(self), indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "BotConfig":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                known = {f for f in asdict(cls())}
                data = {k: v for k, v in data.items() if k in known}
                return cls(**{**asdict(cls()), **data})
            except Exception:
                pass
        return cls()
