"""Config for the Temporal Arbitrage paper-trading strategy — separate from
BotConfig (the momentum directional bot) because this is a fundamentally
different strategy: instead of betting a direction, it tries to assemble a
matched Up+Down pair for under $1.00 by buying the oversold side after a
spike (leg 1) and the other side after a retrace (leg 2).

IMPORTANT — validation status: leg1_trigger_pct and required_margin below are
seeded from a Gate-1 backtest over 7 days of real BTC price data, but Gate 1's
dollar EV numbers turned out to be circular (the same model was used to both
define "cheap" and to score profit — see the conversation this was built in).
Gate 2 (real order-book VWAP fills) could not be run at all: no historical
Polymarket quote data exists to backtest against. These defaults are
therefore placeholders, not validated parameters. This paper engine's actual
job is to collect real fill data so Gate 1/2 can eventually be re-run for
real, on real crossed-the-ask prices instead of a modeled proxy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

ARB_CONFIG_PATH = Path(__file__).resolve().parent.parent / "arb_config.json"

ARB_LANES = ["BTCUSDT-5m", "BTCUSDT-15m"]


@dataclass
class ArbConfig:
    lanes: List[str] = field(default_factory=lambda: list(ARB_LANES))

    # --- position caps ---
    max_leg1_dollars: float = 10.0          # per market
    max_total_naked_dollars: float = 20.0   # ACROSS BOTH lanes combined — the real protection
    max_blocks_per_market: int = 1          # one attempt per market, no stacking

    # --- entry ---
    leg1_trigger_pct: float = 0.10          # % move from round-open that counts as a spike (see Gate-1 caveat above)
    required_margin: float = 0.03           # min locked profit (in $ per $1 pair) required to voluntarily complete leg 2 (see Gate-2 caveat above)
    max_leg1_price: float = 0.45            # above this it isn't "cheap"
    min_leg1_price: float = 0.10            # below this it's cheap-because-dying
    min_seconds_left: int = 60              # no entry you can't hedge in time

    # --- four-layer stop-loss ---
    hard_deadline_sec: int = 45             # seconds left in round: start CHASING leg 2 (relax margin requirement)
    panic_deadline_sec: int = 15            # seconds left in round: complete at any price, or DUMP
    max_pair_cost: float = 1.08             # accept up to -8c to complete during chase/panic
    max_chase_price: float = 0.95           # above this even in panic, DUMP instead of completing
    adverse_move_pct: float = 0.40          # BTC ran this much further against leg1 (as a fraction of the trigger move) -> abandon
    max_naked_loss_pct: float = 0.30        # naked leg1 value down this % from its entry price -> cut

    # --- risk ---
    daily_loss_limit: float = 30.0
    fee_rate: float = 0.02                  # applied to notional on every fill
    slippage: float = 0.005                 # applied to every fill price (worse for the taker)

    # --- execution ---
    paper_trading: bool = True              # always True — no live order path exists for this strategy yet
    poll_interval_sec: float = 1.0

    def save(self, path: Path = ARB_CONFIG_PATH) -> None:
        import os
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            f.write(json.dumps(asdict(self), indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: Path = ARB_CONFIG_PATH) -> "ArbConfig":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                known = {f for f in asdict(cls())}
                data = {k: v for k, v in data.items() if k in known}
                return cls(**{**asdict(cls()), **data})
            except Exception:
                pass
        return cls()
