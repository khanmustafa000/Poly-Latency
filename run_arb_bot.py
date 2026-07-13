"""Standalone process for the Temporal Arbitrage paper bot — separate from
run_bot.py (the momentum directional bot). Paper-only: places no real orders.

Config is reloaded from arb_config.json every cycle, same pattern as the
momentum bot, so tuning leg1_trigger_pct / required_margin / the stop-loss
layers doesn't require a restart.
"""
from __future__ import annotations

import time

from polybot.arb_config import ArbConfig
from polybot.arb_engine import ArbEngine


def main() -> None:
    cfg = ArbConfig.load()
    engine = ArbEngine(cfg)
    engine.start()
    print(f"Arb bot started (PAPER ONLY). Lanes: {cfg.lanes}", flush=True)

    while True:
        time.sleep(5)
        fresh_cfg = ArbConfig.load()
        engine.config = fresh_cfg


if __name__ == "__main__":
    main()
