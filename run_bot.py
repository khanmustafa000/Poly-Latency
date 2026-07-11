"""Standalone trading process — the actual always-on bot.

Run this via systemd so trading continues whether or not anyone has the
Streamlit dashboard open, whether or not your PC is on. The dashboard
(app.py) is a separate process that only *reads* what this writes to
polybot.db; it never needs to be running for the bot to trade.

Config is reloaded from bot_config.json every cycle, so changing settings in
the dashboard's sidebar and clicking "Save config" takes effect within one
poll interval — no restart needed.
"""
from __future__ import annotations

import time

from polybot.config import BotConfig
from polybot.engine import MultiEngine


def main() -> None:
    cfg = BotConfig.load()
    engine = MultiEngine(cfg)
    engine.start()
    print(f"Bot started. Lanes: {cfg.lanes}", flush=True)

    while True:
        time.sleep(5)
        fresh_cfg = BotConfig.load()
        engine.config = fresh_cfg
        engine.broker.config = fresh_cfg


if __name__ == "__main__":
    main()
