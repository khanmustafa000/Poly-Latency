# Polymarket Latency Bot

Binance momentum signal → Polymarket crypto "Up or Down" round → hold to
resolution or stop loss. Every parameter is a slider in the GUI; nothing is
hardcoded.

## Setup

```
pip install -r requirements.txt
streamlit run app.py
```

Paper trading is the default and requires no credentials — it simulates fills
against the real live order book but places no real orders.

## Going live (real money)

1. Copy `.env.example` to `.env`, fill in `POLY_PRIVATE_KEY` (and
   `POLY_FUNDER_ADDRESS` if you're using a proxy/Polymarket-hosted wallet).
2. `pip install py-clob-client` (already in requirements.txt).
3. Flip "Paper trading" off in the sidebar. Start small.

Polymarket blocks US persons from real-money trading — check your eligibility
before enabling live mode.

## Tabs

- **Live** — current price, momentum vs. your threshold, open positions, event log.
- **Trade history** — closed trades, cumulative PnL, win rate.
- **Edge finder / backtest** — replays your exact configured threshold/window/exit
  rule against the last N *real* resolved rounds (pulled live from Binance +
  Polymarket) so you can see whether a given setting has any edge before risking
  anything on it.

## How the pieces fit together

- `polybot/binance_feed.py` — WebSocket trade stream, rolling buffer, momentum calc.
- `polybot/market_finder.py` — finds the currently-open 5m/15m BTC/ETH Up-or-Down round.
- `polybot/polymarket_client.py` — order-book reads (public) + `py_clob_client` (live orders).
- `polybot/broker.py` — paper/live position tracking, stop-loss, resolution settlement.
- `polybot/engine.py` — background-thread loop wiring the above together.
- `polybot/edge_finder.py` — historical backtest of the exact configured rule.
- `app.py` — Streamlit GUI.
