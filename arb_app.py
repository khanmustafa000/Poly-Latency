"""Streamlit dashboard — Temporal Arbitrage bot only, a pure VIEWER.

The actual engine runs as its own always-on process (run_arb_bot.py, via the
polybot-arb systemd service), independent of this dashboard. This file only
reads/writes arb_config.json and reads what the engine has written to
polybot.db, so the bot keeps trading whether or not this page is open.

Run with:  streamlit run arb_app.py   (viewing only — does not start trading)
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from polybot import arb_store, store
from polybot.arb_config import ARB_LANES, ArbConfig
from polybot.config import lane_label

LIVE_STALE_AFTER_SEC = 20

st.set_page_config(page_title="Temporal Arb Bot", page_icon="⚖️", layout="wide")
store.init_db()
arb_store.init_arb_db()


def arb_config_ui() -> ArbConfig:
    """Every Temporal Arb parameter, editable. Deliberately NOT inside the
    auto-refreshing fragment below — a Save click racing an auto-refresh
    timer can silently get cancelled before the write completes."""
    st.subheader("Configuration")
    st.caption(
        "leg1_trigger_pct and required_margin were seeded from a Gate-1 backtest over 7 days of real BTC "
        "data, but that backtest's dollar EV was circular (no real historical Polymarket quote data exists "
        "to validate against) — treat these as a starting point to tune from live paper results, not "
        "validated parameters."
    )
    saved = ArbConfig.load()

    lanes = []
    for lane in ARB_LANES:
        if st.checkbox(lane_label(lane), value=(lane in saved.lanes), key="arb_lane_" + lane):
            lanes.append(lane)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Position caps**")
        max_leg1_dollars = st.number_input("Max leg1 $ (per market)", 1.0, 1000.0, float(saved.max_leg1_dollars), 1.0, key="arb_max_leg1_dollars")
        max_total_naked_dollars = st.number_input(
            "Max combined naked $ (across both lanes)", 1.0, 2000.0, float(saved.max_total_naked_dollars), 1.0,
            help="The real protection — 5m and 15m naked exposure count against ONE shared cap, not two.",
            key="arb_max_total_naked_dollars",
        )
        max_blocks_per_market = st.number_input("Max blocks per market (no stacking)", 1, 5, int(saved.max_blocks_per_market), 1, key="arb_max_blocks_per_market")

        st.markdown("**Entry**")
        leg1_trigger_pct = st.slider("Leg1 trigger (% move from round-open)", 0.02, 1.0, float(saved.leg1_trigger_pct), 0.01, key="arb_leg1_trigger_pct")
        required_margin = st.slider("Required margin (locked profit per $1 pair)", 0.0, 0.20, float(saved.required_margin), 0.01, key="arb_required_margin")
        max_leg1_price = st.slider("Max leg1 price (\"cheap\" ceiling)", 0.10, 0.99, float(saved.max_leg1_price), 0.01, key="arb_max_leg1_price")
        min_leg1_price = st.slider("Min leg1 price (\"cheap-because-dying\" floor)", 0.0, 0.50, float(saved.min_leg1_price), 0.01, key="arb_min_leg1_price")
        min_seconds_left = st.slider("Min seconds left to enter", 0, 300, int(saved.min_seconds_left), 5, key="arb_min_seconds_left")

    with c2:
        st.markdown("**Four-layer stop-loss**")
        hard_deadline_sec = st.slider("Hard deadline (start chasing leg2)", 0, 180, int(saved.hard_deadline_sec), 5, key="arb_hard_deadline_sec")
        panic_deadline_sec = st.slider("Panic deadline (complete at any price, or DUMP)", 0, 60, int(saved.panic_deadline_sec), 5, key="arb_panic_deadline_sec")
        max_pair_cost = st.slider("Max pair cost accepted while chasing", 1.00, 1.30, float(saved.max_pair_cost), 0.01, key="arb_max_pair_cost")
        max_chase_price = st.slider("Max chase price (above this, DUMP instead)", 0.50, 0.99, float(saved.max_chase_price), 0.01, key="arb_max_chase_price")
        adverse_move_pct = st.slider(
            "Adverse move threshold (BTC runs this much further against leg1)", 0.0, 2.0, float(saved.adverse_move_pct), 0.05,
            help="As a fraction of the original trigger move — 0.40 = BTC ran 40% further past the trigger point.",
            key="arb_adverse_move_pct",
        )
        max_naked_loss_pct = st.slider("Max naked leg1 value decay before cutting", 0.0, 1.0, float(saved.max_naked_loss_pct), 0.05, key="arb_max_naked_loss_pct")

        st.markdown("**Risk / execution**")
        daily_loss_limit = st.number_input("Daily loss limit ($)", 0.0, 10000.0, float(saved.daily_loss_limit), 5.0, key="arb_daily_loss_limit")
        fee_rate = st.slider("Fee rate", 0.0, 0.10, float(saved.fee_rate), 0.005, key="arb_fee_rate")
        slippage = st.slider("Slippage", 0.0, 0.05, float(saved.slippage), 0.001, key="arb_slippage")
        poll_interval_sec = st.slider("Poll interval (sec)", 0.5, 5.0, float(saved.poll_interval_sec), 0.5, key="arb_poll_interval_sec")

    cfg = ArbConfig(
        lanes=lanes or list(ARB_LANES),
        max_leg1_dollars=max_leg1_dollars,
        max_total_naked_dollars=max_total_naked_dollars,
        max_blocks_per_market=max_blocks_per_market,
        leg1_trigger_pct=leg1_trigger_pct,
        required_margin=required_margin,
        max_leg1_price=max_leg1_price,
        min_leg1_price=min_leg1_price,
        min_seconds_left=min_seconds_left,
        hard_deadline_sec=hard_deadline_sec,
        panic_deadline_sec=panic_deadline_sec,
        max_pair_cost=max_pair_cost,
        max_chase_price=max_chase_price,
        adverse_move_pct=adverse_move_pct,
        max_naked_loss_pct=max_naked_loss_pct,
        daily_loss_limit=daily_loss_limit,
        fee_rate=fee_rate,
        slippage=slippage,
        paper_trading=True,
        poll_interval_sec=poll_interval_sec,
    )
    if st.button("Save arb config"):
        cfg.save()
        st.success("Saved — the arb bot picks this up within ~5 seconds, no restart needed.")
    return cfg


@st.fragment(run_every=3)
def render_arb_monitor() -> None:
    positions = arb_store.load_arb_positions()
    events = arb_store.load_arb_events(150)

    if events:
        age = time.time() - events[0]["ts"]
        if age <= LIVE_STALE_AFTER_SEC:
            st.success(f"🟢 Arb engine is running (last event {age:.0f}s ago)")
        else:
            st.error(f"🔴 Arb engine looks stopped — last event was {age:.0f}s ago. Check `sudo systemctl status polybot-arb` on the server.")
    else:
        st.warning("No data from the arb engine yet — it may still be starting up, or isn't running.")

    naked = [p for p in positions if p["status"] == "naked"]
    paired = [p for p in positions if p["status"] == "paired"]
    dumped = [p for p in positions if p["status"] in ("dumped", "expired_naked")]
    naked_exposure = sum(p["leg1_cost"] or 0.0 for p in naked)
    realized = sum(p["realized_pnl"] or 0.0 for p in positions if p["realized_pnl"] is not None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open naked legs", f"{len(naked)}", f"${naked_exposure:.2f} exposure")
    c2.metric("Paired (completed)", f"{len(paired)}")
    c3.metric("Dumped / expired naked", f"{len(dumped)}")
    c4.metric("Total realized PnL", f"${realized:+.2f}")

    st.divider()
    st.subheader("Positions")
    if positions:
        rows = [{
            "id": p["id"], "lane": lane_label(f"{p['symbol']}-{p['duration']}"), "market": p["market_slug"],
            "status": p["status"], "leg1": f"{p['leg1_side']} @ {p['leg1_price']:.3f}" if p["leg1_price"] is not None else "—",
            "leg2": (f"{p['leg2_side']} @ {p['leg2_price']:.3f} [{p['leg2_fill_kind']}]" if p["leg2_price"] is not None else "—"),
            "pair_cost": p["pair_cost"], "dump_reason": p["dump_reason"],
            "realized_pnl_$": p["realized_pnl"],
            "would_have_pnl_$": p["would_have_pnl"],
            "opened": time.strftime("%H:%M:%S", time.localtime(p["leg1_ts"])) if p["leg1_ts"] else "—",
        } for p in positions]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No positions yet.")

    st.divider()
    st.subheader("Event log")
    for e in events[:150]:
        ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        icon = {"trade": "💰", "reject": "🚫", "stop": "🛑", "chase": "⏳", "counterfactual": "🔍", "error": "⚠️", "info": "ℹ️"}.get(e["kind"], "ℹ️")
        tag = f"[{lane_label(e['lane'])}] " if e["lane"] else ""
        st.text(f"{ts} {icon} {tag}{e['text']}")


def main():
    st.title("⚖️ Temporal Arbitrage Bot")
    st.caption(
        "Buys the oversold side after a BTC spike (leg 1), then the other side after a retrace (leg 2), "
        "locking in the gap if the pair costs under $1.00. Paper-only; runs as its own always-on server "
        "process (polybot-arb), independent of whether this page is open."
    )
    arb_config_ui()
    st.divider()
    render_arb_monitor()


if __name__ == "__main__":
    main()
