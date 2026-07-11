"""Streamlit GUI for the Polymarket latency-momentum bot.
Runs BTC 5m + 15m ("lanes") in parallel under one shared portfolio.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polybot.config import ALL_LANES, BotConfig, lane_label, lane_parts
from polybot.engine import MultiEngine

# --- palette (validated, see dataviz skill) ---
C_BLUE = "#2a78d6"     # BTC
C_GOOD = "#0ca30c"
C_CRITICAL = "#d03b3b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"

SYMBOL_COLOR = {"BTCUSDT": C_BLUE}

st.set_page_config(page_title="Polymarket Latency Bot", page_icon="⚡", layout="wide")


@st.cache_resource
def get_engine() -> MultiEngine:
    # No arguments -> exactly one cache entry for the process lifetime, so the
    # same engine (and its running background thread) survives every rerun,
    # including a full page refresh. Config is live-patched onto it below instead
    # of being part of the cache key — keying on the config previously meant any
    # difference in sidebar state (e.g. widget defaults resetting on a fresh
    # session) silently created a brand new, never-started engine.
    return MultiEngine(BotConfig())


def sidebar_config() -> BotConfig:
    st.sidebar.header("Lanes")
    saved = BotConfig.load()

    lanes = []
    for lane in ALL_LANES:
        if st.sidebar.checkbox(lane_label(lane), value=(lane in saved.lanes)):
            lanes.append(lane)

    st.sidebar.header("Momentum signal")
    threshold = st.sidebar.slider("Threshold (%)", 0.02, 2.0, float(saved.momentum_threshold_pct), 0.02)
    window = st.sidebar.slider("Window (seconds)", 5, 300, int(saved.momentum_window_sec), 5)

    st.sidebar.subheader("Multi-window scan")
    multi_window = st.sidebar.toggle(
        "Scan several lookback windows, trade the strongest",
        value=saved.multi_window_scan,
        help="Instead of one fixed window, checks all windows below each tick and fires on whichever shows the biggest |% move|.",
    )
    scan_windows_str = st.sidebar.text_input(
        "Windows to scan (sec, comma-separated)",
        value=",".join(str(w) for w in saved.scan_windows_sec),
        disabled=not multi_window,
    )
    try:
        scan_windows = sorted({int(x.strip()) for x in scan_windows_str.split(",") if x.strip()})
    except ValueError:
        scan_windows = list(saved.scan_windows_sec)
    if not scan_windows:
        scan_windows = list(saved.scan_windows_sec)

    st.sidebar.subheader("Entry timing")
    max_in = st.sidebar.slider("Don't enter after N sec into round", 10, 890, int(saved.max_seconds_into_window), 10)
    min_left = st.sidebar.slider("Don't enter with < N sec left", 0, 120, int(saved.min_seconds_left), 5)
    cooldown = st.sidebar.slider("Cooldown per lane (sec)", 0, 120, int(saved.cooldown_sec), 5)
    warmup = st.sidebar.slider("Warmup after start (sec, ignore signals)", 0, 120, int(saved.warmup_sec), 5)

    st.sidebar.subheader("Entry quality gates")
    max_entry_price = st.sidebar.slider(
        "Max entry price (implied win prob)", 0.50, 0.99, float(saved.max_entry_price), 0.01,
        help="Skip the trade if the side is already priced above this — the move is likely already priced in.",
    )
    cooldown_after_loss = st.sidebar.slider("Cooldown after a loss (sec, per lane)", 0, 600, int(saved.cooldown_after_loss_sec), 15)
    daily_loss_limit = st.sidebar.number_input(
        "Daily loss circuit breaker (USD, 0 = off)", 0.0, 1_000_000.0, float(saved.daily_loss_limit_usd), 10.0,
        help="Once today's realized loss hits this, no new trades open (any lane) until tomorrow.",
    )

    use_confidence_gate = st.sidebar.toggle(
        "Confidence gate on momentum signals", value=saved.use_confidence_gate,
        help="Extra check on top of the momentum trigger: models win probability as Phi(|z|), "
             "z = ln(current/round_open) / (volatility * sqrt(seconds_left)). Only enters if that "
             "modeled probability clears the threshold below.",
    )
    with st.sidebar.expander("Confidence gate settings", expanded=False):
        confidence_threshold = st.slider(
            "Min confidence to enter", 0.50, 0.99, float(saved.confidence_threshold), 0.01, disabled=not use_confidence_gate,
        )
        confidence_vol_lookback = st.slider(
            "Volatility lookback (sec)", 30, 900, int(saved.confidence_vol_lookback_sec), 30, disabled=not use_confidence_gate,
        )

    st.sidebar.subheader("Risk / exit")
    stop_loss = st.sidebar.slider("Stop loss (%)", 5, 100, int(saved.stop_loss_pct), 5)
    hold = st.sidebar.checkbox("Hold to resolution (ignore take-profit)", value=saved.hold_to_resolution)
    take_profit = st.sidebar.slider("Take profit (%)", 0, 200, int(saved.take_profit_pct), 5, disabled=hold)

    st.sidebar.header("Portfolio")
    bankroll = st.sidebar.number_input("Bankroll (USD)", 10.0, 1_000_000.0, float(saved.bankroll_usd), 10.0)
    per_trade = st.sidebar.number_input("Per-trade stake (USD)", 1.0, 100_000.0, float(saved.per_trade_usd), 1.0)
    max_conc = st.sidebar.slider("Max concurrent positions (all lanes)", 1, 20, int(saved.max_concurrent_positions))
    max_conc_lane = st.sidebar.slider("Max concurrent per lane", 1, 5, int(saved.max_concurrent_per_lane))

    use_kelly = st.sidebar.toggle(
        "Dynamic Kelly sizing (per lane)", value=saved.use_kelly_sizing,
        help="Replaces the flat per-trade stake with quarter-Kelly, scaled down in drawdown / loss streaks, scaled up on win streaks. Needs enough closed trades per lane first.",
    )
    with st.sidebar.expander("Kelly sizing settings", expanded=False):
        kelly_fraction = st.slider("Kelly fraction", 0.05, 1.0, float(saved.kelly_fraction), 0.05, disabled=not use_kelly)
        kelly_max_pct = st.slider("Max stake (% of bankroll)", 0.01, 0.5, float(saved.kelly_max_pct), 0.01, disabled=not use_kelly)
        kelly_min_trades = st.slider("Min closed trades before Kelly kicks in", 1, 50, int(saved.kelly_min_trades), 1, disabled=not use_kelly)

    st.sidebar.header("Execution")
    paper = st.sidebar.toggle("Paper trading (simulated, no real orders)", value=saved.paper_trading)
    if not paper:
        st.sidebar.error(
            "LIVE mode places real orders with real funds via py-clob-client. "
            "Requires POLY_PRIVATE_KEY set in your environment. Use with extreme caution."
        )
    poll = st.sidebar.slider("Poll interval (sec)", 0.5, 5.0, float(saved.poll_interval_sec), 0.5)

    cfg = BotConfig(
        lanes=lanes or list(ALL_LANES),
        momentum_threshold_pct=threshold,
        momentum_window_sec=window,
        multi_window_scan=multi_window,
        scan_windows_sec=scan_windows,
        max_seconds_into_window=max_in,
        min_seconds_left=min_left,
        cooldown_sec=cooldown,
        warmup_sec=warmup,
        max_entry_price=max_entry_price,
        cooldown_after_loss_sec=cooldown_after_loss,
        daily_loss_limit_usd=daily_loss_limit,
        use_confidence_gate=use_confidence_gate,
        confidence_threshold=confidence_threshold,
        confidence_vol_lookback_sec=confidence_vol_lookback,
        stop_loss_pct=stop_loss,
        hold_to_resolution=hold,
        take_profit_pct=take_profit,
        bankroll_usd=bankroll,
        per_trade_usd=per_trade,
        max_concurrent_positions=max_conc,
        max_concurrent_per_lane=max_conc_lane,
        use_kelly_sizing=use_kelly,
        kelly_fraction=kelly_fraction,
        kelly_max_pct=kelly_max_pct,
        kelly_min_trades=kelly_min_trades,
        paper_trading=paper,
        poll_interval_sec=poll,
    )
    if st.sidebar.button("Save config"):
        cfg.save()
        st.sidebar.success("Saved.")
    return cfg


def render_portfolio_header(engine: MultiEngine):
    broker = engine.broker
    exposure = broker.exposure_usd()
    available = broker.available_cash()
    realized = broker.realized_pnl_usd()

    today_pnl = broker.realized_pnl_today_usd()
    pnl_24h, trades_24h = broker.realized_pnl_last_24h_usd()
    drawdown = broker._portfolio_drawdown_pct()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bankroll", f"${engine.config.bankroll_usd:,.2f}")
    c2.metric("Available cash", f"${available:,.2f}")
    c3.metric("Open exposure", f"${exposure:,.2f}", f"{len(broker.open_positions())} positions")
    c4.metric("Realized PnL (all-time)", f"${realized:+,.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("PnL — last 24h", f"${pnl_24h:+,.2f}", f"{trades_24h} trades")
    c6.metric("Today's PnL", f"${today_pnl:+,.2f}")
    c7.metric("Drawdown from peak", f"{drawdown:.1f}%")
    if engine.config.daily_loss_limit_usd > 0 and today_pnl <= -abs(engine.config.daily_loss_limit_usd):
        st.error(f"Daily loss circuit breaker tripped (${today_pnl:+,.2f}) — no new trades will open until tomorrow.")


def render_revenue_breakdown(engine: MultiEngine):
    closed = [p for p in engine.broker.positions if p.status == "closed"]
    if not closed:
        st.caption("No closed trades yet — revenue breakdown will populate as trades resolve.")
        return

    rows = [{"symbol": p.symbol.replace("USDT", ""), "duration": p.duration, "pnl_usd": p.pnl_usd or 0.0} for p in closed]
    df = pd.DataFrame(rows)

    grouped = df.groupby(["symbol", "duration"])["pnl_usd"].agg(["sum", "count"]).reset_index()
    grouped.columns = ["symbol", "duration", "total_pnl_usd", "trades"]
    st.dataframe(grouped, use_container_width=True)

    fig = go.Figure()
    for symbol in sorted(df["symbol"].unique()):
        sub = grouped[grouped["symbol"] == symbol]
        fig.add_trace(go.Bar(
            x=sub["duration"], y=sub["total_pnl_usd"], name=symbol,
            marker_color=SYMBOL_COLOR.get(symbol + "USDT", C_MUTED),
        ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10), barmode="group",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=C_GRID, title="round duration"),
        yaxis=dict(gridcolor=C_GRID, title="PnL (USD)"),
        legend=dict(orientation="h", y=1.15),
        title="Revenue by coin x round duration",
    )
    st.plotly_chart(fig, use_container_width=True)

    by_symbol = df.groupby("symbol")["pnl_usd"].sum()
    by_duration = df.groupby("duration")["pnl_usd"].sum()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**By coin**")
        st.dataframe(by_symbol.reset_index().rename(columns={"pnl_usd": "total_pnl_usd"}), use_container_width=True)
    with c2:
        st.write("**By round duration**")
        st.dataframe(by_duration.reset_index().rename(columns={"pnl_usd": "total_pnl_usd"}), use_container_width=True)


def lane_stats(engine: MultiEngine, lane: str) -> dict:
    closed = [p for p in engine.broker.positions if p.status == "closed" and p.lane == lane]
    open_pos = [p for p in engine.broker.open_positions() if p.lane == lane]
    wins = [p for p in closed if (p.pnl_usd or 0.0) > 0]
    losses = [p for p in closed if (p.pnl_usd or 0.0) <= 0]
    total_pnl = sum(p.pnl_usd or 0.0 for p in closed)
    return {
        "closed": closed,
        "open_pos": open_pos,
        "n_trades": len(closed),
        "n_open": len(open_pos),
        "total_pnl": total_pnl,
        "win_rate": (len(wins) / len(closed) * 100) if closed else None,
        "avg_trade": (total_pnl / len(closed)) if closed else None,
        "avg_win": (sum(p.pnl_usd or 0.0 for p in wins) / len(wins)) if wins else None,
        "avg_loss": (sum(p.pnl_usd or 0.0 for p in losses) / len(losses)) if losses else None,
        "best_trade": max((p.pnl_usd or 0.0 for p in closed), default=None),
        "worst_trade": min((p.pnl_usd or 0.0 for p in closed), default=None),
    }


def render_lane_card(engine: MultiEngine, lane: str, key_prefix: str = "combined"):
    symbol, duration = lane_parts(lane)
    color = SYMBOL_COLOR.get(symbol, C_MUTED)

    price = engine.latest_price(symbol)
    reading = engine.momentum(lane)
    market = engine.active_market(lane)
    stats = lane_stats(engine, lane)
    window_label = f"{reading.window_sec:.0f}s" if (reading.ok and engine.config.multi_window_scan) else f"{engine.config.momentum_window_sec}s"

    with st.container(border=True):
        st.markdown(f"### 🤖 {lane_label(lane)} agent")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${price:,.2f}" if price else "—")
        c2.metric(f"Momentum ({window_label})", f"{reading.pct_change:+.3f}%" if reading.ok else "—")
        c3.metric("Agent PnL", f"${stats['total_pnl']:+,.2f}", f"{stats['n_trades']} trades")
        c4.metric("Win rate", f"{stats['win_rate']:.0f}%" if stats["win_rate"] is not None else "—")

        if market:
            st.caption(f"Round `{market.slug}` — {market.seconds_left:.0f}s left ({market.seconds_elapsed:.0f}s elapsed)")
        else:
            st.caption("No active round found.")

        if stats["open_pos"]:
            for p in stats["open_pos"]:
                st.info(f"OPEN: {p.side} @ {p.entry_price:.3f} — ${p.size_usd:.2f} staked ({p.shares:.1f} contracts)")
        else:
            st.caption("No open position.")

        hist = engine.snapshot_momentum(lane)
        if hist:
            df = pd.DataFrame(hist)
            df["time"] = pd.to_datetime(df["ts"], unit="s")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["time"], y=df["price"], mode="lines", line=dict(color=color, width=2)))
            fig.update_layout(
                height=160, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                xaxis=dict(gridcolor=C_GRID, title=None, showticklabels=False),
                yaxis=dict(gridcolor=C_GRID, title=None),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}_{lane}")


def render_agent_tab(engine: MultiEngine, lane: str):
    symbol, duration = lane_parts(lane)
    stats = lane_stats(engine, lane)

    render_lane_card(engine, lane, key_prefix="agent")
    st.divider()

    st.subheader("Agent stats")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PnL", f"${stats['total_pnl']:+,.2f}")
    c2.metric("Trades closed", f"{stats['n_trades']}")
    c3.metric("Win rate", f"{stats['win_rate']:.1f}%" if stats["win_rate"] is not None else "—")
    c4.metric("Avg trade PnL", f"${stats['avg_trade']:+,.2f}" if stats["avg_trade"] is not None else "—")
    c5.metric("Open positions", f"{stats['n_open']}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Avg win", f"${stats['avg_win']:+,.2f}" if stats["avg_win"] is not None else "—")
    c7.metric("Avg loss", f"${stats['avg_loss']:+,.2f}" if stats["avg_loss"] is not None else "—")
    c8.metric(
        "Best / worst trade",
        f"${stats['best_trade']:+,.2f}" if stats["best_trade"] is not None else "—",
        f"worst ${stats['worst_trade']:+,.2f}" if stats["worst_trade"] is not None else None,
    )

    st.divider()
    st.subheader(f"{lane_label(lane)} — open positions")
    if stats["open_pos"]:
        rows = [{
            "market": p.market_slug, "side": p.side, "entry_price": p.entry_price,
            "implied_prob_%": p.entry_price * 100, "size_$": p.size_usd, "contracts": round(p.shares, 2),
            "opened": time.strftime("%H:%M:%S", time.localtime(p.entry_ts)),
        } for p in stats["open_pos"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No open position for this agent.")

    st.subheader(f"{lane_label(lane)} — closed trades")
    if stats["closed"]:
        rows = [{
            "market": p.market_slug, "side": p.side,
            "entry_price": p.entry_price, "exit_price": p.exit_price,
            "implied_prob_%": p.entry_price * 100, "size_$": p.size_usd, "contracts": round(p.shares, 2),
            "pnl_%": p.pnl_pct, "pnl_$": p.pnl_usd, "exit_reason": p.exit_reason,
            "opened": time.strftime("%H:%M:%S", time.localtime(p.entry_ts)),
            "closed": time.strftime("%H:%M:%S", time.localtime(p.exit_ts)) if p.exit_ts else "—",
        } for p in sorted(stats["closed"], key=lambda p: p.entry_ts, reverse=True)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No closed trades yet for this agent.")

    st.subheader(f"{lane_label(lane)} — event log")
    events = list(reversed(engine.snapshot_events(lane)))[:100]
    for e in events:
        ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
        icon = {"signal": "🎯", "trade": "💰", "error": "⚠️", "info": "ℹ️"}.get(e.kind, "ℹ️")
        st.text(f"{ts} {icon} {e.text}")


def render_trade_log_tab(engine: MultiEngine):
    st.subheader("Trade log — every trade, every agent")
    all_positions = sorted(engine.broker.positions, key=lambda p: p.entry_ts, reverse=True)
    if not all_positions:
        st.caption("No trades yet.")
        return

    rows = []
    for p in all_positions:
        rows.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.entry_ts)),
            "agent": lane_label(p.lane),
            "market": p.market_slug,
            "side": p.side,
            "status": p.status,
            "size_$": round(p.size_usd, 2),
            "contracts": round(p.shares, 3),
            "entry_price": round(p.entry_price, 4),
            "win_probability_%": round(p.entry_price * 100, 1),
            "exit_price": round(p.exit_price, 4) if p.exit_price is not None else None,
            "pnl_%": round(p.pnl_pct, 2) if p.pnl_pct is not None else None,
            "pnl_$": round(p.pnl_usd, 2) if p.pnl_usd is not None else None,
            "exit_reason": p.exit_reason or "—",
            "mode": "PAPER" if engine.config.paper_trading else "LIVE",
        })
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    agents = ["All"] + sorted(df["agent"].unique().tolist())
    agent_filter = c1.selectbox("Agent", agents)
    status_filter = c2.selectbox("Status", ["All", "open", "closed"])
    side_filter = c3.selectbox("Side", ["All", "Up", "Down"])

    view = df
    if agent_filter != "All":
        view = view[view["agent"] == agent_filter]
    if status_filter != "All":
        view = view[view["status"] == status_filter]
    if side_filter != "All":
        view = view[view["side"] == side_filter]

    st.dataframe(view, use_container_width=True)

    closed_df = df[df["status"] == "closed"]
    if len(closed_df):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total trades", f"{len(df)}")
        c2.metric("Closed / open", f"{len(closed_df)} / {len(df) - len(closed_df)}")
        c3.metric("Overall win rate", f"{(closed_df['pnl_$'] > 0).mean() * 100:.1f}%")
        c4.metric("Total realized PnL", f"${closed_df['pnl_$'].sum():+,.2f}")


def render_live_tab(engine: MultiEngine):
    render_portfolio_header(engine)
    st.divider()

    cols = st.columns(2)
    for i, lane in enumerate(engine.config.lanes):
        with cols[i % 2]:
            render_lane_card(engine, lane)

    st.divider()
    st.subheader("Event log (all lanes)")
    events = list(reversed(engine.snapshot_events()))[:150]
    for e in events:
        ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
        icon = {"signal": "🎯", "trade": "💰", "error": "⚠️", "info": "ℹ️"}.get(e.kind, "ℹ️")
        tag = f"[{lane_label(e.lane)}] " if e.lane else ""
        st.text(f"{ts} {icon} {tag}{e.text}")


def render_history_tab(engine: MultiEngine):
    st.subheader("Revenue breakdown")
    render_revenue_breakdown(engine)

    st.divider()
    st.subheader("All closed trades")
    closed = [p for p in engine.broker.positions if p.status == "closed"]
    if not closed:
        st.caption("No closed trades yet.")
        return
    rows = [{
        "lane": lane_label(p.lane), "market": p.market_slug, "side": p.side,
        "entry": p.entry_price, "exit": p.exit_price, "reason": p.exit_reason,
        "pnl_%": p.pnl_pct, "pnl_$": p.pnl_usd,
        "opened": time.strftime("%H:%M:%S", time.localtime(p.entry_ts)),
    } for p in closed]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    win_rate = (df["pnl_%"] > 0).mean() * 100
    st.metric("Overall win rate", f"{win_rate:.1f}%")
    st.metric("Total realized PnL", f"${df['pnl_$'].sum():+.2f}")


def main():
    st.title("⚡ Polymarket Latency Bot")
    st.caption("Binance momentum → Polymarket BTC Up/Down markets. 5m + 15m, in parallel. Paper-trades by default.")

    cfg = sidebar_config()
    engine = get_engine()
    engine.config = cfg
    engine.broker.config = cfg

    # Auto-start on first load of a fresh engine (e.g. right after a service
    # restart/deploy) so the bot is always trading without a manual click —
    # this is a server-hosted bot, it shouldn't need a human to press Start.
    if not engine.running:
        engine.start()

    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("▶ Start bot", disabled=engine.running, type="primary"):
        engine.start()
        st.rerun()
    if c2.button("■ Stop bot", disabled=not engine.running):
        engine.stop()
        st.rerun()
    c3.write("🟢 Running" if engine.running else "🔴 Stopped")

    agent_tab_labels = [f"🤖 {lane_label(lane)}" for lane in cfg.lanes]
    tab_names = ["Combined dashboard"] + agent_tab_labels + ["Trade log", "Trade history / revenue"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_live_tab(engine)
    for i, lane in enumerate(cfg.lanes):
        with tabs[1 + i]:
            render_agent_tab(engine, lane)
    with tabs[1 + len(cfg.lanes)]:
        render_trade_log_tab(engine)
    with tabs[2 + len(cfg.lanes)]:
        render_history_tab(engine)

    if engine.running:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
