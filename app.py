"""Streamlit dashboard for the Polymarket latency-momentum bot — a pure VIEWER.

The actual trading engine runs as its own always-on process (run_bot.py, via
systemd), completely independent of this dashboard. This file only reads what
that process has written to polybot.db, so the bot keeps trading whether or
not anyone has this page open, whether or not your PC is even on.

Run with:  streamlit run app.py   (viewing only — does not start trading)
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polybot import arb_store, calibration, store
from polybot.arb_config import ARB_LANES, ArbConfig
from polybot.broker import Broker, Position
from polybot.config import ALL_LANES, BotConfig, lane_label, lane_parts

# --- palette (validated, see dataviz skill) ---
C_BLUE = "#2a78d6"     # BTC
C_GOOD = "#0ca30c"
C_CRITICAL = "#d03b3b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"

SYMBOL_COLOR = {"BTCUSDT": C_BLUE}
LIVE_STALE_AFTER_SEC = 20  # if the trading process hasn't written live_state in this long, flag it

st.set_page_config(page_title="Polymarket Latency Bot", page_icon="⚡", layout="wide")
store.init_db()


def load_broker(cfg: BotConfig) -> Broker:
    """Read-only Broker populated from the DB, so we can reuse its aggregate
    calculations (exposure, drawdown, Kelly-relevant stats) without duplicating
    that logic here."""
    broker = Broker(cfg)
    broker.positions = [Position(**row) for row in store.load_positions()]
    return broker


def sidebar_config() -> BotConfig:
    st.sidebar.header("Lanes")
    saved = BotConfig.load()

    lanes = []
    for lane in ALL_LANES:
        if st.sidebar.checkbox(lane_label(lane), value=(lane in saved.lanes)):
            lanes.append(lane)

    st.sidebar.header("Momentum signal")
    dynamic_threshold = st.sidebar.toggle(
        "Dynamic threshold (relative to live BTC volatility)", value=saved.dynamic_threshold,
        help="Instead of a fixed %, trigger when the move is unusual relative to BTC's own recent "
             "volatility (z-score of sigma * sqrt(window)). Adapts to calm vs. volatile regimes "
             "instead of a threshold that never fires in a quiet market or fires constantly in a busy one.",
    )
    dynamic_threshold_z = st.sidebar.slider(
        "Dynamic threshold strength (sigma multiple)", 0.5, 4.0, float(saved.dynamic_threshold_z), 0.1,
        disabled=not dynamic_threshold,
        help="Trigger at |move| >= z * volatility * sqrt(window). Lower = more signals, more noise. Higher = fewer, stronger signals.",
    )
    threshold = st.sidebar.slider(
        "Fixed threshold (%)", 0.02, 2.0, float(saved.momentum_threshold_pct), 0.02, disabled=dynamic_threshold,
    )
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

    use_edge_gate = st.sidebar.toggle(
        "Edge gate (fair value vs current price)", value=saved.use_edge_gate,
        help="After a signal fires and picks Up/Down, checks the modeled fair value (same win-probability "
             "model as the confidence gate) against the side's CURRENT market price. Only buys if fair value "
             "sits at least the required % above what we'd pay — i.e. the BTC move should still move this "
             "price, not already be priced in.",
    )
    min_edge_pct = st.sidebar.slider(
        "Min required edge (%)", 1.0, 100.0, float(saved.min_edge_pct), 1.0, disabled=not use_edge_gate,
    )

    use_skew_signal = st.sidebar.toggle(
        "Skew-as-signal (contrarian probability adjustment)", value=saved.use_skew_signal,
        help="Folds how far the market's own price already sits from 50/50 into the modeled win probability "
             "itself: discounts confidence when the triggered side is already priced in, boosts it when the "
             "market is crowded the other way. Complements the edge gate rather than replacing it.",
    )
    skew_signal_weight = st.sidebar.slider(
        "Skew signal weight (max probability points)", 0.0, 0.30, float(saved.skew_signal_weight), 0.01,
        disabled=not use_skew_signal,
    )

    st.sidebar.subheader("Convergence filter")
    use_convergence_filter = st.sidebar.toggle(
        "Require indicator agreement before entering", value=saved.use_convergence_filter,
        help="Computes RSI mean-reversion, VWAP deviation, and SMA crossover alongside the momentum signal "
             "and requires several of them to agree on direction before trading — filters out noise-level "
             "momentum blips that a single indicator alone can't distinguish from a real move.",
    )
    convergence_min_agree = st.sidebar.slider(
        "Min agreeing indicators (of 4, incl. momentum)", 1, 4, int(saved.convergence_min_agree), 1,
        disabled=not use_convergence_filter,
    )
    with st.sidebar.expander("Convergence filter settings", expanded=False):
        convergence_rsi_period = st.slider("RSI period (buckets)", 5, 30, int(saved.convergence_rsi_period), 1, disabled=not use_convergence_filter)
        convergence_sma_short_sec = st.slider("SMA short window (sec)", 10, 120, int(saved.convergence_sma_short_sec), 5, disabled=not use_convergence_filter)
        convergence_sma_long_sec = st.slider("SMA long window (sec)", 60, 600, int(saved.convergence_sma_long_sec), 10, disabled=not use_convergence_filter)
        convergence_vwap_lookback_sec = st.slider("VWAP lookback (sec)", 30, 600, int(saved.convergence_vwap_lookback_sec), 10, disabled=not use_convergence_filter)
        convergence_bucket_sec = st.slider("Resample bucket width (sec)", 1, 30, int(saved.convergence_bucket_sec), 1, disabled=not use_convergence_filter)

    st.sidebar.subheader("Risk / exit")
    use_btc_reversal_stop = st.sidebar.toggle(
        "BTC-reversal stop (primary)", value=saved.use_btc_reversal_stop,
        help="Exits based on whether BTC itself has moved against the position (z-score vs the round's open "
             "price), not the option's own noisy bid price. Fixes stop-losses that were shaking out correct "
             "calls purely on option-price whipsaw before BTC had actually reversed.",
    )
    btc_reversal_z = st.sidebar.slider(
        "BTC reversal sigma", 0.25, 3.0, float(saved.btc_reversal_z), 0.25, disabled=not use_btc_reversal_stop,
    )
    btc_reversal_min_elapsed = st.sidebar.slider(
        "Min seconds before reversal check", 0, 60, int(saved.btc_reversal_min_elapsed_sec), 5,
        disabled=not use_btc_reversal_stop,
    )
    stop_loss = st.sidebar.slider(
        "Stop loss (%) — price-based safety net", 5, 100, int(saved.stop_loss_pct), 5,
        help="Loose fallback floor for genuine liquidity/mispricing blowouts, not the primary exit trigger.",
    )
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
        dynamic_threshold=dynamic_threshold,
        dynamic_threshold_z=dynamic_threshold_z,
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
        use_edge_gate=use_edge_gate,
        min_edge_pct=min_edge_pct,
        use_skew_signal=use_skew_signal,
        skew_signal_weight=skew_signal_weight,
        use_convergence_filter=use_convergence_filter,
        convergence_min_agree=convergence_min_agree,
        convergence_rsi_period=convergence_rsi_period,
        convergence_sma_short_sec=convergence_sma_short_sec,
        convergence_sma_long_sec=convergence_sma_long_sec,
        convergence_vwap_lookback_sec=convergence_vwap_lookback_sec,
        convergence_bucket_sec=convergence_bucket_sec,
        use_btc_reversal_stop=use_btc_reversal_stop,
        btc_reversal_z=btc_reversal_z,
        btc_reversal_min_elapsed_sec=btc_reversal_min_elapsed,
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
        st.sidebar.success("Saved — the trading process picks this up within ~5 seconds, no restart needed.")
    return cfg


def render_engine_status():
    live = store.load_live_state()
    if not live:
        st.warning("No data from the trading process yet — it may still be starting up, or isn't running. Check `sudo systemctl status polybot-engine` on the server.")
        return
    newest = max(v["updated_ts"] for v in live.values())
    age = time.time() - newest
    if age <= LIVE_STALE_AFTER_SEC:
        st.success(f"🟢 Trading engine is running (last update {age:.0f}s ago)")
    else:
        st.error(f"🔴 Trading engine looks stopped — last update was {age:.0f}s ago. Check `sudo systemctl status polybot-engine` on the server.")


def render_portfolio_header(cfg: BotConfig, broker: Broker):
    exposure = broker.exposure_usd()
    available = broker.available_cash()
    realized = broker.realized_pnl_usd()

    today_pnl = broker.realized_pnl_today_usd()
    pnl_24h, trades_24h = broker.realized_pnl_last_24h_usd()
    drawdown = broker._portfolio_drawdown_pct()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bankroll", f"${cfg.bankroll_usd:,.2f}")
    c2.metric("Available cash", f"${available:,.2f}")
    c3.metric("Open exposure", f"${exposure:,.2f}", f"{len(broker.open_positions())} positions")
    c4.metric("Realized PnL (all-time)", f"${realized:+,.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("PnL — last 24h", f"${pnl_24h:+,.2f}", f"{trades_24h} trades")
    c6.metric("Today's PnL", f"${today_pnl:+,.2f}")
    c7.metric("Drawdown from peak", f"{drawdown:.1f}%")
    if cfg.daily_loss_limit_usd > 0 and today_pnl <= -abs(cfg.daily_loss_limit_usd):
        st.error(f"Daily loss circuit breaker tripped (${today_pnl:+,.2f}) — no new trades will open until tomorrow.")


def render_revenue_breakdown(broker: Broker):
    closed = [p for p in broker.positions if p.status == "closed"]
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


def lane_stats(broker: Broker, lane: str) -> dict:
    closed = [p for p in broker.positions if p.status == "closed" and p.lane == lane]
    open_pos = [p for p in broker.open_positions() if p.lane == lane]
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


def render_lane_card(cfg: BotConfig, broker: Broker, lane: str, key_prefix: str = "combined"):
    symbol, duration = lane_parts(lane)
    color = SYMBOL_COLOR.get(symbol, C_MUTED)

    live = store.load_live_state().get(lane, {})
    price = live.get("price")
    pct_change = live.get("pct_change")
    window_sec = live.get("window_sec")
    stats = lane_stats(broker, lane)
    window_label = f"{window_sec:.0f}s" if (pct_change is not None and cfg.multi_window_scan) else f"{cfg.momentum_window_sec}s"

    with st.container(border=True):
        st.markdown(f"### 🤖 {lane_label(lane)} agent")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${price:,.2f}" if price else "—")
        c2.metric(f"Momentum ({window_label})", f"{pct_change:+.3f}%" if pct_change is not None else "—")
        c3.metric("Agent PnL", f"${stats['total_pnl']:+,.2f}", f"{stats['n_trades']} trades")
        c4.metric("Win rate", f"{stats['win_rate']:.0f}%" if stats["win_rate"] is not None else "—")

        if live.get("market_slug"):
            st.caption(f"Round `{live['market_slug']}` — {live.get('seconds_left', 0):.0f}s left ({live.get('seconds_elapsed', 0):.0f}s elapsed)")
        else:
            st.caption("No active round found.")

        if stats["open_pos"]:
            for p in stats["open_pos"]:
                st.info(f"OPEN: {p.side} @ {p.entry_price:.3f} — ${p.size_usd:.2f} staked ({p.shares:.1f} contracts)")
        else:
            st.caption("No open position.")

        hist = store.load_price_history(lane, since_ts=time.time() - 3600)
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


def render_agent_tab(cfg: BotConfig, broker: Broker, lane: str):
    stats = lane_stats(broker, lane)

    render_lane_card(cfg, broker, lane, key_prefix="agent")
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
    events = [e for e in store.load_events(300) if e["lane"] == lane or e["lane"] is None][:100]
    for e in events:
        ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        icon = {"signal": "🎯", "trade": "💰", "error": "⚠️", "info": "ℹ️"}.get(e["kind"], "ℹ️")
        st.text(f"{ts} {icon} {e['text']}")


def render_trade_log_tab(cfg: BotConfig, broker: Broker):
    st.subheader("Trade log — every trade, every agent")
    all_positions = sorted(broker.positions, key=lambda p: p.entry_ts, reverse=True)
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
            "mode": "PAPER" if cfg.paper_trading else "LIVE",
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


def render_live_tab(cfg: BotConfig, broker: Broker):
    render_engine_status()
    render_portfolio_header(cfg, broker)
    st.divider()

    cols = st.columns(2)
    for i, lane in enumerate(cfg.lanes):
        with cols[i % 2]:
            render_lane_card(cfg, broker, lane)

    st.divider()
    st.subheader("Event log (all lanes)")
    events = store.load_events(150)
    for e in events:
        ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        icon = {"signal": "🎯", "trade": "💰", "error": "⚠️", "info": "ℹ️"}.get(e["kind"], "ℹ️")
        tag = f"[{lane_label(e['lane'])}] " if e["lane"] else ""
        st.text(f"{ts} {icon} {tag}{e['text']}")


def render_calibration(broker: Broker):
    closed = [
        {"entry_confidence": p.entry_confidence, "pnl_usd": p.pnl_usd}
        for p in broker.positions if p.status == "closed"
    ]
    scored = [c for c in closed if c["entry_confidence"] is not None]
    if not scored:
        st.caption("No trades with a stated entry confidence yet — calibration data accumulates as new trades close.")
        return

    brier = calibration.brier_score(closed)
    c1, c2 = st.columns(2)
    c1.metric("Brier score (lower is better)", f"{brier:.3f}" if brier is not None else "—",
               help="0 = perfectly calibrated, 0.25 = no better than a coin flip, higher = worse.")
    c2.metric("Trades with a stated confidence", f"{len(scored)}")

    rows = calibration.calibration_buckets(closed)
    if rows:
        st.caption("Compares each confidence bucket's *stated* win probability to its *realized* win rate. "
                   "A large positive/negative gap means the model is over/under-confident in that band.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_history_tab(broker: Broker):
    st.subheader("Revenue breakdown")
    render_revenue_breakdown(broker)

    st.divider()
    st.subheader("Confidence calibration")
    render_calibration(broker)

    st.divider()
    st.subheader("All closed trades")
    closed = [p for p in broker.positions if p.status == "closed"]
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


@st.fragment(run_every=3)
def render_dashboard_body(cfg: BotConfig) -> None:
    """Everything that needs to auto-refresh lives in this fragment, which reruns
    itself every 3s in isolation. Keeping this separate from the sidebar means a
    "Save config" click there never races the auto-refresh and gets cancelled —
    that was silently dropping saves before this was split out.
    """
    broker = load_broker(cfg)

    agent_tab_labels = [f"🤖 {lane_label(lane)}" for lane in cfg.lanes]
    tab_names = ["Combined dashboard"] + agent_tab_labels + ["Trade log", "Trade history / revenue"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_live_tab(cfg, broker)
    for i, lane in enumerate(cfg.lanes):
        with tabs[1 + i]:
            render_agent_tab(cfg, broker, lane)
    with tabs[1 + len(cfg.lanes)]:
        render_trade_log_tab(cfg, broker)
    with tabs[2 + len(cfg.lanes)]:
        render_history_tab(broker)


def arb_config_ui() -> ArbConfig:
    """Every Temporal Arb parameter, editable. Deliberately NOT inside the
    auto-refreshing fragment below — same reason the momentum bot's config
    lives in the sidebar rather than the fragment: a Save click racing an
    auto-refresh timer can silently get cancelled before the write completes.
    """
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
    st.title("⚡ Polymarket Latency Bot")

    top_tabs = st.tabs(["🚀 Momentum Bot", "⚖️ Temporal Arb"])

    with top_tabs[0]:
        st.caption(
            "Binance momentum → Polymarket BTC Up/Down markets. 5m + 15m, in parallel. "
            "This dashboard is a viewer — the bot trades in its own always-on server process "
            "regardless of whether this page is open."
        )
        cfg = sidebar_config()
        render_dashboard_body(cfg)

    with top_tabs[1]:
        st.caption(
            "Temporal Arbitrage — buys the oversold side after a BTC spike (leg 1), then the other side "
            "after a retrace (leg 2), locking in the gap if the pair costs under $1.00. Paper-only; "
            "runs as its own always-on server process (polybot-arb)."
        )
        arb_config_ui()
        st.divider()
        render_arb_monitor()


if __name__ == "__main__":
    main()
