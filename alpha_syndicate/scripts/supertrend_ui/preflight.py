"""Pre-flight (go / no-go) before a session. It never sends an order: it reads the account, the market and the data, and compares them with what the settings assume.

Status per check: PASS · WARN (advice, never a gate) · FAIL (blocks LIVE / armed sessions) · INFO · SKIP.  Verdict: NO-GO if any FAIL, CAUTION if any WARN, else GO."""
from __future__ import annotations

import json
import math
import os
import queue
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import supertrend_gui as GU

from . import data as D
from . import params as P
from .params import Check, LiveSettings, Report

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
STATE_NAME = "preflight_state.json"


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def state_path() -> str:
    return os.path.join(GU.REPORTS, STATE_NAME)


def load_state() -> dict:
    try:
        with open(state_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                           # noqa: BLE001
        return {}


def save_state(st: dict) -> None:
    try:
        os.makedirs(GU.REPORTS, exist_ok=True)
        with open(state_path(), "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=2, default=str)
    except Exception:                                           # noqa: BLE001
        pass


def record_session_end(why: str, mode: str, pnl: float = 0.0, trades: int = 0) -> None:
    """Remember how the last session ended, so the next pre-flight can warn after a KILL or a crash."""
    st = load_state()
    st["last_session"] = dict(why=why, mode=mode, pnl=pnl, trades=trades, ended=datetime.now(timezone.utc).isoformat())
    save_state(st)


def save_report(rep: Report) -> str:
    try:
        os.makedirs(GU.REPORTS, exist_ok=True)
        path = os.path.join(GU.REPORTS, f"preflight_{datetime.fromtimestamp(rep.created):%Y%m%d_%H%M%S}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(rep.text())
        rep.path = path
        return path
    except Exception:                                           # noqa: BLE001
        return ""


def run_preflight(values: dict, ls: LiveSettings, log: Callable[[str], None] = lambda m: None, *, creds_fn: Optional[Callable] = None, trading_factory: Optional[Callable] = None,
                  data_factory: Optional[Callable] = None, bars_fn: Optional[Callable] = None, backtest_fn: Optional[Callable] = None, now: Optional[datetime] = None) -> Report:
    """Run every check. The broker / data clients and the backtest are injectable so the logic is testable without a network."""
    creds_fn = creds_fn or GU.creds
    trading_factory = trading_factory or (lambda k, s, paper: GU.TradingClient(k, s, paper=paper))
    data_factory = data_factory or (lambda k, s: GU.StockHistoricalDataClient(k, s))
    now_dt = now or D.now_et()
    rep = Report(created=time.time(), live=not ls.paper, armed=ls.armed, capital=ls.capital)
    add = lambda g, n, st, d, key="": rep.checks.append(Check(g, n, st, d, key or n))     # noqa: E731
    cfg = P.to_config(values)
    frac = float(values["equity_frac"])
    acct_name = "Alpaca PAPER account" if ls.paper else "Alpaca LIVE account"

    # ── 1. credentials, connection, endpoint ──
    tc = dc = acct = None
    log("checking credentials …")
    try:
        k, s = creds_fn(ls.paper)
        add("Account", "API keys", PASS, f"keys for the {acct_name} found in ~/.claude.json", "creds")
    except Exception as exc:                                    # noqa: BLE001
        k = s = None
        add("Account", "API keys", FAIL, f"{exc}. Add the keys of the {acct_name} to the alpaca-{'paper' if ls.paper else 'live'} entry in ~/.claude.json.", "creds")
    if k:
        log("connecting to Alpaca …")
        try:
            t0 = time.time()
            tc = trading_factory(k, s, ls.paper)
            acct = tc.get_account()
            lat = (time.time() - t0) * 1000
            dc = data_factory(k, s)
            base = str(getattr(tc, "_base_url", ""))
            is_paper_ep = "PAPER" in base.upper()
            if ls.paper and not is_paper_ep:
                add("Account", "Endpoint", FAIL, f"PAPER selected but the endpoint is {base}: the engine refuses to run", "endpoint")
            elif not ls.paper and is_paper_ep:
                add("Account", "Endpoint", FAIL, f"LIVE selected but the keys connected to the PAPER endpoint ({base}) - wrong keys in alpaca-live?", "endpoint")
            else:
                add("Account", "Endpoint", PASS, f"{'paper' if is_paper_ep else 'live'} endpoint, latency {lat:.0f} ms", "endpoint")
        except Exception as exc:                                # noqa: BLE001
            tc = acct = None
            add("Account", "Connection", FAIL, f"could not connect / read the account: {type(exc).__name__}: {str(exc)[:160]}", "connect")
    else:
        add("Account", "Connection", SKIP, "no keys", "connect")

    equity = bp = 0.0
    if acct is not None:
        equity, bp = _f(acct.equity), _f(acct.buying_power)
        status = str(getattr(acct, "status", "")).upper()
        blocked = [n for n in ("trading_blocked", "account_blocked") if getattr(acct, n, False)]
        num_ = str(getattr(acct, "account_number", ""))
        rep.account = dict(number=num_[:4] + "****", equity=equity, buying_power=bp, status=status)
        if blocked or "ACTIVE" not in status:
            add("Account", "Status", FAIL, f"status {status or '?'}" + (f", blocked: {', '.join(blocked)}" if blocked else ""), "status")
        else:
            add("Account", "Status", PASS, f"{num_[:4]}**** ACTIVE, equity ${equity:,.2f}, buying power ${bp:,.2f}", "status")
        if ls.capital > equity * 1.02:
            add("Account", "Capital vs equity", FAIL if (ls.armed and not ls.paper) else WARN,
                f"session capital ${ls.capital:,.0f} is above the account equity ${equity:,.0f}: orders sized off the capital can exceed what the account holds", "capital")
        else:
            add("Account", "Capital vs equity", PASS, f"session capital ${ls.capital:,.0f} ≤ account equity ${equity:,.0f}", "capital")
        budget = ls.capital * frac / 100.0
        if bp < budget:
            add("Account", "Buying power", WARN, f"buying power ${bp:,.0f} is below one full-size order (${budget:,.0f}) - the broker would reject it", "bp")
        pdt = bool(getattr(acct, "pattern_day_trader", False))
        dtc = int(_f(getattr(acct, "daytrade_count", 0)))
        if pdt or (equity < 25_000 and dtc >= 3):
            add("Account", "Day-trade rule", WARN, f"{'flagged as a pattern day trader' if pdt else f'{dtc} day trades in the last 5 days with equity under $25,000'}: the broker may block further same-day round trips "
                                                   "(take-profits and reversals can be same-day)", "pdt")
        if not getattr(acct, "shorting_enabled", True):
            add("Account", "Shorting", WARN, "shorting is disabled on this account: shorts are routed to long SOXS", "shorting")

    # ── 2. market and instruments ──
    clk = None
    if tc is not None:
        log("reading the market clock and the instruments …")
        try:
            clk = tc.get_clock()
            add("Market", "Clock", INFO, ("OPEN, closes " if clk.is_open else "CLOSED, opens ") + str(clk.next_close if clk.is_open else clk.next_open)[:16], "clock")
        except Exception as exc:                                # noqa: BLE001
            add("Market", "Clock", WARN, f"could not read the market clock: {exc}", "clock")
        for sym in (GU.SYMBOL_LONG, GU.SYMBOL_SHORT):
            try:
                a = tc.get_asset(sym)
                if not getattr(a, "tradable", True):
                    add("Market", f"{sym} tradable", FAIL, f"{sym} is not tradable at this broker", f"asset_{sym}")
                elif sym == GU.SYMBOL_LONG:
                    short_ok = bool(getattr(a, "shortable", False)) and bool(getattr(acct, "shorting_enabled", True))
                    htb = "" if getattr(a, "easy_to_borrow", True) else " (hard to borrow: a borrow fee applies)"
                    route = (f"shorts go DIRECT (short SOXL){htb}" if short_ok else "SOXL is not shortable here: shorts are routed to long SOXS") if ls.prefer_short else "shorts always go to long SOXS (your setting)"
                    add("Market", f"{sym} tradable", PASS, f"tradable; {route}", f"asset_{sym}")
                else:
                    add("Market", f"{sym} tradable", PASS, "tradable", f"asset_{sym}")
            except Exception as exc:                            # noqa: BLE001
                add("Market", f"{sym} tradable", WARN, f"could not read the asset: {str(exc)[:100]}", f"asset_{sym}")

    # ── 3. data: the SAME fetch the live engine uses ──
    bars = None
    if dc is not None or bars_fn is not None:
        log("loading the latest bars (same call the live engine makes) …")
        try:
            if bars_fn is not None:
                bars = bars_fn()
            else:
                eng = GU.Engine(queue.Queue(), ls.paper, ls.capital, frac / 100.0, ls.start_date or datetime.now().strftime("%Y-%m-%d"), False)
                eng.dclient = dc
                bars = eng.fetch_bars()
        except Exception as exc:                                # noqa: BLE001
            add("Data", "Latest bars", FAIL, f"could not load {GU.SYMBOL_LONG} bars: {type(exc).__name__}: {str(exc)[:140]}", "bars")
    if bars is not None:
        n = len(bars)
        need = cfg.atrPeriod + 2
        if n < need:
            add("Data", "Warm-up", FAIL, f"only {n} bars loaded but the ATR period is {cfg.atrPeriod}: the engine would sit in 'warming up' forever", "warmup")
        else:
            add("Data", "Warm-up", PASS, f"{n} bars loaded for an ATR period of {cfg.atrPeriod} (the engine loads at most {P.LIVE_WARMUP_BARS})", "warmup")
        last = bars.index[-1]
        age = now_dt - last
        open_now = bool(getattr(clk, "is_open", D.is_rth(now_dt))) if clk is not None else D.is_rth(now_dt)
        if open_now and age > timedelta(minutes=45):
            add("Data", "Freshness", WARN, f"newest bar is {last:%H:%M} ET, {int(age.total_seconds() // 60)} min old while the market is open: the feed may be delayed", "fresh")
        else:
            add("Data", "Freshness", PASS, f"newest bar {last:%Y-%m-%d %H:%M} ET" + ("" if open_now else " (market closed)"), "fresh")
        sig = D.compute_signal(bars, cfg, now_dt) if n >= need else None
        if sig:
            add("Data", "Signal now", INFO, f"{'UP → long SOXL' if sig['want'] > 0 else 'DOWN → short SOXL'} on the {sig['bar']:%H:%M} bar; close {sig['close']:,.2f}, Supertrend {sig['st']:,.2f}, "
                                          f"re-entry limit {sig['limit']:,.2f}", "signal")
            px = sig["close"]
            q_long = math.floor(ls.capital * frac / 100.0 / px) if px > 0 else 0
            if q_long < 1:
                add("Sizing", "One order", FAIL, f"capital ${ls.capital:,.0f} x {frac:g}% buys 0 shares at {px:,.2f}", "size")
            else:
                add("Sizing", "One order", INFO, f"a long today = {q_long:,} SOXL (${q_long * px:,.0f}); a short = {math.floor(ls.capital * frac / 100.0 * cfg.short_mult / px):,} shares "
                                                f"({cfg.short_mult:g}x size)", "size")

    # ── 4. what the account already holds, and whether this app manages it ──
    if tc is not None:
        try:
            held = {p.symbol: p for p in tc.get_all_positions() if abs(_f(p.qty)) > 1e-9}
            orders = tc.get_orders(GU.GetOrdersRequest(status=GU.QueryOrderStatus.OPEN))
            try:
                with open(GU.STATE_FILE, encoding="utf-8") as fh:
                    state = json.load(fh)
            except Exception:                                   # noqa: BLE001
                state = {}
            managed = bool(state.get("pos")) and state.get("symbol") in held
            if not held:
                add("Exposure", "Open positions", PASS, "flat" + (f"; {len(orders)} open order(s)" if orders else ""), "positions")
                if state.get("pos"):
                    add("Exposure", "Saved exit levels", INFO, "a leftover record of a closed position exists; the engine clears it on the first poll", "state")
            elif managed:
                sym = state["symbol"]
                add("Exposure", "Open positions", INFO, f"holding {sym} ({state.get('side')}) with frozen exit levels" +
                    ("; SOXS exits are polled by THIS app only - nothing rests at the broker" if sym == GU.SYMBOL_SHORT else "; its take-profit rests at the broker"), "positions")
            else:
                add("Exposure", "Open positions", WARN, "the account holds " + ", ".join(f"{s} {_f(p.qty):+,.0f}" for s, p in held.items()) +
                    " but this app has no saved exit levels for it (opened elsewhere, or the state file was cleared): its take-profit / trailing stop will NOT be managed", "positions")
            if orders and not held:
                add("Exposure", "Open orders", WARN, f"{len(orders)} order(s) are resting: {', '.join(f'{o.symbol} {o.side}' for o in orders[:4])}", "orders")
        except Exception as exc:                                # noqa: BLE001
            add("Exposure", "Open positions", WARN, f"could not read positions / orders: {str(exc)[:120]}", "positions")

    # ── 5. app state, settings and risk ──
    kill = GU.KILL_FILE
    if os.path.exists(kill):
        add("Safety", "Kill switch", FAIL, f"a KILL file exists ({kill}): the engine halts the moment it starts. Use 'Clear kill latch' (or delete the file) once you are sure.", "kill")
    else:
        add("Safety", "Kill switch", PASS, "no KILL file", "kill")
    last = load_state().get("last_session", {})
    if last and any(w in str(last.get("why", "")).upper() for w in ("KILL", "HALT")):
        add("Safety", "Previous session", WARN, f"the last session ended with: {last.get('why')} ({str(last.get('ended', ''))[:16]})", "prev")
    elif last:
        add("Safety", "Previous session", PASS, f"ended normally: {last.get('why')}", "prev")
    if ls.daily_loss_pct > 8.0 + 1e-9 or ls.max_trades > 12:
        add("Safety", "Risk limits", WARN, f"daily loss limit {ls.daily_loss_pct:g}% / {ls.max_trades} trades a day: above the defaults (8% / 12)", "limits")
    else:
        add("Safety", "Risk limits", PASS, f"new entries pause at -{ls.daily_loss_pct:g}% on the day or after {ls.max_trades} trades; orders only while the market is open", "limits")
    if ls.armed and not ls.paper and frac > P.SUGGESTED_FRAC + 1e-9:
        rep.suggested_frac = P.SUGGESTED_FRAC
        add("Safety", "Position size", WARN, f"{frac:g}% of equity per trade on a 3x ETF is about {frac * 3 / 100:.1f}x effective exposure; the reference backtest drew down about -50%. "
                                              f"Consider {P.SUGGESTED_FRAC:g}% while the strategy is unvalidated.", "size_risk")
    if ls.start_date:
        try:
            days = (datetime.now() - datetime.strptime(ls.start_date, "%Y-%m-%d")).days
            if days > 120:
                add("Settings", "Start date", WARN, f"the start date is {days} days back: while the account has no fills the engine re-simulates the whole period on EVERY poll (slow, a lot of data)", "start")
            elif days < 0:
                add("Settings", "Start date", FAIL, "the start date is in the future", "start")
        except ValueError:
            add("Settings", "Start date", FAIL, f"'{ls.start_date}' is not a date (YYYY-MM-DD)", "start")
    for msg in P.notices(values):
        add("Settings", "Parameters", WARN, msg, "param")
    if not any(c.key == "param" for c in rep.checks):
        add("Settings", "Parameters", PASS, "the reference settings (Always Fixed, ATR 500, factor 5.0)", "param")

    # ── 6. is the strategy doing anything lately? (cached bars, current settings) ──
    log("running a recent backtest on the cached bars …")
    try:
        run = backtest_fn or _recent_backtest
        res = run(values)
        if res:
            r12, r90, sh, dd, nt = res
            add("Strategy", "Recent backtest", WARN if r12 < 0 else INFO,
                f"last 12 months: {r12 * 100:+,.0f}%, Sharpe {sh:.2f}, max drawdown {dd * 100:.0f}%, {nt} trades; last 90 days: {r90 * 100:+,.0f}%" +
                ("  (negative year: nothing in the history says the edge is on right now)" if r12 < 0 else ""), "recent")
        else:
            add("Strategy", "Recent backtest", SKIP, "no cached bars to run on - press 'Update data'", "recent")
    except Exception as exc:                                    # noqa: BLE001
        add("Strategy", "Recent backtest", SKIP, f"could not run: {str(exc)[:100]}", "recent")

    # ── 7. validation status (team rule: Valid signs off before a strategy is called ready for live) ──
    if P.VALID_SIGNED_OFF:
        add("Validation", "Valid sign-off", PASS, "Valid has signed off", "valid")
    else:
        add("Validation", "Valid sign-off", WARN if (ls.armed and not ls.paper) else INFO, P.VALID_NOTE, "valid")

    rep.notes = ["A pre-flight never sends an order. CAUTION is advice, not a gate; NO-GO blocks armed sessions (a DRY RUN is always allowed).",
                 "Backtest results are in-sample and the Python port does not reconcile with the TradingView Pine run; nothing here is a guarantee."]
    return rep


def _recent_backtest(values: dict):
    """(12-month return, 90-day return, Sharpe, max drawdown, trades) of the current settings on the cached bars, or None."""
    from . import backtest as B
    try:
        bars = D.read_cache("SOXL", 15)
    except FileNotFoundError:
        return None
    end = bars.index[-1]
    out = []
    for days in (365, 90):
        bt = P.BacktestSettings(symbol="SOXL", tf=15, start=(end - timedelta(days=days)).strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), capital=100_000.0, prime=True, benchmark=False)
        r = B.run_backtest(bars, values, bt)
        out.append(r)
    a, b = out
    return a.metrics["total_return"], b.metrics["total_return"], a.metrics.get("sharpe", float("nan")), a.metrics["max_drawdown"], a.metrics["trades"]
