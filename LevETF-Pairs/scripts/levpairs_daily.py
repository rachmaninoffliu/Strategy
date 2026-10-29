"""
The headless daily job for the month-long PAPER test (no GUI, nothing to click):  python levpairs_daily.py <command> ...

  day     wait for 09:25 ET, run the session (decide every bar, send both legs together, hard flatten at 15:52), wait for 16:20 ET, then do `post`.   <- schedule this one
  run     the session only
  post    the evening routine for one session: the broker's own record of the day (orders + fills), top up the cached 1-minute SIP and IEX bars, then the live-vs-backtest comparison
          (levpairs_compare.py) -> reports/live_compare/REPORT_<date>.md and MONTH_TO_DATE.md
  status  what the engine has done today (orders journaled, open ledger, last decision), nothing is sent
  kill    put reports/KILL in place: the running engine stops entering and exits its positions at its next decision (within one bar); `unkill` removes it (the next session refuses to start while it exists)
  flatten close what the ledger holds on the PAPER account right now, for a DEAD engine (needs --arm "ARM PAPER"; refuses while an engine heartbeat is fresh unless --force; --orphans "CLOSE ORPHANS" also
          closes unexplained positions in the six ETFs). Ctrl+C in the console of a running `day` / `run` flattens by itself on the way out.

Safety:  PAPER accounts only (there is no flag for a live account here). Without --arm it is a DRY RUN: signals and a shadow book, nothing is sent. Sending orders needs the exact phrase
         --arm "ARM PAPER"; with the IEX feed (the only real-time feed this account may read) it also needs --accept-iex "ACCEPT IEX" - the IEX s-scores of these pairs are mostly noise
         (LEVPAIRS_REPORT.md section 4F): the paper test then measures the plumbing, the fills and the feed gap, not the strategy.
Example:  python levpairs_daily.py day --arm "ARM PAPER" --accept-iex "ACCEPT IEX" --capital 20000
Schedule: Windows Task Scheduler, weekdays, at any local time that is BEFORE 09:25 ET (the job waits for the open itself, so a daylight-saving shift cannot make it late); LIVE_TEST_RUNBOOK.md.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import replace
from typing import Callable, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levpairs_data as D
import levpairs_live as LV
import levpairs_notify as NF

ARM_PHRASE, IEX_PHRASE = "ARM PAPER", "ACCEPT IEX"


def build_settings(a, need_iex_phrase: bool = True) -> LV.LiveSettings:
    """The live settings from the command line; refuses to arm without the exact phrases. ``need_iex_phrase=False`` for the commands that never enter a trade (flatten)."""
    armed = a.arm is not None
    if armed and a.arm != ARM_PHRASE:
        raise SystemExit(f'--arm must be exactly "{ARM_PHRASE}"')
    if a.accept_iex is not None and a.accept_iex != IEX_PHRASE:
        raise SystemExit(f'--accept-iex must be exactly "{IEX_PHRASE}"')
    if armed and need_iex_phrase and a.feed == "iex" and a.accept_iex is None:
        raise SystemExit(f'sending orders on IEX signals needs --accept-iex "{IEX_PHRASE}": the IEX s-scores of these pairs correlate only 0.2-0.57 with the consolidated tape and the rules trade 8x as often')
    return LV.LiveSettings(paper=True, armed=armed, feed=a.feed, accept_iex=a.accept_iex is not None or not need_iex_phrase, capital_override=float(a.capital or 0.0), equity_fraction=float(a.equity_fraction),
                           flatten_orphans_eod=armed and not getattr(a, "no_orphan_eod", False))                  # an order that was in flight when a call hung or the process died must not carry overnight


def require_own_keys(live: LV.LiveSettings, a) -> None:
    """An ARMED job sends and closes orders on whatever account its keys belong to. Without the project's key file (or ALPACA_* variables) the keys would come from ~/.claude.json: the
    account the Claude tools use. Refuse that unless the user says so."""
    if live.armed and a.command in ("day", "run", "flatten") and not D.own_keys() and getattr(a, "shared_keys", None) != "SHARED ACCOUNT":
        raise SystemExit("an ARMED job needs the project's own keys: 'Alpaca Key.txt' (Key ID / Secret Key) in the project folder, or ALPACA_API_KEY / ALPACA_SECRET_KEY. Without them it would use "
                         "~/.claude.json - the account the Claude tools use - and the 15:52 orphan close, the watchdog and a KILL would act on THAT account's positions. To use it anyway: --shared-keys \"SHARED ACCOUNT\"")


def make_logger(date: pd.Timestamp) -> Callable[[str, str], None]:
    os.makedirs(D.REPORTS, exist_ok=True)
    path = os.path.join(D.REPORTS, f"levpairs_daily_{date:%Y%m%d}.log")

    def log(msg: str, kind: str = "info") -> None:
        line = f"{pd.Timestamp.now(tz=D.NY):%H:%M:%S} {kind.upper():5s} {msg}"
        print(line, flush=True)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    return log


def wait_until_et(hhmm: str, log: Callable = print, now_fn: Callable = LV.clock_now, sleep: Callable = time.sleep) -> None:
    """Sleep until today's ``hhmm`` New York time (returns at once if it has passed)."""
    while True:
        now = now_fn()
        target = pd.Timestamp(f"{now:%Y-%m-%d} {hhmm}", tz=D.NY)
        if now >= target:
            return
        left = (target - now).total_seconds()
        if left > 120:
            log(f"waiting for {hhmm} ET ({left / 60:.0f} min)", "info")
        sleep(min(left, 60.0))


def flatten_on_signals() -> None:
    """Ctrl+C already flattens (run_session's finally). Make Ctrl+Break and a plain `taskkill` / SIGTERM do the same instead of ending the process with positions open. (Closing the console window, a forced
    kill, logoff or a reboot cannot be caught: that is what the watchdog is for.)"""
    def handler(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), handler)
            except (ValueError, OSError):                                                # not the main thread
                pass


def keep_awake(on: bool) -> None:
    """Windows: ask the system not to sleep while the job runs (a sleeping PC sends no orders and, worse, does not flatten)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 if on else 0x80000000)
    except Exception:                                                                    # noqa: BLE001 - not Windows
        pass


def leftovers(live: LV.LiveSettings, today: pd.Timestamp) -> tuple[list, set]:
    """(open pairs of an EARLIER session in the ledger, symbols of an order that was in flight): what makes the engine refuse to start. Nothing is meant to be held overnight."""
    import json
    f = LV.state_path(LV.mode_name(live))
    if not os.path.exists(f):
        return [], set()
    try:
        st = json.load(open(f, encoding="utf-8"))
    except (OSError, ValueError):
        return [], set()
    if st.get("date") == f"{pd.Timestamp(today):%Y-%m-%d}":
        return [], set()
    pend = {leg[0] for leg in ((st.get("pending") or {}).get("legs") or [])}
    return [k for k, v in st.get("pairs", {}).items() if v.get("side")], pend


def clean_leftovers(live: LV.LiveSettings, preset: str, today: pd.Timestamp, log, engine_factory: Optional[Callable] = None, wait: Callable = None, notify: Optional[Callable[[str], None]] = None) -> bool:
    """Armed jobs only. A crash, a kill or a logoff yesterday leaves positions in the ledger (the engine then refuses to start and the day is lost): close them as soon as the market is open, with the same safe
    closer as KILL, and close the legs of an entry that was in flight (verified against the broker). Returns True when the ledger is clean."""
    open_pairs, pend = leftovers(live, today)
    if not live.armed or not (open_pairs or pend):
        return True
    log(f"yesterday's leftovers: open pairs {open_pairs or 'none'}, an entry was in flight on {sorted(pend) or 'nothing'} - closing them once the market is open", "warn")
    (wait or wait_until_et)("09:30:30", log)
    eng = (engine_factory or (lambda: make_engine(live, preset, log)))()
    if eng.tc is None:
        eng.connect()
    eng.state = eng._latest_state()
    if open_pairs:
        eng.flatten("KILL", persist=True)
    if pend:
        eng.flatten_orphans(only=pend)
    still_pairs, still_pend = leftovers(live, today)
    if still_pairs or still_pend:
        msg = f"the leftovers could NOT be cleared (open {still_pairs}, in flight {sorted(still_pend)}): close them by hand, then start again - no trading today"
        log(msg, "err")
        if notify:
            notify("Lev-ETF paper job: " + msg)
        return False
    log("the leftovers are closed", "ok")
    return True


def make_engine(live: LV.LiveSettings, preset: str, log) -> LV.LiveEngine:
    from levpairs_ui.presets import PRESETS, cfgs
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    sc, c = cfgs(preset)
    return LV.LiveEngine(sc, c, live, log=log)


def post(live: LV.LiveSettings, preset: str, date: pd.Timestamp, log, wait: bool = True, tape=None, notify: Optional[Callable[[str], None]] = None) -> Optional[dict]:
    """The evening routine. Every step is independent: one failing does not stop the others."""
    mode = LV.mode_name(live)
    d = pd.Timestamp(date).normalize()
    if not any(os.path.exists(os.path.join(D.REPORTS, f"levpairs_{n}_{mode}.jsonl")) for n in ("decisions", "fills")):
        log(f"nothing was recorded for {d.date()} - no comparison", "warn")
        return None
    if wait:
        wait_until_et("16:20", log)                                                      # SIP history is only readable 15 minutes after the fact
    if live.armed:
        try:
            eng = make_engine(live, preset, log)
            eng.connect()
            out = eng.sync_broker_day(d)
            log(f"broker record: {len(out['orders'])} orders, {len(out['fills'])} fills written", "ok")
        except Exception as e:                                                           # noqa: BLE001
            log(f"broker sync failed ({type(e).__name__}: {str(e)[:120]}) - the comparison uses the journal only", "err")
    try:
        D.update_cache(feeds=("sip", "iex"), log=lambda m: log(m, "info"))
    except Exception as e:                                                               # noqa: BLE001
        log(f"data top-up failed ({type(e).__name__}: {str(e)[:120]}) - the comparison may lack today's bars", "err")
    try:
        import levpairs_compare as K
        out = K.run_day(d, mode, tape if tape is not None else K.AlpacaTape(), log=lambda m: log(m, "warn"))
        s = out["summary"]
        log(f"compared: {s['live_trips']} live trips, {s['matched']} matched with the SIP twin, live P&L ${s['live_pnl_usd']:,.2f}, slippage vs NBBO mid {s['slip_vs_mid_bp']:.2f} bp - reports/live_compare/REPORT_{d:%Y-%m-%d}.md", "ok")
        if live.armed and notify:
            notify(f"Lev-ETF paper day {d:%Y-%m-%d}: {s['live_trips']} trips, P&L ${s['live_pnl_usd']:,.2f}, {s['matched']} matched with the backtest, fill slippage vs NBBO mid {s['slip_vs_mid_bp']:.2f} bp")
        return out
    except Exception as e:                                                               # noqa: BLE001
        log(f"comparison failed ({type(e).__name__}: {str(e)[:160]})", "err")
        if live.armed and notify:
            notify(f"Lev-ETF paper job: the evening comparison for {d:%Y-%m-%d} FAILED ({type(e).__name__}) - look at reports/levpairs_daily_{d:%Y%m%d}.log")
        return None


def status(live: LV.LiveSettings) -> str:
    import json
    mode = LV.mode_name(live)
    lines = [f"mode {mode.upper()} · reports {D.REPORTS}"]
    f = LV.state_path(mode)
    if os.path.exists(f):
        st = json.load(open(f, encoding="utf-8"))
        open_pairs = {k: (v["shares_L"], v["shares_B"]) for k, v in st["pairs"].items() if v.get("side")}
        lines.append(f"ledger {st['date']}: last bar {st['last_j']}, open {open_pairs or 'none'}, halted {st.get('halted')}, alarms {st.get('alarms') or 'none'}, orders sent {st.get('orders', 0)}")
    lock = os.path.join(D.REPORTS, f"levpairs_engine_{mode}.lock")
    lines.append("engine heartbeat: " + (f"{time.time() - os.path.getmtime(lock):.0f} s ago" if os.path.exists(lock) else "no engine running"))
    for n in ("fills", "decisions"):
        p = os.path.join(D.REPORTS, f"levpairs_{n}_{mode}.jsonl")
        lines.append(f"{n}: " + (f"{sum(1 for _ in open(p, encoding='utf-8'))} rows in {os.path.basename(p)}" if os.path.exists(p) else "none yet"))
    return "\n".join(lines)


def flatten_now(live: LV.LiveSettings, a, engine_factory: Optional[Callable] = None) -> str:
    """The manual closer for a dead engine. Never while another engine looks alive (unless --force)."""
    if not live.armed:
        raise SystemExit(f'flatten sends orders: add --arm "{ARM_PHRASE}" (PAPER account)')
    if a.orphans is not None and a.orphans != "CLOSE ORPHANS":
        raise SystemExit('--orphans must be exactly "CLOSE ORPHANS"')
    lock = os.path.join(D.REPORTS, f"levpairs_engine_{LV.mode_name(live)}.lock")
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 90 and not a.force:
        return "an engine heartbeat is fresh: an engine is running. Stop it first (Ctrl+C in its console flattens on the way out) or use --force"
    eng = (engine_factory or (lambda: make_engine(live, a.preset, make_logger(LV.clock_now().tz_localize(None).normalize()))))()
    if eng.tc is None:
        eng.connect()
    eng.state = eng._latest_state()
    res = eng.flatten("KILL", persist=True)
    if a.orphans is not None:
        res += eng.flatten_orphans()
    return "; ".join(f"{r['symbol']}:{r['status']}" for r in res) or "nothing to close"


def main(argv=None) -> None:
    from levpairs_ui.presets import DEFAULT_PRESET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["day", "run", "post", "status", "kill", "unkill", "flatten"])
    ap.add_argument("--arm", help=f'send orders to the PAPER account: the exact phrase "{ARM_PHRASE}" (without it: DRY RUN)')
    ap.add_argument("--accept-iex", dest="accept_iex", help=f'allow orders on IEX signals: the exact phrase "{IEX_PHRASE}"')
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"], help="feed of the live signals (this account may only read IEX in real time)")
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--capital", type=float, default=0.0, help="deploy this many dollars instead of equity x fraction")
    ap.add_argument("--equity-fraction", dest="equity_fraction", type=float, default=1.0)
    ap.add_argument("--date", help="post: the session to process (default today)")
    ap.add_argument("--shared-keys", dest="shared_keys", help='allow an ARMED job on the default keys of ~/.claude.json (the account the Claude tools use): the exact phrase "SHARED ACCOUNT"')
    ap.add_argument("--now", action="store_true", help="post: do not wait for 16:20 ET")
    ap.add_argument("--mode", choices=["paper", "dry"], help="post / status: read and write the files of this mode (paper = what an ARMED job wrote) without arming anything; default: dry unless --arm is given")
    ap.add_argument("--no-orphan-eod", dest="no_orphan_eod", action="store_true", help="armed jobs close unexplained positions in the six ETFs at the 15:52 flatten (a dedicated account assumed); this switches that off")
    ap.add_argument("--force", action="store_true", help="flatten: even if an engine heartbeat looks fresh (two closers on one account can double-close: stop the engine first)")
    ap.add_argument("--orphans", help='flatten: also close unexplained positions in the six ETFs - the exact phrase "CLOSE ORPHANS"')
    a = ap.parse_args(argv)
    live = build_settings(a, need_iex_phrase=a.command != "flatten")
    require_own_keys(live, a)
    if a.mode == "paper" and a.command in ("post", "status"):
        live = replace(live, armed=True, accept_iex=True)                                # the files of the paper session; post / status never send an order
    if a.command == "status":
        print(status(live))
        return
    if a.command in ("kill", "unkill"):
        f = os.path.join(D.REPORTS, "KILL")
        if a.command == "kill":
            os.makedirs(D.REPORTS, exist_ok=True)
            open(f, "w").write(f"{pd.Timestamp.now(tz=D.NY)}\n")
            print(f"{f} created: a running engine exits its positions at its next decision and enters nothing; no session starts until `unkill`")
        elif os.path.exists(f):
            os.remove(f)
            print("KILL file removed")
        return
    if a.command == "flatten":
        print(flatten_now(live, a))
        return
    date = pd.Timestamp(a.date) if a.date else LV.clock_now().tz_localize(None).normalize()
    log = make_logger(date)
    if a.command == "post":
        post(live, a.preset, date, log, wait=not a.now, notify=NF.send)
        return
    keep_awake(True)
    flatten_on_signals()
    try:
        if a.command == "day":
            wait_until_et("09:25", log)
        if live.armed and not clean_leftovers(live, a.preset, date, log, notify=NF.send):
            raise SystemExit(2)
        eng = make_engine(live, a.preset, log)
        log(f"{LV.mode_name(live).upper()} session, feed {live.feed.upper()}, preset {a.preset!r}", "ok")
        try:
            eng.run_session()
        except Exception as e:                                                           # noqa: BLE001
            if live.armed:
                NF.send(f"Lev-ETF paper job: the session ENDED WITH AN ERROR ({type(e).__name__}: {str(e)[:150]}) - the ledger was flattened on the way out; look at reports/levpairs_daily_{date:%Y%m%d}.log")
            raise
        if a.command == "day":
            post(live, a.preset, date, log, notify=NF.send)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
