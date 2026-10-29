"""
An independent watchdog for the paper test:  python levpairs_watchdog.py [--flatten "FLATTEN PAPER"] [--close-orphans "CLOSE ORPHANS"]

The engine flattens itself at 15:52 ET. This is a SEPARATE process that does not trust it: during the session it looks, every 20 seconds, at the engine's heartbeat (reports/levpairs_engine_<mode>.lock,
touched every second), at the engine's ledger and at what the PAPER account really holds in the six ETFs.

  engine alive                      never touches anything (two closers on one account double-close); raises an ALERT if positions are still open after 15:56:30
  engine dead / hung (no heartbeat for 90 s, seen on two looks 20 s apart, with no sleep / clock jump in between)
                                    ALERT (a pop-up window and reports/ALERT_<date>.txt); with --flatten "FLATTEN PAPER" it closes what the engine's ledger holds, and the legs of an entry that was in flight when the
                                    engine died (the intent on disk) - market orders through the same safe closer as KILL (re-reads the broker before each order, never more than the broker holds, never while the market
                                    is closed, one closer at a time across processes through reports/levpairs_flatten_<mode>.lock). With --close-orphans "CLOSE ORPHANS" it also closes positions in the six ETFs the ledger
                                    does not explain (on a shared account that would close another strategy's position - leave it off unless the account is dedicated).
                                    It then leaves reports/levpairs_watchdog_<mode>.json: an engine that wakes up later reloads the flat ledger and stops trading for the day.
  engine never started              ALERT after 09:40 ET on a trading day (the market clock is asked, so holidays do not alarm)
  after 16:00                       nothing can be closed any more: positions still open are reported loudly (they will carry overnight)

Schedule it with Windows Task Scheduler at any local time before 09:30 ET, like levpairs_daily.py (run_levpairs_watchdog.bat). PAPER accounts only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from typing import Callable, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levpairs_data as D
import levpairs_live as LV
import levpairs_notify as NF

FLATTEN_PHRASE, ORPHAN_PHRASE = "FLATTEN PAPER", "CLOSE ORPHANS"
STALE_S = 90.0                                                                           # no heartbeat for this long = the engine is dead or hung (it touches the lock every second)
WARN_AFTER = "15:56:30"


class Confirm:
    """'Dead' must be seen on two looks at least ``after_s`` apart before the watchdog acts, and a PC that slept (or whose clock jumped) between two looks starts the count again: a machine that has just
    resumed from sleep shows a stale heartbeat for a second or two although nothing is wrong, and the engine - frozen the same way - wakes up and touches it again."""

    def __init__(self, after_s: float = 20.0, max_gap_s: float = 60.0, wall: Callable[[], float] = time.time):
        self.after_s, self.max_gap_s, self.wall = after_s, max_gap_s, wall
        self._first: Optional[float] = None
        self._prev: Optional[float] = None

    def look(self, dead: bool) -> bool:
        now = self.wall()
        if self._prev is not None and (now - self._prev > self.max_gap_s or now < self._prev):
            self._first = None                                                           # slept, stalled or the clock moved since the last look: forget what we saw
        self._prev = now
        if not dead:
            self._first = None
            return False
        if self._first is None:
            self._first = now
            return False
        return now - self._first >= self.after_s


def decide(hhmmss: str, weekday: int, heartbeat_age: Optional[float], held: dict, ledger_open: dict, flatten: bool, close_orphans: bool, pending_syms=(), market_open: Optional[bool] = None,
           engine_expected: bool = False) -> tuple[str, str]:
    """(verdict, detail). Verdicts: idle | ok | alert | flatten | late. Pure: all the inputs are arguments. ``pending_syms``: the symbols of an entry that was in flight (intent on disk) - positions in them
    that the ledger does not explain belong to the engine and may be closed with --flatten alone; ``market_open``: the broker's clock (None = unknown)."""
    if weekday >= 5 or hhmmss < "09:30:00":
        return "idle", ""
    if market_open is False and hhmmss < "16:00:00":
        return "idle", "the market is closed today"
    held = {s: q for s, q in held.items() if q}
    alive = heartbeat_age is not None and heartbeat_age < STALE_S
    if not held:
        if engine_expected and "09:40:00" <= hhmmss < "15:50:00" and not alive:
            return "alert", f"ENGINE NOT RUNNING: no heartbeat ({'never' if heartbeat_age is None else f'{heartbeat_age:.0f} s'}) at {hhmmss} ET on a trading day - nothing is trading"
        return "ok", "the account holds none of the six ETFs"
    if hhmmss >= "16:00:00":
        return "late", f"the market is closed and the account still holds {held}: it carries overnight - close it by hand when the market opens"
    if alive:
        if hhmmss >= WARN_AFTER:
            return "alert", f"the engine is running but the account still holds {held} at {hhmmss} ET - it should be flattening"
        return "ok", "the engine is running"
    unexplained = {s: q for s, q in held.items() if s not in ledger_open}
    in_flight = {s: q for s, q in unexplained.items() if s in set(pending_syms)}
    why = f"NO ENGINE HEARTBEAT ({'never' if heartbeat_age is None else f'{heartbeat_age:.0f} s'}) and the account holds {held} (ledger: {ledger_open or 'empty'}" + (f", an entry was in flight: {sorted(pending_syms)}" if pending_syms else "") + ")"
    if flatten and (ledger_open or in_flight or (close_orphans and unexplained)):
        return "flatten", why
    return "alert", why + ("" if flatten else " - run with --flatten to let the watchdog close it")


def read_inputs(tc, mode: str) -> tuple[Optional[float], dict, dict, set, Optional[bool]]:
    """(heartbeat age in s or None, the six ETFs the account holds, the engine's ledger, the symbols of an entry that was in flight, market open?)"""
    lock = os.path.join(D.REPORTS, f"levpairs_engine_{mode}.lock")
    age = time.time() - os.path.getmtime(lock) if os.path.exists(lock) else None
    ledger, pend = {}, set()
    f = LV.state_path(mode)
    if os.path.exists(f):
        try:
            st = json.load(open(f, encoding="utf-8"))
            for name, ps in st["pairs"].items():
                if ps.get("side"):
                    L, B = name.split("/")
                    ledger[L], ledger[B] = ps["shares_L"], ps["shares_B"]
            pend = {leg[0] for leg in ((st.get("pending") or {}).get("legs") or [])}
        except (OSError, ValueError, KeyError):
            pass
    held = {p.symbol: float(p.qty) for p in tc.get_all_positions() if p.symbol in D.SYMBOLS}
    try:
        mopen = bool(tc.get_clock().is_open)
    except Exception:                                                                    # noqa: BLE001
        mopen = None
    return age, held, ledger, pend, mopen


_last_popup: dict = {}


def notify_popup(msg: str) -> None:
    """A topmost message box on the desktop (Windows), in its own thread so that it never blocks the watchdog; the same message at most once per 10 minutes. A dead engine cannot tell you itself."""
    key, now = re.sub(r"\d+", "#", msg)[:60], time.time()                               # the digits (ages, quantities, times) change at every look: the same ALERT must not pop up again and again
    if now - _last_popup.get(key, 0.0) < 600.0:
        return
    _last_popup[key] = now

    def show():
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg[:900], "Lev-ETF pairs watchdog", 0x30 | 0x40000)         # warning icon, topmost
        except Exception:                                                                # noqa: BLE001 - not Windows
            pass
    threading.Thread(target=show, daemon=True).start()


def notify_all(msg: str) -> None:
    """The pop-up window on this PC and, when LEVPAIRS_NOTIFY_URL is set, a push to your phone."""
    notify_popup(msg)
    NF.send(msg, title="Lev-ETF watchdog")


def alert(msg: str, log: Callable = print, notify: Optional[Callable[[str], None]] = None) -> None:
    line = f"{pd.Timestamp.now(tz=D.NY):%Y-%m-%d %H:%M:%S} ET  {msg}"
    log(line)
    try:
        with open(os.path.join(D.REPORTS, f"ALERT_{pd.Timestamp.now(tz=D.NY):%Y%m%d}.txt"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    if notify is not None:
        notify(msg)


def check(tc, mode: str, now: pd.Timestamp, flatten: bool, close_orphans: bool, engine_factory: Callable, log: Callable = print, confirm: Optional[Confirm] = None,
          notify: Optional[Callable[[str], None]] = None, engine_expected: bool = False) -> tuple[str, str]:
    """One look. ``engine_factory()`` returns an ARMED paper LiveEngine used only for the closing (tests pass a fake). With ``confirm`` the 'engine is dead' verdicts wait for a second look."""
    age, held, ledger, pend, mopen = read_inputs(tc, mode)
    verdict, detail = decide(now.strftime("%H:%M:%S"), now.weekday(), age, held, ledger, flatten, close_orphans, pend, mopen, engine_expected)
    dead = age is None or age >= STALE_S
    if confirm is not None:
        seen_twice = confirm.look(dead and verdict in ("alert", "flatten") and "ENGINE" in detail.upper())
        if dead and verdict in ("alert", "flatten") and "ENGINE" in detail.upper() and not seen_twice:
            return "watch", "no heartbeat on this look: confirming in 20 s - " + detail
    if verdict in ("alert", "late"):
        alert(("ALERT " if verdict == "alert" else "LATE ") + detail, log, notify)
    elif verdict == "flatten":
        alert("ENGINE DOWN - flattening: " + detail, log, notify)
        eng = engine_factory()
        res = eng.flatten("watchdog", persist=True)
        if close_orphans:
            res += eng.flatten_orphans()
        elif pend:
            res += eng.flatten_orphans(only=pend)                                        # the legs of the entry that was in flight when the engine died
        try:                                                                             # tell a revived engine: this ledger was flattened, reload it and stop trading today
            json.dump(dict(date=f"{now:%Y-%m-%d}", at=str(now), results=[dict(symbol=r["symbol"], status=r["status"]) for r in res[:20]]), open(os.path.join(D.REPORTS, f"levpairs_watchdog_{mode}.json"), "w"))
        except OSError:
            pass
        alert("watchdog closed: " + "; ".join(f"{r['symbol']}:{r['status']}" for r in res[:12]), log, notify)
        _, held2, _, _, _ = read_inputs(tc, mode)
        if any(held2.values()):
            alert(f"STILL HELD after the watchdog's flatten: {held2}", log, notify)
    return verdict, detail


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flatten", help=f'let the watchdog close the ledger positions of a dead engine: the exact phrase "{FLATTEN_PHRASE}"')
    ap.add_argument("--close-orphans", dest="close_orphans", help=f'also close unexplained positions in the six ETFs: the exact phrase "{ORPHAN_PHRASE}"')
    ap.add_argument("--mode", default="paper", choices=["paper"], help="the mode of the engine to watch (PAPER only)")
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--no-popup", dest="no_popup", action="store_true", help="no message box and no push notification, only the console and the ALERT file")
    ap.add_argument("--shared-keys", dest="shared_keys", help='allow --flatten with the default keys of ~/.claude.json (the account the Claude tools use): the exact phrase "SHARED ACCOUNT"')
    ap.add_argument("--no-engine-expected", dest="no_engine_expected", action="store_true", help="do not alert when no engine is running (a day on which you do not trade)")
    a = ap.parse_args(argv)
    if a.flatten is not None and a.flatten != FLATTEN_PHRASE:
        raise SystemExit(f'--flatten must be exactly "{FLATTEN_PHRASE}"')
    if a.close_orphans is not None and a.close_orphans != ORPHAN_PHRASE:
        raise SystemExit(f'--close-orphans must be exactly "{ORPHAN_PHRASE}"')
    if (a.flatten is not None or a.close_orphans is not None) and not D.own_keys() and a.shared_keys != "SHARED ACCOUNT":
        raise SystemExit("--flatten / --close-orphans close positions: give the watchdog the project's own keys ('Alpaca Key.txt' in the project folder, or ALPACA_API_KEY / ALPACA_SECRET_KEY) "
                         "(without them it would use ~/.claude.json, the account the Claude tools use), or knowingly accept the shared account with --shared-keys \"SHARED ACCOUNT\"")
    from alpaca.trading.client import TradingClient
    k, s = D.creds(True)
    tc = D.harden(TradingClient(k, s, paper=True))

    def factory() -> LV.LiveEngine:
        from levpairs_ui.presets import DEFAULT_PRESET, cfgs
        sc, c = cfgs(DEFAULT_PRESET)
        eng = LV.LiveEngine(sc, c, LV.LiveSettings(paper=True, armed=True, feed="iex", accept_iex=True), log=lambda m, k="info": alert(f"[engine] {m}"))
        eng.tc = tc
        eng.acct = dict(blocked=False)
        eng.state = eng._latest_state()
        return eng
    confirm = Confirm(after_s=20.0, max_gap_s=max(3 * a.every, 60.0))
    last = ("", "")
    print(f"watchdog on the PAPER account, flatten={'on' if a.flatten else 'off'}, orphans={'on' if a.close_orphans else 'off'}, popup={'off' if a.no_popup else 'on'}", flush=True)
    while True:
        now = LV.clock_now()
        if now.strftime("%H:%M:%S") > "16:10:00":
            return
        try:
            v = check(tc, "paper", now, a.flatten is not None, a.close_orphans is not None, factory, log=lambda m: print(m, flush=True), confirm=confirm,
                      notify=None if a.no_popup else notify_all, engine_expected=not a.no_engine_expected)
            if v != last and v[0] not in ("alert", "late", "flatten"):
                print(f"{now:%H:%M:%S} {v[0]} {v[1]}", flush=True)
            last = v
        except Exception as e:                                                           # noqa: BLE001 - the watchdog itself must not die on a network blip
            alert(f"watchdog could not look ({type(e).__name__}: {str(e)[:100]})", notify=None if a.no_popup else notify_all)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
