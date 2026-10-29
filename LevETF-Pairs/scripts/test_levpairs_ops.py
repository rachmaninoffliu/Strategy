"""
Tests for the daily job and the watchdog of the paper test:  python test_levpairs_ops.py   (fakes only: no network, no orders)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levpairs_compare as K
import levpairs_daily as DY
import levpairs_data as D
import levpairs_live as LV
import levpairs_notify as NF
import levpairs_watchdog as WD
import test_levpairs_compare as TC
from levpairs_ui.presets import DEFAULT_PRESET, cfgs
from test_levpairs import FakeTC

NY = D.NY


def args(**kw):
    base = dict(arm=None, accept_iex=None, feed="iex", capital=0.0, equity_fraction=1.0)
    return types.SimpleNamespace(**{**base, **kw})


class TestDaily(unittest.TestCase):
    def setUp(self):
        self._rd = D.REPORTS
        self.tmp = tempfile.mkdtemp(prefix="levpairs_ops_")
        D.REPORTS = self.tmp
        LV.D.REPORTS = self.tmp

    def tearDown(self):
        D.REPORTS = self._rd
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_arming_needs_the_exact_phrases_and_is_paper_only(self):
        live = DY.build_settings(args())
        self.assertFalse(live.armed)
        self.assertTrue(live.paper)
        with self.assertRaises(SystemExit):
            DY.build_settings(args(arm="arm paper"))                                          # not the phrase
        with self.assertRaises(SystemExit):
            DY.build_settings(args(arm="ARM PAPER"))                                          # armed on IEX without the second phrase
        with self.assertRaises(SystemExit):
            DY.build_settings(args(arm="ARM PAPER", accept_iex="yes"))
        live = DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX", capital=20000.0))
        self.assertTrue(live.armed and live.paper and live.accept_iex)
        self.assertEqual(live.capital_override, 20000.0)
        self.assertTrue(DY.build_settings(args(arm="ARM PAPER", feed="sip")).armed, "on SIP signals the IEX phrase is not needed")
        self.assertFalse(hasattr(DY.build_settings(args()), "live_account"), "there is no way to ask for a real-money account")

    def test_waiting_for_the_open_uses_new_york_time(self):
        clock = {"t": pd.Timestamp("2026-10-26 06:00", tz=NY)}
        sleeps = []

        def sleep(s):
            sleeps.append(s)
            clock["t"] += pd.Timedelta(seconds=s)
        DY.wait_until_et("09:25", log=lambda *a: None, now_fn=lambda: clock["t"], sleep=sleep)
        self.assertGreaterEqual(clock["t"], pd.Timestamp("2026-10-26 09:25", tz=NY))
        self.assertLess(clock["t"], pd.Timestamp("2026-10-26 09:27", tz=NY))
        self.assertLessEqual(max(sleeps), 60.0)
        n = len(sleeps)
        DY.wait_until_et("09:25", log=lambda *a: None, now_fn=lambda: clock["t"], sleep=sleep)       # already past: returns at once
        self.assertEqual(len(sleeps), n)

    def test_status_reads_the_ledger_the_heartbeat_and_the_journal(self):
        live = DY.build_settings(args())
        self.assertIn("no engine running", DY.status(live))
        eng = LV.LiveEngine(*cfgs(DEFAULT_PRESET), live, log=lambda *a: None)
        st = eng.blank_state("2026-10-26")
        st["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30)
        eng.state = st
        eng.save_state(st)
        open(os.path.join(self.tmp, "levpairs_engine_dry.lock"), "w").write("x")
        eng.append_log("fills", [dict(date="2026-10-26", pair="TQQQ/QQQ", action="ENTER", symbol="TQQQ")])
        text = DY.status(live)
        self.assertIn("(100, -30)", text)
        self.assertIn("engine heartbeat: 0 s ago", text.replace("1 s", "0 s").replace("2 s", "0 s"))
        self.assertIn("fills: 1 rows", text)

    def test_the_evening_routine_runs_every_step_and_survives_a_failing_one(self):
        date, sc, c, tw = TC.session()
        tape, trades, pnl = TC.fabricate(self.tmp, date, sc, c, tw)
        live = replace(DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX")))
        calls = []
        real = (LV.LiveEngine.connect, LV.LiveEngine.sync_broker_day, D.update_cache)
        LV.LiveEngine.connect = lambda self: calls.append("connect") or {}
        LV.LiveEngine.sync_broker_day = lambda self, d, write=True: calls.append("sync") or {"orders": pd.DataFrame(), "fills": pd.DataFrame()}

        def boom(*a, **k):
            calls.append("update")
            raise ConnectionError("data API down")
        D.update_cache = boom
        logs = []
        try:
            out = DY.post(live, DEFAULT_PRESET, pd.Timestamp(date), lambda m, k="info": logs.append((k, m)), wait=False, tape=tape)
        finally:
            LV.LiveEngine.connect, LV.LiveEngine.sync_broker_day, D.update_cache = real
        self.assertEqual(calls, ["connect", "sync", "update"])
        self.assertTrue(any("data top-up failed" in m for _, m in logs))
        self.assertIsNotNone(out, "the comparison still ran")
        self.assertEqual(out["summary"]["matched"], len(trades))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "live_compare", f"REPORT_{date}.md")))

    def test_kill_file_and_the_manual_flatten(self):
        f = os.path.join(self.tmp, "KILL")
        buf = io.StringIO()
        with redirect_stdout(buf):
            DY.main(["kill"])
        self.assertTrue(os.path.exists(f))
        with redirect_stdout(buf):
            DY.main(["unkill"])
        self.assertFalse(os.path.exists(f))
        with self.assertRaises(SystemExit):                                                  # flatten sends orders: it needs the arming phrase
            DY.flatten_now(DY.build_settings(args()), types.SimpleNamespace(orphans=None, force=False, preset=DEFAULT_PRESET))
        live = DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX"))
        ns = types.SimpleNamespace(orphans=None, force=False, preset=DEFAULT_PRESET)
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})

        def engine():
            e = LV.LiveEngine(*cfgs(DEFAULT_PRESET), replace(live, feed="sip", flatten_backoff_s=0.0, flatten_verify_s=1.0), log=lambda *a: None)
            e.tc = tc
            e.acct = dict(blocked=False, equity=100000.0)
            return e
        st = engine().blank_state("2026-10-22")
        st["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30, N_L=7000.0)
        e0 = engine()
        e0.state = st
        e0.save_state(st)
        lock = os.path.join(self.tmp, "levpairs_engine_paper.lock")
        open(lock, "w").write("x")                                                           # a fresh heartbeat: an engine is running
        self.assertIn("heartbeat is fresh", DY.flatten_now(live, ns, engine_factory=engine))
        self.assertEqual(tc.sent, [])
        os.utime(lock, (1, 1))                                                               # a dead engine
        out = DY.flatten_now(live, ns, engine_factory=engine)
        self.assertEqual(sorted((r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent), [("QQQ", "buy", 30), ("TQQQ", "sell", 100)])
        self.assertIn("TQQQ", out)

    # ── RM review: yesterday's leftovers, the commands, the signals ──────────────────────────────────────────────────────────────────────────────────
    def test_the_15_52_orphan_flatten_is_on_for_an_armed_job_and_flatten_needs_no_iex_phrase(self):
        live = DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX"))
        self.assertTrue(live.flatten_orphans_eod)
        self.assertFalse(DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX", no_orphan_eod=True)).flatten_orphans_eod)
        self.assertFalse(DY.build_settings(args()).flatten_orphans_eod, "a dry run sends nothing")
        self.assertTrue(DY.build_settings(args(arm="ARM PAPER"), need_iex_phrase=False).armed, "flatten sends no entry: the IEX phrase is not asked for")
        with self.assertRaises(SystemExit):
            DY.build_settings(args(arm="ARM PAPER"))

    def test_post_and_status_can_read_the_paper_files_without_arming(self):
        seen = []
        real = DY.post
        DY.post = lambda live, preset, date, log, wait=True, tape=None, notify=None: seen.append((live, wait))
        try:
            DY.main(["post", "--mode", "paper", "--date", "2026-10-23", "--now"])
            DY.main(["post", "--date", "2026-10-23", "--now"])
        finally:
            DY.post = real
        self.assertEqual([(l.armed, LV.mode_name(l), w) for l, w in seen], [(True, "paper", False), (False, "dry", False)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            DY.main(["status", "--mode", "paper"])
        self.assertIn("mode PAPER", buf.getvalue())

    def test_a_leftover_from_yesterday_is_closed_before_the_session_starts(self):
        live = DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX"))
        today = pd.Timestamp("2026-10-26")
        self.assertEqual(DY.leftovers(live, today), ([], set()))
        e0 = LV.LiveEngine(*cfgs(DEFAULT_PRESET), replace(live, feed="sip"), log=lambda *a: None)
        st = e0.blank_state("2026-10-23")
        st["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30, N_L=7000.0)
        st["pending"] = dict(pair="SOXL/SMH", closing=False, legs=[["SOXL", 50, True], ["SMH", 20, False]], ts="x", unknown=True)
        e0.state = st
        e0.save_state(st)
        self.assertEqual(DY.leftovers(live, today), (["TQQQ/QQQ"], {"SOXL", "SMH"}))
        self.assertEqual(DY.leftovers(live, pd.Timestamp("2026-10-23")), ([], set()), "the same day's ledger is the running session's, not a leftover")
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30, "SOXL": 50, "SMH": -20, "SPY": 7})
        waited = []

        def engine():
            e = LV.LiveEngine(*cfgs(DEFAULT_PRESET), replace(live, feed="sip", flatten_backoff_s=0.0, flatten_verify_s=1.0), log=lambda *a: None)
            e.tc = tc
            e.acct = dict(blocked=False, equity=100000.0)
            return e
        logs = []
        ok = DY.clean_leftovers(live, DEFAULT_PRESET, today, lambda m, k="info": logs.append((k, m)), engine_factory=engine, wait=lambda hhmm, log=None: waited.append(hhmm))
        self.assertTrue(ok)
        self.assertEqual(waited, ["09:30:30"], "it waits for the open: a close order before it would queue")
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {"SPY": 7}, "the ledger pair and the in-flight legs are closed; an unrelated SPY is not touched")
        self.assertEqual(DY.leftovers(live, today), ([], set()))
        tc.market_open = False                                                                   # a leftover that cannot be closed stops the start instead of hiding
        st2 = e0.blank_state("2026-10-23")
        st2["pairs"]["SPXL/SPY"].update(side=1, shares_L=5, shares_B=0, N_L=1000.0)
        e0.state = st2
        e0.save_state(st2)
        tc.positions = {"SPXL": 5}
        self.assertFalse(DY.clean_leftovers(live, DEFAULT_PRESET, today, lambda m, k="info": logs.append((k, m)), engine_factory=engine, wait=lambda *a, **k: None))
        self.assertTrue(any(k == "err" and "could NOT be cleared" in m for k, m in logs))
        self.assertTrue(DY.clean_leftovers(DY.build_settings(args()), DEFAULT_PRESET, today, lambda *a: None), "a dry run has nothing to close")

    def test_ctrl_break_and_sigterm_end_like_ctrl_c(self):
        old = {n: signal.getsignal(getattr(signal, n)) for n in ("SIGTERM", "SIGBREAK") if hasattr(signal, n)}
        try:
            DY.flatten_on_signals()
            for n in old:
                handler = signal.getsignal(getattr(signal, n))
                with self.assertRaises(KeyboardInterrupt):
                    handler(getattr(signal, n), None)
        finally:
            for n, h in old.items():
                signal.signal(getattr(signal, n), h)

    # ── RM re-verification: N5 (pop-up dedupe), own keys, the push notification ──────────────────────────────────────────────────────────────────────
    def test_n5_the_same_alert_with_a_growing_heartbeat_age_pops_up_once(self):
        made = []
        real = WD.threading.Thread
        WD.threading.Thread = lambda target=None, daemon=None: made.append(target) or types.SimpleNamespace(start=lambda: None)
        WD._last_popup.clear()
        try:
            for age in (95, 115, 135, 155, 175, 195, 215, 235):
                WD.notify_popup(f"ALERT NO ENGINE HEARTBEAT ({age} s) and the account holds {{'TQQQ': {100 + age}.0}}")
            WD.notify_popup("ENGINE NOT RUNNING at 10:00")
        finally:
            WD.threading.Thread = real
        self.assertEqual(len(made), 2, "8 looks of the same alert = one window; a different alert = another")
        bat = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_levpairs_watchdog.bat"), encoding="utf-8").read()
        self.assertIn("--no-engine-expected", bat, "the launcher documents the switch for DRY days")

    def test_an_armed_job_and_a_closing_watchdog_need_the_accounts_own_keys(self):
        keep = {k: os.environ.pop(k, None) for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_KEY_FILE")}
        os.environ["ALPACA_KEY_FILE"] = "none"                                                   # no key file, no variables: only the ~/.claude.json fallback is left
        try:
            live = DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX"))
            for cmd in ("day", "run", "flatten"):
                with self.assertRaises(SystemExit):
                    DY.require_own_keys(live, types.SimpleNamespace(command=cmd, shared_keys=None))
            DY.require_own_keys(live, types.SimpleNamespace(command="post", shared_keys=None))           # post / status / kill never send an order
            DY.require_own_keys(DY.build_settings(args()), types.SimpleNamespace(command="day", shared_keys=None))   # a dry run sends nothing
            DY.require_own_keys(live, types.SimpleNamespace(command="day", shared_keys="SHARED ACCOUNT"))            # knowingly
            with self.assertRaises(SystemExit):
                WD.main(["--flatten", "FLATTEN PAPER"])
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"] = "k", "s"
            self.assertTrue(D.own_keys())
            DY.require_own_keys(live, types.SimpleNamespace(command="day", shared_keys=None))
            import tempfile                                                                       # the project's key file counts as own keys too
            kf = os.path.join(tempfile.mkdtemp(), "Alpaca Key.txt")
            with open(kf, "w", encoding="utf-8") as fh:
                fh.write("Key ID: PKFAKEFAKEFAKEFAKE\nSecret Key: fakefakefakefakefake\n")
            for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
                os.environ.pop(k, None)
            os.environ["ALPACA_KEY_FILE"] = kf
            self.assertTrue(D.own_keys())
            self.assertEqual(D.creds(True), ("PKFAKEFAKEFAKEFAKE", "fakefakefakefakefake"))
            DY.require_own_keys(live, types.SimpleNamespace(command="day", shared_keys=None))
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_a_push_notification_reaches_the_phone_without_ever_breaking_the_caller(self):
        keep = os.environ.pop(NF.URL_ENV, None)
        sent = []
        post = lambda url, data=None, headers=None, timeout=None: sent.append((url, data, headers))
        try:
            NF._last.clear()
            self.assertFalse(NF.send("x", post=post, wait=True), "nothing configured: nothing happens")
            os.environ[NF.URL_ENV] = "https://ntfy.example/secret-topic"
            self.assertTrue(NF.send("ENGINE DOWN: the account holds {'TQQQ': 100.0}", title="Lev-ETF watchdog", post=post, wait=True))
            self.assertEqual(sent[0][0], "https://ntfy.example/secret-topic")
            self.assertIn(b"ENGINE DOWN", sent[0][1])
            self.assertEqual(sent[0][2]["Title"], "Lev-ETF watchdog")
            self.assertFalse(NF.send("ENGINE DOWN: the account holds {'TQQQ': 250.0}", post=post, wait=True), "the same alert (digits ignored) once in 10 minutes")
            self.assertTrue(NF.send("something else", post=post, wait=True))

            def boom(*a, **k):
                raise ConnectionError("no network")
            self.assertTrue(NF.send("a third thing", post=boom, wait=True), "a failing service is swallowed")
        finally:
            os.environ.pop(NF.URL_ENV, None)
            if keep is not None:
                os.environ[NF.URL_ENV] = keep
        got = []                                                                                 # the daily job and the watchdog use it
        live = DY.build_settings(args(arm="ARM PAPER", accept_iex="ACCEPT IEX"))
        tc = FakeTC(positions={"TQQQ": 100}, )
        tc.market_open = False                                                                   # a leftover that cannot be closed: the phone is told
        e0 = LV.LiveEngine(*cfgs(DEFAULT_PRESET), replace(live, feed="sip"), log=lambda *a: None)
        st = e0.blank_state("2026-10-23")
        st["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=0, N_L=7000.0)
        e0.state = st
        e0.save_state(st)

        def engine():
            e = LV.LiveEngine(*cfgs(DEFAULT_PRESET), replace(live, feed="sip", flatten_backoff_s=0.0, flatten_verify_s=0.5), log=lambda *a: None)
            e.tc = tc
            e.acct = dict(blocked=False, equity=100000.0)
            return e
        self.assertFalse(DY.clean_leftovers(live, DEFAULT_PRESET, pd.Timestamp("2026-10-26"), lambda *a: None, engine_factory=engine, wait=lambda *a, **k: None, notify=got.append))
        self.assertTrue(got and "could NOT be cleared" in got[0])

    def test_nothing_recorded_means_no_comparison(self):
        logs = []
        out = DY.post(DY.build_settings(args()), DEFAULT_PRESET, pd.Timestamp("2026-10-26"), lambda m, k="info": logs.append(m), wait=False)
        self.assertIsNone(out)
        self.assertTrue(any("nothing was recorded" in m for m in logs))


class TestWatchdog(unittest.TestCase):
    NOW = pd.Timestamp("2026-10-23 11:00", tz=NY)                                              # a Friday at 11:00 ET: the tests must not depend on the time they run at (after 15:58:30 the real clock refuses closes)

    def setUp(self):
        self._rd, self._clock = D.REPORTS, LV.clock_now
        self.tmp = tempfile.mkdtemp(prefix="levpairs_wd_")
        D.REPORTS = self.tmp
        LV.D.REPORTS = self.tmp
        LV.clock_now = lambda: self.NOW
        self.logs = []

    def tearDown(self):
        D.REPORTS, LV.clock_now = self._rd, self._clock
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_decision_table(self):
        held = {"TQQQ": 100.0, "QQQ": -30.0}
        led = {"TQQQ": 100, "QQQ": -30}
        d = WD.decide
        self.assertEqual(d("11:00:00", 5, None, held, led, True, False)[0], "idle", "weekend")
        self.assertEqual(d("08:00:00", 1, None, held, led, True, False)[0], "idle", "before the open")
        self.assertEqual(d("11:00:00", 1, 400.0, {}, {}, True, False)[0], "ok", "flat")
        self.assertEqual(d("11:00:00", 1, 0.5, held, led, True, False)[0], "ok", "engine alive: never touched")
        self.assertEqual(d("15:57:00", 1, 0.5, held, led, True, False)[0], "alert", "alive but still holding at 15:57: alert, no action")
        self.assertEqual(d("11:00:00", 1, 200.0, held, led, False, False)[0], "alert", "dead, flatten off")
        self.assertEqual(d("11:00:00", 1, 200.0, held, led, True, False)[0], "flatten", "dead, flatten on")
        self.assertEqual(d("11:00:00", 1, None, held, led, True, False)[0], "flatten", "no heartbeat file at all")
        self.assertEqual(d("11:00:00", 1, 200.0, {"TQQQ": 10.0}, {}, True, False)[0], "alert", "dead, only unexplained positions, orphans not allowed")
        self.assertEqual(d("11:00:00", 1, 200.0, {"TQQQ": 10.0}, {}, True, True)[0], "flatten", "dead, orphans allowed")
        self.assertEqual(d("16:01:00", 1, 200.0, held, led, True, True)[0], "late", "after the close nothing can be closed")

    def engine(self, tc):
        sc, c = cfgs(DEFAULT_PRESET)
        e = LV.LiveEngine(sc, c, LV.LiveSettings(paper=True, armed=True, feed="sip", flatten_backoff_s=0.0, flatten_verify_s=1.0, flatten_attempts=3), log=lambda m, k="info": self.logs.append(m))
        e.tc = tc
        e.acct = dict(blocked=False, equity=100000.0)
        e.state = e._latest_state()
        return e

    def ledger(self):
        eng = LV.LiveEngine(*cfgs(DEFAULT_PRESET), LV.LiveSettings(paper=True, armed=True, feed="sip"), log=lambda *a: None)
        st = eng.blank_state("2026-10-22")
        st["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30, N_L=7000.0)
        eng.state = st
        eng.save_state(st)

    def test_a_dead_engine_with_open_positions_is_closed_but_a_live_one_is_left_alone(self):
        self.ledger()
        now = pd.Timestamp("2026-10-23 11:00", tz=NY)
        lock = os.path.join(self.tmp, "levpairs_engine_paper.lock")
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        open(lock, "w").write("x")                                                           # alive: a fresh heartbeat
        v, _ = WD.check(tc, "paper", now, True, False, lambda: self.engine(tc), log=self.logs.append)
        self.assertEqual((v, tc.sent), ("ok", []))
        os.utime(lock, (1, 1))                                                               # dead: the heartbeat is ancient
        v, _ = WD.check(tc, "paper", now, False, False, lambda: self.engine(tc), log=self.logs.append)
        self.assertEqual((v, tc.sent), ("alert", []), "flatten off: an alert and nothing else")
        alerts = open(os.path.join(self.tmp, f"ALERT_{pd.Timestamp.now(tz=NY):%Y%m%d}.txt"), encoding="utf-8").read()
        self.assertIn("NO ENGINE HEARTBEAT", alerts)
        v, _ = WD.check(tc, "paper", now, True, False, lambda: self.engine(tc), log=self.logs.append)
        self.assertEqual(v, "flatten")
        self.assertEqual(sorted((r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent), [("QQQ", "buy", 30), ("TQQQ", "sell", 100)])
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})
        self.assertTrue(any("watchdog closed" in m for m in self.logs))

    def test_unexplained_positions_are_only_closed_with_the_second_phrase(self):
        now = pd.Timestamp("2026-10-23 11:00", tz=NY)
        tc = FakeTC(positions={"SPXL": 40, "AAPL": 9})                                       # not the ledger's, and an AAPL that is nobody's business
        v, _ = WD.check(tc, "paper", now, True, False, lambda: self.engine(tc), log=self.logs.append)
        self.assertEqual((v, tc.sent), ("alert", []))
        v, _ = WD.check(tc, "paper", now, True, True, lambda: self.engine(tc), log=self.logs.append)
        self.assertEqual(v, "flatten")
        self.assertEqual([(r.symbol, str(r.side.value)) for r in tc.sent], [("SPXL", "sell")])
        self.assertEqual(tc.positions.get("AAPL"), 9)


    # ── RM review: confirmation, an entry in flight, the marker, alerts ─────────────────────────────────────────────────────────────────────────────
    def test_the_decision_table_with_pending_entries_holidays_and_an_engine_that_never_started(self):
        d = WD.decide
        held = {"TQQQ": 357.0, "QQQ": -107.0}
        self.assertEqual(d("10:30:00", 1, None, {}, {}, False, False, engine_expected=True)[0], "alert", "nothing is trading at 10:30")
        self.assertIn("ENGINE NOT RUNNING", d("10:30:00", 1, None, {}, {}, False, False, engine_expected=True)[1])
        self.assertEqual(d("10:30:00", 1, 0.5, {}, {}, False, False, engine_expected=True)[0], "ok", "a running engine with no position is fine")
        self.assertEqual(d("09:35:00", 1, None, {}, {}, False, False, engine_expected=True)[0], "ok", "before 09:40 the engine may still be starting")
        self.assertEqual(d("15:55:00", 1, None, {}, {}, False, False, engine_expected=True)[0], "ok", "after 15:50 an engine that has ended is normal")
        self.assertEqual(d("10:30:00", 1, None, {}, {}, False, False, engine_expected=False)[0], "ok")
        self.assertEqual(d("10:30:00", 1, None, {}, {}, False, False, market_open=False, engine_expected=True)[0], "idle", "a holiday: the broker's clock says closed")
        self.assertEqual(d("11:00:00", 1, 400.0, held, {}, True, False, pending_syms={"TQQQ", "QQQ"})[0], "flatten", "an entry that was in flight when the engine died: --flatten alone closes its legs")
        self.assertEqual(d("11:00:00", 1, 400.0, {"SPXL": 5.0}, {}, True, False, pending_syms={"TQQQ", "QQQ"})[0], "alert", "a position that is not one of the in-flight legs needs --close-orphans")

    def test_the_watchdog_waits_for_a_second_look_and_forgets_after_a_sleep(self):
        t = {"w": 1000.0}
        c = WD.Confirm(after_s=20.0, max_gap_s=60.0, wall=lambda: t["w"])
        self.assertFalse(c.look(True))
        t["w"] += 10
        self.assertFalse(c.look(True), "10 s after the first look is not enough")
        t["w"] += 15
        self.assertTrue(c.look(True), "dead on two looks 25 s apart")
        t["w"] += 20
        self.assertFalse(c.look(False), "alive again: forgotten")
        self.assertFalse(c.look(True))
        t["w"] += 400                                                                          # the PC slept for 400 s: the engine was frozen the same way, a stale heartbeat proves nothing
        self.assertFalse(c.look(True))
        t["w"] += 25
        self.assertTrue(c.look(True))
        t["w"] -= 1000                                                                         # the wall clock stepped back
        self.assertFalse(c.look(True))

    def test_check_confirms_before_it_closes_and_tells_a_revived_engine(self):
        lock = os.path.join(self.tmp, "levpairs_engine_paper.lock")
        open(lock, "w").write("x")
        os.utime(lock, (1, 1))                                                                 # no heartbeat for ages
        eng0 = LV.LiveEngine(*cfgs(DEFAULT_PRESET), LV.LiveSettings(paper=True, armed=True, feed="sip"), log=lambda *a: None)
        st = eng0.blank_state("2026-10-22")
        st["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30, N_L=7000.0)
        eng0.state = st
        eng0.save_state(st)
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        t = {"w": 5000.0}
        confirm = WD.Confirm(after_s=20.0, max_gap_s=60.0, wall=lambda: t["w"])
        told = []
        v, why = WD.check(tc, "paper", self.NOW, True, False, lambda: self.engine(tc), log=self.logs.append, confirm=confirm, notify=told.append)
        self.assertEqual((v, tc.sent), ("watch", []), "the first look only notes it")
        t["w"] += 25
        v, why = WD.check(tc, "paper", self.NOW, True, False, lambda: self.engine(tc), log=self.logs.append, confirm=confirm, notify=told.append)
        self.assertEqual(v, "flatten")
        self.assertEqual(sorted((r.symbol, str(r.side.value)) for r in tc.sent), [("QQQ", "buy"), ("TQQQ", "sell")])
        marker = json.load(open(os.path.join(self.tmp, "levpairs_watchdog_paper.json")))
        self.assertEqual(marker["date"], "2026-10-23")
        self.assertTrue(any("ENGINE DOWN" in m for m in told), "a pop-up goes out")

    def test_an_entry_that_was_in_flight_is_closed_by_flatten_alone_and_nothing_else_is_touched(self):
        eng0 = LV.LiveEngine(*cfgs(DEFAULT_PRESET), LV.LiveSettings(paper=True, armed=True, feed="sip"), log=lambda *a: None)
        st = eng0.blank_state("2026-10-22")
        st["pending"] = dict(pair="TQQQ/QQQ", closing=False, legs=[["TQQQ", 357, True], ["QQQ", 107, False]], ts="x")
        eng0.state = st
        eng0.save_state(st)
        tc = FakeTC(positions={"TQQQ": 357, "QQQ": -107, "SPXL": 5})                         # the legs of the crashed entry and an unrelated SPXL
        v, _ = WD.check(tc, "paper", self.NOW, True, False, lambda: self.engine(tc), log=self.logs.append)
        self.assertEqual(v, "flatten")
        self.assertEqual({k: v_ for k, v_ in tc.positions.items() if v_}, {"SPXL": 5}, "the in-flight legs are closed, the SPXL is not")

    def test_the_engine_never_started_alert_and_the_popup_is_not_repeated(self):
        told = []
        tc = FakeTC(positions={})
        v, why = WD.check(tc, "paper", pd.Timestamp("2026-10-23 10:00", tz=NY), False, False, lambda: self.engine(tc), log=self.logs.append, notify=told.append, engine_expected=True)
        self.assertEqual(v, "alert")
        self.assertIn("ENGINE NOT RUNNING", told[0])
        made = []
        real = WD.threading.Thread
        WD.threading.Thread = lambda target=None, daemon=None: made.append(target) or types.SimpleNamespace(start=lambda: None)
        WD._last_popup.clear()
        try:
            WD.notify_popup("ENGINE NOT RUNNING at 10:00")
            WD.notify_popup("ENGINE NOT RUNNING at 10:00")
            WD.notify_popup("something else")
        finally:
            WD.threading.Thread = real
        self.assertEqual(len(made), 2, "the same message once in 10 minutes")

    def test_the_phrases(self):
        with self.assertRaises(SystemExit):
            WD.main(["--flatten", "yes"])
        with self.assertRaises(SystemExit):
            WD.main(["--close-orphans", "yes"])


class PriceTC(FakeTC):
    """A fake broker that fills every market order at a price the test sets (here: the open of the next bar)."""

    def __init__(self):
        super().__init__()
        self.px = {}

    def submit_order(self, req):
        o = super().submit_order(req)
        o.filled_avg_price = self.px.get(req.symbol, 100.0)
        return o

    def get_account(self):                                                                       # a big account: whole-share rounding and the exposure caps stay out of the way
        a = super().get_account()
        a.equity = a.cash = "100000000"
        return a


class TestRehearsal(unittest.TestCase):
    """The whole chain without the network: the ARMED engine runs a real past session bar by bar against a fake broker that fills at the next bar's open (what the backtest assumes), writes its journal, and
    levpairs_compare must then find the live session identical to the SIP twin - same trips, same positions, no slippage."""

    def setUp(self):
        self._rd = D.REPORTS
        self.tmp = tempfile.mkdtemp(prefix="levpairs_reh_")
        D.REPORTS = self.tmp
        LV.D.REPORTS = self.tmp

    def tearDown(self):
        D.REPORTS = self._rd
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_engine_to_journal_to_comparison(self):
        date, sc, c, tw = TC.session()
        c = replace(c, vol_on=False, g_init=1.0)
        d = pd.Timestamp(date)
        tc = PriceTC()
        eng = LV.LiveEngine(sc, c, LV.LiveSettings(armed=True, paper=True, feed="sip", fill_wait_s=2.0, flatten_backoff_s=0.0, flatten_verify_s=1.0, max_leg_nav=1.0, max_gross_nav=1.0, max_gross_x=10.0),
                            log=lambda m, k="info": None, source=LV.ParquetSource("sip"))
        eng.tc = tc
        eng.acct = {"equity": 1e8, "blocked": False}
        eng.start_session(d, 1e8)
        eng.acct = {"equity": 1e8, "blocked": False}
        bars = {s_: pd.read_parquet(D.path1(s_, "sip")) for s_ in D.SYMBOLS}
        nb = len(D.session_grid(sc.tf))
        for j in range(nb):
            t_next = d + pd.Timedelta(minutes=D.OPEN_MIN + (j + 1) * sc.tf)
            tc.px = {s_: float(b.open.get(t_next, b.close.loc[:t_next].iloc[-1])) for s_, b in bars.items()}          # an order sent at the bar's end fills at the next bar's open
            eng.step(d, t_next + pd.Timedelta(seconds=4))
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {}, "flat at the end of the session")
        out = K.run_day(date, "paper", None, feeds=("sip",), reports=self.tmp, log=lambda m: None)
        s, n = out["summary"], len(tw["trades"])
        self.assertEqual((s["live_trips"], s["live_complete"], s["matched"], s["live_only"], s["twin_only"]), (n, n, n, 0, 0))
        self.assertEqual(s["pos_agree_sip"], 1.0)
        self.assertAlmostEqual(s["s_corr_sip"], 1.0, places=6)
        self.assertAlmostEqual(s["slip_entry_bp"], 0.0, delta=1e-6)
        self.assertAlmostEqual(s["slip_exit_bp"], 0.0, delta=1e-6)
        self.assertAlmostEqual(s["d_price_bp"], 0.0, delta=1e-6)
        j = K.read_jsonl(os.path.join(self.tmp, "levpairs_fills_paper.jsonl"))
        self.assertEqual(len(j), 4 * n, "two legs in, two legs out, for every trip")
        self.assertTrue(j.trip.notna().all() and j.order_id.notna().all() and j.t_submit_ns.notna().all())
        self.assertTrue(os.path.exists(os.path.join(self.tmp, f"levpairs_run_paper_{d:%Y%m%d}.json")))
        self.assertEqual(sorted(out["trips"].position.dropna().str.slice(0, 4).unique()), ["LONG"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
