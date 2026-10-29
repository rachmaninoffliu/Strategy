"""
Tests for the live-vs-backtest comparison:  python test_levpairs_compare.py

A whole live session is FABRICATED from the SIP backtest of a real past day: every trade of the twin becomes a pair of entry orders and a pair of exit orders filled a known number of basis points worse than the
twin's price, with a fake tape around them. The comparison must then report exactly those numbers - the matching, the fixed-share P&L, the slippage against every reference, the latencies and the decision
agreement - and must cope with a KILL exit, a failed entry, the broker's record overriding the journal, and a re-run of the same day.   No network, no orders.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levpairs_compare as K
import levpairs_data as D
from levpairs_ui.presets import DEFAULT_PRESET, cfgs

NY = D.NY
DELTA = 0.5                                                                                         # bp: every fill is this much worse than the twin's price
_cache: dict = {}


def session():
    """A recent session whose SIP twin has trades, with its config and twin."""
    if "s" not in _cache:
        sc, c = cfgs(DEFAULT_PRESET)
        dates = D.load_pairdata(("TQQQ", "QQQ"), sc.tf, "sip").dates[-14:][::-1]
        for d in dates:
            tw = K.twin(d, sc, c, "sip")
            if tw is not None and len(tw["trades"]) >= 3:
                _cache["s"] = (str(d.date()), sc, c, tw)
                break
        else:
            raise unittest.SkipTest("no recent session with at least 3 trades in the cached bars")
    return _cache["s"]


class FakeTape(K.Tape):
    def __init__(self):
        self.pts = []                                                                               # (symbol, time, mid)

    def add(self, sym, t, mid):
        self.pts.append((sym, t, mid))

    def _find(self, sym, t):
        for s, t0, mid in self.pts:
            if s == sym and abs((t - t0).total_seconds()) < 0.5:
                return mid
        return None

    def quote_at(self, sym, t):
        mid = self._find(sym, t)
        return None if mid is None else dict(t=t, bid=mid * (1 - 0.3e-4), ask=mid * (1 + 0.3e-4))

    def trade_after(self, sym, t):
        mid = self._find(sym, t)
        return None if mid is None else dict(t=t, price=mid * (1 + 0.1e-4))


def fabricate(reports, date, sc, c, tw, delta=DELTA, kill_last=False, close_action="KILL", fail_first=False, broker_delta=None, split_first_exit=False, shift_exit_first=False):
    """Write the journal, decisions, run record and (optionally) the broker's order list of a live session that traded exactly like the SIP twin, `delta` bp worse. Options: the last trip ends in a flatten
    (``close_action``: KILL, watchdog ...), the first trip's exit is split over two orders at different prices, the first trip's exit comes one bar later than the twin's (and 10 bp worse), a failed entry.
    Returns (tape, trades, expected P&L in dollars)."""
    tape, rows, brows, pnl_usd = FakeTape(), [], [], 0.0
    trades = tw["trades"]
    for n, t in enumerate(trades.itertuples()):
        trip = f"{date}-{t.pair.replace('/', '')}-{n + 1}"
        L, B = t.pair.split("/")
        side = 1 if t.side == "long" else -1
        qL = 100
        qB = int(round(t.beta * qL * t.L_entry / t.B_entry))
        fills = {}
        last = kill_last and n == len(trades) - 1
        for act, tm, (pxL, pxB) in (("ENTER", t.entry_time, (t.L_entry, t.B_entry)), ("EXIT", t.exit_time, (t.L_exit, t.B_exit))):
            buyL = (side > 0) if act == "ENTER" else (side < 0)
            shifted = shift_exit_first and n == 0 and act == "EXIT"
            bar_end = pd.Timestamp(tm).tz_localize(NY) + (pd.Timedelta(minutes=sc.tf) if shifted else pd.Timedelta(0))
            for sym, q, px, buy in ((L, qL, pxL, buyL), (B, qB, pxB, not buyL)):
                d_ = (broker_delta if (broker_delta is not None) else delta) + (10.0 if shifted else 0.0)
                jd_ = delta + (10.0 if shifted else 0.0)
                parts = [(q, 0.0)]
                if split_first_exit and n == 0 and act == "EXIT":
                    q1 = int(round(0.6 * q))
                    parts = [(q1, 1.0), (q - q1, -1.0)]                                         # 60% at 1 bp worse, 40% at 1 bp better than the nominal fill
                for k, (pq, extra) in enumerate(parts):
                    fill = px * (1 + (1 if buy else -1) * (d_ + extra) * 1e-4)
                    jfill = px * (1 + (1 if buy else -1) * (jd_ + extra) * 1e-4)
                    sub = bar_end + pd.Timedelta(seconds=4.4 + k)
                    cid = f"LP-{date.replace('-', '')}-{t.pair.replace('/', '')}-{act}-{sym}-{n}-{k}"
                    action = close_action if (last and act == "EXIT") else act
                    flat = action not in ("ENTER", "EXIT")
                    rows.append(dict(date=date, time=f"{bar_end:%H:%M}" if not flat else f"{(bar_end + pd.Timedelta(seconds=5)):%H:%M:%S}", pair=t.pair, action=action, trip=None if flat else trip,
                                     j=0, s=t.s_entry, decided_at=(bar_end + pd.Timedelta(seconds=4.1)).isoformat(), symbol=sym, side="BUY" if buy else "SELL", qty=pq, px=None if flat else jfill,
                                     status="sent" if flat else "filled", ref=px, order_id=f"id-{cid}", client_order_id=cid, submitted_at=(sub + pd.Timedelta(milliseconds=50)).isoformat(),
                                     filled_at=(sub + pd.Timedelta(milliseconds=300)).isoformat(), t_submit_ns=int(sub.value), t_ack_ns=int(sub.value) + 30_000_000, t_seen_ns=int(sub.value) + 400_000_000))
                    brows.append(dict(order_id=f"id-{cid}", client_order_id=cid, symbol=sym, side="buy" if buy else "sell", qty=pq, filled_qty=pq, filled_avg_price=fill, status="filled",
                                      submitted_at=(sub + pd.Timedelta(milliseconds=50)).isoformat(), filled_at=(sub + pd.Timedelta(milliseconds=300)).isoformat()))
                    tape.add(sym, sub + pd.Timedelta(milliseconds=50), px)
                    fills.setdefault((act, sym), []).append((fill, pq))
        Le, Be = fills[("ENTER", L)][0][0], fills[("ENTER", B)][0][0]
        pnl_usd += side * (sum(q_ * (p_ - Le) for p_, q_ in fills[("EXIT", L)]) - sum(q_ * (p_ - Be) for p_, q_ in fills[("EXIT", B)]))
    if fail_first:                                                                                    # a trip whose second leg was rejected and whose first leg was unwound
        t = trades.iloc[0]
        L, B = t.pair.split("/")
        rows.append(dict(date=date, time="09:40", pair=t.pair, action="ENTER", trip=f"{date}-FAIL-1", j=1, s=-2.5, symbol=L, side="BUY", qty=10, px=100.0, status="filled", client_order_id="LP-FAIL-1"))
        rows.append(dict(date=date, time="09:40", pair=t.pair, action="ENTER", trip=f"{date}-FAIL-1", j=1, s=-2.5, symbol=B, side="SELL", qty=3, status="rejected", client_order_id="LP-FAIL-2", note="not shortable"))
        rows.append(dict(date=date, time="09:40", pair=t.pair, action="UNWIND", trip=f"{date}-FAIL-1", j=1, symbol=L, side="SELL", qty=10, status="sent", client_order_id="LP-FAIL-3"))
    key = date.replace("-", "")
    with open(os.path.join(reports, "levpairs_fills_paper.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if v is not None}) + "\n")
    sg = tw["sigs"]
    dec = []
    for name, sig in sg.items():
        for j in range(sig.nb):
            dec.append(dict(date=date, j=j, time=f"{pd.Timestamp(date) + pd.Timedelta(minutes=D.OPEN_MIN + (j + 1) * sc.tf):%H:%M}", wall=None, pair=name, mode="paper", s=float(sig.s[-1, j]),
                            pos_after=int(tw["res"].per_pair[name].dec[-1, j])))
    with open(os.path.join(reports, "levpairs_decisions_paper.jsonl"), "w", encoding="utf-8") as fh:
        for r in dec:
            fh.write(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}) + "\n")
    json.dump(dict(date=date, mode="paper", sc=asdict(sc), c=asdict(c), live={"feed": "iex"}, starts=[dict(feed="iex", equity=100000.0)]), open(os.path.join(reports, f"levpairs_run_paper_{key}.json"), "w"), default=str)
    if broker_delta is not None:
        pd.DataFrame(brows).to_csv(os.path.join(reports, f"levpairs_broker_orders_paper_{key}.csv"), index=False)
    return tape, trades, pnl_usd


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="levpairs_cmp_")
        self.date, self.sc, self.c, self.tw = session()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_session_that_traded_like_the_twin_is_reported_exactly(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        s = out["summary"]
        n = len(trades)
        self.assertEqual((s["live_trips"], s["live_complete"], s["matched"], s["live_only"], s["twin_only"]), (n, n, n, 0, 0))
        self.assertAlmostEqual(s["live_pnl_usd"], pnl, places=4, msg="fixed-share P&L from the fill prices")
        self.assertAlmostEqual(s["d_price_bp"], -8 * DELTA, delta=0.06, msg="every leg 0.5 bp worse at both ends: 2 x 0.5 on the 3x leg plus 3 x 2 x 0.5 on the hedge")
        self.assertAlmostEqual(s["slip_entry_bp"], DELTA, delta=1e-6)
        self.assertAlmostEqual(s["slip_exit_bp"], DELTA, delta=1e-6)
        self.assertAlmostEqual(s["slip_vs_signal_bp"], DELTA, delta=1e-6)
        self.assertAlmostEqual(s["slip_vs_mid_bp"], DELTA, delta=1e-6)
        self.assertAlmostEqual(s["slip_vs_touch_bp"], DELTA - 0.3, delta=1e-6)
        self.assertAlmostEqual(s["slip_vs_next_trade_bp"], DELTA, delta=0.11)
        self.assertAlmostEqual(s["spread_bp"], 0.6, delta=1e-6)
        self.assertAlmostEqual(s["lat_bar_to_decision_s"], 4.1, delta=1e-3)
        self.assertAlmostEqual(s["lat_decision_to_submit_s"], 0.3, delta=1e-3)
        self.assertAlmostEqual(s["lat_submit_to_fill_s"], 0.25, delta=1e-3)
        self.assertAlmostEqual(s["lat_local_to_broker_ms"], 50.0, delta=1e-3)
        self.assertEqual(s["pos_agree_sip"], 1.0)
        self.assertAlmostEqual(s["s_corr_sip"], 1.0, places=9)
        t = out["trips"]
        self.assertTrue(t.position.str.match(r"^LONG \w+ / SHORT \w+$").all(), "which product was bought and which sold")

    def test_the_brokers_own_fill_prices_override_the_journal(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, broker_delta=1.0)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        s = out["summary"]
        self.assertAlmostEqual(s["slip_entry_bp"], 1.0, delta=1e-6)
        self.assertAlmostEqual(s["d_price_bp"], -8.0, delta=0.1)
        self.assertAlmostEqual(s["live_pnl_usd"], pnl, places=4)

    def test_a_kill_exit_a_failed_entry_and_a_rerun(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, kill_last=True, fail_first=True, broker_delta=DELTA)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        t = out["trips"]
        self.assertIn("KILL", set(t.kind.dropna()), "the trip that ended in a KILL takes its exit from the KILL order (priced from the broker's record)")
        self.assertIn("entry failed", set(t.kind.dropna()))
        self.assertEqual(out["summary"]["live_only"], 1, "the failed entry has no twin")
        self.assertEqual(out["summary"]["live_complete"], len(trades))
        self.assertAlmostEqual(out["summary"]["live_pnl_usd"], pnl, places=4)
        K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        sm = pd.read_csv(os.path.join(self.tmp, "live_compare", "summary.csv"))
        self.assertEqual(len(sm), 1, "running a day again replaces its row")
        for f in (f"REPORT_{self.date}.md", "MONTH_TO_DATE.md", f"{self.date}_trips.csv", f"{self.date}_legs.csv", f"{self.date}_decisions.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, "live_compare", f)), f)
        self.assertIn("month to date", open(os.path.join(self.tmp, "live_compare", "MONTH_TO_DATE.md"), encoding="utf-8").read())

    def test_c1_a_trip_closed_by_the_watchdog_is_complete(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, kill_last=True, close_action="watchdog", broker_delta=DELTA)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        self.assertIn("watchdog", set(out["trips"].kind.dropna()))
        self.assertEqual(out["summary"]["live_complete"], len(trades))
        self.assertAlmostEqual(out["summary"]["live_pnl_usd"], pnl, places=4)

    def test_c2_an_exit_split_over_two_orders_is_priced_with_both(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, split_first_exit=True, broker_delta=DELTA)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        self.assertAlmostEqual(out["summary"]["live_pnl_usd"], pnl, places=4, msg="the quantity-weighted price of both exit orders, not the last order alone")
        t = out["trips"].set_index("trip")
        first = t.loc[f"{self.date}-{trades.iloc[0].pair.replace('/', '')}-1"]
        self.assertFalse(first.qty_mismatch)
        self.assertEqual(first.qty_L_exit, first.qty_L)

    def test_c3_slippage_is_only_judged_on_trips_whose_entry_and_exit_bars_agree(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, shift_exit_first=True)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        s, n = out["summary"], len(trades)
        self.assertEqual((s["matched"], s["comparable"]), (n, n - 1), "the trip that left one bar late is matched but not comparable")
        self.assertAlmostEqual(s["slip_exit_bp"], DELTA, delta=1e-6, msg="the late trip's 10.5 bp is timing, not fill quality, and stays out of the slippage")
        self.assertAlmostEqual(s["d_price_bp"], -8 * DELTA, delta=0.06)
        self.assertNotAlmostEqual(s["d_price_bp_all"], s["d_price_bp"], places=1, msg="all matched trips, timing included, are reported separately")
        self.assertFalse(bool(out["trips"].set_index("trip").loc[f"{self.date}-{trades.iloc[0].pair.replace('/', '')}-1", "comparable"]))

    def test_c4_a_trip_id_that_appears_twice_does_not_break_the_day(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw)
        f = os.path.join(self.tmp, "levpairs_fills_paper.jsonl")
        rows = [json.loads(ln) for ln in open(f, encoding="utf-8")]
        dup = [dict(r, client_order_id=r["client_order_id"] + "-again", order_id=r["order_id"] + "-again") for r in rows if r["action"] == "ENTER" and r["trip"].endswith("-1")]
        with open(f, "a", encoding="utf-8") as fh:
            for r in dup:
                fh.write(json.dumps(r) + "\n")
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)                  # an older journal reused a trip id after a failed entry: the entries are merged, nothing raises
        self.assertEqual(out["summary"]["live_trips"], len(trades))

    def test_c5_dry_run_days_do_not_mix_into_the_paper_month(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw)
        K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        shutil.copy(os.path.join(self.tmp, "levpairs_decisions_paper.jsonl"), os.path.join(self.tmp, "levpairs_decisions_dry.jsonl"))
        K.run_day(self.date, "dry", None, reports=self.tmp, log=lambda m: None)
        sm = pd.read_csv(os.path.join(self.tmp, "live_compare", "summary.csv"))
        self.assertEqual(sorted(sm["mode"]), ["dry", "paper"], "one row per (date, mode)")
        month = open(os.path.join(self.tmp, "live_compare", "MONTH_TO_DATE.md"), encoding="utf-8").read()
        self.assertIn("Live paper test", month)
        self.assertIn(f"live {len(trades)}", month, "the paper month counts only the paper trips")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "live_compare", "MONTH_TO_DATE_dry.md")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "live_compare", f"REPORT_{self.date}_dry.md")))
        K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        self.assertEqual(len(pd.read_csv(os.path.join(self.tmp, "live_compare", "summary.csv"))), 2, "re-running a day replaces ITS row only")

    def rewrite_journal(self, fn):
        f = os.path.join(self.tmp, "levpairs_fills_paper.jsonl")
        rows = [json.loads(ln) for ln in open(f, encoding="utf-8")]
        rows = fn(rows)
        with open(f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def test_n6_a_flatten_close_belongs_to_one_trip_the_one_it_follows(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, broker_delta=DELTA)
        t0, t1 = f"{self.date}-TQQQQQQ-1", f"{self.date}-TQQQQQQ-2"

        def fn(rows):
            out = []
            for r in rows:
                if r.get("trip") == t0 and r["action"] == "EXIT":
                    continue                                                                     # the first trip of the pair has no exit of its own ...
                if r.get("trip") == t1 and r["action"] == "EXIT":
                    r = dict(r, action="KILL", trip=None, status="sent", time=r["time"] + ":05")  # ... and the second ends in a KILL, sent long after BOTH entries
                    r.pop("px", None)
                out.append(r)
            return out
        self.rewrite_journal(fn)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        t = out["trips"].set_index("trip")
        self.assertFalse(bool(t.loc[t0, "complete"]), "the earlier trip must not claim the later trip's close orders")
        self.assertTrue(bool(t.loc[t1, "complete"]))
        self.assertEqual(t.loc[t1, "kind"], "KILL")
        self.assertEqual(out["summary"]["live_complete"], len(trades) - 1)

    def test_n6_an_exit_spread_over_several_bars_is_not_comparable_and_legs_follow_the_trips(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw, split_first_exit=True, broker_delta=DELTA)

        def fn(rows):
            for r in rows:
                if r["action"] == "EXIT" and r["trip"].endswith("-1") and r["client_order_id"].endswith("-1"):     # the second part of the first trip's exit goes out one bar later
                    r["time"] = f"{pd.Timestamp(self.date + ' ' + r['time']) + pd.Timedelta(minutes=self.sc.tf):%H:%M}"
            return rows
        self.rewrite_journal(fn)
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        t = out["trips"].set_index("trip")
        first = t.loc[f"{self.date}-{trades.iloc[0].pair.replace('/', '')}-1"]
        self.assertEqual(first.exit_bars, 2)
        self.assertFalse(bool(first.comparable))
        self.assertEqual(out["summary"]["comparable"], len(trades) - 1)
        legs = out["legs"]
        bad = legs[legs.trip == first.name]
        good = legs[legs.trip.notna() & (legs.trip != first.name)]
        self.assertTrue(bad.slip_vs_twin_bp.isna().all(), "the legs of a trip that is not comparable carry no slippage against the twin")
        self.assertTrue(good.slip_vs_twin_bp.notna().all())

    def test_a_day_without_orders_still_compares_the_decisions(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw)
        os.remove(os.path.join(self.tmp, "levpairs_fills_paper.jsonl"))
        out = K.run_day(self.date, "paper", None, reports=self.tmp, log=lambda m: None)
        self.assertEqual(out["summary"]["live_trips"], 0)
        self.assertEqual(out["summary"]["pos_agree_sip"], 1.0)

    def test_a_torn_last_line_after_a_crash_is_skipped(self):
        tape, trades, pnl = fabricate(self.tmp, self.date, self.sc, self.c, self.tw)
        with open(os.path.join(self.tmp, "levpairs_fills_paper.jsonl"), "a", encoding="utf-8") as fh:
            fh.write('{"date": "' + self.date + '", "pair": "TQQ')
        out = K.run_day(self.date, "paper", tape, reports=self.tmp, log=lambda m: None)
        self.assertEqual(out["summary"]["live_trips"], len(trades))


if __name__ == "__main__":
    unittest.main(verbosity=2)
