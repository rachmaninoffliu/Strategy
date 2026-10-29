"""
Tests for the leveraged-ETF pairs project:  python test_levpairs.py

  signals   rolling OU fit == a brute-force window loop; no look-ahead; the hedge ratio only uses earlier sessions; gap + body == close-to-close
  rules     the bar-by-bar state machine on hand-made cases (entry, target, stop + cooldown, time stop, end of day, skip-open, filters)
  accounting a hand-computed trade under both fill models, costs on both legs, nothing held overnight, volatility target uses only the past
  live      live signals == backtest signals; a replayed session == the backtest positions; DRY RUN never touches the broker; the two-leg order flow, unwind, KILL, restart
  GUI       the window builds offscreen, starts on PAPER + DRY RUN, switching account disarms, settings -> configs
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levpairs_bridge as BR
import levpairs_core as C
import levpairs_data as D
import levpairs_live as LV
import levpairs_sim as S
from levpairs_ui.presets import DEFAULT_PRESET, PRESETS, cfgs

_cache: dict = {}


def pdata(pair=("TQQQ", "QQQ"), tf=5, start="2024-06-01"):
    k = (pair, tf, start)
    if k not in _cache:
        _cache[k] = D.load_pairdata(pair, tf, start=start)
    return _cache[k]


def fake_sig(s, Lc=None, Bc=None, Lo=None, Bo=None, tf=5, beta=3.0) -> C.PairSignals:
    """Signals with PRICES: close (and open) of both legs, default a flat 100 / 50 with every open equal to the previous close. eps, gap and body follow from the prices as in build_signals."""
    s = np.asarray(s, float)
    n, nb = s.shape
    px = lambda x, d: np.full((n, nb), d, float) if x is None else np.array(np.broadcast_to(np.asarray(x, float), (n, nb)))
    Lc, Bc = px(Lc, 100.0), px(Bc, 50.0)

    def opens(o, c):
        if o is not None:
            return px(o, 0.0)
        o = c.copy()
        o[:, 1:] = c[:, :-1]
        return o
    Lo, Bo = opens(Lo, Lc), opens(Bo, Bc)

    def rets(c, o):
        gap, full = np.zeros_like(c), np.empty_like(c)
        gap[:, 1:] = o[:, 1:] / c[:, :-1] - 1.0
        body = c / o - 1.0
        full[:, 1:] = c[:, 1:] / c[:, :-1] - 1.0
        full[:, 0] = body[:, 0]
        return gap, body, full
    gL, bL, fL = rets(Lc, Lo)
    gB, bB, fB = rets(Bc, Bo)
    z = np.zeros((n, nb))
    return C.PairSignals(("TQQQ", "QQQ"), C.SigCfg(tf=tf), pd.DatetimeIndex(pd.date_range("2025-01-02", periods=n)), nb, s, np.full((n, nb), 10.0), np.full((n, nb), 5.0), fL - beta * fB, gL - beta * gB, bL - beta * bB,
                         np.full(n, beta), z, z, Lc, Bc, Lo, Bo)


def one(x, dt=np.int8):
    return np.array([x], dtype=dt)


class TestSignals(unittest.TestCase):
    def test_rolling_ou_equals_bruteforce(self):
        rng = np.random.default_rng(1)
        nb, n = 10, 80
        X = np.zeros(nb * n)
        for i in range(1, len(X)):
            X[i] = 0.002 + 0.85 * X[i - 1] + rng.normal() * 0.3
        win = 3 * nb
        s, hl, sig = C.ou_scores(X, nb, win, 10)
        checked = 0
        for t in (45, 100, 251, 399, 640, 799):
            pos = [p for p in range(t - win + 1, t + 1) if p >= 1 and p % nb != 0]
            x, y = X[[p - 1 for p in pos]], X[pos]
            vx, cv = x.var(), np.mean(x * y) - x.mean() * y.mean()
            b = cv / vx
            a = y.mean() - b * x.mean()
            vz = np.mean((y - a - b * x) ** 2)
            if 0 < b < 1:
                want = (X[t] - a / (1 - b)) / np.sqrt(vz / (1 - b * b))
                self.assertAlmostEqual(s[t], want, places=7)
                self.assertAlmostEqual(hl[t], np.log(2) / -np.log(b), places=7)
                checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_ou_recovers_known_parameters(self):
        rng = np.random.default_rng(2)
        X = np.zeros(20_000)
        for i in range(1, len(X)):
            X[i] = 0.9 * X[i - 1] + rng.normal() * 0.1
        s, hl, sig = C.ou_scores(X, 10_000_000, 5000, 1000)                             # one 'session': no boundaries inside the window
        self.assertAlmostEqual(float(np.nanmedian(hl[6000:])), np.log(2) / -np.log(0.9), delta=1.0)
        self.assertAlmostEqual(float(np.nanmedian(sig[6000:])), 0.1 / np.sqrt(1 - 0.81), delta=0.01)

    def test_no_lookahead_and_beta_uses_earlier_sessions(self):
        d = pdata()
        n = len(d.dates)
        base = C.build_signals(d, C.SigCfg(tf=5, spread="reset", beta_mode="rolling", win_days=5.0))
        k, t = n - 40, 30
        rng = np.random.default_rng(3)
        mut = {f: getattr(d, f).copy() for f in ("L_open", "L_close", "B_open", "B_close")}
        for f in mut:
            a = mut[f]
            a[k, t + 1:] *= 1 + rng.normal(size=a.shape[1] - t - 1) * 0.01                # the rest of session k ...
            a[k + 1:] *= 1 + rng.normal(size=a[k + 1:].shape) * 0.01                       # ... and every later session
        d2 = replace(d, **mut)
        alt = C.build_signals(d2, C.SigCfg(tf=5, spread="reset", beta_mode="rolling", win_days=5.0))
        np.testing.assert_allclose(alt.s[:k], base.s[:k], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(alt.s[k, :t + 1], base.s[k, :t + 1], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(alt.beta[:k + 1], base.beta[:k + 1], atol=1e-12)         # session k's beta only saw sessions before k
        self.assertFalse(np.allclose(alt.s[k + 3], base.s[k + 3], equal_nan=True), "the perturbation should change LATER sessions")

    def test_gap_plus_body_is_close_to_close(self):
        d = pdata()
        sg = C.build_signals(d, C.SigCfg(tf=5, spread="reset"))
        diff = (sg.eps_gap + sg.eps_body - sg.eps)[:, 1:]
        self.assertLess(float(np.median(np.abs(diff))), 1e-7)                              # only the cross term of two returns (rare large moves) is missing
        self.assertLess(float(np.abs(diff).max()), 1e-3)
        self.assertTrue(np.all(sg.eps_gap[:, 0] == 0))

    def test_no_lookahead_chain_and_zscore_modes(self):
        d = pdata()
        n = len(d.dates)
        k, t = n - 30, 20
        rng = np.random.default_rng(6)
        for cfg in (C.SigCfg(tf=5, spread="chain", beta_mode="rolling", win_days=5.0), C.SigCfg(tf=5, spread="reset", s_mode="z", win_days=3.0)):
            base = C.build_signals(d, cfg)
            mut = {f: getattr(d, f).copy() for f in ("L_open", "L_close", "B_open", "B_close")}
            for f in mut:
                a = mut[f]
                a[k, t + 1:] *= 1 + rng.normal(size=a.shape[1] - t - 1) * 0.01
                a[k + 1:] *= 1 + rng.normal(size=a[k + 1:].shape) * 0.01
            alt = C.build_signals(replace(d, **mut), cfg)
            np.testing.assert_allclose(alt.s[:k], base.s[:k], equal_nan=True, atol=1e-12)
            np.testing.assert_allclose(alt.s[k, :t + 1], base.s[k, :t + 1], equal_nan=True, atol=1e-12)

    def test_beta_is_close_to_three_for_the_nasdaq_pair(self):
        sg = C.build_signals(pdata(), C.SigCfg(tf=5, beta_mode="rolling", beta_shrink=0.0))
        self.assertAlmostEqual(float(np.median(sg.beta)), 3.0, delta=0.15)

    def test_resample_matches_manual_aggregation(self):
        b1 = D.load_bars("SPY", 1).loc["2026-10-20":"2026-10-21"]
        r = D.resample(b1, 5)
        first = b1.iloc[:5]
        row = r.iloc[0]
        self.assertEqual(row.open, first.open.iloc[0]); self.assertEqual(row.close, first.close.iloc[-1])
        self.assertEqual(row.high, first.high.max()); self.assertEqual(row.low, first.low.min()); self.assertAlmostEqual(row.volume, first.volume.sum())
        self.assertAlmostEqual(row.vwap, (first.vwap * first.volume).sum() / first.volume.sum(), places=9)
        self.assertEqual(len(r), 2 * 78)

    def test_missing_bars_are_carried_forward_and_flagged(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2025-03-03 09:30"), pd.Timestamp("2025-03-03 09:50")])
        b = pd.DataFrame({"open": [10.0, 11.0], "high": [10.0, 11.0], "low": [10.0, 11.0], "close": [10.0, 11.0], "volume": [5.0, 6.0], "vwap": [10.0, 11.0], "trade_count": [1.0, 1.0]}, index=idx)
        o, c, v, m = D._matrix(b, pd.DatetimeIndex([pd.Timestamp("2025-03-03")]), 78, 5)
        self.assertTrue(m[0, 1] and m[0, 3] and not m[0, 0] and not m[0, 4])
        self.assertEqual(c[0, 2], 10.0); self.assertEqual(c[0, 4], 11.0); self.assertEqual(v[0, 2], 0.0)


class TestRules(unittest.TestCase):
    cfg = S.SimCfg(s_open=2.0, s_close_long=0.5, s_close_short=0.5, s_stop=4.0, max_hold_min=20, skip_open_min=0, no_entry_min=0, flat_min=0, cooldown_min=10, max_halflife_min=0.0)

    def step(self, t, st, pos=0, age=0, cool=0, stopped=False, fin=True, hl=True, edge=True, move=False, c=None, nb=78):
        return S.decide_bar(t, nb, 5, c or self.cfg, one(pos), one(age, int), one(cool, int), one(stopped, bool), one(st, float), one(fin, bool), one(hl, bool), one(edge, bool), one(move, bool))

    def test_entries_and_targets(self):
        self.assertEqual(self.step(10, -2.5)[0][0], 1)
        self.assertEqual(self.step(10, 2.5)[0][0], -1)
        self.assertEqual(self.step(10, -1.9)[0][0], 0)
        p, _, _, r = self.step(11, -0.4, pos=1, age=1)
        self.assertEqual((p[0], r[0]), (0, 1))                                              # long closes once s > -0.5
        p, *_ = self.step(11, -1.0, pos=1, age=1)
        self.assertEqual(p[0], 1)
        p, _, _, r = self.step(11, 0.3, pos=-1, age=1)
        self.assertEqual((p[0], r[0]), (0, 1))

    def test_stop_cooldown_then_reentry(self):
        p, _, cool, r = self.step(20, -4.5, pos=1, age=1)
        self.assertEqual((p[0], r[0], cool[0]), (0, 2, 2))
        p, _, cool, _ = self.step(21, -3.0, pos=0, cool=2)
        self.assertEqual((p[0], cool[0]), (0, 1))                                             # still cooling down although |s| > open
        p, _, cool, _ = self.step(22, -3.0, pos=0, cool=1)
        self.assertEqual(p[0], 0)
        p, _, _, _ = self.step(23, -3.0, pos=0, cool=0)
        self.assertEqual(p[0], 1)

    def test_time_stop_and_end_of_day_and_dead_signal(self):
        p, age, _, r = self.step(30, -1.0, pos=1, age=3)                                       # max hold 20 min = 4 bars: the 4th bar held exits
        self.assertEqual((p[0], r[0]), (0, 3))
        p, _, _, r = self.step(77, -1.0, pos=1, age=1)                                        # flat_min 0: eod_dec = nb - 1
        self.assertEqual((p[0], r[0]), (0, 4))
        self.assertEqual(self.step(77, -3.0)[0][0], 0)                                         # and no entry on the last decision
        p, _, _, r = self.step(30, np.nan, pos=1, age=1, fin=False)
        self.assertEqual((p[0], r[0]), (0, 5))
        p, _, _, r = self.step(30, -1.0, pos=1, age=1, stopped=True)
        self.assertEqual((p[0], r[0]), (0, 6))

    def test_entry_windows_and_filters(self):
        c = replace(self.cfg, skip_open_min=15, no_entry_min=30, flat_min=10)
        self.assertEqual(self.step(2, -3.0, c=c)[0][0], 0)                                    # first 3 bars: skip
        self.assertEqual(self.step(3, -3.0, c=c)[0][0], 1)
        self.assertEqual(self.step(77 - 6, -3.0, c=c)[0][0], 1)                              # last_entry = nb - 1 - 6
        self.assertEqual(self.step(77 - 5, -3.0, c=c)[0][0], 0)
        self.assertEqual(self.step(10, -3.0, hl=False)[0][0], 0)
        self.assertEqual(self.step(10, -3.0, edge=False)[0][0], 0)
        self.assertEqual(self.step(10, -3.0, move=True)[0][0], 0)
        self.assertEqual(self.step(10, -3.0, stopped=True)[0][0], 0)

    def test_edge_filter_uses_both_legs_costs(self):
        c = replace(self.cfg, min_edge_ratio=2.0, s_close_long=0.5, s_close_short=0.5)
        s = np.array([[2.5]]); se = np.array([[5.0]]); hl = np.array([[10.0]])
        _, edge_ok = S.entry_filters(s, hl, se, np.array([3.0]), c, "ou", cL_bp=1.0, cB_bp=0.5)
        self.assertEqual(float((2.5 - 0.5) * 5.0), 10.0)
        self.assertTrue(edge_ok[0, 0])                                                         # round trip = 2 * (1 + 3 * 0.5) = 5 bp; 10 >= 2 * 5
        _, edge_ok = S.entry_filters(s, hl, se, np.array([3.0]), replace(c, min_edge_ratio=2.5), "ou", cL_bp=1.0, cB_bp=0.5)
        self.assertFalse(edge_ok[0, 0])


class TestAccounting(unittest.TestCase):
    def setUp(self):
        self._lc = S.leg_costs
        S.leg_costs = lambda pair, c: (1.0, 0.5)                                              # unit cost = (1 + 3 * 0.5) bp = 2.5 bp per unit change

    def tearDown(self):
        S.leg_costs = self._lc

    # bars 0..9; the spread signal is -2.5 at bar 2 (enter), -0.2 at bar 4 (leave). Prices of the 3x leg (the 1x leg is flat unless stated): bar 3 opens 100.10 and closes 100.30, bar 4 opens 100.32 and closes 100.50,
    # bar 5 opens 100.60 - so a long-spread trade is filled at bar 3's open (100.10, next_open), halfway (100.05, mid) or at bar 2's close (100.00, close) and closed at bar 5's open (100.60), 100.55 or bar 4's close (100.50)
    PX_L = dict(Lc=[[100.0, 100.0, 100.0, 100.30, 100.50, 100.70, 100.70, 100.70, 100.70, 100.70]], Lo=[[100.0, 100.0, 100.0, 100.10, 100.32, 100.60, 100.70, 100.70, 100.70, 100.70]])
    CFG = dict(s_open=2.0, s_close_long=0.5, s_close_short=0.5, skip_open_min=0, no_entry_min=0, flat_min=0, cooldown_min=0, max_halflife_min=0.0, vol_on=False)
    S_LONG = np.array([[np.nan, 0, -2.5, -1.0, -0.2, 0, 0, 0, 0, 0]])

    def test_hand_computed_trade_under_three_fill_models(self):
        sig = fake_sig(self.S_LONG, **self.PX_L)
        c = S.SimCfg(**self.CFG)
        uc = 2.5e-4
        for fill, le, lx in (("next_open", 100.10, 100.60), ("mid", 100.05, 100.55), ("close", 100.00, 100.50)):
            r = S.simulate_pair(sig, replace(c, fill=fill))
            self.assertEqual(list(r.hold[0]), [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], "the decisions do not depend on the fill price")
            want = lx / le - 1.0                                                                      # fixed shares bought at le and sold at lx
            self.assertAlmostEqual(r.gross_bar.sum(), want, places=12, msg=fill)
            self.assertAlmostEqual(r.net_bar.sum(), want - 2 * uc, places=12, msg=fill)
            tr = r.trades
            self.assertEqual(len(tr), 1)
            self.assertEqual((tr.side.iloc[0], tr.reason.iloc[0], int(tr.bars.iloc[0])), ("long", "target", 2))
            self.assertAlmostEqual(tr.L_entry.iloc[0], le, places=9); self.assertAlmostEqual(tr.L_exit.iloc[0], lx, places=9)
            self.assertAlmostEqual(tr.gross_bp.iloc[0], want * 1e4, places=6, msg=fill)
            self.assertAlmostEqual(tr.price_bp.iloc[0], want * 1e4, places=6, msg=fill)                 # the table's P&L can be reproduced from its own fill prices
            self.assertAlmostEqual(tr.net_bp.iloc[0], want * 1e4 - 5.0, places=6, msg=fill)
        self.assertAlmostEqual(S.simulate_pair(sig, c).trades.gap_in_bp.iloc[0], (100.10 / 100.0 - 1.0) * 1e4, places=6)         # what a close fill would collect at entry
        self.assertAlmostEqual(S.simulate_pair(sig, c).trades.gap_out_bp.iloc[0], (100.60 / 100.50 - 1.0) * 1e4, places=6)       # ... and give up at exit

    def test_both_legs_are_priced_with_fixed_shares_and_the_hedge_ratio(self):
        Bc = [[50.0, 50.0, 50.0, 50.10, 50.20, 50.30, 50.30, 50.30, 50.30, 50.30]]
        Bo = [[50.0, 50.0, 50.0, 50.05, 50.10, 50.30, 50.30, 50.30, 50.30, 50.30]]
        sig = fake_sig(self.S_LONG, Bc=Bc, Bo=Bo, **self.PX_L)
        c = S.SimCfg(**self.CFG)
        for fill, (le, lx), (be, bx) in (("next_open", (100.10, 100.60), (50.05, 50.30)), ("mid", (100.05, 100.55), (50.025, 50.25)), ("close", (100.00, 100.50), (50.00, 50.20))):
            r = S.simulate_pair(sig, replace(c, fill=fill))
            want = (lx / le - 1.0) - 3.0 * (bx / be - 1.0)                                              # long the 3x leg, short beta x the 1x leg
            tr = r.trades.iloc[0]
            self.assertAlmostEqual(r.gross_bar.sum(), want, places=12, msg=fill)
            self.assertAlmostEqual(tr.gross_bp, want * 1e4, places=6, msg=fill)
            self.assertAlmostEqual(tr.price_bp, want * 1e4, places=6, msg=fill)
            self.assertAlmostEqual(tr.B_entry, be, places=9); self.assertAlmostEqual(tr.B_exit, bx, places=9)
            self.assertAlmostEqual(tr.net_bp, want * 1e4 - 5.0, places=6, msg=fill)

    def test_trade_table_names_the_products_the_times_and_the_prices(self):
        sig = fake_sig(self.S_LONG, **self.PX_L)
        tr = S.simulate_pair(sig, S.SimCfg(**self.CFG)).trades
        for col in S.TRADE_COLS:
            self.assertIn(col, tr.columns)
        x = tr.iloc[0]
        self.assertEqual((x.position, x.long_leg, x.short_leg), ("LONG TQQQ / SHORT QQQ", "TQQQ", "QQQ"))
        self.assertEqual((x.entry_time, x.exit_time, int(x.hold_min)), (pd.Timestamp("2025-01-02 09:45"), pd.Timestamp("2025-01-02 09:55"), 10))      # decided at bar 2's close, in at bar 3, out at bar 5
        self.assertEqual(x.pair, "TQQQ/QQQ")
        short = S.simulate_pair(fake_sig(-self.S_LONG, **self.PX_L), S.SimCfg(**self.CFG)).trades.iloc[0]
        self.assertEqual((short.side, short.position, short.long_leg, short.short_leg), ("short", "LONG QQQ / SHORT TQQQ", "QQQ", "TQQQ"))

    def test_extra_latency_moves_the_trade_one_bar(self):
        s = np.array([[np.nan, 0, -2.5, -1.0, -0.2, 0, 0, 0, 0, 0]])
        sig = fake_sig(s)
        c = S.SimCfg(s_open=2.0, skip_open_min=0, no_entry_min=0, flat_min=0, cooldown_min=0, max_halflife_min=0.0, vol_on=False)
        self.assertEqual(list(S.simulate_pair(sig, replace(c, delay_bars=1)).hold[0]), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_nothing_is_held_overnight_in_a_real_backtest(self):
        d = pdata()
        for tf in (5,):
            sg = C.build_signals(d, C.SigCfg(tf=tf, spread="reset", beta_mode="fixed"))
            for fill in ("next_open", "mid", "close"):
                r = S.simulate_pair(sg, S.SimCfg(s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0, fill=fill, vol_on=False))
                self.assertTrue((r.hold[:, 0] == 0).all())
                self.assertTrue((r.hold[:, -2:] == 0).all(), "flat for the last 10 minutes")
                self.assertTrue((r.dec[:, -1] == 0).all())
                self.assertGreater(len(r.trades), 50)
                self.assertTrue((r.trades.bars > 0).all())

    def test_costs_charge_both_legs_per_unit_change(self):
        d = pdata()
        sg = C.build_signals(d, C.SigCfg(tf=5, spread="reset", beta_mode="fixed"))
        S.leg_costs = self._lc
        c = S.SimCfg(s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0, vol_on=False)
        r = S.simulate_pair(sg, c)
        cL, cB = S.leg_costs(sg.pair, c)
        changes = np.abs(np.diff(np.concatenate([np.zeros((len(r.hold), 1)), r.hold.astype(int)], axis=1), axis=1)).sum(axis=1)
        np.testing.assert_allclose(r.cost_bar.sum(axis=1), changes * (cL + sg.beta * cB) / 1e4, rtol=1e-12, atol=1e-15)

    def test_fill_at_the_close_overstates_the_edge(self):
        d = pdata()
        sg = C.build_signals(d, C.SigCfg(tf=5, spread="reset", beta_mode="fixed"))
        S.leg_costs = self._lc
        c = S.SimCfg(s_open=2.0, min_edge_ratio=0.0, max_halflife_min=60.0, vol_on=False)
        g_close = S.simulate_pair(sg, replace(c, fill="close")).gross_bar.sum()
        g_mid = S.simulate_pair(sg, replace(c, fill="mid")).gross_bar.sum()
        g_open = S.simulate_pair(sg, replace(c, fill="next_open")).gross_bar.sum()
        self.assertGreater(g_close, g_open, "a bar-close fill collects the bid-ask bounce")
        self.assertTrue(g_open < g_mid < g_close, "the midpoint fill sits between the two")
        with self.assertRaises(ValueError):
            S.simulate_pair(sg, replace(c, fill="vwap"))

    def test_volatility_target_uses_only_past_returns(self):
        rng = np.random.default_rng(4)
        u = pd.Series(rng.normal(0, 0.002, 300), index=pd.bdate_range("2024-01-02", periods=300))
        c = S.SimCfg(vol_on=True, vol_target=0.10, g_min=0.1, g_max=5.0, vol_warmup=20)
        g1 = S.unit_gross_leverage(u, c)
        u2 = u.copy(); u2.iloc[200:] *= 10
        g2 = S.unit_gross_leverage(u2, c)
        pd.testing.assert_series_equal(g1.iloc[:201], g2.iloc[:201])                       # the gross on day 200 cannot depend on day 200's return
        self.assertNotEqual(float(g1.iloc[205]), float(g2.iloc[205]))

    def test_pairs_can_be_switched_off(self):
        sigs = {k: C.build_signals(pdata(p), C.SigCfg(tf=5, spread="reset", beta_mode="fixed")) for k, p in zip(S.PAIR_ORDER, D.PAIRS)}
        c = S.SimCfg(start="2025-01-01", vol_on=False, s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0)
        r_all = S.simulate(sigs, c)
        r_one = S.simulate(sigs, replace(c, pair_w=(1.0, 0.0, 0.0)))
        self.assertEqual(list(r_one.per_pair), ["TQQQ/QQQ"])
        self.assertEqual(set(r_all.per_pair), set(S.PAIR_ORDER))
        np.testing.assert_allclose(r_all.weights.sum(axis=1), 1.0)
        with self.assertRaises(ValueError):
            S.simulate(sigs, replace(c, pair_w=(0, 0, 0)))

    def test_short_side_hand_computed(self):
        sig = fake_sig(-self.S_LONG, **self.PX_L)
        r = S.simulate_pair(sig, S.SimCfg(**self.CFG))
        self.assertEqual(list(r.hold[0]), [0, 0, 0, -1, -1, 0, 0, 0, 0, 0])
        want = -(100.60 / 100.10 - 1.0)
        self.assertAlmostEqual(r.net_bar.sum(), want - 2 * 2.5e-4, places=12)                           # short the spread: the price move is the position's loss, the costs are the same
        tr = r.trades
        self.assertEqual((tr.side.iloc[0], tr.reason.iloc[0]), ("short", "target"))
        self.assertAlmostEqual(tr.net_bp.iloc[0], want * 1e4 - 5.0, places=6)
        self.assertAlmostEqual(tr.price_bp.iloc[0], want * 1e4, places=6)

    def test_trade_table_equals_the_bar_accounting_on_real_data(self):
        sg = C.build_signals(pdata(), C.SigCfg(tf=5, spread="reset", beta_mode="fixed"))
        S.leg_costs = self._lc
        for fill in ("next_open", "mid", "close"):
            r = S.simulate_pair(sg, S.SimCfg(s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0, vol_on=False, fill=fill))
            self.assertGreater(len(r.trades), 50)
            np.testing.assert_allclose(r.trades.gross_bp.to_numpy(), r.trades.price_bp.to_numpy(), atol=1e-6, err_msg=f"{fill}: every trade's P&L is reproducible from its four fill prices")
            self.assertAlmostEqual(float(r.trades.net_bp.sum()), float(r.net_bar.sum() * 1e4), delta=1e-6 * max(1.0, abs(float(r.net_bar.sum() * 1e4))))   # the blind spot of the old exit-gap bug

    def test_fill_comparison_uses_identical_trades_and_orders_the_fills(self):
        panel = {k: pdata(p) for k, p in zip(S.PAIR_ORDER, D.PAIRS)}
        sigs = {k: C.build_signals(panel[k], C.SigCfg(tf=5, spread="reset", beta_mode="fixed")) for k in panel}
        S.leg_costs = self._lc
        base = S.SimCfg(start="2025-01-01", vol_on=False, s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0)
        tab, gaps = BR.fill_comparison(panel, sigs, base, split="2025-09-01")
        self.assertEqual(list(tab.index), ["next_open", "mid", "close"])
        self.assertEqual(len(set(tab.trades)), 1, "the decisions do not depend on the fill, so the trade list is the same")
        self.assertTrue(tab.gross_bp.loc["next_open"] < tab.gross_bp.loc["mid"] < tab.gross_bp.loc["close"])
        self.assertEqual(set(gaps), {"gap_in", "gap_out", "trades"})
        self.assertEqual(gaps["trades"], int(tab.trades.iloc[0]))
        # the whole difference between the fills is the move between the signal close and the next open, at entry and at exit
        self.assertAlmostEqual(tab.gross_bp.loc["close"] - tab.gross_bp.loc["next_open"], gaps["gap_in"] - gaps["gap_out"], delta=0.2)

    def test_sweeping_costs_does_not_change_which_trades_are_taken(self):
        sg = C.build_signals(pdata(), C.SigCfg(tf=5, spread="reset", beta_mode="fixed"))
        S.leg_costs = self._lc
        c = S.SimCfg(s_open=1.5, min_edge_ratio=1.5, max_halflife_min=60.0, vol_on=False)
        a, b = S.simulate_pair(sg, replace(c, slip_bp=0.0)), S.simulate_pair(sg, replace(c, slip_bp=1.0, cost_mult=2.0))
        np.testing.assert_array_equal(a.dec, b.dec)
        self.assertGreater(float(b.cost_bar.sum()), float(a.cost_bar.sum()) * 2)

    def test_half_spread_is_half_the_sampled_full_spread(self):
        sp = D.load_spreads()
        for sym in D.SYMBOLS:
            self.assertAlmostEqual(D.half_spread_bp(sym), float(sp.loc[sym, "all"]) / 2.0)

    def test_metrics_sanity(self):
        r = pd.Series(np.tile([0.001, -0.0005], 200), index=pd.bdate_range("2020-01-01", periods=400))
        m = S.metrics(r)
        self.assertAlmostEqual(m["sharpe"], r.mean() / r.std() * np.sqrt(252), places=9)
        self.assertGreater(m["psr0"], 0.99)


# ── live ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class FakeAPIError(Exception):
    """What alpaca-py raises for an HTTP error: the status code is a property of the exception."""

    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


class FakeTC:
    def __init__(self, positions=None):
        self.sent, self.cancelled, self.reject, self.n = [], [], set(), 0
        self.positions = dict(positions or {})
        self.fill_status = "filled"
        self.market_open = True
        self._orders = {}
        self._lk = threading.Lock()
        self.delay = 0.0                                                                           # simulated API latency of a submit
        self.arrivals = []                                                                         # (symbol, time_ns) when each submit reached the broker
        self.activities = []                                                                       # what the REST endpoint /account/activities/FILL returns

    def get_account(self):
        return types.SimpleNamespace(account_number="PA1234567", status="ACTIVE", equity="100000", cash="100000", buying_power="400000", multiplier="4", shorting_enabled=True, trading_blocked=False,
                                     account_blocked=False, pattern_day_trader=False, last_equity="100000", daytrading_buying_power="400000")

    def get_clock(self):
        now = pd.Timestamp.now(tz="America/New_York")
        return types.SimpleNamespace(is_open=self.market_open, next_open=now, next_close=now, timestamp=now)

    def submit_order(self, req):
        self.arrivals.append((req.symbol, time.time_ns()))
        if self.delay:
            time.sleep(self.delay)
        if getattr(self, "delays", {}).get(req.symbol):
            time.sleep(self.delays[req.symbol])
        if req.symbol in self.reject:
            raise FakeAPIError("asset is not shortable", 403)
        with self._lk:
            self.n += 1
            self.sent.append(req)
            o = types.SimpleNamespace(id=f"o{self.n}", symbol=req.symbol, qty=req.qty, side=req.side, client_order_id=req.client_order_id, status=self.fill_status, filled_qty=req.qty if self.fill_status == "filled" else 0,
                                      filled_avg_price=100.0 if self.fill_status == "filled" else None, submitted_at="2026-10-22 10:00:00.100", filled_at="2026-10-22 10:00:00.180" if self.fill_status == "filled" else None)
            self._orders[o.id] = o
            if self.fill_status == "filled":                                                      # the fake broker books the position
                self.positions[req.symbol] = self.positions.get(req.symbol, 0) + (req.qty if str(req.side.value) == "buy" else -req.qty)
            return o

    def get_order_by_id(self, oid):
        return self._orders[oid]

    def get_order_by_client_id(self, cid):
        self.lookups = getattr(self, "lookups", 0) + 1
        for o in list(self._orders.values()):
            if o.client_order_id == cid:
                return o
        raise FakeAPIError("order not found", 404)

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def get_orders(self, filter=None):
        if filter is not None:                                                                     # the end-of-day listing: every order of the day
            return list(self._orders.values())
        return []

    def get(self, path, data=None):
        return list(self.activities)

    def get_all_positions(self):
        return [types.SimpleNamespace(symbol=s, qty=q) for s, q in self.positions.items()]


class VClock:
    """A virtual clock: sleep() advances it, so a six-minute flatten runs in milliseconds and the broker-call rate can be measured in virtual minutes."""

    def __init__(self, t0=1_000_000.0):
        self.t = t0
        self.wall = 0.0                                                                           # a step of the PC's wall clock, which monotonic() does not see

    def time(self):
        return self.t + self.wall

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def time_ns(self):
        return int(self.time() * 1e9)


def peak_per_minute(stamps) -> int:
    stamps = sorted(stamps)
    peak, j = 0, 0
    for i, t in enumerate(stamps):
        while stamps[j] <= t - 60.0:
            j += 1
        peak = max(peak, i - j + 1)
    return peak


def view(name="TQQQ/QQQ", s=-2.5, L=70.0, B=700.0, beta=3.0, eps_body=0.0, eps_gap=0.0, **kw) -> LV.PairView:
    return LV.PairView(name, 20, s, 10.0, 5.0, beta, L, B, 0.0, eps_gap, eps_body, 0, 0.0, False, np.array([s]), **kw)


class TestLive(unittest.TestCase):
    def setUp(self):
        self._rd = D.REPORTS
        self.tmp = tempfile.mkdtemp(prefix="levpairs_test_")
        D.REPORTS = self.tmp                                                                   # state files and logs go to a throw-away folder
        LV.D.REPORTS = self.tmp
        sc, c = cfgs(DEFAULT_PRESET)
        self.sc, self.c = sc, replace(c, vol_on=False, g_init=1.0)
        self.logs = []

    def tearDown(self):
        D.REPORTS = self._rd

    def engine(self, armed=False, tc=None, **kw) -> LV.LiveEngine:
        kw = {**dict(feed="sip", fill_wait_s=2.0, flatten_verify_s=1.0, flatten_backoff_s=0.0, flatten_attempts=3), **kw}
        e = LV.LiveEngine(self.sc, self.c, LV.LiveSettings(armed=armed, **kw), log=lambda m, k="info": self.logs.append((k, m)))
        e.tc = tc
        e.acct = {"equity": 100_000.0, "blocked": False}
        e.deploy = 100_000.0
        e.G = 1.0
        e.state = e.blank_state("2026-10-22")
        e.state["date"] = "2026-10-22"
        return e

    def test_live_signals_equal_backtest_signals(self):
        sc = C.SigCfg(tf=5, spread="reset", beta_mode="fixed", win_days=5.0)
        full = D.load_pairdata(("TQQQ", "QQQ"), 5, start="2026-01-01")
        sg = C.build_signals(full, sc)
        src = LV.ParquetSource("sip")
        rng = np.random.default_rng(5)
        n = 0
        for d in rng.choice(np.arange(len(full.dates) - 60, len(full.dates)), 4, replace=False):
            for j in (3, 15, 40, 77):
                now = full.dates[d] + pd.Timedelta(minutes=D.OPEN_MIN + (j + 1) * 5, seconds=4)
                views, _ = LV.pair_views(src, [("TQQQ", "QQQ")], sc, self.c, full.dates[d], now)
                v = views["TQQQ/QQQ"]
                self.assertEqual(v.j, j)
                for a, b in ((v.s, sg.s[d, j]), (v.hl_min, sg.hl_min[d, j]), (v.sig_eq_bp, sg.sig_eq_bp[d, j]), (v.beta, sg.beta[d]), (v.x_bp, sg.X[d, j] * 1e4)):
                    if np.isfinite(b):
                        self.assertAlmostEqual(a, b, places=6)
                n += 1
        self.assertEqual(n, 16)

    def test_replayed_session_equals_backtest_positions(self):
        sc, c = self.sc, replace(self.c, s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0)
        full = D.load_pairdata(("TQQQ", "QQQ"), 5, start="2026-01-01")
        sigs = {k: C.build_signals(D.load_pairdata(p, 5, start="2026-01-01"), sc) for k, p in zip(S.PAIR_ORDER, D.PAIRS)}
        res = S.simulate(sigs, replace(c, start="2026-01-01"))
        total_trades = 0
        for d in (full.dates[-3], full.dates[-12]):
            eng, rows = LV.replay_session(d, sc, c, LV.LiveSettings(feed="sip"), equity=100_000_000.0)          # a huge book: whole-share rounding is negligible
            self.assertIn("position", rows.columns)
            self.assertTrue(rows[rows.pos_after == 0].position.eq("FLAT").all())
            self.assertTrue(rows[rows.pos_after != 0].position.str.match(r"^(LONG|SHORT) ").all())
            for k in S.PAIR_ORDER:
                di = list(sigs[k].dates).index(d)
                want = res.per_pair[k].dec[di]
                got = rows[rows.pair == k].set_index("j").pos_after.reindex(range(78)).ffill().fillna(0).to_numpy()
                np.testing.assert_array_equal(got[:77], want[:77])
                total_trades += int((np.abs(np.diff(want.astype(int))) > 0).sum())
                unit = res.per_pair[k].net_bar[di].sum()
                st = eng.state["pairs"][k]
                if st["n_trades"]:
                    self.assertAlmostEqual(st["pnl_usd"] / st["N_L"], unit, delta=3e-5 * max(st["n_trades"], 1))
        self.assertGreater(total_trades, 6, "the test days should contain trades")

    def test_dry_run_never_touches_the_broker(self):
        class Boom:
            def __getattr__(self, name):
                raise AssertionError(f"DRY RUN called the broker: {name}")
        e = self.engine(armed=False, tc=Boom())
        row = {}
        p = ("TQQQ", "QQQ")
        v = view()
        ps = e.state["pairs"]["TQQQ/QQQ"]
        row["time"] = "10:00"
        self.assertTrue(e.trade_pair(p, "TQQQ/QQQ", ps, 1, v, 1.0, 2.5e-4, row))
        self.assertEqual(ps["side"], 1)
        self.assertGreater(ps["shares_L"], 0); self.assertLess(ps["shares_B"], 0)
        self.assertTrue(e.trade_pair(p, "TQQQ/QQQ", ps, 0, v, 1.0, 2.5e-4, row))
        self.assertEqual((ps["side"], ps["shares_L"], ps["shares_B"]), (0, 0, 0))
        e.flatten("test")                                                                       # DRY RUN flatten clears the shadow ledger, no broker call

    def test_position_and_action_text_say_which_product_is_long_and_which_short(self):
        self.assertEqual(LV.pair_position_text("TQQQ/QQQ", {"side": 0}), "FLAT")
        self.assertEqual(LV.pair_position_text("TQQQ/QQQ", {"side": 1, "shares_L": 357, "shares_B": -107}), "LONG TQQQ 357 sh / SHORT QQQ 107 sh")
        self.assertEqual(LV.pair_position_text("SPXL/SPY", {"side": -1, "shares_L": -200, "shares_B": 60}), "SHORT SPXL 200 sh / LONG SPY 60 sh")

        class Boom:
            def __getattr__(self, name):
                raise AssertionError(f"DRY RUN called the broker: {name}")
        e = self.engine(armed=False, tc=Boom())
        p, name = ("TQQQ", "QQQ"), "TQQQ/QQQ"
        ps = e.state["pairs"][name]
        row = {"time": "10:00"}
        self.assertTrue(e.trade_pair(p, name, ps, 1, view(L=70.0, B=700.0), 1.0, 2.5e-4, row))
        self.assertTrue(row["action"].startswith("ENTER long spread: BUY "), row["action"])
        self.assertIn(" TQQQ @ 70.00", row["action"]); self.assertIn("SELL ", row["action"]); self.assertIn(" QQQ @ 700.00", row["action"])
        pos = LV.pair_position_text(name, ps)
        self.assertTrue(pos.startswith("LONG TQQQ ") and " / SHORT QQQ " in pos, pos)
        row = {"time": "10:30"}
        self.assertTrue(e.trade_pair(p, name, ps, 0, view(L=71.0, B=700.0), 1.0, 2.5e-4, row))
        self.assertTrue(row["action"].startswith("EXIT: SELL "), row["action"])
        self.assertIn(" TQQQ @ 71.00", row["action"]); self.assertIn("BUY ", row["action"]); self.assertIn(" QQQ @ 700.00", row["action"])
        self.assertEqual(LV.pair_position_text(name, ps), "FLAT")
        row = {"time": "11:00"}
        self.assertTrue(e.trade_pair(p, name, ps, -1, view(s=2.5, L=70.0, B=700.0), 1.0, 2.5e-4, row))
        self.assertTrue(row["action"].startswith("ENTER short spread: SELL "), row["action"])
        pos = LV.pair_position_text(name, ps)
        self.assertTrue(pos.startswith("SHORT TQQQ ") and " / LONG QQQ " in pos, pos)
        row = {"time": "11:30"}
        self.assertTrue(e.trade_pair(p, name, ps, 0, view(s=0.0, L=69.0, B=700.0), 1.0, 2.5e-4, row))
        self.assertTrue(row["action"].startswith("EXIT: BUY "), row["action"])

    def test_two_leg_entry_and_exit_armed(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc, max_leg_nav=1.0, max_gross_nav=1.0, max_gross_x=10.0)       # caps wide open: this test is about the order flow
        ps = e.state["pairs"]["TQQQ/QQQ"]
        v = view(L=70.0, B=700.0)
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, v, 1.0, 2.5e-4, {"time": "10:00"}))
        syms = {r.symbol: (str(r.side.value), int(r.qty)) for r in tc.sent}                        # both legs go out together: their order of arrival is not defined
        n_L = int(100_000.0 / 4 // 70.0)
        self.assertEqual(syms["TQQQ"], ("buy", n_L))                                              # long the spread: buy the 3x ETF, sell the hedge
        self.assertEqual(syms["QQQ"][0], "sell")
        self.assertAlmostEqual(syms["QQQ"][1] * 700.0, 3.0 * n_L * 70.0, delta=700.0)
        self.assertEqual(len(tc.sent), 2)
        self.assertTrue(all(str(r.client_order_id).startswith("LP-") for r in tc.sent))
        self.assertEqual((ps["side"], ps["shares_L"], -ps["shares_B"] > 0), (1, n_L, True))
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, v, 1.0, 2.5e-4, {"time": "10:30"}))
        self.assertEqual(sorted((r.symbol, str(r.side.value)) for r in tc.sent[2:]), [("QQQ", "buy"), ("TQQQ", "sell")])
        self.assertEqual(ps["side"], 0)

    def test_failed_leg_unwinds_the_other_and_marks_the_pair_broken(self):
        tc = FakeTC()
        tc.reject = {"QQQ"}
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, row))
        sides = [(r.symbol, str(r.side.value)) for r in tc.sent]
        self.assertEqual(sides, [("TQQQ", "buy"), ("TQQQ", "sell")], "the filled leg is closed again")
        self.assertEqual(ps["side"], 0)
        self.assertTrue(ps["broken"])
        self.assertIn("orders failed", row["note"])

    def test_entry_refused_if_the_broker_already_holds_a_leg_and_exit_never_exceeds_the_broker(self):
        tc = FakeTC(positions={"QQQ": 40})
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, row))
        self.assertIn("already holds", row["note"])
        self.assertEqual(tc.sent, [])
        tc.positions = {}
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        n_sent = len(tc.sent)
        tc.positions = {"TQQQ": ps["shares_L"] - 10, "QQQ": ps["shares_B"]}                      # someone sold 10 TQQQ by hand
        want_L = tc.positions["TQQQ"]
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, view(), 1.0, 2.5e-4, {"time": "10:30"}))
        closed = {r.symbol: int(r.qty) for r in tc.sent[n_sent:]}
        self.assertEqual(closed["TQQQ"], want_L)

    def test_partial_fill_that_times_out_is_cancelled_and_unwound(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        orig = tc.submit_order

        def submit(req):
            o = orig(req)
            if req.symbol == "QQQ":                                                              # the hedge leg never fills completely
                o.status, o.filled_qty = "partially_filled", 5
            return o
        tc.submit_order = submit
        ps = e.state["pairs"]["TQQQ/QQQ"]
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, row))
        self.assertTrue(tc.cancelled)
        unwind = [(r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent[2:]]
        self.assertIn(("QQQ", "buy", 5), unwind)                                                   # the 5 QQQ shares that were sold are bought back
        self.assertTrue(ps["broken"])

    def test_fill_log_records_slippage_against_the_signal_price(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(L=99.0, B=101.0), 1.0, 2.5e-4, {"time": "10:00"}))
        f = pd.read_csv(os.path.join(self.tmp, "levpairs_fills_paper.csv"))
        buy = f[f.symbol == "TQQQ"].iloc[0]
        sell = f[f.symbol == "QQQ"].iloc[0]
        self.assertAlmostEqual(buy.slip_bp, (100.0 - 99.0) / 99.0 * 1e4, places=6)                # bought at 100 vs a 99 signal: worse
        self.assertAlmostEqual(sell.slip_bp, -(100.0 - 101.0) / 101.0 * 1e4, places=6)             # sold at 100 vs a 101 signal: worse


    # ── the RM review: scenarios that used to leave the ledger blind or the account exposed ─────────────────────────────────────────
    def test_polling_error_leaves_the_pair_broken_and_the_orphan_visible(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc, resolve_unknown=False)
        def boom(oid):
            raise ConnectionError("api down")
        tc.get_order_by_id = boom
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        self.assertTrue(ps["broken"])
        self.assertEqual(ps["side"], 0)
        self.assertTrue(e.state["alarms"])
        orph = e.orphans()                                                                         # the fake broker booked both fills although the ledger is flat: reconcile sees it
        self.assertEqual(set(orph), {"TQQQ", "QQQ"})

    def test_flatten_that_cannot_close_keeps_the_pair_in_the_ledger(self):
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, shares_L=100, shares_B=-30)
        tc.reject = {"QQQ"}                                                                        # the hedge leg cannot be closed (halted, rejected ...)
        e.flatten("KILL")
        self.assertEqual(ps["side"], 1, "not confirmed flat: it must stay in the ledger")
        self.assertTrue(ps["broken"])
        self.assertTrue(e.state["alarms"])
        self.assertEqual(tc.positions["TQQQ"], 0)
        tc.reject = set()
        e.flatten("KILL again")
        self.assertEqual((ps["side"], tc.positions.get("QQQ", 0)), (0, 0))

    def test_write_ahead_intent_survives_a_crash_between_the_legs(self):
        class Crash(BaseException):
            pass
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        def die(oid):
            raise Crash()
        tc.get_order_by_id = die
        with self.assertRaises(Crash):
            e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", e.state["pairs"]["TQQQ/QQQ"], 1, view(), 1.0, 2.5e-4, {"time": "10:00"})
        st = json.load(open(LV.state_path("paper")))
        self.assertEqual(st["pending"]["pair"], "TQQQ/QQQ")                                         # a restart sees an order in flight and refuses to start (run_session)
        self.assertEqual([x[0] for x in st["pending"]["legs"]], ["TQQQ", "QQQ"])

    def test_single_instance_lock(self):
        e1, e2 = self.engine(), self.engine()
        e1._acquire_lock()
        with self.assertRaises(RuntimeError):
            e2._acquire_lock()
        e1._release_lock()
        e2._acquire_lock()
        e2._release_lock()

    def test_iex_guard_is_in_the_order_path_not_only_in_run_session(self):
        tc = FakeTC()
        e = LV.LiveEngine(self.sc, self.c, LV.LiveSettings(armed=True, feed="iex"), log=lambda *a: None)
        e.tc, e.acct, e.state = tc, {"equity": 100_000.0, "blocked": False}, e.blank_state("2026-10-22")
        with self.assertRaises(RuntimeError):
            e.send("TQQQ/QQQ", [("TQQQ", 1, True)], {"time": "10:00"})
        self.assertEqual(tc.sent, [])
        e.live = replace(e.live, accept_iex=True)
        self.assertIsNotNone(e.send("TQQQ/QQQ", [("TQQQ", 1, True)], {"time": "10:00"}))

    def test_flatten_works_on_yesterdays_ledger_without_a_session(self):
        e = self.engine(armed=True, tc=FakeTC())
        e.state["date"] = "2026-10-20"
        e.state["pairs"]["SPXL/SPY"].update(side=-1, shares_L=-50, shares_B=60)
        e.save_state(e.state)
        tc = FakeTC(positions={"SPXL": -50, "SPY": 60})
        e2 = LV.LiveEngine(self.sc, self.c, LV.LiveSettings(armed=True, feed="sip", flatten_verify_s=1.0), log=lambda *a: None)
        e2.tc = tc                                                                                   # a fresh engine: no start_session, no state in memory
        e2.flatten("KILL")
        self.assertEqual(sorted((r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent), [("SPXL", "buy", 50), ("SPY", "sell", 60)])
        self.assertEqual(e2.state["pairs"]["SPXL/SPY"]["side"], 0)

    def test_reset_broken_refuses_while_the_broker_disagrees(self):
        tc = FakeTC(positions={"QQQ": 40})
        e = self.engine(armed=True, tc=tc)
        e.state["pairs"]["TQQQ/QQQ"]["broken"] = True
        with self.assertRaises(RuntimeError):
            e.reset_broken()
        tc.positions = {}
        e.reset_broken()
        self.assertFalse(e.state["pairs"]["TQQQ/QQQ"]["broken"])

    def test_an_open_order_of_the_engine_blocks_a_second_entry(self):
        tc = FakeTC()
        tc.get_orders = lambda: [types.SimpleNamespace(client_order_id="LP-20261022-TQQQQQQ-1-1", symbol="QQQ", id="z")]
        e = self.engine(armed=True, tc=tc)
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", e.state["pairs"]["TQQQ/QQQ"], 1, view(), 1.0, 2.5e-4, row))
        self.assertIn("still open", row["note"])
        self.assertEqual(tc.sent, [])

    def test_exposure_caps_use_each_legs_own_price_and_the_open_gross(self):
        e = self.engine(armed=False)                                                                # equity 100k: leg cap 10k, gross cap 25k
        sz, why = e.size_pair("TQQQ/QQQ", 1.0, view(L=70.0, B=700.0))
        n_L, n_B, _ = sz
        self.assertLessEqual(n_B * 700.0, 10_000.0 + 700.0)
        self.assertLessEqual(n_L * 70.0 + n_B * 700.0, 25_000.0)
        self.assertGreaterEqual(n_L, 1)
        e.state["pairs"]["SPXL/SPY"].update(side=1, gross_usd=24_900.0)                              # the book is already almost full
        sz, why = e.size_pair("TQQQ/QQQ", 1.0, view(L=70.0, B=700.0))
        self.assertIsNone(sz)
        self.assertIn("exposure caps", why)

    def test_sip_data_older_than_five_minutes_is_not_a_signal(self):
        class Lagged(LV.ParquetSource):
            def bars1(self, sym, start, end):
                if pd.Timestamp(end) - pd.Timestamp(start) < pd.Timedelta(days=1):                   # today's bars arrive 20 minutes late, like SIP on a non-real-time account
                    end = pd.Timestamp(end) - pd.Timedelta(minutes=20)
                return super().bars1(sym, start, end)
        sc = C.SigCfg(tf=5, spread="reset", beta_mode="fixed", win_days=5.0)
        d = D.load_pairdata(("TQQQ", "QQQ"), 5, start="2026-09-01").dates[-3]
        now = d + pd.Timedelta(minutes=D.OPEN_MIN + 40 * 5, seconds=4)
        fresh, _ = LV.pair_views(LV.ParquetSource("sip"), [("TQQQ", "QQQ")], sc, self.c, d, now)
        late, _ = LV.pair_views(Lagged("sip"), [("TQQQ", "QQQ")], sc, self.c, d, now)
        self.assertTrue(fresh["TQQQ/QQQ"].fin)
        self.assertFalse(late["TQQQ/QQQ"].fin)

    def test_lp_pnl_in_armed_mode_uses_the_engines_own_fills(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(L=99.0, B=101.0), 1.0, 2.5e-4, {"time": "10:00"}))   # fills at 100 / 100
        e.views = {"TQQQ/QQQ": view(L=101.0, B=99.0)}
        want = ps["shares_L"] * (101.0 - 100.0) + ps["shares_B"] * (99.0 - 100.0)
        self.assertAlmostEqual(e.lp_pnl(), want)
        tc.positions = {"TQQQ": ps["shares_L"], "QQQ": ps["shares_B"]}
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, view(L=101.0, B=99.0), 1.0, 2.5e-4, {"time": "10:30"}))   # closes at 100 / 100 (the fake)
        self.assertAlmostEqual(e.lp_pnl(), 0.0, places=6)
        # another strategy winning in the same account does not move the halt
        e.acct["equity"] = 120_000.0
        self.assertFalse(e.loss_halt_check(-5000.0)[0])

    # ── the RM re-verification: gaps left by the first round of fixes ───────────────────────────────────────────────────────────
    def test_an_unknown_leg_is_never_unwound_against_the_other(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc, resolve_unknown=False)
        orig = tc.get_order_by_id
        def get(oid):
            if oid == "o2":                                                                         # the big hedge leg: its state cannot be read
                raise ConnectionError("api down")
            return orig(oid)
        tc.get_order_by_id = get
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        self.assertEqual(len(tc.sent), 2, "no unwind order: selling the filled leg would leave the unknown hedge leg naked")
        self.assertTrue(ps["broken"])
        self.assertTrue(any("unknown" in a for a in e.state["alarms"]))
        e.tc.get_order_by_id = orig
        e.flatten_orphans()                                                                         # the explicit, confirmed route that clears what the ledger does not explain
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})

    def test_flatten_retries_through_a_transient_broker_outage(self):
        tc = FakeTC(positions={"TQQQ": 47, "QQQ": -14})
        e = self.engine(armed=True, tc=tc, flatten_attempts=6)
        e.state["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=47, shares_B=-14)
        real = tc.get_all_positions
        calls = {"n": 0}
        def flaky():
            calls["n"] += 1
            if calls["n"] <= 3:
                raise ConnectionError("outage")
            return real()
        tc.get_all_positions = flaky
        e.flatten("end of day")
        self.assertEqual(e.state["pairs"]["TQQQ/QQQ"]["side"], 0)
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})

    def test_flatten_does_not_stack_an_order_on_a_symbol_that_still_has_one_open(self):
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        tc.get_orders = lambda: [types.SimpleNamespace(client_order_id="LP-20261022-X-1", symbol="QQQ", id="stuck")]
        def cancel(oid):
            raise ConnectionError("cancel failed")
        tc.cancel_order_by_id = cancel
        e = self.engine(armed=True, tc=tc)
        e.state["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30)
        e.flatten("KILL")
        self.assertEqual([r.symbol for r in tc.sent], ["TQQQ"], "QQQ already has an open order: a second close would flip the position")
        self.assertTrue(e.state["pairs"]["TQQQ/QQQ"]["broken"])
        self.assertEqual(e.state["pairs"]["TQQQ/QQQ"]["side"], 1)

    def test_kill_flag_stops_entries_in_flight(self):
        e = self.engine(armed=False)
        e.kill_flag.set()
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", e.state["pairs"]["TQQQ/QQQ"], 1, view(), 1.0, 2.5e-4, row))
        self.assertIn("KILL", row["note"])

    def test_whole_shares_that_miss_the_hedge_ratio_are_not_traded(self):
        e = self.engine(armed=False)
        e.deploy = 3000.0
        sz, why = e.size_pair("TQQQ/QQQ", 1.0, view(L=70.0, B=1500.0))
        self.assertIsNone(sz)
        self.assertIn("hedge ratio", why)
        sz, why = e.size_pair("TQQQ/QQQ", 1.0, view(L=70.0, B=700.0))
        self.assertIsNotNone(sz)

    def test_flatten_orphans_closes_unexplained_positions_only_when_armed(self):
        tc = FakeTC(positions={"TQQQ": 47, "QQQ": -14, "AAPL": 10})
        self.assertEqual(self.engine(armed=False, tc=tc).flatten_orphans(), [])
        e = self.engine(armed=True, tc=tc)
        res = e.flatten_orphans()
        self.assertEqual(sorted((r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent), [("QQQ", "buy", 14), ("TQQQ", "sell", 47)])
        self.assertEqual(tc.positions.get("AAPL"), 10)                                              # nothing outside the six ETFs

    # ── RM round 2 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def test_orphan_flatten_never_moves_the_broker_toward_the_ledger(self):
        def run(ledger, broker):
            tc = FakeTC(positions=dict(broker))
            e = self.engine(armed=True, tc=tc)
            e.state["pairs"]["TQQQ/QQQ"].update(side=1 if ledger else 0, shares_L=ledger.get("TQQQ", 0), shares_B=ledger.get("QQQ", 0))
            e.flatten_orphans()
            return tc
        tc = run({"TQQQ": 119, "QQQ": -36}, {"QQQ": -36})                                           # D2: the ledger holds more than the broker: nothing to close, nothing to re-create
        self.assertEqual(tc.sent, [])
        tc = run({"TQQQ": 100, "QQQ": -300}, {})                                                   # D3: the pair was closed by hand: the ledger is wrong, the broker is flat
        self.assertEqual(tc.sent, [])
        tc = run({"TQQQ": 100}, {"TQQQ": -50})                                                      # D4: opposite sides: close what the broker holds, do not buy 150
        self.assertEqual([(r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent], [("TQQQ", "buy", 50)])
        tc = run({"TQQQ": 100, "QQQ": -30}, {"TQQQ": 150, "QQQ": -30})                              # same side: only the excess
        self.assertEqual([(r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent], [("TQQQ", "sell", 50)])

    def test_end_of_day_flatten_keeps_retrying_until_the_cutoff_not_for_a_fixed_count(self):
        import unittest.mock as mock
        class Clock:
            t = pd.Timestamp("2026-10-23 15:52:00", tz="America/New_York")
            def __call__(self):
                self.t += pd.Timedelta(seconds=20)
                return self.t
        clk = Clock()
        tc = FakeTC(positions={"TQQQ": 47, "QQQ": -14})
        e = self.engine(armed=True, tc=tc, flatten_attempts=3)
        e.state["date"] = "2026-10-23"
        e.state["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=47, shares_B=-14)
        real, calls = tc.get_all_positions, {"n": 0}
        def flaky():
            calls["n"] += 1
            if calls["n"] <= 5:                                                                    # an outage much longer than 3 attempts
                raise ConnectionError("outage")
            return real()
        tc.get_all_positions = flaky
        with mock.patch.object(LV, "clock_now", clk):
            e.flatten("end of day")
        self.assertEqual(e.state["pairs"]["TQQQ/QQQ"]["side"], 0)
        # the manual KILL keeps the short fixed-attempt behaviour
        tc2 = FakeTC(positions={"TQQQ": 47, "QQQ": -14})
        e2 = self.engine(armed=True, tc=tc2, flatten_attempts=3)
        e2.state["date"] = "2026-10-23"
        e2.state["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=47, shares_B=-14)
        real2, c2 = tc2.get_all_positions, {"n": 0}
        def flaky2():
            c2["n"] += 1
            if c2["n"] <= 12:
                raise ConnectionError("outage")
            return real2()
        tc2.get_all_positions = flaky2
        clk2 = Clock()
        with mock.patch.object(LV, "clock_now", clk2):
            e2.flatten("KILL")
        self.assertEqual(e2.state["pairs"]["TQQQ/QQQ"]["side"], 1)
        self.assertTrue(e2.state["pairs"]["TQQQ/QQQ"]["broken"])

    def test_kill_while_an_entry_is_in_flight_unwinds_it(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        orig = tc.submit_order

        def submit(req):
            o = orig(req)
            e.kill_flag.set()                                                                      # KILL pressed while the two orders are on their way
            return o
        tc.submit_order = submit
        ps = e.state["pairs"]["TQQQ/QQQ"]
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, row))
        self.assertEqual(sorted((r.symbol, str(r.side.value)) for r in tc.sent), [("QQQ", "buy"), ("QQQ", "sell"), ("TQQQ", "buy"), ("TQQQ", "sell")], "both legs were sent, both are unwound")
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})
        self.assertEqual(ps["side"], 0)
        self.assertIn("KILL was pressed while the entry was in flight", row["note"])

    def test_kill_before_the_entry_is_sent_sends_nothing(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        e.kill_flag.set()
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        self.assertEqual(tc.sent, [])

    def test_oversized_or_tiny_entries_are_skipped(self):
        e = self.engine(armed=False)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        e.deploy = 100.0                                                                         # a single share costs more than the leg
        row = {}
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, row))
        self.assertIn("minimum ticket", row["note"])

    def test_kill_closes_only_the_ledger_and_never_more_than_the_broker_holds(self):
        tc = FakeTC(positions={"TQQQ": 120, "QQQ": -30, "SPY": 500, "AAPL": 77})
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, shares_L=100, shares_B=-35)
        res = e.flatten("KILL")
        sent = {r.symbol: (str(r.side.value), int(r.qty)) for r in tc.sent}
        self.assertEqual(sent, {"TQQQ": ("sell", 100), "QQQ": ("buy", 30)})                   # min(ledger, broker), same direction; SPY / AAPL untouched
        self.assertEqual(ps["side"], 0)
        self.assertEqual(len(res), 2)

    def test_state_resume_and_overnight_guard(self):
        e = self.engine(armed=True, tc=FakeTC())
        ps = e.state["pairs"]["SPXL/SPY"]
        ps.update(side=-1, shares_L=-50, shares_B=60)
        e.save_state(e.state)
        e2 = LV.LiveEngine(self.sc, self.c, LV.LiveSettings(armed=True, feed="sip"), log=lambda *a: None)
        self.assertEqual(e2.load_state("2026-10-22")["pairs"]["SPXL/SPY"]["shares_L"], -50)            # same day: resumed
        with self.assertRaises(RuntimeError):
            e2.load_state("2026-10-23")                                                       # next day with an open position: refuses to start
        e3 = LV.LiveEngine(self.sc, self.c, LV.LiveSettings(armed=False, feed="sip"), log=lambda *a: None)
        self.assertEqual(e3.load_state("2026-10-23")["pairs"]["SPXL/SPY"]["side"], 0)          # the dry-run book has its own file

    def test_corrupt_state_is_an_error_not_an_empty_ledger(self):
        open(LV.state_path("paper"), "w").write("{not json")
        e = LV.LiveEngine(self.sc, self.c, LV.LiveSettings(armed=True, paper=True), log=lambda *a: None)
        with self.assertRaises(RuntimeError):
            e.load_state("2026-10-22")

    def test_loss_halt_blocks_new_risk_and_flattens(self):
        e = self.engine(armed=False)
        e.state["pairs"]["TQQQ/QQQ"]["pnl_usd"] = -1500.0                                      # -1.5% of 100k deployed, limit 1%
        halt, why = e.loss_halt_check(-1500.0)
        self.assertTrue(halt)
        self.assertIn("below", why)
        self.assertFalse(e.loss_halt_check(-500.0)[0])

    def test_step_on_a_session_halts_and_goes_flat_after_a_loss(self):
        c = replace(self.c, s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0)
        eng = LV.LiveEngine(self.sc, c, LV.LiveSettings(feed="sip", daily_loss_halt=1e-9), log=lambda *a: None, source=LV.ParquetSource("sip"))
        eng.save_state = lambda st: None
        eng.append_log = lambda n, r: None
        d = pd.Timestamp(D.load_pairdata(("TQQQ", "QQQ"), 5, start="2026-09-01").dates[-4])
        eng.state = eng.blank_state(d); eng.deploy = 100_000.0; eng.G = 1.0; eng.history = {}
        seen_pos, halted_at = [], None
        for j in range(78):
            eng.step(d, d + pd.Timedelta(minutes=D.OPEN_MIN + (j + 1) * 5, seconds=4))
            seen_pos.append(sum(abs(p["side"]) for p in eng.state["pairs"].values()))
            if eng.halted and halted_at is None:
                halted_at = j
        self.assertEqual(seen_pos[-1], 0, "flat at the end of the session")
        if halted_at is not None:                                                              # once halted: flat for the rest of the session, no new entries
            self.assertTrue(all(x == 0 for x in seen_pos[halted_at:]), seen_pos[halted_at:])

    def test_no_order_call_outside_send_and_flatten(self):
        src = open(LV.__file__, encoding="utf-8").read()
        lines = src.splitlines()
        owners = []
        for i, ln in enumerate(lines):
            if "submit_order(" in ln:
                j = i
                while not lines[j].lstrip().startswith("def "):
                    j -= 1
                owners.append(lines[j].split("def ")[1].split("(")[0])
        self.assertEqual(sorted(owners), ["_flatten", "_flatten_orphans", "one"])                          # the one submit inside _submit_all (entry, exit, unwind), the verified flatten, the orphan flatten - nowhere else
        callers = []
        for i, ln in enumerate(lines):
            if "self._submit_all(" in ln:
                j = i
                while not lines[j].lstrip().startswith("def "):
                    j -= 1
                callers.append(lines[j].split("def ")[1].split("(")[0])
        self.assertEqual(sorted(callers), ["_send_failed", "send"], "_submit_all is called only by send (the order) and its failure path (the unwind)")
        body = src.split("def send(")[1].split("# ── emergency")[0]
        self.assertIn("if not self.live.armed:", body)
        self.assertIn("paper=self.live.paper", src)


    # ── RM third pass (D1-D7): one closer at a time, no orders outside market hours, honest text, a P&L that cannot be NaN ──────────────────────────────
    def closable_pair(self, tc, **kw):
        e = self.engine(armed=True, tc=tc, **kw)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, shares_L=100, shares_B=-30, N_L=7000.0, entry_px={"TQQQ": 70.0, "QQQ": 233.0}, ent_L=70.0, ent_B=233.0)
        return e, ps

    def test_d1_a_second_flatten_waits_for_the_running_one_instead_of_closing_beside_it(self):
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        e, ps = self.closable_pair(tc, kill_lock_wait_s=0.2, flatten_join_s=20.0)
        active, peak = {"n": 0}, {"n": 0}
        orig = e._flatten

        def slow(reason, persist=False):
            active["n"] += 1
            peak["n"] = max(peak["n"], active["n"])
            try:
                time.sleep(0.8)                                                                   # the end-of-day flatten is still looping when KILL is pressed
                return orig(reason, persist)
            finally:
                active["n"] -= 1
        e._flatten = slow
        out = {}
        a = threading.Thread(target=lambda: out.setdefault("a", e.flatten("end of day")))
        b = threading.Thread(target=lambda: out.setdefault("b", e.flatten("KILL")))
        a.start(); time.sleep(0.2); b.start(); a.join(30); b.join(30)
        self.assertEqual(peak["n"], 1, "never two closers on one ledger at once")
        self.assertEqual(sorted((r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent), [("QQQ", "buy", 30), ("TQQQ", "sell", 100)], "closed once, not twice")
        self.assertEqual((tc.positions.get("TQQQ", 0), tc.positions.get("QQQ", 0)), (0, 0))
        self.assertTrue(e.kill_flag.is_set())
        self.assertTrue(any("already running" in m for _, m in self.logs))

    def test_d1_a_flatten_re_reads_the_broker_before_each_order(self):
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        e, ps = self.closable_pair(tc)
        orig = tc.submit_order

        def submit(req):
            o = orig(req)
            if req.symbol == "TQQQ":
                tc.positions["QQQ"] = 0                                                           # the hedge is closed by hand while the first leg is being closed
            return o
        tc.submit_order = submit
        e.flatten("KILL")
        self.assertEqual([r.symbol for r in tc.sent], ["TQQQ"], "a stale read would have BOUGHT 30 QQQ and opened a long")
        self.assertEqual((tc.positions.get("QQQ", 0), ps["side"]), (0, 0))

    def test_d2_a_state_file_of_the_previous_build_seeds_entry_prices_and_a_missing_price_books_nothing(self):
        e = self.engine(armed=False)
        st = e.blank_state("2026-10-22")
        ps = st["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, shares_L=100, shares_B=-30, N_L=7000.0, side_prev=1, entry_px={"TQQQ": 70.0, "QQQ": 233.0})
        del ps["ent_L"], ps["ent_B"]                                                             # what an older build wrote
        json.dump(st, open(LV.state_path("dry"), "w"))
        loaded = e.load_state(pd.Timestamp("2026-10-22"))
        ps = loaded["pairs"]["TQQQ/QQQ"]
        self.assertEqual((ps["ent_L"], ps["ent_B"]), (70.0, 233.0))
        e.state = loaded
        e._book_shadow(ps, view(L=70.07, B=233.1, L_prev=70.0, B_prev=233.0, L_open_j=70.0, B_open_j=233.0), 5, 4)
        self.assertLess(abs(ps["pnl_usd"]), 10.0, "a bar of 7 cents books a few dollars, not the -$2,000 of a 1.0 entry price")
        self.assertFalse(e.loss_halt_check(sum(x["pnl_usd"] for x in loaded["pairs"].values()))[0])
        before = ps["pnl_usd"]
        e._book_shadow(ps, view(L=70.07, B=233.1, L_prev=70.0, B_prev=float("nan"), L_open_j=70.0, B_open_j=233.0), 6, 5)
        self.assertEqual(ps["pnl_usd"], before)
        self.assertTrue(np.isfinite(ps["pnl_usd"]))
        self.assertTrue(e.loss_halt_check(float("nan"))[0], "a P&L that is not a number must halt, not let a real loss through")

    def test_d2_bars_skipped_by_a_restart_are_caught_up_with_the_fill_price_of_the_position(self):
        e = self.engine(armed=False)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, side_prev=0, N_L=1000.0, ent_L=0.0, ent_B=0.0)                         # decided at the close of bar 10, then the engine was down for bars 11 and 12
        Lc = np.array([100.0] * 11 + [100.5, 100.8, 101.0]); Lo = np.array([100.0] * 11 + [100.2, 100.5, 100.8])
        B = np.full(14, 50.0)
        v = view(L=101.0, B=50.0, L_prev=100.8, B_prev=50.0, L_open_j=100.8, B_open_j=50.0, L_open_h=Lo, L_close_h=Lc, B_open_h=B, B_close_h=B)
        e._book_shadow(ps, v, 13, 10)
        self.assertAlmostEqual(ps["pnl_usd"], (101.0 / 100.2 - 1.0) * 1000.0, places=6)         # in at the open of bar 11 (next_open), marked at the close of bar 13
        self.assertAlmostEqual(ps["ent_L"], 100.2, places=9)

    def test_d3_the_exit_text_says_what_was_sent_and_at_what_price(self):
        tc = FakeTC(positions={"TQQQ": 60, "QQQ": -30})                                           # the broker holds only 60 of the 100 TQQQ
        e, ps = self.closable_pair(tc)
        row = {"time": "10:30"}
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, view(L=71.0, B=233.0), 1.0, 2.5e-4, row))
        self.assertEqual({r.symbol: int(r.qty) for r in tc.sent}, {"TQQQ": 60, "QQQ": 30})
        self.assertIn("SELL 60 TQQQ @ 100.00", row["action"]); self.assertIn("BUY 30 QQQ @ 100.00", row["action"])            # the fill price, not the 71.00 signal
        self.assertIn("(the ledger held +100 / -30)", row["action"])
        self.assertEqual((row["shares_L"], row["shares_B"]), (-60, 30))
        self.assertNotIn("dry run", row["action"])
        tc2 = FakeTC(positions={})                                                                # the broker is already flat: nothing is sent and the text says so
        e2, ps2 = self.closable_pair(tc2)
        row2 = {"time": "10:30"}
        self.assertTrue(e2.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps2, 0, view(L=71.0, B=233.0), 1.0, 2.5e-4, row2))
        self.assertEqual(tc2.sent, [])
        self.assertEqual(row2["action"], "EXIT: nothing to close at the broker (the ledger was cleared)")
        self.assertEqual((row2["shares_L"], row2["shares_B"]), (0, 0))
        tc3 = FakeTC()                                                                            # an ARMED entry shows the fill price, a dry-run entry says it shows the signal close
        e3 = self.engine(armed=True, tc=tc3)
        row3 = {"time": "10:00"}
        self.assertTrue(e3.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", e3.state["pairs"]["TQQQ/QQQ"], 1, view(L=99.0, B=101.0), 1.0, 2.5e-4, row3))
        self.assertIn(" TQQQ @ 100.00", row3["action"]); self.assertNotIn("99.00", row3["action"])
        row4 = {"time": "10:00"}
        e4 = self.engine(armed=False)
        self.assertTrue(e4.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", e4.state["pairs"]["TQQQ/QQQ"], 1, view(L=99.0, B=101.0), 1.0, 2.5e-4, row4))
        self.assertIn("TQQQ @ 99.00", row4["action"]); self.assertIn("dry run", row4["action"])

    def test_d3_a_halt_row_says_flat_in_the_position_column_and_the_decision_log_is_readable(self):
        full = D.load_pairdata(("TQQQ", "QQQ"), 5, start="2026-01-01")
        c = replace(self.c, s_open=1.5, min_edge_ratio=0.0, max_halflife_min=0.0)
        found = None
        for d in full.dates[-10:]:
            eng, rows = LV.replay_session(d, self.sc, c, LV.LiveSettings(feed="sip", daily_loss_halt=1e-7), equity=100_000_000.0)    # a loss of about $10 halts
            halted = rows[rows.note.fillna("") == "daily-loss halt"] if "note" in rows else rows.iloc[0:0]
            if len(halted):
                found = halted
                break
        self.assertIsNotNone(found, "one of the last ten sessions should trip a hair-trigger halt")
        self.assertTrue((found.pos_after == 0).all() and (found.position == "FLAT").all(), found[["time", "pair", "pos_after", "position", "note"]])
        e = self.engine(armed=False)                                                              # rows with different fields land in one readable file with one header
        e.append_log("decisions", [dict(date="2026-10-22", j=3, time="10:00", pair="TQQQ/QQQ", mode="dry", note="no signal", pos_before=0, pos_after=0)])
        e.append_log("decisions", [dict(date="2026-10-22", j=4, time="10:05", pair="TQQQ/QQQ", mode="dry", s=-2.5, action="ENTER long spread: BUY 5 TQQQ @ 70.00", position="LONG TQQQ 5 sh / SHORT QQQ 1 sh",
                                        shares_L=5, shares_B=-1, fills={"TQQQ": {"qty": 5}}, pos_before=0, pos_after=1, target=1, note="")])
        f = os.path.join(self.tmp, "levpairs_decisions_dry.csv")
        df = pd.read_csv(f)
        self.assertEqual(list(df.columns), LV.LOG_COLS["decisions"])
        self.assertEqual(len(df), 2)
        open(f, "w").write("date,j,time\n2026-10-21,1,10:00\n")                                   # a log of another layout is kept aside, not mixed
        e.append_log("decisions", [dict(date="2026-10-22", j=5, time="10:10", pair="TQQQ/QQQ", mode="dry", note="x")])
        self.assertEqual(list(pd.read_csv(f).columns), LV.LOG_COLS["decisions"])
        self.assertTrue([n for n in os.listdir(self.tmp) if n.startswith("levpairs_decisions_dry.csv.") and n.endswith(".old")])

    def test_d5_no_close_order_is_sent_while_the_market_is_closed_or_after_the_cutoff(self):
        tc = FakeTC(positions={"TQQQ": 47, "QQQ": -14})
        tc.market_open = False
        e = self.engine(armed=True, tc=tc)
        res = e.flatten_orphans()
        self.assertEqual(tc.sent, [])
        self.assertTrue(any("market is closed" in r["status"] for r in res), res)
        tc2 = FakeTC(positions={"TQQQ": 100, "QQQ": -30})                                          # KILL on a ledger pair with the market closed: nothing queues for the open, the pair stays and alarms
        tc2.market_open = False
        queued = [types.SimpleNamespace(id="q1", symbol="TQQQ", client_order_id="LP-20261022-TQQQQQQ-1")]
        tc2.get_orders = lambda: list(queued)
        e2, ps2 = self.closable_pair(tc2)
        e2.flatten("KILL")
        self.assertEqual(tc2.sent, [])
        self.assertEqual(ps2["side"], 1)
        self.assertTrue(ps2["broken"] and e2.state["alarms"])
        self.assertIn("q1", tc2.cancelled, "the engine's own queued order must not fill at the open")
        tc3 = FakeTC(positions={"TQQQ": 47, "QQQ": -14})                                           # past the last useful moment of today's session
        e3 = self.engine(armed=True, tc=tc3, flatten_until="00:00:01")
        e3.state["date"] = str(LV.clock_now().date())
        res3 = e3.flatten_orphans()
        self.assertEqual(tc3.sent, [])
        self.assertTrue(any("last useful moment" in r["status"] for r in res3), res3)

    def test_d5_an_orphan_close_that_does_not_fill_is_cancelled_not_left_to_queue(self):
        tc = FakeTC(positions={"TQQQ": 47})
        tc.fill_status = "new"
        open_ = []
        orig = tc.submit_order

        def submit(req):
            o = orig(req)
            open_.append(types.SimpleNamespace(id=o.id, symbol=req.symbol, client_order_id=req.client_order_id))
            return o
        tc.submit_order = submit
        tc.get_orders = lambda: list(open_)
        e = self.engine(armed=True, tc=tc, flatten_verify_s=0.3, flatten_attempts=2)
        e.flatten_orphans()
        self.assertEqual(len(tc.sent), 1, "the second attempt must not stack another order on the open one")
        self.assertEqual(tc.cancelled, ["o1"], "the unfilled orphan close is cancelled at the end")

    def test_d6_the_heartbeat_runs_inside_a_long_flatten(self):
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        tc.reject = {"QQQ"}
        e, ps = self.closable_pair(tc, flatten_attempts=3, flatten_verify_s=0.5)
        lf = os.path.join(self.tmp, "engine.lock")
        open(lf, "w").write("x")
        os.utime(lf, (1, 1))
        e._lockfile = lf
        e.flatten("KILL")
        self.assertGreater(os.path.getmtime(lf), time.time() - 60, "the lock must not look stale while a flatten is looping")
        e2 = self.engine(armed=True, tc=FakeTC(positions={"TQQQ": 47}))
        os.utime(lf, (1, 1))
        e2._lockfile = lf
        e2.flatten_orphans()
        self.assertGreater(os.path.getmtime(lf), time.time() - 60)

    def test_d7_sizing_never_overshoots_the_caps_by_a_share(self):
        rng = np.random.default_rng(7)
        e = self.engine(armed=False)
        n_ok = 0
        for _ in range(3000):
            eq = float(rng.uniform(20_000, 2_000_000))
            e.acct["equity"], e.deploy = eq, eq
            L, B = float(rng.uniform(20, 150)), float(rng.uniform(100, 700))
            sz, _ = e.size_pair("TQQQ/QQQ", float(rng.uniform(0.2, 1.0)), view(L=L, B=B))
            if sz is None:
                continue
            n_L, n_B, _ = sz
            n_ok += 1
            self.assertLessEqual(max(n_L * L, n_B * B), e.live.max_leg_nav * eq + 1e-9, (eq, L, B, n_L, n_B))
            self.assertLessEqual(n_L * L + n_B * B, min(e.live.max_gross_nav * eq, e.live.max_gross_x * eq) + 1e-9, (eq, L, B, n_L, n_B))
        self.assertGreater(n_ok, 500)


    # ── the order flow of the month-long paper test: both legs together, every order journaled, the broker's own record synced ───────────────────────────
    def read_journal(self, mode="paper") -> pd.DataFrame:
        f = os.path.join(self.tmp, f"levpairs_fills_{mode}.jsonl")
        return pd.DataFrame([json.loads(ln) for ln in open(f, encoding="utf-8")])

    def test_both_legs_are_sent_at_the_same_moment(self):
        tc = FakeTC()
        tc.delay = 0.3                                                                             # a slow API: one after the other the two submits would take 0.6 s
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        t0 = time.time()
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(L=70.0, B=700.0), 1.0, 2.5e-4, {"time": "10:00"}))
        took = time.time() - t0
        arr = dict(tc.arrivals)
        self.assertEqual(set(arr), {"TQQQ", "QQQ"})
        self.assertLess(abs(arr["TQQQ"] - arr["QQQ"]) / 1e9, 0.05, "the two orders reach the broker within milliseconds of each other")
        self.assertLess(took, 0.55, "the entry takes one round trip, not two")
        j = self.read_journal()
        self.assertLess(abs(int(j.t_submit_ns.iloc[0]) - int(j.t_submit_ns.iloc[1])) / 1e9, 0.05)

    def test_every_order_is_journaled_with_its_decision_and_timestamps(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        row = {"time": "10:00", "wall": "2026-10-22T10:00:04.000-04:00"}
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(L=99.0, B=101.0, s=-2.7), 1.0, 2.5e-4, row))
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, view(L=99.5, B=101.0, s=-0.2), 1.0, 2.5e-4, {"time": "10:30", "wall": "2026-10-22T10:30:04.000-04:00"}))
        j = self.read_journal()
        self.assertEqual(sorted(j.action), ["ENTER", "ENTER", "EXIT", "EXIT"])
        self.assertEqual(set(j.trip), {"2026-10-22-TQQQQQQ-1"}, "the entry and the exit of one round trip share a trip id")
        enter = j[j.action == "ENTER"].set_index("symbol")
        self.assertEqual((enter.loc["TQQQ", "side"], enter.loc["QQQ", "side"]), ("BUY", "SELL"))
        for col in ("order_id", "client_order_id", "t_submit_ns", "t_ack_ns", "t_seen_ns", "submitted_at", "filled_at", "px", "ref", "slip_bp", "decided_at", "s"):
            self.assertTrue(j[col].notna().all(), col)
        self.assertTrue((j.t_ack_ns >= j.t_submit_ns).all() and (j.t_seen_ns >= j.t_ack_ns).all())
        self.assertAlmostEqual(float(enter.loc["TQQQ", "slip_bp"]), (100.0 - 99.0) / 99.0 * 1e4, places=5)
        self.assertEqual(list(pd.read_csv(os.path.join(self.tmp, "levpairs_fills_paper.csv")).columns), LV.LOG_COLS["fills"])

    def test_rejected_unknown_and_unwound_orders_are_journaled_too(self):
        tc = FakeTC()
        tc.reject = {"QQQ"}
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertFalse(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        j = self.read_journal()
        st = {(r.action, r.symbol): r.status for r in j.itertuples()}
        self.assertEqual(st[("ENTER", "TQQQ")], "filled")
        self.assertEqual(st[("ENTER", "QQQ")], "rejected")
        self.assertEqual(st[("UNWIND", "TQQQ")], "sent")
        self.assertIn("not shortable", j[(j.action == "ENTER") & (j.symbol == "QQQ")].note.iloc[0])
        tc2 = FakeTC()                                                                             # a submit call that never returns: both legs are 'unknown', nothing is unwound
        tc2.delay = 1.5
        e2 = self.engine(armed=True, tc=tc2, submit_timeout_s=0.3, resolve_unknown=False)
        ps2 = e2.state["pairs"]["TQQQ/QQQ"]
        self.assertFalse(e2.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps2, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        self.assertTrue(ps2["broken"] and e2.state["alarms"])
        self.assertEqual(sorted(self.read_journal().status.tolist()).count("unknown"), 2, "both legs of the entry whose submit never returned are journaled as unknown")

    def test_an_open_excel_window_cannot_cost_a_journal_row(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        real = pd.DataFrame.to_csv

        def locked(self_, *a, **k):
            raise PermissionError("the file is open in another program")
        pd.DataFrame.to_csv = locked
        try:
            ps = e.state["pairs"]["TQQQ/QQQ"]
            self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        finally:
            pd.DataFrame.to_csv = real
        self.assertEqual(len(self.read_journal()), 2, "the rows are in the jsonl")
        self.assertTrue(any("Excel" in m for _, m in self.logs))

    def test_the_brokers_record_of_the_day_is_synced_and_written(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(), 1.0, 2.5e-4, {"time": "10:00"}))
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, view(), 1.0, 2.5e-4, {"time": "10:30"}))
        tc._orders["x"] = types.SimpleNamespace(id="x", client_order_id="SOMEONE-ELSE-1", symbol="AAPL", qty=5, side=types.SimpleNamespace(value="buy"), status="filled", filled_qty=5, filled_avg_price=200.0)
        tc.activities = [dict(order_id="o1", symbol="TQQQ", side="buy", qty="357", price="100.0", transaction_time="2026-10-22T14:00:00.2Z", type="fill", cum_qty="357", leaves_qty="0"),
                         dict(order_id="x", symbol="AAPL", side="buy", qty="5", price="200.0", transaction_time="2026-10-22T14:00:00.3Z", type="fill", cum_qty="5", leaves_qty="0")]
        out = e.sync_broker_day("2026-10-22")
        self.assertEqual(len(out["orders"]), 4)
        self.assertTrue(out["orders"].client_order_id.str.startswith("LP-").all(), "only this engine's orders")
        self.assertEqual(list(out["fills"].order_id), ["o1"])
        for k in ("orders", "fills"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, f"levpairs_broker_{k}_paper_20261022.csv")))

    # ── RM review of the paper-test machinery ───────────────────────────────────────────────────────────────────────────────────────────────────────
    def book_then_raise(self, tc, sym, exc):
        """submit_order for ``sym`` books the order at the broker and THEN raises: what alpaca-py's retry of a POST that had succeeded, or a timeout after the broker accepted the order, looks like."""
        orig = tc.submit_order

        def submit(req):
            o = orig(req)
            if req.symbol == sym:
                raise exc
            return o
        tc.submit_order = submit

    def raise_before_booking(self, tc, sym, exc):
        orig = tc.submit_order

        def submit(req):
            if req.symbol == sym:
                raise exc
            return orig(req)
        tc.submit_order = submit

    def entry(self, e, row=None):
        ps = e.state["pairs"]["TQQQ/QQQ"]
        return e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 1, view(L=70.0, B=700.0), 1.0, 2.5e-4, row if row is not None else {"time": "10:00"}), ps

    def test_d1_a_retried_post_that_had_succeeded_is_not_a_rejection(self):
        tc = FakeTC()
        self.book_then_raise(tc, "QQQ", FakeAPIError("client_order_id must be unique", 422))
        e = self.engine(armed=True, tc=tc)
        ok, ps = self.entry(e)
        self.assertTrue(ok, "the order exists at the broker: the entry goes on with it")
        self.assertEqual(len(tc.sent), 2, "nothing was unwound")
        self.assertEqual(ps["side"], 1)
        self.assertEqual(set(self.read_journal().status), {"filled"})
        self.assertTrue(any("found by its client id" in m for _, m in self.logs))

    def test_d1_a_timeout_after_the_broker_accepted_the_order_continues_with_it(self):
        import requests
        tc = FakeTC()
        self.book_then_raise(tc, "TQQQ", requests.exceptions.ReadTimeout("read timed out"))
        e = self.engine(armed=True, tc=tc)
        ok, ps = self.entry(e)
        self.assertTrue(ok)
        self.assertEqual(len(tc.sent), 2)

    def test_d1_a_timeout_before_the_broker_got_it_is_a_rejection_and_the_other_leg_is_unwound(self):
        import requests
        tc = FakeTC()
        self.raise_before_booking(tc, "QQQ", requests.exceptions.ConnectTimeout("connect timed out"))
        e = self.engine(armed=True, tc=tc)
        ok, ps = self.entry(e)
        self.assertFalse(ok)
        self.assertEqual(sorted((r.symbol, str(r.side.value)) for r in tc.sent), [("TQQQ", "buy"), ("TQQQ", "sell")], "the leg that is not at the broker is not there; the filled one is closed again")
        self.assertIsNone(e.state["pending"], "nothing is in doubt")
        self.assertEqual(tc.lookups, 1)

    def test_d1_when_even_the_lookup_fails_the_leg_is_unknown_and_the_intent_stays_on_disk(self):
        import requests
        tc = FakeTC()
        self.book_then_raise(tc, "QQQ", requests.exceptions.ReadTimeout("read timed out"))

        def lookup_down(cid):
            raise ConnectionError("api down")
        tc.get_order_by_client_id = lookup_down
        e = self.engine(armed=True, tc=tc, resolve_unknown=False, lookup_retry_s=0.0)
        ok, ps = self.entry(e)
        self.assertFalse(ok)
        self.assertEqual(sorted(r.symbol for r in tc.sent), ["QQQ", "TQQQ"], "nothing was unwound: a leg in doubt must not be answered with an order on the other one")
        self.assertTrue(ps["broken"] and e.state["alarms"])
        self.assertTrue(e.state["pending"]["unknown"])
        self.assertTrue(json.load(open(LV.state_path("paper")))["pending"]["unknown"], "on disk: a restart refuses to start over it")
        e._clear_pending()
        self.assertIsNotNone(e.state["pending"], "an unknown order is not forgotten by the ordinary clean-up")
        j = self.read_journal().set_index("symbol")
        self.assertEqual(j.loc["QQQ", "status"], "unknown")
        self.assertTrue(str(j.loc["QQQ", "client_order_id"]).startswith("LP-"), "the client id of the order in doubt is journaled")
        e.state["pairs"]["TQQQ/QQQ"].update(side=0)                                          # the ledger is flat, the broker holds both legs: the verified orphan flatten closes them and only then clears the intent
        e.flatten_orphans()
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})
        self.assertIsNone(e.state["pending"])

    def test_d1_a_plain_4xx_is_a_rejection_without_a_lookup(self):
        tc = FakeTC()
        tc.reject = {"QQQ"}
        e = self.engine(armed=True, tc=tc)
        ok, ps = self.entry(e)
        self.assertFalse(ok)
        self.assertEqual(getattr(tc, "lookups", 0), 0)
        self.assertEqual(self.read_journal().set_index("symbol").loc["QQQ", "status"], "rejected")

    def test_d1_a_submit_that_hangs_leaves_a_known_intent_and_the_late_order_is_closed_by_the_orphan_flatten(self):
        tc = FakeTC()
        tc.delay = 1.0
        e = self.engine(armed=True, tc=tc, submit_timeout_s=0.3, fill_wait_s=0.2, resolve_unknown=False)
        ok, ps = self.entry(e)
        self.assertFalse(ok)
        self.assertTrue(e.state["pending"]["unknown"])
        j = self.read_journal()
        self.assertEqual(set(j.status), {"unknown"})
        self.assertTrue(j.client_order_id.notna().all(), "the client ids of the orders in doubt are journaled")
        time.sleep(1.2)                                                                       # the hung calls come back: both orders land at the broker, the ledger is flat
        self.assertEqual({k: abs(v) for k, v in tc.positions.items() if v}.keys(), {"TQQQ", "QQQ"})
        e.flatten_orphans()
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})
        self.assertIsNone(e.state["pending"])

    def test_d2_leaving_a_session_with_an_order_in_flight_closes_what_it_left(self):
        tc = FakeTC(positions={"TQQQ": 357, "QQQ": -107})                                    # the entry went out, Ctrl+C came before the ledger heard of it
        e = self.engine(armed=True, tc=tc, fill_wait_s=0.1)
        e.state["pending"] = dict(pair="TQQQ/QQQ", closing=False, legs=[["TQQQ", 357, True], ["QQQ", 107, False]], ts="x")
        e._leave_session()
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {})
        self.assertIsNone(e.state["pending"])
        tc2 = FakeTC(positions={"SPY": 50})                                                  # no intent in flight: another strategy's SPY is none of its business
        e2 = self.engine(armed=True, tc=tc2, fill_wait_s=0.1)
        e2._leave_session()
        self.assertEqual(tc2.sent, [])

    def test_d3_every_http_call_gets_a_default_timeout(self):
        import requests
        from requests.adapters import HTTPAdapter
        seen = []
        orig = HTTPAdapter.send

        def spy(self_, request, **kw):
            seen.append(kw.get("timeout"))
            r = requests.Response()
            r.status_code, r._content, r.url = 200, b"{}", request.url
            return r
        HTTPAdapter.send = spy
        try:
            client = D.harden(types.SimpleNamespace(_session=requests.Session()))
            client._session.get("http://alpaca.invalid/x")
            client._session.get("http://alpaca.invalid/x", timeout=3)
            client._session.get("https://alpaca.invalid/x")
        finally:
            HTTPAdapter.send = orig
        self.assertEqual(seen, [(5.0, 20.0), 3, (5.0, 20.0)], "a default for a call that sets none, the caller's own timeout left alone")
        self.assertEqual(type(D.data_client()._session.get_adapter("https://data.alpaca.markets")).__name__, "TimeoutAdapter")

    def test_the_clock_skew_between_the_pc_and_the_broker_is_measured_and_judged(self):
        import alpaca.trading.client as atc
        self.assertEqual([LV.LiveEngine._clock_check(x).level for x in (0.4, -1.9, 3.0, -8.0, 45.0)], ["ok", "ok", "warn", "warn", "fail"])
        tc = FakeTC()
        tc.get_clock = lambda: types.SimpleNamespace(is_open=True, next_open=pd.Timestamp.now(tz="UTC"), next_close=pd.Timestamp.now(tz="UTC"), timestamp=pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=5))
        real_cls, real_creds = atc.TradingClient, D.creds
        atc.TradingClient, D.creds = (lambda k, s, paper=True: tc), (lambda paper=True: ("k", "s"))
        try:
            e = self.engine(armed=True, tc=None)
            acct = e.connect()
        finally:
            atc.TradingClient, D.creds = real_cls, real_creds
        self.assertAlmostEqual(acct["clock_skew_s"], 5.0, delta=1.5)

    def test_a_save_that_windows_refuses_for_a_moment_is_retried(self):
        e = self.engine(armed=False)
        real, calls = os.replace, {"n": 0}

        def flaky(a, b):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError("the file is open in another program")
            return real(a, b)
        os.replace = flaky
        try:
            e.save_state(e.state)
        finally:
            os.replace = real
        self.assertEqual(calls["n"], 3)
        self.assertTrue(os.path.exists(LV.state_path("dry")))
        os.replace = lambda a, b: (_ for _ in ()).throw(PermissionError("locked"))
        try:
            with self.assertRaises(PermissionError):
                e.save_state(e.state)
        finally:
            os.replace = real

    def test_a_second_process_cannot_flatten_beside_the_first(self):
        tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30})
        A = self.engine(armed=True, tc=tc, flatten_join_s=10.0)
        B = self.engine(armed=True, tc=tc, flatten_join_s=0.6)                               # the watchdog: the same files, another process
        for e in (A, B):
            e.state["pairs"]["TQQQ/QQQ"].update(side=1, shares_L=100, shares_B=-30, N_L=7000.0)
        active, peak = {"n": 0}, {"n": 0}
        for e in (A, B):
            orig = e._flatten

            def wrap(reason, persist=False, _o=orig):
                active["n"] += 1
                peak["n"] = max(peak["n"], active["n"])
                try:
                    time.sleep(1.2)
                    return _o(reason, persist)
                finally:
                    active["n"] -= 1
            e._flatten = wrap
        out = {}
        a = threading.Thread(target=lambda: out.setdefault("a", A.flatten("end of day")))
        a.start()
        time.sleep(0.3)
        out["b"] = B.flatten("watchdog")
        a.join(20)
        self.assertEqual(peak["n"], 1, "never two closers on one account, even from two processes")
        self.assertIn("another process", out["b"][0]["status"])
        self.assertEqual(sorted((r.symbol, str(r.side.value), int(r.qty)) for r in tc.sent), [("QQQ", "buy", 30), ("TQQQ", "sell", 100)], "closed exactly once")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "levpairs_flatten_paper.lock")), "the lock is released")
        lock = os.path.join(self.tmp, "levpairs_flatten_paper.lock")                           # a dead closer's lock does not block for ever
        open(lock, "w").write("1234 x")
        os.utime(lock, (1, 1))
        tc.positions = {"SPXL": 5}
        C = self.engine(armed=True, tc=tc)
        C.state["pairs"]["SPXL/SPY"].update(side=1, shares_L=5, shares_B=0, N_L=1000.0)
        C.flatten("KILL")
        self.assertEqual(tc.positions.get("SPXL", 0), 0)

    def test_a_revived_engine_honours_the_watchdogs_flatten(self):
        tc = FakeTC(positions={})
        e = self.engine(armed=True, tc=tc)
        e.state["date"] = "2026-10-22"
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, shares_L=100, shares_B=-30)                                        # the hung engine still believes it holds this
        flat = e.blank_state("2026-10-22")                                                    # ... but the watchdog flattened and saved a flat ledger
        e.save_state(flat)
        e.state = {**e.state, "pairs": {**e.state["pairs"]}}
        json.dump(dict(date="2026-10-22", at="x"), open(os.path.join(self.tmp, "levpairs_watchdog_paper.json"), "w"))
        e._honour_watchdog()
        self.assertTrue(e.halted)
        self.assertEqual(e.state["pairs"]["TQQQ/QQQ"]["side"], 0, "the flat ledger is reloaded from disk")
        self.assertTrue(e.state["halted"])
        e._honour_watchdog()                                                                  # once only
        self.assertTrue(any("watchdog flattened" in m for _, m in self.logs))

    def test_bars_that_passed_during_a_restart_still_age_the_position(self):
        full = D.load_pairdata(("TQQQ", "QQQ"), 5, start="2026-01-01")
        d = full.dates[-3]
        outs = []
        for last_j in (30, 39):                                                               # a gap of 9 bars, then no gap
            e = self.engine(armed=False)
            e.state = e.blank_state(d)
            e.deploy, e.G = 1e6, 1.0
            ps = e.state["pairs"]["TQQQ/QQQ"]
            ps.update(side=1, pos=1, age=20, shares_L=100, shares_B=-30, N_L=7000.0, side_prev=1, ent_L=70.0, ent_B=700.0, entry_px={"TQQQ": 70.0, "QQQ": 700.0})
            e.state["last_j"] = last_j
            e.source = types.SimpleNamespace(feed="sip")
            real = LV.pair_views
            LV.pair_views = lambda *a, **k: ({"TQQQ/QQQ": LV.PairView("TQQQ/QQQ", 40, -1.0, 10.0, 5.0, 3.0, 70.0, 700.0, 0.0, 0.0, 0.0, 0, 0.0, False, np.array([-1.0]), L_prev=70.0, B_prev=700.0, L_open_j=70.0, B_open_j=700.0)}, {})
            try:
                rows = e.step(d, d + pd.Timedelta(minutes=D.OPEN_MIN + 41 * 5, seconds=4))
            finally:
                LV.pair_views = real
            outs.append([r for r in rows if r["pair"] == "TQQQ/QQQ"][0])
        self.assertEqual(outs[0]["target"], 0, "20 bars held + 9 bars of gap = 29 >= the 24-bar time stop: out")
        self.assertEqual(outs[1]["target"], 1, "the same position without the gap is still inside the time stop")

    def test_each_entry_attempt_has_its_own_trip_id_and_the_exit_uses_the_one_of_its_entry(self):
        tc = FakeTC()
        tc.reject = {"QQQ"}
        e = self.engine(armed=True, tc=tc)
        ok1, ps = self.entry(e)
        tc.reject = set()
        ok2, ps = self.entry(e)
        self.assertEqual((ok1, ok2), (False, True))
        self.assertTrue(e.trade_pair(("TQQQ", "QQQ"), "TQQQ/QQQ", ps, 0, view(L=70.0, B=700.0), 1.0, 2.5e-4, {"time": "10:30"}))
        j = self.read_journal()
        self.assertEqual(sorted(set(j[j.action.isin(["ENTER", "UNWIND"])].trip)), ["2026-10-22-TQQQQQQ-1", "2026-10-22-TQQQQQQ-2"])
        self.assertEqual(set(j[j.action == "EXIT"].trip), {"2026-10-22-TQQQQQQ-2"})

    # ── RM re-verification: N1 (leave a session: only the legs in flight), N2 (a slow submit), N3 (a revived engine mid-step), N4 (the OS lock), N7 (an order in doubt is cleaned up) ────
    def test_n1_leaving_a_session_closes_only_the_legs_of_the_entry_in_flight(self):
        tc = FakeTC(positions={"TQQQ": 357, "QQQ": -107, "SPY": 25})                           # SPY is somebody else's
        e = self.engine(armed=True, tc=tc, fill_wait_s=0.1)
        e.state["pending"] = dict(pair="TQQQ/QQQ", closing=False, legs=[["TQQQ", 357, True], ["QQQ", 107, False]], ts="x")
        e._leave_session()
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {"SPY": 25})
        self.assertIsNone(e.state["pending"])

    def test_n2_leaving_a_session_waits_for_a_submit_call_that_is_still_out(self):
        tc = FakeTC()
        tc.delays = {"QQQ": 2.0}                                                               # the hedge leg's call is slow: it lands 2 s after the interrupt
        e = self.engine(armed=True, tc=tc, fill_wait_s=0.1, submit_timeout_s=6.0)
        legs = [("TQQQ", 357, True), ("QQQ", 107, False)]
        e.state["pending"] = dict(pair="TQQQ/QQQ", closing=False, legs=[[s_, q, b] for s_, q, b in legs], ts="x")
        reqs = [(s_, q, b, f"LP-X-{s_}") for s_, q, b in legs]
        th = threading.Thread(target=lambda: e._submit_all(reqs), daemon=True)                  # what an interrupted send leaves behind: submit threads still running
        th.start()
        time.sleep(0.3)
        t0 = time.time()
        e._leave_session()
        self.assertGreater(time.time() - t0, 1.0, "it waited for the slow submit")
        self.assertEqual({k: v for k, v in tc.positions.items() if v}, {}, "the late leg was closed, not left naked")
        self.assertIsNone(e.state["pending"])
        tc2 = FakeTC()                                                                         # a call that never comes back inside the wait: the intent stays on disk
        tc2.delays = {"QQQ": 3.0}
        e2 = self.engine(armed=True, tc=tc2, fill_wait_s=0.1, submit_timeout_s=0.4)
        e2.state["pending"] = dict(pair="TQQQ/QQQ", closing=False, legs=[[s_, q, b] for s_, q, b in legs], ts="x")
        th2 = threading.Thread(target=lambda: e2._submit_all([(s_, q, b, f"LP-Y-{s_}") for s_, q, b in legs]), daemon=True)
        th2.start()
        time.sleep(0.2)
        e2._leave_session()
        self.assertTrue(e2.state["pending"]["unknown"])

    def test_n3_a_revived_engine_cannot_open_a_pair_or_overwrite_the_flat_ledger_mid_step(self):
        tc = FakeTC()
        e = self.engine(armed=True, tc=tc)
        e.state["date"] = "2026-10-22"
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=1, shares_L=100, shares_B=-30)                                         # what the sleeping engine still believes
        held = e.state                                                                          # the running step holds a reference to this very dict
        e.save_state(e.blank_state("2026-10-22"))                                              # the watchdog's flat ledger on disk
        json.dump(dict(date="2026-10-22", at="x"), open(os.path.join(self.tmp, "levpairs_watchdog_paper.json"), "w"))
        row = {"time": "10:00"}
        self.assertFalse(e.trade_pair(("SPXL", "SPY"), "SPXL/SPY", e.state["pairs"]["SPXL/SPY"], 1, view("SPXL/SPY", L=70.0, B=700.0), 1.0, 2.5e-4, row), "no new pair after the watchdog flattened")
        self.assertEqual(tc.sent, [])
        self.assertIn("watchdog", row["note"])
        self.assertIs(e.state, held, "the ledger was reloaded IN PLACE: the step's own reference sees the flat one")
        self.assertEqual(held["pairs"]["TQQQ/QQQ"]["side"], 0)
        e.save_state(held)                                                                       # what the end of the step does
        self.assertEqual(json.load(open(LV.state_path("paper")))["pairs"]["TQQQ/QQQ"]["side"], 0, "the flat ledger was not overwritten with the stale one")
        self.assertTrue(json.load(open(LV.state_path("paper")))["halted"])

    def test_n4_the_cross_process_lock_is_an_os_lock_that_cannot_be_stolen_or_guessed_stale(self):
        A, B = self.engine(armed=True, tc=FakeTC()), self.engine(armed=True, tc=FakeTC())
        lock = os.path.join(self.tmp, "levpairs_flatten_paper.lock")
        open(lock, "w").write("999 a long dead closer")                                        # a file left by a crashed process: nobody holds the OS lock, so it is free at once
        os.utime(lock, (1, 1))
        t0 = time.time()
        self.assertTrue(A._xlock_acquire(2.0))
        self.assertLess(time.time() - t0, 0.5)
        os.utime(lock, (1, 1))                                                                 # however old the file looks, a HELD lock is respected: nobody steals it
        t0 = time.time()
        self.assertFalse(B._xlock_acquire(0.8))
        self.assertTrue(0.7 <= time.time() - t0 < 1.6, "it waits for its timeout and gives up - no busy loop, no double hold")
        A._xlock_release()
        self.assertTrue(B._xlock_acquire(2.0), "released: the next closer gets it")
        B._xlock_release()
        wins = []                                                                              # two closers racing for a free lock: exactly one wins, every time
        for k in range(30):
            a, b = self.engine(armed=True, tc=FakeTC()), self.engine(armed=True, tc=FakeTC())
            gate, res = threading.Barrier(2), {}

            def go(name, eng):
                gate.wait()
                res[name] = eng._xlock_acquire(0.0)
            ts_ = [threading.Thread(target=go, args=("a", a)), threading.Thread(target=go, args=("b", b))]
            [t.start() for t in ts_]
            [t.join() for t in ts_]
            wins.append(sum(res.values()))
            a._xlock_release()
            b._xlock_release()
        self.assertEqual(set(wins), {1})

    def test_n7_an_order_in_doubt_is_looked_up_three_times_and_cleaned_up_in_the_same_step(self):
        import requests
        tc = FakeTC()
        self.book_then_raise(tc, "QQQ", requests.exceptions.ReadTimeout("read timed out"))
        calls = {"n": 0}
        real = tc.get_order_by_client_id

        def flaky(cid):                                                                        # the lookup fails twice, then answers
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ConnectionError("api down")
            return real(cid)
        tc.get_order_by_client_id = flaky
        e = self.engine(armed=True, tc=tc, lookup_retry_s=0.0)
        ok, ps = self.entry(e)
        self.assertTrue(ok, "the third try found the order: the entry goes on")
        self.assertEqual(calls["n"], 3)
        tc2 = FakeTC()                                                                         # the lookup never answers: unknown -> the same step waits, cancels, closes what both legs left
        self.book_then_raise(tc2, "QQQ", requests.exceptions.ReadTimeout("read timed out"))

        def down(cid):
            raise ConnectionError("api down")
        tc2.get_order_by_client_id = down
        e2 = self.engine(armed=True, tc=tc2, lookup_retry_s=0.0, unknown_resolve_wait_s=0.1)
        ok, ps = self.entry(e2)
        self.assertFalse(ok)
        self.assertEqual({k: v for k, v in tc2.positions.items() if v}, {}, "nothing stays naked until 15:52")
        self.assertIsNone(e2.state["pending"], "and the intent is cleared only because the broker is verified flat")
        self.assertTrue(ps["broken"], "the pair stays BROKEN until someone has looked at it")
        self.assertTrue(any("in doubt" in m for _, m in self.logs))

    # ── RM final pass: N1 (orphan flatten needs the step lock), N2 (broker-call budget), N3 (old state after an exit decision, reset_broken), clock-read fallback ──────────
    def test_n1_the_orphan_flatten_does_not_go_ahead_without_the_step_lock(self):
        tc = FakeTC(positions={"TQQQ": 357})                                                       # the first leg of an entry is on the broker, the ledger learns about it when the step ends
        e = self.engine(armed=True, tc=tc, kill_lock_wait_s=0.2)
        release, taken = threading.Event(), threading.Event()

        def hold():
            with e._lock:
                taken.set()
                release.wait(10)
        t = threading.Thread(target=hold)
        t.start()
        self.assertTrue(taken.wait(5))
        res = e.flatten_orphans()
        release.set(); t.join(5)
        self.assertEqual(tc.sent, [], "closing the first leg of an entry in flight leaves the second as a naked hedge")
        self.assertTrue(any("step is running" in r["status"] for r in res), res)
        self.assertEqual(tc.positions["TQQQ"], 357)
        e.flatten_orphans()                                                                         # between bars the same call works
        self.assertEqual([(r.symbol, int(r.qty)) for r in tc.sent], [("TQQQ", 357)])

    def never_filling_flatten(self, **kw):
        """Three pairs open, a broker whose close orders never fill: the flatten loops for 40 attempts (about 11 virtual minutes). Returns (peak broker calls in any 60 s, orders sent)."""
        vc, real, stamps = VClock(), LV.time, []
        LV.time = vc
        try:
            tc = FakeTC(positions={"TQQQ": 100, "QQQ": -30, "SOXL": 50, "SMH": -20, "SPXL": 80, "SPY": -25})
            tc.fill_status = "new"
            open_ = []
            orig = tc.submit_order

            def submit(req):
                o = orig(req)
                open_.append(types.SimpleNamespace(id=o.id, symbol=req.symbol, client_order_id=req.client_order_id))
                return o

            def cancel(oid):
                tc.cancelled.append(oid)
                open_[:] = [o for o in open_ if o.id != oid]
            tc.submit_order, tc.get_orders, tc.cancel_order_by_id = submit, (lambda: list(open_)), cancel
            for name in ("get_orders", "get_all_positions", "submit_order", "cancel_order_by_id", "get_clock"):
                def wrap(fn):
                    def f(*a, **k):
                        stamps.append(vc.t)
                        return fn(*a, **k)
                    return f
                setattr(tc, name, wrap(getattr(tc, name)))
            e = self.engine(armed=True, tc=tc, flatten_attempts=40, flatten_verify_s=12.0, flatten_backoff_s=5.0, **kw)
            for name, (a, b) in {"TQQQ/QQQ": (100, -30), "SOXL/SMH": (50, -20), "SPXL/SPY": (80, -25)}.items():
                e.state["pairs"][name].update(side=1, shares_L=a, shares_B=b)
            e.flatten("KILL")
            self.assertTrue(all(ps["broken"] for ps in e.state["pairs"].values()))
            return peak_per_minute(stamps), len(tc.sent)
        finally:
            LV.time = real

    def test_n2_a_flatten_that_cannot_close_stays_under_the_broker_call_budget(self):
        free_peak, free_sent = self.never_filling_flatten(max_calls_per_min=0, flatten_poll_s=0.5)       # no limit, the old polling: what RM measured (177 a minute)
        peak, sent = self.never_filling_flatten()                                                         # the defaults: 120 a minute, 1 s polling
        self.assertGreater(free_peak, 140, "the scenario must be heavy enough to show the problem")
        self.assertLessEqual(peak, 120)
        self.assertGreater(sent, 0)

    def test_n2_the_throttle_holds_any_60_second_window_under_the_limit(self):
        vc, real = VClock(), LV.time
        LV.time = vc
        try:
            e = self.engine(armed=True, tc=FakeTC(), max_calls_per_min=50)
            stamps = []
            for _ in range(500):
                e._throttle()
                stamps.append(vc.t)
            self.assertLessEqual(peak_per_minute(stamps), 50)
            self.assertGreaterEqual(vc.t - 1_000_000.0, 500 / 50 * 60 - 120)                              # 500 calls at 50 a minute take about nine virtual minutes, not none
        finally:
            LV.time = real

    def test_n4_a_pc_clock_stepped_back_does_not_stall_the_throttle(self):
        vc, real = VClock(), LV.time
        LV.time = vc
        try:
            e = self.engine(armed=True, tc=FakeTC(), max_calls_per_min=5)
            for _ in range(5):
                e._throttle()
            vc.wall -= 3600.0                                                                      # the PC clock is set back an hour with the whole budget spent
            t0 = vc.t
            e._throttle()
            self.assertLessEqual(vc.t - t0, 61.0, "the wait is at most one window, not an hour")
        finally:
            LV.time = real

    def test_n3_an_old_state_saved_after_an_exit_decision_books_the_move_not_a_bogus_loss(self):
        e = self.engine(armed=False)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps.update(side=0, side_prev=1, N_L=7000.0, ent_L=0.0, ent_B=0.0)                           # the exit was decided, the previous build kept no entry prices
        e._book_shadow(ps, view(L=70.04, B=233.0, L_prev=70.0, B_prev=233.0, L_open_j=70.05, B_open_j=233.0), 6, 5)
        self.assertAlmostEqual(ps["pnl_usd"], 7000.0 * (70.05 / 70.0 - 1.0), delta=0.01)             # closed at the open: +$5, not the +$350 of a 1.0 entry price

    def test_n3_clearing_the_broken_flags_reads_the_broker_first(self):
        e = self.engine(armed=True, tc=None)
        ps = e.state["pairs"]["TQQQ/QQQ"]
        ps["broken"] = True
        with self.assertRaises(RuntimeError) as cm:
            e.reset_broken()
        self.assertIn("connect first", str(cm.exception))
        self.assertTrue(ps["broken"])
        e.tc = FakeTC(positions={"TQQQ": 5})                                                          # the broker still holds something the ledger does not explain
        with self.assertRaises(RuntimeError) as cm:
            e.reset_broken()
        self.assertIn("still differ", str(cm.exception))
        self.assertTrue(ps["broken"])
        e.tc.positions = {}
        e.reset_broken()
        self.assertFalse(ps["broken"])

    def test_the_market_hours_fallback_when_the_broker_clock_cannot_be_read(self):
        tc = FakeTC(positions={"TQQQ": 47})

        def down():
            raise ConnectionError("clock endpoint down")
        tc.get_clock = down
        e = self.engine(armed=True, tc=tc)
        real = LV.clock_now
        try:
            for stamp, want in (("2026-10-20 11:00", True), ("2026-10-20 15:59:00", True), ("2026-10-20 15:59:45", False), ("2026-10-20 09:29", False), ("2026-10-20 17:00", False), ("2026-10-24 11:00", False)):   # Tue ... Sat
                LV.clock_now = lambda st=stamp: pd.Timestamp(st, tz=LV.NY)
                ok, why = e._closes_allowed()
                self.assertEqual(ok, want, (stamp, why))
            LV.clock_now = lambda: pd.Timestamp("2026-10-20 17:00", tz=LV.NY)
            e.flatten_orphans()
            self.assertEqual(tc.sent, [], "with the clock endpoint down at 17:00 nothing is sent")
            e.flatten("KILL")
            self.assertEqual(tc.sent, [])
        finally:
            LV.clock_now = real


class TestGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["LEVPAIRS_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(), "t.ini")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        from supertrend_ui.theme import apply_theme
        apply_theme(cls.app)

    def test_window_builds_and_settings_roundtrip(self):
        from levpairs_ui.main_window import MainWindow
        w = MainWindow()
        w.show()
        sc, c = w.settings.sig_cfg(), w.settings.sim_cfg()
        self.assertEqual((sc.tf, sc.spread, c.fill), (5, "reset", "next_open"))
        self.assertFalse(w.live.armed.isChecked(), "must start DRY RUN")
        self.assertTrue(w.live.paper(), "must start on PAPER")
        self.assertEqual(c.pair_w, (1.0, 1.0, 1.0))
        w.live.armed.setChecked(True)
        self.assertTrue(w.live.live_settings().armed)
        w.live.account.setCurrentIndex(1)
        self.assertFalse(w.live.armed.isChecked(), "switching to LIVE must disarm")
        self.assertFalse(w.live.live_settings().armed)
        w.live.account.setCurrentIndex(0)
        w.settings.w["s_open"].setValue(1.6)
        w.settings.w["tf"].setCurrentText("15")
        w.settings.w["w_soxl"].setValue(0.0)
        self.assertAlmostEqual(w.settings.sim_cfg().s_open, 1.6)
        self.assertEqual(w.settings.sig_cfg().tf, 15)
        self.assertEqual(w.settings.sim_cfg().pair_w[1], 0.0)
        v = w.settings.values()
        w.settings.set_values({**v, "s_open": 1.1, "tf": 30})
        self.assertAlmostEqual(w.settings.sim_cfg().s_open, 1.1)
        self.assertEqual(w.settings.sig_cfg().tf, 30)
        w.close()

    def test_controls_lock_while_a_session_runs_and_the_status_follows_the_engine(self):
        from levpairs_ui.main_window import MainWindow
        w = MainWindow()
        w.show()
        lv = w.live
        self.assertEqual(lv.effective_mode(), (False, True))
        lv.armed.setChecked(True)
        self.assertEqual(lv.effective_mode(), (True, True))
        lv._set_locked(True)
        for ctl in (lv.account, lv.armed, lv.feed, lv.accept_iex, lv.eq_frac, lv.max_gross, lv.max_leg):
            self.assertFalse(ctl.isEnabled())
        lv._set_locked(False)
        self.assertTrue(lv.account.isEnabled())
        lv.armed.setChecked(False)
        ls = lv.live_settings()
        self.assertEqual((ls.max_gross_nav, ls.max_leg_nav, ls.accept_iex), (0.25, 0.10, False))
        w.close()

    def test_d4_kill_and_flatten_never_act_through_a_replay_engine(self):
        from levpairs_ui.main_window import MainWindow
        w = MainWindow()
        w.show()
        lv = w.live
        eng = LV.LiveEngine(w.settings.sig_cfg(), w.settings.sim_cfg(), LV.LiveSettings(feed="sip"))
        eng.is_replay = True                                                                      # what replay_session sets on the engine it returns
        lv.eng = eng
        self.assertTrue(eng.is_replay)
        lv.armed.setChecked(True)                                                                 # PAPER + ARMED in the controls, a replay engine (unarmed, no broker) left from before
        e = lv.engine_for_action()
        self.assertIsNot(e, eng)
        self.assertTrue(e.live.armed and e.live.paper and not e.is_replay)
        lv.eng = e
        self.assertIs(lv.engine_for_action(), e, "an engine of the same account and mode is reused")
        lv.armed.setChecked(False)
        self.assertIsNot(lv.engine_for_action(), e, "a different mode gets a fresh engine")
        lv._show_rows(pd.DataFrame([dict(time="10:00", pair="TQQQ/QQQ", action="ENTER long spread: BUY 5 TQQQ @ 70.00  +  SELL 1 QQQ @ 233.00")]))     # a frame without a note column
        self.assertEqual(lv.t_dec.rowCount(), 1)
        w.close()

    def test_n3_the_clear_broken_flags_button_goes_through_the_engine_for_action(self):
        from PyQt6.QtCore import QEventLoop, QTimer
        from levpairs_ui.main_window import MainWindow
        w = MainWindow()
        lv = w.live
        replay = LV.LiveEngine(w.settings.sig_cfg(), w.settings.sim_cfg(), LV.LiveSettings(feed="sip"))
        replay.is_replay = True
        replay.state = replay.blank_state("2026-10-22")
        replay.state["pairs"]["TQQQ/QQQ"]["broken"] = True
        lv.eng = replay
        calls = []
        sentinel = types.SimpleNamespace(live=types.SimpleNamespace(armed=False), tc=None, state={"pairs": {}}, reset_broken=lambda: calls.append("reset"))
        real = lv.engine_for_action
        lv.engine_for_action = lambda: sentinel
        lv.reset_broken()
        loop = QEventLoop()
        t = QTimer(); t.timeout.connect(lambda: loop.quit() if not lv._busy else None); t.start(100)
        QTimer.singleShot(10_000, loop.quit)
        loop.exec(); t.stop()
        self.assertEqual(calls, ["reset"])
        self.assertTrue(replay.state["pairs"]["TQQQ/QQQ"]["broken"], "the replay engine is never the one that gets cleared")
        lv.engine_for_action = real
        self.assertIsNot(real(), replay)
        w.close()

    def test_every_preset_maps_to_the_panel_and_to_cfgs(self):
        from levpairs_ui.settings_panel import SettingsPanel
        p = SettingsPanel()
        for name in PRESETS:
            p.apply_preset(name)
            sc, c = cfgs(name)
            self.assertEqual(p.sig_cfg().tf, sc.tf, name)
            self.assertEqual(p.sig_cfg().spread, sc.spread, name)
            self.assertAlmostEqual(p.sim_cfg().s_open, c.s_open, msg=name)
            self.assertEqual(p.sim_cfg().fill, c.fill, name)
            self.assertEqual(p.sim_cfg().vol_on, c.vol_on, name)

    def test_backtest_tab_runs_end_to_end(self):
        from PyQt6.QtCore import QEventLoop, QTimer
        from levpairs_ui.main_window import MainWindow
        w = MainWindow()
        w.show()
        w.backtest.d0.setDate(w.backtest.d0.date().addYears(8))
        w.backtest.split.setDate(w.backtest.d0.date().addMonths(10))
        w.backtest.run()
        loop = QEventLoop()
        t = QTimer(); t.timeout.connect(lambda: loop.quit() if not w.backtest._busy else None); t.start(300)
        QTimer.singleShot(240_000, loop.quit)
        loop.exec()
        self.assertIsNotNone(w.backtest.out, w.backtest.verdict.text())
        self.assertGreater(len(w.backtest.out.ret), 100)
        self.assertGreater(w.backtest.tbl_year.rowCount(), 0)
        self.assertEqual(w.backtest.tbl_pair.rowCount(), 3)
        tr = w.backtest.tbl_tr                                                                  # the trade table says which product is long and which short, when, and at what price
        self.assertGreater(tr.rowCount(), 0)
        heads = [tr.horizontalHeaderItem(i).text() for i in range(tr.columnCount())]
        self.assertEqual(heads[:6], ["Date", "Pair", "You hold", "Orders at entry (per $10k of the 3x ETF)", "Entry", "Exit"])
        for r in range(min(tr.rowCount(), 25)):
            cell = lambda name: tr.item(r, heads.index(name)).text()
            L_, B_ = cell("Pair").split("/")
            hold, orders = cell("You hold"), cell("Orders at entry (per $10k of the 3x ETF)")
            self.assertIn(hold, (f"LONG {L_} / SHORT {B_}", f"LONG {B_} / SHORT {L_}"))
            self.assertTrue(orders.startswith(f"BUY {L_} $10,000  +  SELL {B_} $") if hold.startswith(f"LONG {L_}") else orders.startswith(f"SELL {L_} $10,000  +  BUY {B_} $"), orders)
            self.assertRegex(cell("Entry"), r"^\d\d:\d\d$"); self.assertRegex(cell("Exit"), r"^\d\d:\d\d$")
            self.assertRegex(cell("Entry price  3x / 1x"), r"^[\d,]+\.\d\d / [\d,]+\.\d\d$")
        w.backtest.run_fill()
        loop = QEventLoop()
        t = QTimer(); t.timeout.connect(lambda: loop.quit() if not w.backtest._busy else None); t.start(300)
        QTimer.singleShot(240_000, loop.quit)
        loop.exec()
        self.assertEqual(w.backtest.tbl_fill.rowCount(), 3)
        self.assertEqual(len({w.backtest.tbl_fill.item(r, 3).text() for r in range(3)}), 1, "the same number of round trips under every fill")
        self.assertIn("identical in all three rows", w.backtest.fill_note.text())
        w.backtest.show_session(busiest=True)
        w.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
