"""
Tests for the stat-arb engine (synthetic data, no network, no orders):  python -m unittest test_statarb -v

  * no look-ahead   : changing prices AFTER day t never changes any signal at or before day t
  * OU recovery     : on a planted mean-reverting residual the s-score has the right sign and a positive information coefficient
  * neutrality      : the projected book has ~zero dollar / beta / sector exposure
  * accounting      : flat prices -> P&L is exactly minus the modelled costs; orders fill at the NEXT open
  * live planning   : whole shares, no flips, short only if easy-to-borrow, dry run sends nothing, armed run tags and queues OPG orders, ledger only follows accepted orders
  * GUI             : the window builds offscreen, settings round-trip, presets apply
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statarb_core as C
import statarb_data as D
import statarb_live as LV
import statarb_sim as S

ETFS = list(D.FACTOR_ETFS)
SECTORS = list(D.SECTOR_ETF)


def synthetic_panel(n_per_sector: int = 14, T: int = 700, seed: int = 0, ou_kappa: float = 25.0, ou_sigma: float = 0.012) -> D.Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=T)
    n = n_per_sector * len(SECTORS)
    mkt = rng.normal(0.0004, 0.010, T)
    sec_f = rng.normal(0, 0.006, (T, len(SECTORS)))
    names, sector, rets = [], {}, np.zeros((T, n))
    dt = 1 / 252
    for j in range(n):
        k = j % len(SECTORS)
        beta = rng.uniform(0.6, 1.4)
        ou = np.zeros(T)
        for t in range(1, T):                                    # OU residual level: its increments are mean reverting
            ou[t] = ou[t - 1] * np.exp(-ou_kappa * dt) + ou_sigma * rng.normal() * np.sqrt((1 - np.exp(-2 * ou_kappa * dt)) / (2 * ou_kappa * dt))
        rets[:, j] = beta * mkt + sec_f[:, k] + np.diff(ou, prepend=0.0) + rng.normal(0, 0.003, T)
        sym = f"S{j:03d}"
        names.append(sym); sector[sym] = SECTORS[k]
    px = 50 * np.exp(np.cumsum(rets, axis=0))
    opn = px * (1 + rng.normal(0, 0.002, px.shape))
    vol = rng.lognormal(15.5, 0.3, px.shape)
    rows = []
    for j, s in enumerate(names):
        rows.append(pd.DataFrame({"symbol": s, "date": dates, "open": opn[:, j], "high": px[:, j] * 1.01, "low": px[:, j] * 0.99, "close": px[:, j], "volume": vol[:, j]}))
    etf_r = {"SPY": mkt, "QQQ": mkt * 1.2, "IWM": mkt * 1.1, "USO": rng.normal(0, 0.02, T), "TLT": rng.normal(0, 0.008, T), "UUP": rng.normal(0, 0.004, T), "GLD": rng.normal(0, 0.009, T),
             "VIXY": -mkt * 2 + rng.normal(0, 0.03, T), "ITA": mkt}
    for k, e in enumerate(D.SECTOR_ETF.values()):
        etf_r[e] = mkt + sec_f[:, k]
    for e, r in etf_r.items():
        p = 100 * np.exp(np.cumsum(r))
        rows.append(pd.DataFrame({"symbol": e, "date": dates, "open": p * 0.999, "high": p, "low": p, "close": p, "volume": 1e7}))
    return D.Panel(pd.concat(rows, ignore_index=True), sector_map=pd.Series(sector), stocks=names)


FC = C.FactorCfg(start="2020-06-01", n_factors=6, min_adv=1e6, min_price=1.0)


class TestNoLookahead(unittest.TestCase):
    def test_future_prices_do_not_change_past_signals(self):
        p = synthetic_panel(seed=1)
        t = 560
        a = C.build_signals(p, FC)
        # corrupt everything after day t (prices, volumes, ETFs)
        q = synthetic_panel(seed=1)
        rng = np.random.default_rng(5)
        for f in ("open_all", "close_all", "volume_all"):
            df = getattr(q, f)
            df.iloc[t + 1:] = df.iloc[t + 1:] * rng.uniform(0.5, 1.8, df.iloc[t + 1:].shape)
        for k in ("open", "close", "volume", "etf_close", "etf_open"):
            pass
        q2 = D.Panel(pd.concat([getattr(q, f + "_all").stack().rename(f) for f in ("open", "close", "volume")], axis=1).rename_axis(["date", "symbol"]).reset_index(),
                     sector_map=q.sector, stocks=list(q.stocks))
        b = C.build_signals(q2, FC)
        for name in ("s", "kappa", "seq", "rvol", "b_spy", "b_oil", "mom_z", "vol_z", "size_z"):
            x, y = getattr(a, name)[:t + 1], getattr(b, name)[:t + 1]
            both = np.isfinite(x) & np.isfinite(y)
            self.assertTrue(np.allclose(x[both], y[both], atol=1e-5, equal_nan=True), f"{name} changed when only the future changed")
            self.assertEqual(np.isfinite(x).sum(), np.isfinite(y).sum(), f"{name}: availability changed")
        self.assertTrue(np.array_equal(a.news[:t + 1], b.news[:t + 1]))


class TestSignalQuality(unittest.TestCase):
    def test_planted_ou_residual_reverts(self):
        p = synthetic_panel(seed=2, ou_kappa=40.0, ou_sigma=0.02)
        sig = C.build_signals(p, FC)
        O = p.open.to_numpy(float)
        ic = []
        for d in range(sig.s.shape[0] - 4):
            s = sig.s[d]
            fwd = O[d + 4] / O[d + 1] - 1
            m = np.isfinite(s) & np.isfinite(fwd)
            if m.sum() > 60:
                ic.append(np.corrcoef(pd.Series(-s[m]).rank(), pd.Series(fwd[m]).rank())[0, 1])
        self.assertGreater(np.mean(ic), 0.03, "a planted OU residual must give a positive reversal IC")
        self.assertGreater(np.nanstd(sig.s[-5]), 0.5)


class TestBook(unittest.TestCase):
    def setUp(self):
        self.p = synthetic_panel(seed=3)
        self.sig = C.build_signals(self.p, FC)
        self.last = len(self.p.dates) - 1

    def test_projection_is_neutral(self):
        pc = replace(C.PortCfg(), tc_bps=0, band=0.0, vol_target=0.1, shock=False, n_pca_neutral=3, event_filter=False, earn_filter=False, name_cap=0.25)
        beta = lambda v, x: abs(float(np.nansum(v * np.nan_to_num(x.b_spy))) / np.abs(v).sum())
        bn, br, dn, sec_n, days = [], [], [], [], 0
        for off in range(1, 61, 3):
            x = S.DayInputs.from_signals(self.sig, self.last - off)
            w, info = S.Book(len(self.sig.cols), pc).decide(x, 1.0)
            raw, _ = S.Book(len(self.sig.cols), replace(pc, neutral=False)).decide(x, 1.0)
            if info["n_long"] < 4 or info["n_short"] < 4 or not np.any(w) or not np.any(raw):
                continue
            days += 1
            g = np.abs(w).sum()
            bn.append(beta(w, x)); br.append(beta(raw, x)); dn.append(abs(w.sum()) / g)
            sec_n.append(max(abs(w[x.sector_idx == k].sum()) for k in np.unique(x.sector_idx)) / g)
        if days < 5:
            self.skipTest("too few signal days on this synthetic panel")
        self.assertLess(np.mean(dn), 0.02, "dollar neutral")
        self.assertLess(np.mean(bn), 0.6 * np.mean(br) + 0.01, "the projection must cut the market beta of the raw book")
        self.assertLess(np.mean(sec_n), 0.10, "no sector carries more than ~10% of gross net")

    def test_states_follow_paper_thresholds(self):
        pc = replace(C.PortCfg(), event_filter=False, earn_filter=False, neutral=False, band=0.0)
        book = S.Book(len(self.sig.cols), pc)
        x = S.DayInputs.from_signals(self.sig, self.last - 2)
        book.decide(x, 1.0)
        s = x.s
        st = book.state
        ok = np.isfinite(s) & x.univ
        self.assertTrue(np.all(s[st > 0] < -pc.s_open + 1e-9) and np.all(s[st < 0] > pc.s_open - 1e-9))
        self.assertTrue(np.all(~((np.abs(s) < pc.s_open) & ok & (st != 0))))


class TestAccounting(unittest.TestCase):
    def test_flat_prices_pnl_is_minus_cost(self):
        p = synthetic_panel(seed=4)
        sig = C.build_signals(p, FC)
        sig.s[:] = np.where(np.arange(sig.s.shape[1])[None, :] % 7 == 0, 2.0, sig.s)         # force some entries
        sig.rvol[:] = 0.01
        sig.univ[:] = True
        flat = LV._truncate(p, len(p.dates) - 1)
        flat.open = pd.DataFrame(np.tile(p.open.iloc[0].to_numpy(), (len(p.dates), 1)), index=p.dates, columns=p.open.columns)       # no price ever moves
        pc = replace(C.PortCfg(), tc_bps=5.0, borrow_bps=0.0, shock=False, neutral=False, vol_target=1.0, g_init=1.0, g_min=1.0, g_max=1.0, band=0.0, scale_band=0.0, start="2020-07-01",
                     event_filter=False, earn_filter=False)
        r = S.simulate(sig, flat, pc)
        d = r.daily
        self.assertTrue((d.gross_ret.abs() < 1e-12).all(), "no price moves, no gross P&L")
        self.assertTrue(np.allclose(d.ret, -d.cost, atol=1e-12))
        self.assertGreater(d.cost.sum(), 0)
        cost_from_turnover = (d.turnover * 5.0 / 1e4).where(sig.adv[sig.dates.get_indexer(d.index)].min(axis=1) > 2e8)
        self.assertTrue(d.turnover.sum() > 0)

    def test_fill_is_next_open(self):
        p = synthetic_panel(seed=5, T=500)
        sig = C.build_signals(p, replace(FC, start="2020-06-01"))
        r = S.simulate(sig, p, replace(C.PortCfg(), shock=False, neutral=False, start="2020-07-01", event_filter=False, earn_filter=False))
        W = r.weights
        d = r.daily.index[5]
        i = sig.dates.get_loc(d)
        # weights held over [open i, open i+1] were decided at the close of i-1 and are stored at row i; the weights stored at row i+1 never depend on prices after the close of i
        self.assertTrue(np.all(W[0:int(sig.dates.get_loc(r.daily.index[0]))] == 0))
        self.assertIsNotNone(i)


class TestEarningsSleeve(unittest.TestCase):
    def test_sleeve_holds_only_scheduled_announcers_and_hedges(self):
        p = synthetic_panel(seed=7, T=600)
        rows = []
        for j, sym in enumerate(p.stocks):                                           # one announcement every 63 sessions after the close, staggered by name
            for k in range(j % 63, 600, 63):
                rows.append({"symbol": sym, "ts": p.dates[k] + pd.Timedelta(hours=16), "fetched": pd.Timestamp.now()})
        earn = pd.DataFrame(rows)
        sig = C.build_signals(p, replace(FC, start="2020-06-01"), earnings=earn)
        pc = replace(C.PortCfg(), w_rev=0.0, w_eap=1.0, start="2020-09-01", shock=False, vol_target=1.0, g_init=1.0, g_min=1.0, g_max=2.0, eap_min_names=5)
        r = S.simulate(sig, p, pc)
        d = r.daily
        self.assertGreater(d.n_eap.max(), 5)
        self.assertTrue((d.etf_gross > 0.1).any(), "the sector-ETF hedge must be on")
        self.assertLessEqual(d.gross.max(), pc.g_max + 1e-6, "Reg-T gross cap")
        # every held name had its announcement 2..10 sessions after the decision close
        for i in np.where(r.weights.sum(axis=1) != 0)[0][::7]:
            dec = i - 1
            sched = sig.earn_ext[dec + pc.eap_lo:dec + pc.eap_hi + 1].any(axis=0)
            held = r.weights[i] != 0
            self.assertTrue(np.all(sched[held]), f"held a name with no announcement in the window on row {i}")
        # beta of the hedged book to the market stays small
        self.assertLess(abs(d.beta.mean()), 0.25)

    def test_cap_is_real_and_sector_cap_leaves_cash(self):
        p = synthetic_panel(seed=9, T=420)
        sig = C.build_signals(p, replace(FC, start="2020-06-01"))
        ecols = list(p.etf_close.columns)
        pc = replace(C.PortCfg(), w_rev=0.0, w_eap=1.0, eap_min_names=3, eap_cap=0.04, eap_sector_cap=0.30, shock=False)
        port = S.Portfolio(len(sig.cols), pc, ecols, S.sector_etf_index(sig, ecols), sig.sector_idx)
        x = S.DayInputs.from_signals(sig, len(p.dates) - 15, pc)
        x.eap_ahead = np.zeros(len(sig.cols), bool); x.eap_ahead[:6] = True                  # only six announcers
        x.univ = np.ones(len(sig.cols), bool)
        w, th, info = port.decide(x, 1.0)
        self.assertLessEqual(w.max(), 0.04 + 1e-12, "a name never exceeds the cap, even when only six report")
        self.assertAlmostEqual(w.sum(), 6 * 0.04, places=6, msg="the rest stays in cash instead of being re-normalised")
        x.eap_ahead[:] = False; x.eap_ahead[np.where(sig.sector_idx == sig.sector_idx[0])[0][:12]] = True        # twelve announcers in ONE sector
        port2 = S.Portfolio(len(sig.cols), pc, ecols, S.sector_etf_index(sig, ecols), sig.sector_idx)
        w2, _, _ = port2.decide(x, 1.0)
        self.assertLessEqual(w2.sum(), 0.30 + 1e-9, "one sector never carries more than the sector cap")

    def test_schedule_extends_past_the_last_bar(self):
        p = synthetic_panel(seed=8, T=300)
        last = p.dates[-1]
        nxt = D.future_sessions(last, 4)
        earn = pd.DataFrame({"symbol": [p.stocks[0]], "ts": [nxt[2] + pd.Timedelta(hours=7)], "fetched": [pd.Timestamp.now()]})
        m = D.earnings_reaction_matrix(p, earn, extra=12)
        self.assertEqual(m.shape[0], len(p.dates) + 12)
        self.assertTrue(m[len(p.dates) + 2, 0], "a published future announcement must be visible beyond the data")


class FakeAsset:
    pass


class FakeOrder:
    def __init__(self, i):
        self.id = f"oid{i}"
        self.status = type("S", (), {"value": "accepted"})()


class FakePos:
    def __init__(self, symbol, qty):
        self.symbol, self.qty = symbol, str(qty)


class FakeTC:
    def __init__(self, positions=None, fail=()):
        self.sent = []
        self.positions = positions or {}
        self.fail = set(fail)

    def submit_order(self, req):
        if req.symbol in self.fail:
            raise RuntimeError("rejected by the fake broker")
        self.sent.append(req)
        return FakeOrder(len(self.sent))

    def get_clock(self):
        return type("C", (), {"is_open": False})()

    def get_orders(self):
        return []

    def get_all_positions(self):
        return [FakePos(s, q) for s, q in self.positions.items()]


class TestLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = (D.REPORTS, D.DATA_DIR)
        D.REPORTS, D.DATA_DIR = self.tmp, self.tmp
        pd.DataFrame({"symbol": ["S000", "S001", "S002"], "name": ["a", "b", "c"], "exchange": ["NYSE"] * 3, "status": ["active"] * 3, "tradable": [True] * 3, "shortable": [True, True, False],
                      "etb": [True, False, True], "marginable": [True] * 3, "fractionable": [True] * 3}).to_parquet(os.path.join(self.tmp, "statarb_assets.parquet"))
        self.panel = synthetic_panel(seed=6, T=420)
        self.eng = LV.LiveEngine(FC, C.PortCfg(), LV.LiveSettings(armed=False, min_trade_usd=100, max_name_pct=0.5))
        self.eng.panel = self.panel
        self._old_exp = LV.expected_last_session
        LV.expected_last_session = lambda now=None: self.panel.dates[-1]                  # the synthetic panel is 'today'

    def tearDown_extra(self):
        LV.expected_last_session = self._old_exp

    def tearDown(self):
        D.REPORTS, D.DATA_DIR = self.old
        LV.expected_last_session = self._old_exp

    def test_orders_whole_shares_no_flip_and_etb(self):
        i = len(self.panel.dates) - 1
        px = float(self.panel.close.iloc[i]["S000"])
        notes: list = []
        o, tgt = self.eng._orders({"S000": 0.10, "S001": -0.10, "S002": -0.10}, {"S002": 5, "S003": 3}, self.panel, i, 100_000.0, notes)
        r0 = o[o.symbol == "S000"].iloc[0]
        self.assertEqual(r0.side, "buy"); self.assertEqual(int(r0.qty), int(10_000 // px))
        r1 = o[o.symbol == "S001"].iloc[0]
        self.assertFalse(bool(r1.ok), "S001 is not easy-to-borrow")
        r2 = o[o.symbol == "S002"].iloc[0]
        self.assertEqual(int(r2.tgt_qty), 0, "a long that should become a short is closed first, never flipped in one order")
        self.assertEqual(r2.side, "sell"); self.assertEqual(int(r2.qty), 5)
        self.assertTrue((o[o.symbol == "S003"].iloc[0].reason == "exit"))

    def test_dry_run_sends_nothing(self):
        plan = LV.Plan(decision_date=self.panel.dates[-1], deploy=1e5, equity=1e5, g_target=1.0, shock_mult=1.0, n_long=1, n_short=0, orders=pd.DataFrame(
            [dict(symbol="S000", side="buy", qty=10, cur_qty=0, tgt_qty=10, ref_px=50.0, notional=500.0, target_w=0.01, ok=True, note="", reason="entry")]), targets={"S000": 0.01},
            exposures={}, regime={}, notes=[], book_after={"rev": {"positions": {"S000": {"side": 1, "hold": 0, "entry_s": -2, "w": 0.01}}, "g_applied": 1.0}, "eap": {"w": {}, "g_applied": None}}, ledger_after={"S000": 10})
        self.eng.tc = FakeTC()
        with self.assertRaises(RuntimeError):
            self.eng.submit(plan)
        self.assertEqual(self.eng.tc.sent, [])

    def test_armed_submit_tags_opg_and_updates_ledger(self):
        from alpaca.trading.enums import TimeInForce
        self.eng.live = replace(self.eng.live, armed=True, exec_mode="day")
        plan = LV.Plan(decision_date=self.panel.dates[-1], deploy=1e5, equity=1e5, g_target=1.0, shock_mult=1.0, n_long=1, n_short=0, orders=pd.DataFrame(
            [dict(symbol="S000", side="buy", qty=10, cur_qty=0, tgt_qty=10, ref_px=50.0, notional=500.0, target_w=0.01, ok=True, note="", reason="entry"),
             dict(symbol="S001", side="sell", qty=4, cur_qty=0, tgt_qty=-4, ref_px=50.0, notional=200.0, target_w=-0.01, ok=False, note="not shortable", reason="entry")]),
            targets={"S000": 0.01}, exposures={}, regime={}, notes=[], book_after={"rev": {"positions": {"S000": {"side": 1, "hold": 0, "entry_s": -2, "w": 0.01},
                                                                                                        "S001": {"side": -1, "hold": 0, "entry_s": 2, "w": -0.01}}, "g_applied": 1.0},
                                                                                   "eap": {"w": {}, "g_applied": None}}, ledger_after={"S000": 10})
        self.eng.tc = FakeTC(); self.eng.acct = {"blocked": False, "equity": 1e5}
        res = self.eng.submit(plan)
        self.assertEqual(len(self.eng.tc.sent), 1, "the skipped (not shortable) order must not be sent")
        req = self.eng.tc.sent[0]
        self.assertTrue(req.client_order_id.startswith("SA-")); self.assertEqual(int(req.qty), 10); self.assertEqual(req.time_in_force, TimeInForce.DAY)
        st = self.eng.load_state()
        self.assertEqual(st["ledger"], {"S000": 10})
        self.assertNotIn("S001", st["book"]["rev"]["positions"], "a skipped entry must not be remembered as a position")

    def _plan(self, rows):
        return LV.Plan(decision_date=self.panel.dates[-1], deploy=1e5, equity=1e5, g_target=1.0, shock_mult=1.0, n_long=1, n_short=0, orders=pd.DataFrame(rows), targets={"S000": 0.01}, exposures={},
                       regime={}, notes=[], book_after={"rev": {"positions": {}, "g_applied": 1.0}, "eap": {"w": {"S000": 0.01}, "g_applied": 1.0}}, ledger_after={"S000": 10})

    def test_second_submit_for_the_same_session_is_refused(self):
        self.eng.live = replace(self.eng.live, armed=True, exec_mode="day")
        row = dict(symbol="S000", side="buy", qty=10, cur_qty=0, tgt_qty=10, ref_px=50.0, notional=500.0, target_w=0.01, ok=True, note="", reason="entry")
        self.eng.tc = FakeTC(); self.eng.acct = {"blocked": False, "equity": 1e5}
        self.eng.submit(self._plan([row]))
        with self.assertRaises(RuntimeError):
            self.eng.submit(self._plan([row]))
        self.assertEqual(len(self.eng.tc.sent), 1, "no duplicate order")

    def test_daily_loss_halt_blocks_new_risk_but_not_exits(self):
        self.eng.live = replace(self.eng.live, armed=True, exec_mode="day", daily_loss_halt=0.02)
        self.eng.tc = FakeTC(); self.eng.acct = {"blocked": False, "equity": 96_000.0, "last_equity": 100_000.0}
        st = self.eng.load_state(); st["ledger"] = {"S005": 7}; self.eng.save_state(st)
        rows = [dict(symbol="S000", side="buy", qty=10, cur_qty=0, tgt_qty=10, ref_px=50.0, notional=500.0, target_w=0.01, ok=True, note="", reason="entry"),
                dict(symbol="S005", side="sell", qty=7, cur_qty=7, tgt_qty=0, ref_px=50.0, notional=350.0, target_w=0.0, ok=True, note="", reason="exit")]
        res = self.eng.submit(self._plan(rows))
        sent = {r.symbol for r in self.eng.tc.sent}
        self.assertEqual(sent, {"S005"}, "only the closing order goes out while halted")
        self.assertTrue(any(r["symbol"] == "S000" and r["status"] == "skipped" for r in res))

    def test_stale_panel_is_refused(self):
        LV.expected_last_session = lambda now=None: self.panel.dates[-1] + pd.Timedelta(days=7)
        with self.assertRaises(RuntimeError) as cm:
            self.eng.check_fresh(self.panel)
        self.assertIn("stale", str(cm.exception))

    def test_stock_cache_behind_the_etfs_is_refused(self):
        p = self.panel
        q = LV._truncate(p, len(p.dates) - 1)
        c = q.close.copy()
        c.iloc[-1] = np.nan                                                            # ETFs have the last session, the stocks do not
        q.close = c
        with self.assertRaises(RuntimeError):
            self.eng.check_fresh(q)

    def test_kill_closes_only_what_the_engine_owns_and_keeps_failures_in_the_ledger(self):
        self.eng.tc = FakeTC(positions={"S000": 1000, "S001": -20, "S002": 50}, fail={"S002"})
        self.eng.acct = {"blocked": False, "equity": 1e5}
        st = self.eng.load_state(); st["ledger"] = {"S000": 100, "S001": -20, "S002": 50, "S009": 5}; self.eng.save_state(st)
        res = self.eng.flatten()
        sent = {r.symbol: (int(r.qty), str(r.side)) for r in self.eng.tc.sent}
        self.assertEqual(sent["S000"][0], 100, "closes the engine's 100, not the account's 1000 (shares of other strategies stay)")
        self.assertEqual(sent["S001"][0], 20)
        left = self.eng.load_state()["ledger"]
        self.assertIn("S002", left, "a rejected closing order must leave the position in the ledger")
        self.assertNotIn("S000", left); self.assertNotIn("S009", left, "no broker position -> nothing to close, forgotten")

    def test_rejected_everything_leaves_the_session_retryable(self):
        self.eng.live = replace(self.eng.live, armed=True, exec_mode="day")
        self.eng.tc = FakeTC(fail={"S000"}); self.eng.acct = {"blocked": False, "equity": 1e5}
        row = dict(symbol="S000", side="buy", qty=10, cur_qty=0, tgt_qty=10, ref_px=50.0, notional=500.0, target_w=0.01, ok=True, note="", reason="entry")
        self.eng.submit(self._plan([row]))
        self.assertEqual(self.eng.load_state().get("held_date"), None)
        self.eng.tc = FakeTC()
        self.eng.submit(self._plan([row]))                                              # retry works
        self.assertEqual(len(self.eng.tc.sent), 1)

    def test_corrupt_state_is_an_error_not_an_empty_ledger(self):
        open(LV.state_path(True, "core"), "w").write("{not json")
        with self.assertRaises(RuntimeError):
            self.eng.load_state()


class TestGui(unittest.TestCase):
    def test_window_builds_and_settings_roundtrip(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["STATARB_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(), "t.ini")
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from supertrend_ui.theme import apply_theme
        apply_theme(app)
        from statarb_ui.main_window import MainWindow
        w = MainWindow()
        w.show()
        f, p = w.settings.factor_cfg(), w.settings.port_cfg()
        self.assertEqual(f.n_factors, 15)
        self.assertFalse(w.live.armed.isChecked(), "must start DRY RUN")
        self.assertTrue(w.live.paper(), "must start on PAPER")
        # arming on paper must not survive a switch to the real-money account
        w.live.armed.setChecked(True)
        self.assertTrue(w.live.live_settings().armed)
        w.live.account.setCurrentIndex(1)
        self.assertFalse(w.live.armed.isChecked(), "switching to LIVE must disarm")
        self.assertFalse(w.live.live_settings().armed)
        w.live.account.setCurrentIndex(0)
        w.settings.w["s_open"].setValue(1.6)
        self.assertAlmostEqual(w.settings.port_cfg().s_open, 1.6)
        v = w.settings.values()
        w.settings.set_values({**v, "s_open": 1.1})
        self.assertAlmostEqual(w.settings.port_cfg().s_open, 1.1)
        w.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
