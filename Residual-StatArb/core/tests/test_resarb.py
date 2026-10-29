"""
Tests for the shared residual stat-arb core and the three strategies. FAKES ONLY: synthetic bars in a temp folder, a fake broker; no network, no keys, no orders.

  python -m unittest discover -s tests -v        (from Residual-StatArb/core)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import synth  # noqa: E402  (must come before resarb: it points the data dir at a temp folder)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from resarb import backtest as BT  # noqa: E402
from resarb import data as D  # noqa: E402
from resarb import factors as F  # noqa: E402
from resarb import features as X  # noqa: E402
from resarb import live as LV  # noqa: E402
from resarb import portfolio as P  # noqa: E402
from resarb import strategy as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
for k in ("ResMomentum", "ResMeanReversion", "ResPrediction"):
    sys.path.insert(0, os.path.join(ROOT, k))
from strategy_resmom import ResidualMomentum  # noqa: E402
from strategy_resmr import ResidualMeanReversion  # noqa: E402
from strategy_respred import ResidualPrediction  # noqa: E402

M = synth.make_market()


def panel_from(frames: dict, syms=None) -> D.Panel:
    syms = syms or list(D.ETFS) + M["stocks"]
    return D.Panel(frames, syms)


def ctx_for(strat, frames=None) -> S.Context:
    p = panel_from(frames or M["frames"])
    member_d = D.membership_mask(p, int(strat.v["universe_n"]))
    res = F.build(p, strat.factor_cfg(), member_d)
    member = member_d[p.day] & np.isfinite(res.e_cc) & np.isfinite(p.open)
    return S.Context(p=p, res=res, member=member, member_d=member_d, lag=strat.lag)


def perturb_after(frames: dict, t0: pd.Timestamp, seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for s, f in frames.items():
        g = f.copy()
        m = g.index > t0
        k = 1 + rng.normal(0, 0.03, m.sum())
        for c in ("open", "high", "low", "close", "vwap"):
            g.loc[m, c] = g.loc[m, c].values * k
        g.loc[m, "volume"] = g.loc[m, "volume"].values * (1 + rng.random(m.sum()))
        out[s] = g
    return out


def whole_shares(w: np.ndarray, px: np.ndarray, cap: float, is_etf: np.ndarray) -> np.ndarray:
    """Stock weights of the whole shares the live engine will actually hold (the ETF hedge is sized from these)."""
    r = np.trunc(w * cap / px) * px / cap
    r[is_etf] = 0.0
    return r


def same(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9 * (1 + np.abs(a)))))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestFactorsAndFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strat = ResidualMeanReversion({"universe_n": 50, "beta_days": 126})
        cls.ctx = ctx_for(cls.strat)

    def test_panel_days_and_bars(self):
        p = self.ctx.p
        self.assertEqual(len(p.dates), len(M["days"]))
        self.assertTrue((p.bar[p.day_first] == 0).all())
        self.assertTrue((p.day_last - p.day_first + 1 == 26).all())

    def test_etf_betas_recover_the_truth(self):
        res, p = self.ctx.res, self.ctx.p
        d = len(p.dates) - 1
        err_m, err_s = [], []
        for s, (bm, bs) in M["betas"].items():
            j = p.col[s]
            err_m.append(abs(res.b_m[d, j] - bm))
            err_s.append(abs(res.b_s[d, j] - bs))
        self.assertLess(np.median(err_m), 0.15)
        self.assertLess(np.median(err_s), 0.2)

    def test_hedged_residual_has_no_market_beta(self):
        res, p = self.ctx.res, self.ctx.p
        m = p.col["SPY"]
        rows = slice(26 * 150, None)
        corr = [np.corrcoef(np.nan_to_num(res.e_cc[rows, p.col[s]]), np.nan_to_num(res.r_cc[rows, m]))[0, 1] for s in M["stocks"]]
        self.assertLess(abs(np.median(corr)), 0.08)

    def test_betas_residuals_and_features_never_look_ahead(self):
        p = self.ctx.p
        t0 = p.ts[len(p.ts) - 26 * 10 + 7]                                                    # a bar in the middle of a session, 10 sessions before the end
        f2 = perturb_after(M["frames"], t0)
        c2 = ctx_for(self.strat, f2)
        i0 = int(np.searchsorted(p.ts, t0)) + 1
        d0 = p.day[i0 - 1] + 1
        for name in ("b_m", "b_s", "idio_var"):
            self.assertTrue(same(getattr(self.ctx.res, name)[:d0], getattr(c2.res, name)[:d0]), name)
        self.assertTrue(same(self.ctx.res.e_cc[:i0], c2.res.e_cc[:i0]))
        fa = X.build(self.ctx.p, self.ctx.res, 0)
        fb = X.build(c2.p, c2.res, 0)
        for k in fa:
            self.assertTrue(same(fa[k][:i0], fb[k][:i0]), f"feature {k} changed when only later data changed")
        fa1 = X.build(self.ctx.p, self.ctx.res, 1, names=["hks"])
        fb1 = X.build(c2.p, c2.res, 1, names=["hks"])
        self.assertTrue(same(fa1["hks"][:i0], fb1["hks"][:i0]))

    def test_forward_target_alignment(self):
        e = self.ctx.res.e_oo
        j = self.ctx.p.col[M["stocks"][3]]
        for lag in (0, 1):
            f1 = X.forward(self.ctx.res, 1, lag)
            f3 = X.forward(self.ctx.res, 3, lag)
            t = 4000
            self.assertAlmostEqual(f1[t, j], e[t + 1 + lag, j], places=12)
            self.assertAlmostEqual(f3[t, j], e[t + 1 + lag:t + 4 + lag, j].sum(), places=12)

    def test_ou_score_matches_a_direct_ar1_fit(self):
        W = 60
        Xc = np.cumsum(np.nan_to_num(self.ctx.res.e_cc[:, [self.ctx.p.col[M["stocks"][5]]]]), axis=0)
        s, kap, r2 = X.ou_score(Xc, W)
        t = 5000
        x, y = Xc[t - W:t, 0], Xc[t - W + 1:t + 1, 0]
        b, a = np.polyfit(x, y, 1)
        res_ = y - (a + b * x)
        seq = np.sqrt(res_.var(ddof=0) * W / (W - 2) / (1 - b * b))
        self.assertAlmostEqual(s[t, 0], (Xc[t, 0] - a / (1 - b)) / seq, places=6)
        self.assertAlmostEqual(kap[t, 0], -np.log(b), places=8)

    def test_residual_momentum_window(self):
        ed = self.ctx.res.e_d
        m = X.res_momentum(self.ctx.res, 60, 5, vol_scale=False, min_frac=0.5)
        d, j = 200, self.ctx.p.col[M["stocks"][2]]
        self.assertAlmostEqual(m[d, j], np.nansum(ed[d - 60:d - 5, j]), places=10)


class TestKeyFile(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.f = os.path.join(self.d, "Alpaca Kay.txt")
        with open(self.f, "w", encoding="utf-8") as fh:
            fh.write("Key ID: PKTESTTESTTESTTESTTEST01\r\n\r\nSecret Key: sEcReTsEcReTsEcReTsEcReTsEcReTsEcReT0000\r\n")

    def tearDown(self):
        os.environ["ALPACA_KEY_FILE"] = "none"
        for k in ("RESMOM_ALPACA_API_KEY", "RESMOM_ALPACA_SECRET_KEY"):
            os.environ.pop(k, None)

    def test_the_key_file_is_used_for_paper_and_counts_as_own_keys(self):
        from resarb import alpaca_keys as K
        os.environ["ALPACA_KEY_FILE"] = self.f
        self.assertEqual(K.creds(True), ("PKTESTTESTTESTTESTTEST01", "sEcReTsEcReTsEcReTsEcReTsEcReTsEcReT0000"))
        self.assertEqual(K.source(True), "file")
        self.assertEqual(D.creds("RESMOM", True)[0], "PKTESTTESTTESTTESTTEST01")
        self.assertTrue(D.own_keys("RESMOM"))
        self.assertIn("Alpaca Kay.txt", D.key_source("RESMOM"))
        self.assertNotEqual(K.source(False), "file", "a paper key never serves a live connection")

    def test_a_strategy_variable_overrides_the_file_and_none_switches_the_file_off(self):
        from resarb import alpaca_keys as K
        os.environ["ALPACA_KEY_FILE"] = self.f
        os.environ["RESMOM_ALPACA_API_KEY"], os.environ["RESMOM_ALPACA_SECRET_KEY"] = "PKOTHER", "other"
        self.assertEqual(D.creds("RESMOM", True), ("PKOTHER", "other"))
        self.assertEqual(D.creds("RESMR", True)[0], "PKTESTTESTTESTTESTTEST01")
        os.environ["ALPACA_KEY_FILE"] = "none"
        self.assertIsNone(K.find_key_file())
        self.assertNotEqual(K.source(True), "file")
        self.assertFalse(D.own_keys("RESMR"), "without the file and variables only ~/.claude.json is left: not the project's own keys")

    def test_the_key_file_is_found_by_walking_up_from_the_code(self):
        from resarb import alpaca_keys as K
        os.environ.pop("ALPACA_KEY_FILE", None)
        deep = os.path.join(self.d, "a", "b", "c")
        os.makedirs(deep)
        self.assertEqual(K.find_key_file(deep), self.f)


class TestPriceFixes(unittest.TestCase):
    def test_an_unadjusted_distribution_is_back_adjusted_when_read_and_the_file_stays_raw(self):
        s = M["stocks"][11]
        raw = D.load_bars(s, fix=False)
        day = M["days"][100]
        try:
            pd.DataFrame([dict(symbol=s, ex_date=day, factor=0.8, why="test")]).to_parquet(D.FIX_FILE)
            D._FIXES = None
            f = D.load_bars(s)
            before, after = f.index < day, f.index >= day
            self.assertTrue(np.allclose(f.close[before], raw.close[before] * 0.8))
            self.assertTrue(np.allclose(f.close[after], raw.close[after]))
            self.assertTrue(np.allclose(f.volume, raw.volume))
            self.assertTrue(np.allclose(D.load_bars(s, fix=False).close, raw.close), "the cache file itself is never rewritten")
        finally:
            os.remove(D.FIX_FILE)
            D._FIXES = None


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.sec = np.repeat(np.arange(4), 5)
        self.score = np.linspace(-1, 1, 20)
        self.mem = np.ones(20, bool)

    def test_quantile_enters_top_and_bottom_equal_weight_dollar_neutral(self):
        cfg = P.PortCfg(method="quantile", q_in=0.2, q_out=0.2, gross=1.0, max_w=1.0)
        w = P.decide(self.score, self.mem, np.zeros(20), cfg, self.sec)
        self.assertEqual((w > 0).sum(), 4); self.assertEqual((w < 0).sum(), 4)
        self.assertAlmostEqual(w.sum(), 0.0); self.assertAlmostEqual(np.abs(w).sum(), 1.0)
        self.assertTrue((w[-4:] > 0).all() and (w[:4] < 0).all())

    def test_quantile_hysteresis_keeps_a_held_name_until_it_leaves_q_out(self):
        cfg = P.PortCfg(method="quantile", q_in=0.2, q_out=0.4, gross=1.0, max_w=1.0)
        held = np.zeros(20); held[12] = 0.1                                                    # percentile 0.625: outside the top 20%, inside the top 40%
        w = P.decide(self.score, self.mem, held, cfg, self.sec)
        self.assertGreater(w[12], 0)
        held2 = np.zeros(20); held2[8] = 0.1                                                   # percentile 0.425: out
        self.assertEqual(P.decide(self.score, self.mem, held2, cfg, self.sec)[8], 0.0)

    def test_threshold_open_close_and_cap(self):
        cfg = P.PortCfg(method="threshold", s_open=1.25, s_close=0.5, names=2, gross=1.0, max_w=1.0)
        sc = np.array([2.0, 1.5, 1.3, 0.8, 0.2, -0.2, -1.4, -3.0, np.nan, 0.0])
        held = np.array([0, 0, 0, 0.1, 0.1, 0, 0, 0, 0.1, 0])
        w = P.decide(sc, np.ones(10, bool), held, cfg, np.zeros(10, int))
        self.assertGreater(w[3], 0, "held long with score 0.8 > close 0.5 stays")
        self.assertEqual(w[4], 0.0, "held long with score 0.2 < 0.5 closes")
        self.assertEqual(w[8], 0.0, "no score -> closed")
        self.assertEqual(int((w > 0).sum()), 2, "cap: held name first, then the strongest new one")
        self.assertGreater(w[0], 0)
        self.assertTrue(w[6] < 0 and w[7] < 0)

    def test_never_flips_and_keeps_untradable_names(self):
        cfg = P.PortCfg(method="quantile", q_in=0.2, q_out=0.2, gross=1.0, max_w=1.0)
        held = np.zeros(20); held[0] = 0.2; held[19] = 0.05
        trad = np.ones(20, bool); trad[19] = False
        w = P.decide(self.score, self.mem, held, cfg, self.sec, trad)
        self.assertEqual(w[0], 0.0, "a long that should now be short is closed, not flipped")
        self.assertEqual(w[19], 0.05, "a name without a price keeps its position")

    def test_partial_trading_moves_a_fraction_of_the_way_and_closes_dust(self):
        held = np.array([0.10, 0.0, -0.10, 0.004, 0.05])
        tgt = np.array([0.20, 0.10, 0.0, 0.0, 0.05])
        w = P.approach(held, tgt, 0.25, gross=1.0)
        self.assertAlmostEqual(w[0], 0.125); self.assertAlmostEqual(w[1], 0.025); self.assertAlmostEqual(w[2], -0.075)
        self.assertEqual(w[3], 0.0, "a leftover far below an average position is closed outright")
        self.assertAlmostEqual(w[4], 0.05)
        self.assertTrue(np.array_equal(P.approach(held, tgt, 1.0), tgt))
        trad = np.array([True, True, False, True, True])
        self.assertEqual(P.approach(held, tgt, 0.25, tradable=trad)[2], -0.10)

    def test_etf_hedge_removes_market_and_sector_beta(self):
        strat = ResidualMeanReversion({"universe_n": 50})
        ctx = ctx_for(strat)
        p, res = ctx.p, ctx.res
        d = len(p.dates) - 1
        rng = np.random.default_rng(1)
        w = np.zeros(len(p.syms))
        for s in M["stocks"]:
            w[p.col[s]] = rng.normal(0, 0.05)
        h = P.hedge_etf(w, res, d)
        bm, bs = np.nan_to_num(res.b_m[d]), np.nan_to_num(res.b_s[d])
        self.assertAlmostEqual(w @ bm + h.sum(), 0.0, places=10)                               # every ETF has market beta 1 in this model
        for k, c in enumerate(res.etf_idx[1:], start=1):
            m = res.sector_col == c
            self.assertAlmostEqual(w[m] @ bs[m] + h[k], 0.0, places=10)

    def test_projection_is_factor_and_dollar_neutral(self):
        strat = ResidualMeanReversion({"universe_n": 50})
        ctx = ctx_for(strat)
        d = len(ctx.p.dates) - 1
        w = np.zeros(len(ctx.p.syms))
        for i, s in enumerate(M["stocks"]):
            w[ctx.p.col[s]] = (-1) ** i * 0.05
        w2 = P.project(w, ctx.res, d)
        B = P.loadings(ctx.res, d)
        self.assertLess(np.abs(w2 @ B).max(), 1e-10)
        self.assertAlmostEqual(np.abs(w2).sum(), np.abs(w).sum(), places=10)


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strat = ResidualMeanReversion({"universe_n": 50})
        cls.ctx = ctx_for(cls.strat)

    def two_name_score(self):
        p = self.ctx.p
        sc = np.full(p.close.shape, np.nan)
        a, b = p.col[M["stocks"][1]], p.col[M["stocks"][2]]
        sc[:, a], sc[:, b] = 1.0, -1.0
        return sc, a, b

    def run_bt(self, lag=0, cost=0.0, every=1, flat_eod=False, hedge="none"):
        sc, a, b = self.two_name_score()
        cfg = P.PortCfg(method="quantile", q_in=0.5, q_out=0.5, gross=1.0, max_w=1.0, hedge=hedge)
        start = str(self.ctx.p.dates[200].date())
        bc = BT.BtCfg(start=start, lag=lag, cost_model="flat", cost_bp=cost, etf_cost_bp=cost, borrow_bp_yr=0.0, flat_eod=flat_eod, split="2100-01-01")
        return BT.run(self.ctx.p, self.ctx.res, sc, self.ctx.member, BT.schedule(self.ctx.p, every=every), cfg, bc), a, b, bc

    def test_pnl_equals_weights_times_open_to_open_returns(self):
        bt, a, b, bc = self.run_bt()
        p = self.ctx.p
        k0 = int(np.searchsorted(p.ts, pd.Timestamp(bc.start)))
        r = self.ctx.res.r_oo
        exp = 0.5 * r[k0 + 5:k0 + 400, a] - 0.5 * r[k0 + 5:k0 + 400, b]
        self.assertTrue(np.allclose(bt.bar_pnl[k0 + 5:k0 + 400], exp, atol=1e-12))

    def test_first_trade_is_one_bar_after_the_data_and_lag_adds_a_bar(self):
        p = self.ctx.p
        for lag in (0, 1):
            bt, a, b, bc = self.run_bt(lag=lag, cost=10.0)
            k0 = int(np.searchsorted(p.ts, pd.Timestamp(bc.start)))
            first = int(np.argmax(bt.bar_pnl != 0))
            self.assertEqual(first, max(k0, 1 + lag), "the first decision executes at the first simulated bar")
            self.assertAlmostEqual(bt.bar_cost[first], 1.0 * 10.0 / 1e4, places=12, msg="entry cost = gross x cost per side")
            self.assertTrue(np.all(bt.bar_cost[:first] == 0))

    def test_spread_cost_model(self):
        s = BT.SPREAD
        mid = BT.half_spread_bp(np.array([np.log(s["adv_med"])]), np.array([s["px_med"]]), 10)[0]
        self.assertAlmostEqual(mid, np.exp(s["c"]), places=9)
        self.assertAlmostEqual(BT.half_spread_bp(np.array([np.log(s["adv_med"])]), np.array([s["px_med"]]), 1)[0] / mid, np.exp(s["first_hour"]), places=9)
        self.assertAlmostEqual(BT.half_spread_bp(np.array([np.log(s["adv_med"])]), np.array([s["px_med"]]), 25)[0] / mid, np.exp(s["last_half_hour"]), places=9)
        thin = BT.half_spread_bp(np.array([np.log(s["adv_med"] / 10)]), np.array([s["px_med"]]), 10)[0]
        self.assertGreater(thin, 2 * mid, "a name with a tenth of the dollar volume is much wider")
        sc, a, b = self.two_name_score()
        cfg = P.PortCfg(method="quantile", q_in=0.5, q_out=0.5, gross=1.0, max_w=1.0, hedge="none")
        start = str(self.ctx.p.dates[200].date())
        bc = BT.BtCfg(start=start, cost_model="spread", cost_mult=2.0, slip_bp=0.5, borrow_bp_yr=0.0, split="2100-01-01")
        bt = BT.run(self.ctx.p, self.ctx.res, sc, self.ctx.member, BT.schedule(self.ctx.p, every=26), cfg, bc)
        k = int(np.argmax(bt.bar_cost != 0))
        p = self.ctx.p
        la = BT.log_adv(p)[p.day[k]]
        hs = BT.half_spread_bp(la, p.open[k], int(p.bar[k]), 26, int(p.dates[p.day[k]].year))
        self.assertLess(BT.spread_c(2017), BT.spread_c(2026), "spreads were tighter before 2020")
        self.assertAlmostEqual(bt.bar_cost[k], 0.5 * (hs[a] * 2 + 0.5) / 1e4 + 0.5 * (hs[b] * 2 + 0.5) / 1e4, places=12)

    def test_vwap_fill_charges_the_distance_from_the_open(self):
        bt0, a, b, bc = self.run_bt(every=26)
        sc, a, b = self.two_name_score()
        cfg = P.PortCfg(method="quantile", q_in=0.5, q_out=0.5, gross=1.0, max_w=1.0, hedge="none")
        bc1 = BT.BtCfg(start=bc.start, lag=0, cost_model="flat", cost_bp=0.0, etf_cost_bp=0.0, borrow_bp_yr=0.0, split="2100-01-01", fill="vwap")
        bt1 = BT.run(self.ctx.p, self.ctx.res, sc, self.ctx.member, BT.schedule(self.ctx.p, every=26), cfg, bc1)
        k = int(np.argmax(bt1.bar_cost != 0))
        p = self.ctx.p
        exp = 0.5 * (p.vwap[k, a] / p.open[k, a] - 1) - 0.5 * (p.vwap[k, b] / p.open[k, b] - 1)
        self.assertAlmostEqual(bt1.bar_cost[k], exp, places=12)

    def test_monthly_schedule_uses_the_exchange_calendar_not_the_last_date_in_the_data(self):
        p = self.ctx.p
        ends = D.month_end_sessions(p.dates[0], p.dates[-1])
        full = BT.schedule(p, bar_of_day=23, monthly=True)
        flagged = pd.DatetimeIndex(np.unique(p.ts[full].normalize()))
        self.assertTrue(set(flagged) <= set(ends), "only NYSE month-end sessions decide")
        self.assertGreaterEqual(len(flagged), 10)
        mid = [i for i, d in enumerate(p.dates) if d not in set(ends) and d.day in range(10, 20)][-1]
        cut = p.slice(0, int(p.day_last[mid]) + 1)                                             # data that end in the middle of a month (= live, every day)
        s = BT.schedule(cut, bar_of_day=23, monthly=True)
        self.assertFalse(s[cut.day == cut.day[-1]].any(), "the newest day of the data is not a month end just because it is the newest")
        self.assertEqual(s.sum(), full[:len(s)].sum())

    def test_flat_overnight_holds_nothing_over_the_close(self):
        bt, a, b, bc = self.run_bt(flat_eod=True)
        p = self.ctx.p
        k0 = int(np.searchsorted(p.ts, pd.Timestamp(bc.start)))
        last = p.day_last[np.unique(p.day[k0:])]
        self.assertTrue(np.allclose(bt.bar_pnl[last[:-1]], 0.0), "the last bar of each day (overnight to the next open) is flat")

    def test_cost_curve_reprices_the_same_trades(self):
        bt, a, b, bc = self.run_bt(cost=3.0, every=26)
        cc = BT.cost_curve(bt, [1.0])
        self.assertAlmostEqual(cc.sharpe.iloc[0], bt.stats["all"]["sharpe"], places=3)                    # additive vs compounded within the day

    def test_gross_cap_scales_the_whole_book_like_the_live_engine(self):
        sc, a, b = self.two_name_score()
        cfg = P.PortCfg(method="quantile", q_in=0.5, q_out=0.5, gross=1.0, max_w=1.0, hedge="etf")
        start = str(self.ctx.p.dates[200].date())
        res = {}
        for cap in (0.0, 0.6):
            bc = BT.BtCfg(start=start, cost_model="flat", cost_bp=0.0, etf_cost_bp=0.0, borrow_bp_yr=0.0, split="2100-01-01", max_gross=cap)
            res[cap] = BT.run(self.ctx.p, self.ctx.res, sc, self.ctx.member, BT.schedule(self.ctx.p, every=1), cfg, bc)
        self.assertGreater(res[0.0].daily.gross_all.mean(), 0.9, "uncapped: 1.0 of stocks + the hedge")
        self.assertLess(res[0.6].daily.gross_all.max(), 0.6 * 1.02, "capped: stocks + ETFs never above the cap (after one bar of drift)")
        k = self.ctx.p.col[M["stocks"][1]]
        r0, r6 = res[0.0].bar_pnl, res[0.6].bar_pnl
        nz = np.abs(r0) > 1e-9
        self.assertTrue(np.all(np.abs(r6[nz] / r0[nz]) < 1.0), "the whole book is scaled, not just one side")

    def test_etf_hedged_book_has_little_market_beta(self):
        bt, a, b, bc = self.run_bt(hedge="etf", every=26)
        self.assertLess(abs(bt.stats.get("beta_spy", 1.0)), 0.15)


class TestStrategies(unittest.TestCase):
    def check_no_lookahead(self, strat):
        c1 = ctx_for(strat)
        p = c1.p
        t0 = p.ts[len(p.ts) - 26 * 12 + 9]
        c2 = ctx_for(strat, perturb_after(M["frames"], t0))
        i0 = int(np.searchsorted(p.ts, t0)) + 1
        s1, s2 = strat.score(c1), strat.score(c2)
        self.assertTrue(same(s1[:i0], s2[:i0]), f"{strat.NAME}: the score at a bar changed when only later data changed")
        self.assertTrue(np.isfinite(s1[i0 - 1]).sum() >= 5)

    def test_momentum_no_lookahead(self):
        self.check_no_lookahead(ResidualMomentum({"universe_n": 50, "form_days": 60, "skip_days": 5, "beta_days": 63}))

    def test_mean_reversion_no_lookahead_both_modes(self):
        self.check_no_lookahead(ResidualMeanReversion({"universe_n": 50, "mode": "ou", "ou_window": 52}))
        self.check_no_lookahead(ResidualMeanReversion({"universe_n": 50, "mode": "xs", "xs_bars": 13}))

    def test_prediction_walk_forward_no_lookahead(self):
        d = tempfile.mkdtemp()
        st = ResidualPrediction({"universe_n": 50, "model": "ridge", "horizon": 4, "features": "rr_1,rr_4,rr_26,ou_s,hks,vol_surp", "retrain": "monthly", "train_years": 0.5,
                                 "min_train_years": 0.5, "start": str(M["days"][0].date()), "beta_days": 63}, folder=d)
        self.check_no_lookahead(st)

    def test_ou_mean_reversion_finds_the_planted_reversion(self):
        strat = ResidualMeanReversion({"universe_n": 50, "mode": "ou", "ou_window": 130})
        ctx = ctx_for(strat)
        sc = strat.score(ctx)
        from resarb import research as R
        ic = R.score_ic(ctx, sc, str(ctx.p.dates[150].date()), str(ctx.p.dates[250].date()), horizons=(4,))
        self.assertGreater(ic.IS_ic.iloc[0], 0.02, "the synthetic residuals mean-revert: the s-score must predict it")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class FakeTC:
    def __init__(self, equity=100_000.0, last_equity=None, positions=None, is_open=True, close_at=None):
        self.equity, self.last_equity = equity, last_equity if last_equity is not None else equity
        self.pos = dict(positions or {})
        self.is_open, self.close_at = is_open, close_at
        self.submitted, self.cancelled = [], []
        self.raise_on: dict = {}                                                               # symbol -> (exception, accepted?)
        self.stuck_open = False

    def get_account(self):
        return SimpleNamespace(account_number="PA0TEST01", status="ACTIVE", equity=self.equity, last_equity=self.last_equity, cash=self.equity, buying_power=4 * self.equity,
                               multiplier="4", shorting_enabled=True, trading_blocked=False, account_blocked=False, pattern_day_trader=False)

    def get_clock(self):
        nc = (self.close_at or pd.Timestamp.now().normalize() + pd.Timedelta(hours=16)).tz_localize(LV.NY)
        return SimpleNamespace(is_open=self.is_open, next_open=nc + pd.Timedelta(hours=17, minutes=30), next_close=nc, timestamp=pd.Timestamp.now(tz=LV.NY))

    def get_all_positions(self):
        return [SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.pos.items() if q]

    def get_orders(self, req):
        st = str(getattr(req.status, "value", req.status))
        if st == "open":
            return [o for o in self.submitted if o.status == "new"] if self.stuck_open else []
        return list(self.submitted)

    def submit_order(self, req):
        side = str(getattr(req.side, "value", req.side))
        o = SimpleNamespace(id=uuid.uuid4().hex, client_order_id=req.client_order_id, symbol=req.symbol, qty=req.qty, side=side, status="filled" if not self.stuck_open else "new",
                            filled_qty=req.qty, filled_avg_price=100.0, submitted_at=pd.Timestamp.now(tz="UTC"), filled_at=pd.Timestamp.now(tz="UTC"))
        exc = self.raise_on.get(req.symbol)
        if exc is not None:
            e, accepted = exc
            if accepted:
                self.submitted.append(o)
            raise e
        self.submitted.append(o)
        q = int(req.qty) * (1 if side == "buy" else -1)
        self.pos[req.symbol] = self.pos.get(req.symbol, 0) + q
        return o

    def get_order_by_client_id(self, cid):
        for o in self.submitted:
            if o.client_order_id == cid:
                return o
        raise KeyError(cid)

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)


class APIErr(Exception):
    def __init__(self, code, msg="rejected"):
        super().__init__(msg)
        self.status_code = code


IEX_SHARE = {}


def fake_fetch(client, symbols, start, end=None, feed="sip", minutes=15):
    """The synthetic SIP bars; the 'IEX' feed has the same prices and 2-5% of the volume (a fixed share per symbol)."""
    out = {}
    for s in symbols:
        f = M["frames"].get(s)
        if f is None:
            continue
        f = f[(f.index >= pd.Timestamp(start)) & (f.index < pd.Timestamp(end))]
        if feed == "iex":
            f = f.copy()
            f["volume"] = f["volume"] * IEX_SHARE.setdefault(s, 0.02 + 0.03 * (hash(s) % 100) / 100)
        out[s] = f
    return out


class TestLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tc = FakeTC(close_at=M["days"][-1] + pd.Timedelta(hours=16))
        self._orig = (D.trading_client, D.data_client, D.fetch_bars, D.update_bars)
        D.trading_client = lambda prefix, paper=True: self.tc
        D.data_client = lambda prefix=None: object()
        D.fetch_bars = fake_fetch
        D.update_bars = lambda *a, **k: {}
        reg = os.path.join(D.DATA_DIR, "resarb_accounts.json")
        if os.path.exists(reg):
            os.remove(reg)
        for k in [k for k in os.environ if k.startswith("RESMR_") or k.startswith("RESMOM_")]:
            del os.environ[k]
        self.now = M["days"][-1] + pd.Timedelta(hours=11, minutes=0, seconds=30)

    def tearDown(self):
        D.trading_client, D.data_client, D.fetch_bars, D.update_bars = self._orig
        for k in [k for k in os.environ if k.startswith("RESMR_") or k.startswith("RESMOM_")]:
            del os.environ[k]

    def engine(self, armed=False, values=None, **kw):
        v = {"universe_n": 50, "mode": "xs", "xs_bars": 13, "method": "quantile", "q_in": 0.2, "q_out": 0.3, "rebalance": "15min", "beta_days": 63}
        v.update(values or {})
        st = ResidualMeanReversion(v, folder=self.tmp)
        e = LV.LiveEngine(st, LV.LiveSettings(armed=armed, fill_wait_s=0.0, **kw), log=lambda m, k="info": None)
        e.connect()
        return e

    def own_keys(self):
        os.environ["RESMR_ALPACA_API_KEY"] = "k"
        os.environ["RESMR_ALPACA_SECRET_KEY"] = "s"

    def test_data_bar_matches_the_backtest_timing(self):
        h = M["ts"]
        day = M["days"][-1]
        k = day + pd.Timedelta(hours=11)
        self.assertEqual(LV.data_bar_for(k, 0, h), k - pd.Timedelta(minutes=15))
        self.assertEqual(LV.data_bar_for(k, 1, h), k - pd.Timedelta(minutes=30))
        o = day + pd.Timedelta(hours=9, minutes=30)
        prev = M["days"][-2]
        self.assertEqual(LV.data_bar_for(o, 0, h), prev + pd.Timedelta(hours=15, minutes=45))
        self.assertEqual(LV.data_bar_for(o, 1, h), prev + pd.Timedelta(hours=15, minutes=30))
        self.assertEqual(LV.data_bar_for(o + pd.Timedelta(minutes=15), 1, h), prev + pd.Timedelta(hours=15, minutes=45))

    def test_live_targets_equal_the_backtest_decision(self):
        e = self.engine()
        plan = e.plan(self.now)
        self.assertTrue(plan.decision)
        st = e.s
        ctx = ctx_for(st)
        p = ctx.p
        td = int(np.searchsorted(p.ts, plan.data_bar))
        sc = st.score(ctx)
        is_etf = np.array([s in D.ETFS for s in p.syms])
        sidx = np.where(~is_etf)[0]
        w = P.decide(sc[td, sidx], ctx.member[td, sidx], np.zeros(len(sidx)), st.port_cfg(), ctx.res.sector_col[sidx], np.isfinite(p.open[td, sidx]))
        full = np.zeros(len(p.syms)); full[sidx] = w
        full[ctx.res.etf_idx] += P.hedge_etf(whole_shares(full, p.close[td], plan.capital, is_etf), ctx.res, p.day[td])
        live = plan.book.set_index("symbol").target_w
        for s, j in p.col.items():
            if abs(full[j]) > 1e-12 or s in live.index:
                self.assertAlmostEqual(float(live.get(s, 0.0)), full[j], places=9, msg=s)

    def test_iex_volumes_are_scaled_back_to_sip_size(self):
        e = self.engine()
        e.plan(self.now)
        s = M["stocks"][4]
        self.assertAlmostEqual(e._iex_scale[s], 1 / IEX_SHARE[s], places=6)
        today = e.live_bars(self.now)[s]
        sip = M["frames"][s].loc[today.index, "volume"]
        self.assertTrue(np.allclose(today.volume.values, sip.values, rtol=1e-9))

    def test_live_volume_features_match_the_backtest_on_the_iex_feed(self):
        e = self.engine(values={"mode": "ou", "ou_window": 52})
        plan = e.plan(self.now)
        ctx_l = e.context(self.now)
        ctx_b = ctx_for(e.s)
        t = int(np.searchsorted(ctx_b.p.ts, plan.data_bar))
        tl = int(np.searchsorted(ctx_l.p.ts, plan.data_bar))
        vb = X.build(ctx_b.p, ctx_b.res, 0, names=["vol_surp"])["vol_surp"][t]
        vl = X.build(ctx_l.p, ctx_l.res, 0, names=["vol_surp"])["vol_surp"][tl]
        jb = [ctx_b.p.col[s] for s in M["stocks"]]
        jl = [ctx_l.p.col[s] for s in M["stocks"]]
        self.assertTrue(np.allclose(vb[jb], vl[jl], atol=1e-5))

    def test_a_manual_dry_plan_after_the_close_uses_the_newest_bar(self):
        e = self.engine()
        evening = M["days"][-1] + pd.Timedelta(hours=18)
        self.tc.is_open = False
        plan = e.plan(evening, force=True)
        self.assertEqual(plan.data_bar, M["ts"][-1])
        self.assertTrue(any("outside the session" in n for n in plan.notes))
        with self.assertRaises(RuntimeError):
            e.plan(evening)                                                                    # the loop itself never decides outside the session

    def test_live_targets_equal_the_backtest_with_positions_and_partial_trading(self):
        self.tc.pos = {M["stocks"][2]: 150, M["stocks"][6]: -90, M["stocks"][13]: 60, "SPY": -10}
        e = self.engine(values={"trade_rate": 0.3})
        plan = e.plan(self.now)
        st = e.s
        ctx = ctx_for(st)
        p = ctx.p
        td = int(np.searchsorted(p.ts, plan.data_bar))
        cap = plan.capital
        held = np.zeros(len(p.syms))
        for s, q in self.tc.pos.items():
            held[p.col[s]] = q * p.close[td, p.col[s]] / cap
        is_etf = np.array([s in D.ETFS for s in p.syms])
        sidx = np.where(~is_etf)[0]
        trad = np.isfinite(p.open[td, sidx])
        w = P.approach(held[sidx], P.decide(st.score(ctx)[td, sidx], ctx.member[td, sidx], held[sidx], st.port_cfg(), ctx.res.sector_col[sidx], trad), 0.3, 1.0, trad)
        full = np.zeros(len(p.syms)); full[sidx] = w
        full[ctx.res.etf_idx] += P.hedge_etf(whole_shares(full, p.close[td], cap, is_etf), ctx.res, p.day[td])
        live = plan.book.set_index("symbol").target_w
        for s, j in p.col.items():
            if abs(full[j]) > 1e-12 or s in live.index:
                self.assertAlmostEqual(float(live.get(s, 0.0)), full[j], places=9, msg=s)

    def test_dry_run_sends_nothing(self):
        e = self.engine(armed=False)
        plan = e.plan(self.now)
        with self.assertRaises(RuntimeError):
            e.submit(plan)
        self.assertEqual(self.tc.submitted, [])

    def test_armed_refuses_shared_keys_and_a_claimed_account(self):
        e = self.engine(armed=True)
        self.assertTrue(any("own keys" in x for x in e.arm_problems()))
        plan = e.plan(self.now)
        with self.assertRaises(RuntimeError):
            e.submit(plan)
        self.own_keys()
        self.assertEqual(e.arm_problems(), [])
        reg = os.path.join(D.DATA_DIR, "resarb_accounts.json")
        import json
        json.dump({"paper:PA0TEST01": {"strategy": "ResMomentum"}}, open(reg, "w"))
        self.assertTrue(any("claimed" in x for x in e.arm_problems()))
        with self.assertRaises(RuntimeError):
            e.submit(plan)
        self.assertEqual(self.tc.submitted, [])

    def test_armed_submit_reduces_first_journals_and_respects_etb(self):
        self.own_keys()
        hard = M["stocks"][0]                                                                  # not easy to borrow in the synthetic asset list
        self.tc.pos = {M["stocks"][5]: 40}
        e = self.engine(armed=True)
        plan = e.plan(self.now)
        o = plan.orders
        red = list(o.reason)
        self.assertEqual(red, sorted(red, key=lambda r: r != "reduce"), "reducing orders first")
        if (o.symbol == hard).any():
            r = o[o.symbol == hard].iloc[0]
            if r.side == "sell" and r.tgt_qty < 0:
                self.assertFalse(bool(r.ok))
        res = e.submit(plan)
        self.assertTrue(all(r["cid"].startswith("RESMR-") for r in res))
        self.assertEqual(len(res), int(o.ok.sum()))
        rows = [__import__("json").loads(x) for x in open(e.path("journal", "jsonl"), encoding="utf-8")]
        self.assertEqual(sum(1 for r in rows if r["kind"] == "order"), len(res))
        self.assertIsNone(e.load_state()["pending"])
        self.assertEqual(self.tc.submitted[0].client_order_id, o.cid.iloc[0])

    def test_compare_rebuilds_the_same_decision_from_the_journal(self):
        self.own_keys()
        self.tc.pos = {M["stocks"][5]: 40, M["stocks"][9]: -25}
        e = self.engine(armed=True)
        plan = e.plan(self.now)
        e.submit(plan)
        from resarb import compare as C
        out = C.compare(e.s, "paper", log=lambda m: None)
        s = out["summary"]
        self.assertEqual(s["decisions"], 1)
        self.assertAlmostEqual(abs(plan.expo["beta_after_hedge"]), 0.0, places=9, msg="the hedge covers the whole shares actually held")
        self.assertAlmostEqual(s["target_corr"], 1.0, places=9, msg="same data + same held positions = the same targets")
        self.assertAlmostEqual(s["same_side"], 1.0)
        self.assertLess(s["diff_gross"], 1e-9)
        self.assertAlmostEqual(s["score_corr"], 1.0, places=9)
        self.assertEqual(s["fills"], int(plan.orders.ok.sum()))
        f = out["fills"]
        r = f.iloc[0]
        o = M["frames"][r.symbol].loc[plan.trade_bar, "open"]
        self.assertAlmostEqual(r.bt_open, o, places=9)
        self.assertAlmostEqual(r.slip_vs_bt_open_bp, (100.0 / o - 1) * 1e4 * (1 if r.side == "buy" else -1), places=6)

    def test_ambiguous_submit_is_looked_up_and_a_4xx_is_rejected(self):
        self.own_keys()
        e = self.engine(armed=True)
        plan = e.plan(self.now)
        o = plan.orders[plan.orders.ok]
        self.assertGreaterEqual(len(o), 2)
        a, b = o.symbol.iloc[0], o.symbol.iloc[1]
        self.tc.raise_on = {a: (ConnectionError("reset"), True), b: (APIErr(422, "insufficient qty"), False)}
        res = {r["symbol"]: r for r in e.submit(plan)}
        self.assertEqual(res[a]["status"], "filled", "the order went through although the answer was lost")
        self.assertEqual(res[b]["status"], "rejected")

    def test_daily_loss_halt_allows_only_reducing_orders(self):
        self.own_keys()
        self.tc.equity, self.tc.last_equity = 90_000.0, 100_000.0
        self.tc.pos = {M["stocks"][7]: 30}
        e = self.engine(armed=True)
        plan = e.plan(self.now)
        self.assertTrue(plan.halted)
        bad = plan.orders[(plan.orders.reason != "reduce") & plan.orders.ok]
        self.assertEqual(len(bad), 0)

    def test_drawdown_stop_flattens_and_stays_flat(self):
        self.tc.pos = {M["stocks"][7]: 30, "SPY": -5}
        e = self.engine()
        st = e.load_state(); st["hwm"] = 200_000.0; e.save_state(st)
        plan = e.plan(self.now)
        self.assertTrue(plan.stopped)
        self.assertTrue((plan.orders.tgt_qty == 0).all())
        self.assertEqual(set(plan.orders.symbol), {M["stocks"][7], "SPY"})
        self.assertIsNotNone(e.load_state()["stopped"], "the stop is remembered until the drawdown base is reset")
        self.tc.equity = 300_000.0
        self.assertTrue(e.plan(self.now).stopped, "a recovered equity does not lift the stop by itself")
        e.reset_drawdown()
        self.assertFalse(e.plan(self.now).stopped)

    def test_monthly_strategy_does_not_rebalance_on_an_ordinary_day_in_live(self):
        e = self.engine(values={"rebalance": "monthly", "trade_time": "11:00"})
        plan = e.plan(self.now)
        is_month_end = M["days"][-1] in set(D.month_end_sessions(M["days"][0], M["days"][-1]))
        self.assertEqual(plan.decision, is_month_end)
        if not is_month_end:
            self.assertEqual(len(plan.orders), 0)

    def test_drawdown_stop_flattens_between_scheduled_rebalances(self):
        self.tc.pos = {M["stocks"][7]: 30, "SPY": -5}
        e = self.engine(values={"rebalance": "monthly", "trade_time": "11:00"})
        st = e.load_state(); st["hwm"] = 200_000.0; e.save_state(st)
        plan = e.plan(self.now)
        self.assertTrue(plan.stopped and plan.decision)
        self.assertEqual(set(plan.orders.symbol), {M["stocks"][7], "SPY"})
        self.assertTrue((plan.orders.tgt_qty == 0).all())

    def test_live_momentum_score_equals_the_backtest_score(self):
        v = {"universe_n": 50, "form_days": 60, "skip_days": 5, "beta_days": 63, "rebalance": "daily", "trade_time": "11:00", "feed": "iex"}
        st = ResidualMomentum(v, folder=self.tmp)
        e = LV.LiveEngine(st, LV.LiveSettings(armed=False, fill_wait_s=0.0), log=lambda m, k="info": None)
        e.connect()
        plan = e.plan(self.now)
        ctx_l = e.context(self.now)
        ctx_b = ctx_for(st)
        sl = st.score(ctx_l)[int(np.searchsorted(ctx_l.p.ts, plan.data_bar))]
        sb = st.score(ctx_b)[int(np.searchsorted(ctx_b.p.ts, plan.data_bar))]
        for s in M["stocks"]:
            a, b = sl[ctx_l.p.col[s]], sb[ctx_b.p.col[s]]
            self.assertTrue((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, f"{s}: live {a} vs backtest {b}")

    def test_kill_closes_only_strategy_symbols(self):
        self.tc.pos = {M["stocks"][3]: 10, "SPY": -4, "ZZZZ": 7}
        e = self.engine()
        out = {r["symbol"]: r["status"] for r in e.kill()}
        self.assertEqual(out[M["stocks"][3]], "close sent")
        self.assertEqual(out["SPY"], "close sent")
        self.assertIn("left alone", out["ZZZZ"])
        self.assertEqual(self.tc.pos["ZZZZ"], 7)

    def test_orders_that_stay_open_block_trading(self):
        self.own_keys()
        self.tc.stuck_open = True
        self.tc.submitted.append(SimpleNamespace(id="x1", client_order_id="RESMR-OLD-1", symbol="SAA", status="new"))
        e = self.engine(armed=True)
        plan = e.plan(self.now)
        with self.assertRaises(RuntimeError):
            e.cancel_ours(wait_s=0.0)
        n = len(self.tc.submitted)
        with self.assertRaises(RuntimeError):
            e.submit(plan)
        self.assertEqual(len(self.tc.submitted), n)

    def test_one_armed_engine_per_strategy(self):
        p = os.path.join(self.tmp, "x.lock")
        a, b = LV.FileLock(p), LV.FileLock(p)
        self.assertTrue(a.acquire())
        self.assertFalse(b.acquire())
        a.release()
        self.assertTrue(b.acquire())
        b.release()

    def test_real_money_is_disabled(self):
        st = ResidualMeanReversion({}, folder=self.tmp)
        e = LV.LiveEngine(st, LV.LiveSettings(paper=False), log=lambda m, k="info": None)
        os.environ.pop("RESARB_ALLOW_LIVE", None)
        with self.assertRaises(RuntimeError):
            e.connect()


class TestGui(unittest.TestCase):
    def test_windows_build_for_all_three_strategies(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from resarb.ui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        for cls in (ResidualMomentum, ResidualMeanReversion, ResidualPrediction):
            w = MainWindow(cls, tempfile.mkdtemp())
            vals = w.settings.values()
            self.assertEqual(set(vals), {p.key for p in cls.all_params()})
            for name in cls.PRESETS:
                w.settings.preset.setCurrentText(name)
                w.settings.apply_preset()
                for k, v in cls.PRESETS[name].items():
                    self.assertEqual(str(w.settings.values()[k]), str(v), f"{cls.NAME} preset {name}: {k}")
            self.assertFalse(w.live.armed.isChecked(), "the window must start DRY RUN")
            w.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
