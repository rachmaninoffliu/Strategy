"""
Tests for the mean-reversion models (resarb.mr). FAKES ONLY: synthetic daily + 15-minute bars in a temp folder and a simulated broker; no network, no keys, no orders.

The central test replays a backtest session by session through the LIVE engine (morning step, decision step, auctions, after-close step) and requires every order and fill to equal
the backtest's trades, and the final equity to match.

  python -m unittest tests.test_mr -v        (from Residual-StatArb/core)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import synth  # noqa: E402,F401  (must come before resarb: it points the data dir at a temp folder)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from resarb import data as D  # noqa: E402
from resarb.mr import engine as E  # noqa: E402
from resarb.mr import live as ML  # noqa: E402
from resarb.mr.market import DAILY_FILE, Market, session_closes, snapshot_from_bars  # noqa: E402
from resarb.mr.models import MRModel, Pos, Signals, decide  # noqa: E402

SYMS = [f"MR{c}" for c in "ABCDEFGHIJKL"]
FINAL = ML.FINAL


def make_mr_market(syms: list, start: str = "2023-01-03", end: str = "2024-03-28", seed: int = 7) -> dict:
    """Mean-reverting daily closes with Brownian-bridge 15-minute paths (early-close sessions end at 13:00). Writes the daily file and the bar files into the temp data dir."""
    rng = np.random.default_rng(seed)
    days = D.sessions(start, end)
    closes = session_closes(days[0], days[-1])
    daily, bars = [], {}
    for s in syms:
        p, x = 40 + 80 * rng.random(), 0.0
        rows = []
        for d in days:
            ct = closes.get(d, d + pd.Timedelta(hours=16))
            n = int((ct - (d + pd.Timedelta(hours=9, minutes=30))) / pd.Timedelta(minutes=15))
            xn = 0.6 * x + rng.normal(0, 0.02)
            target = p * np.exp(xn - x + 0.0004)
            o0 = p * np.exp(rng.normal(0, 0.006))
            w = np.cumsum(rng.normal(0, 0.003, n))
            w -= np.linspace(1 / n, 1, n) * w[-1]
            path = np.r_[o0, o0 * np.exp(np.linspace(0, np.log(target / o0), n + 1)[1:] + w)]
            hi = np.maximum(path[:-1], path[1:]) * (1 + np.abs(rng.normal(0, 0.001, n)))
            lo = np.minimum(path[:-1], path[1:]) * (1 - np.abs(rng.normal(0, 0.001, n)))
            idx = d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15) * np.arange(n)
            v = rng.integers(20_000, 90_000, n).astype(float)
            f = pd.DataFrame({"open": path[:-1], "high": hi, "low": lo, "close": path[1:], "volume": v, "vwap": (path[:-1] + path[1:]) / 2, "trades": v / 50}, index=idx)
            rows.append(f)
            daily.append(dict(symbol=s, date=d, open=path[0], high=hi.max(), low=lo.min(), close=path[-1], volume=v.sum()))
            p, x = path[-1], xn
        bars[s] = pd.concat(rows)
        bars[s].to_parquet(D.bars_path(s, "sip"))
    dd = pd.DataFrame(daily)
    old = pd.read_parquet(DAILY_FILE) if os.path.exists(DAILY_FILE) else pd.DataFrame(columns=dd.columns)
    pd.concat([old[~old.symbol.isin(syms)], dd], ignore_index=True).to_parquet(DAILY_FILE)
    return dict(days=days, daily=dd, bars=bars, closes=closes)


MK = make_mr_market(SYMS)


class FakeBroker:
    """A simulated paper account: market-on-open / market-on-close orders fill in the auctions at the official open / close the backtest uses, with the same per-side cost."""
    keys = "own"

    def __init__(self, data: dict, cash: float, cost_bp: float):
        self.daily_df = data["daily"]
        self.dix = data["daily"].set_index(["symbol", "date"]).sort_index()
        self.b = data["bars"]
        self.closes = data["closes"]
        self.cash, self.cost = float(cash), cost_bp / 1e4
        self.pos: dict = {}
        self.ords: list = []
        self.fills: list = []
        self.now = None
        self.n = 0

    def _px(self, s: str) -> float:
        day = self.now.normalize()
        f = self.b[s]
        x = f[(f.index >= day) & (f.index <= self.now - pd.Timedelta(minutes=15))]
        if len(x):
            return float(x.close.iloc[-1])
        d = self.daily_df[(self.daily_df.symbol == s) & (self.daily_df.date < day)]
        return float(d.close.iloc[-1])

    def account(self) -> dict:
        eq = self.cash + sum(q * self._px(s) for s, (q, _) in self.pos.items() if q)
        return dict(number="PA-MR-TEST", status="ACTIVE", equity=eq, last_equity=eq, cash=self.cash, buying_power=2 * eq, multiplier=2.0, shorting=True, blocked=False)

    def clock(self) -> dict:
        day = self.now.normalize()
        c = self.closes.get(day)
        op = day + pd.Timedelta(hours=9, minutes=30)
        return dict(is_open=c is not None and op <= self.now < c, next_open=op, next_close=c if c is not None else op, ts=self.now)

    def positions(self) -> dict:
        return {s: dict(qty=q, avg_px=a, mv=q * self._px(s)) for s, (q, a) in self.pos.items() if q}

    def submit(self, symbol, qty, side, tif, cid) -> dict:
        assert not any(o["cid"] == cid for o in self.ords), f"duplicate client id {cid}"
        self.n += 1
        o = dict(id=str(self.n), cid=cid, symbol=symbol, side=side, qty=float(qty), status="accepted", filled_qty=0.0, filled_avg_price=np.nan, submitted_at=self.now,
                 filled_at=None, tif=tif)
        self.ords.append(o)
        return dict(id=o["id"], status=o["status"])

    def orders(self, after, status: str = "all") -> list:
        out = [dict(o) for o in self.ords if o["submitted_at"] >= after]
        if status == "open":
            out = [o for o in out if o["status"] not in FINAL]
        return out

    def order_by_cid(self, cid):
        return next((dict(o) for o in self.ords if o["cid"] == cid), None)

    def cancel(self, oid) -> None:
        for o in self.ords:
            if o["id"] == oid and o["status"] not in FINAL:
                o["status"] = "canceled"

    def bars(self, symbols, start, end, feed) -> dict:
        return {s: self.b[s][(self.b[s].index >= start) & (self.b[s].index < end)] for s in symbols if s in self.b}

    def daily(self, symbols, start) -> pd.DataFrame:
        d = self.daily_df
        return d[d.symbol.isin(symbols) & (d.date >= start) & (d.date < self.now.normalize())].copy()

    def assets(self) -> pd.DataFrame:
        return pd.DataFrame({"symbol": list(self.b), "shortable": True, "etb": True})

    def auction(self, tif: str, day: pd.Timestamp) -> None:
        for o in self.ords:
            if o["tif"] != tif or o["status"] in FINAL:
                continue
            px = float(self.dix.loc[(o["symbol"], day), "open" if tif == "opg" else "close"])
            q = int(o["qty"]) * (1 if o["side"] == "buy" else -1)
            self.cash -= q * px + abs(q * px) * self.cost
            old_q, old_a = self.pos.get(o["symbol"], (0, np.nan))
            nq = old_q + q
            self.pos[o["symbol"]] = (nq, px if old_q == 0 else old_a)
            o.update(status="filled", filled_qty=float(abs(q)), filled_avg_price=px, filled_at=self.now)
            self.fills.append(dict(day=day, symbol=o["symbol"], side=o["side"], qty=abs(q), px=px, tif=tif, cid=o["cid"]))


def replay(m: MRModel, feed: str, start_i: int = 80, capital: float = 100_000.0):
    """Backtest the model, then drive the live engine through the same sessions with the simulated broker. Returns (backtest result, broker, engine)."""
    days = MK["days"]
    mk = Market(SYMS, minute=m.minute, start=str(days[0].date()))
    bt = E.run(m, mk, start=str(days[start_i].date()), capital=capital)
    br = FakeBroker(MK, capital, m.cost_auction_bp)
    folder = tempfile.mkdtemp(prefix="mr_live_")
    eng = ML.MRLiveEngine(m, ML.MRLiveSettings(armed=True, feed=feed, history_days=4000), folder, log=lambda *a: None, broker=br, clock_fn=lambda: br.now)
    br.now = days[start_i] + pd.Timedelta(hours=8)
    eng.connect()
    for d in days[start_i:]:
        tm = eng.times(d)
        br.now = d + pd.Timedelta(hours=8, minutes=50)
        eng.tick()
        br.now = tm["open"]
        br.auction("opg", d)
        br.now = tm["decision"] + pd.Timedelta(seconds=1)
        assert eng.tick() == "decision", f"no decision on {d:%Y-%m-%d}"
        br.now = tm["close"]
        br.auction("cls", d)
        br.now = tm["close"] + pd.Timedelta(minutes=6)
        eng.tick()
    return bt, br, eng


def trade_sets(bt, br) -> tuple:
    tr = bt.trades
    b_in = {(pd.Timestamp(r.entry_date), r.symbol, int(r.shares)) for r in tr.itertuples()}
    b_out = {(pd.Timestamp(r.exit_date), r.symbol, int(r.shares), "opg" if r.exit_auction == "open" else "cls") for r in tr.itertuples() if pd.notna(r.exit_date)}
    f = pd.DataFrame(br.fills)
    ent = f[f.cid.str.contains(r"-e\d+$")]
    ext = f[~f.cid.str.contains(r"-e\d+$")]
    l_in = {(pd.Timestamp(r.day), r.symbol, int(r.qty)) for r in ent.itertuples()}
    l_out = {(pd.Timestamp(r.day), r.symbol, int(r.qty), r.tif) for r in ext.itertuples()}
    return b_in, l_in, b_out, l_out


class TestParity(unittest.TestCase):
    def check(self, m: MRModel, feed: str):
        bt, br, eng = replay(m, feed)
        self.assertGreater(len(bt.trades), 30, "the synthetic market should make the model trade")
        b_in, l_in, b_out, l_out = trade_sets(bt, br)
        self.assertEqual(sorted(b_in - l_in), [], "entries in the backtest that live did not make")
        self.assertEqual(sorted(l_in - b_in), [], "entries live made that the backtest did not")
        self.assertEqual(sorted(b_out - l_out), [], "exits in the backtest that live did not make")
        self.assertEqual(sorted(l_out - b_out), [], "exits live made that the backtest did not")
        last = MK["days"][-1]
        br.now = last + pd.Timedelta(hours=17)
        live_eq = br.cash + sum(q * float(br.dix.loc[(s, last), "close"]) for s, (q, _) in br.pos.items() if q)
        self.assertAlmostEqual(live_eq / bt.equity.iloc[-1], 1.0, places=9)
        jr = eng.read_journal(("trade_opened",))
        self.assertEqual(len(jr), len(b_in))
        return bt, br, eng

    def test_rule_model_1530_delayed_sip(self):
        m = MRModel(name="ibs rule", universe="list:" + ",".join(SYMS), minute=1530, entry_signal="ibs", entry_op="<", entry_value=0.25, trend_n=0, exit_signal="ibs", exit_op=">",
                    exit_value=0.7, exit_sma_n=5, max_hold=4, slots=3, rank_by="signal")
        self.check(m, "sip_delayed")

    def test_overnight_model_1545_realtime(self):
        m = MRModel(name="rsi2 overnight", universe="list:" + ",".join(SYMS), minute=1545, entry_signal="rsi2", entry_op="<", entry_value=25, hold="overnight", exit_signal="",
                    slots=2, rank_by="vol_low")
        self.check(m, "sip")

    def test_rule_model_1545_hybrid_feed(self):
        """sip+iex stitches delayed SIP bars and real-time IEX bars: no bar may be lost or doubled at the seam (here both feeds carry the same prints)."""
        m = MRModel(name="ibs hybrid", universe="list:" + ",".join(SYMS), minute=1545, entry_signal="ibs", entry_op="<", entry_value=0.3, trend_n=0, exit_signal="ibs", exit_op=">",
                    exit_value=0.6, exit_sma_n=0, max_hold=5, slots=4, rank_by="vol_low")
        self.check(m, "sip+iex")

    def test_ret_model_with_trend_and_time_stop(self):
        m = MRModel(name="ret2 trend", universe="list:" + ",".join(SYMS), minute=1530, entry_signal="ret2", entry_op="<", entry_value=-0.02, trend_n=50, exit_signal="",
                    exit_sma_n=0, max_hold=3, slots=4, rank_by="signal")
        self.check(m, "sip_delayed")


class TestModel(unittest.TestCase):
    def setUp(self):
        self.mk = Market(SYMS, minute=1545, start="2023-01-03")
        self.sig = Signals(self.mk)

    def test_no_leverage_and_ranking(self):
        m = MRModel(universe="list:" + ",".join(SYMS), entry_signal="ret1", entry_op="<", entry_value=10.0, slots=4, rank_by="signal")      # everything qualifies
        t = 150
        dec = decide(m, t, self.sig, np.ones(len(SYMS), bool), {}, 100_000.0)
        self.assertEqual(len(dec.entries), 4)
        cost = sum(abs(sh) * self.sig.last[t][self.mk.col[s]] for s, sh, _, _ in dec.entries)
        self.assertLessEqual(cost, (1 - m.cash_reserve) * 100_000.0 + 1e-6)
        r1 = self.sig.get("ret1")[t]
        want = [self.mk.syms[j] for j in np.argsort(r1)[:4]]                                    # the 4 most negative returns get the slots
        self.assertEqual([e[0] for e in dec.entries], want)
        self.assertEqual(len(dec.candidates), len(SYMS))
        self.assertEqual(sum(c[3] for c in dec.candidates), 4)

    def test_inverse_vol_sizing_stays_unlevered(self):
        m = MRModel(universe="list:" + ",".join(SYMS), entry_signal="ret1", entry_op="<", entry_value=10.0, slots=3, sizing="inv_vol", vol_target=0.5, max_slot_mult=3.0)
        t = 150
        dec = decide(m, t, self.sig, np.ones(len(SYMS), bool), {}, 100_000.0)                  # a huge vol target: every name wants 3 slots' worth
        cost = sum(abs(sh) * self.sig.last[t][self.mk.col[s]] for s, sh, _, _ in dec.entries)
        self.assertLessEqual(cost, (1 - m.cash_reserve) * 100_000.0 + 1e-6)
        self.assertEqual(len(dec.entries), 1)                                                   # the first takes the whole budget (3 x 1/3)
        m2 = MRModel(**{**m.to_dict(), "vol_target": 1e-6})                                     # a tiny target: every slot shrinks
        dec2 = decide(m2, t, self.sig, np.ones(len(SYMS), bool), {}, 100_000.0)
        self.assertLess(sum(abs(sh) * self.sig.last[t][self.mk.col[s]] for s, sh, _, _ in dec2.entries), 0.05 * 100_000.0)

    def test_combined_universe_mask(self):
        m = MRModel(universe="list:MRA,MRB+list:MRC")
        self.assertEqual(E.model_symbols(m), ["MRA", "MRB", "MRC"])
        mk = Market(SYMS, minute=1545, start="2023-06-01")
        mask = E.universe_mask(mk, m)
        self.assertTrue(mask[:, mk.col["MRC"]].all() and mask[:, mk.col["MRA"]].all())
        self.assertFalse(mask[:, mk.col["MRD"]].any())

    def test_default_feed(self):
        self.assertEqual(ML.default_feed(1530), "sip_delayed")
        self.assertEqual(ML.default_feed(1545), "sip+iex")
        self.assertEqual(ML.default_feed(1545, "sip"), "sip")                                   # a paid real-time plan stays
        self.assertEqual(ML.default_feed(1530, "sip+iex"), "sip_delayed")                       # exact data when it is available in time

    def test_short_model_refused_for_live(self):
        br = FakeBroker(MK, 100_000, 1.0)
        br.now = pd.Timestamp("2024-03-01 10:00")
        eng = ML.MRLiveEngine(MRModel(minute=1530, direction="short", entry_op=">", entry_value=0.8), ML.MRLiveSettings(), tempfile.mkdtemp(), log=lambda *a: None, broker=br,
                              clock_fn=lambda: br.now)
        self.assertTrue(any("short" in p for p in eng.model_problems()))

    def test_full_slots_and_held_names(self):
        m = MRModel(universe="list:" + ",".join(SYMS), entry_signal="ret1", entry_op="<", entry_value=10.0, slots=2, exit_signal="", exit_sma_n=0, max_hold=99)
        t = 150
        pos = {s: Pos(s, 10, t - 1, 50.0) for s in SYMS[:2]}
        dec = decide(m, t, self.sig, np.ones(len(SYMS), bool), pos, 100_000.0)
        self.assertEqual(dec.entries, [])
        self.assertTrue(all(c[0] not in pos for c in dec.candidates))

    def test_backtest_trades_have_every_field(self):
        m = MRModel(universe="list:" + ",".join(SYMS), entry_signal="ibs", entry_value=0.2, slots=3)
        r = E.run(m, self.mk, start="2023-05-01")
        need = {"trade_id", "symbol", "direction", "shares", "entry_date", "entry_auction", "entry_px", "entry_value", "signal", "signal_value", "rank", "decision_px", "exit_date",
                "exit_auction", "exit_px", "exit_reason", "held_sessions", "pnl", "ret"}
        self.assertTrue(need <= set(r.trades.columns))
        self.assertLess(r.stats["max_exposure"], 1.0 + 1e-9 + 0.05)
        self.assertGreater(len(r.trades), 10)

    def test_early_close_snapshot(self):
        d = pd.Timestamp("2023-11-24")                                                          # NYSE closes at 13:00
        b = MK["bars"]["MRA"]
        sn = snapshot_from_bars(b[b.index.normalize() == d], 1530, MK["closes"])
        day_bars = b[b.index.normalize() == d]
        self.assertAlmostEqual(sn.loc[d, "last"], day_bars.loc[d + pd.Timedelta(hours=12, minutes=15), "close"])   # 30 minutes before the close
        sn = snapshot_from_bars(day_bars, 1530)                                                  # without the calendar: 15:30 = the whole day (what the old code did)
        self.assertAlmostEqual(sn.loc[d, "last"], day_bars.close.iloc[-1])

    def test_feed_and_time_checks(self):
        br = FakeBroker(MK, 100_000, 1.0)
        br.now = pd.Timestamp("2024-03-01 10:00")
        eng = ML.MRLiveEngine(MRModel(minute=1545), ML.MRLiveSettings(feed="sip_delayed"), tempfile.mkdtemp(), log=lambda *a: None, broker=br, clock_fn=lambda: br.now)
        self.assertTrue(any("cut-off" in p for p in eng.model_problems()))
        eng = ML.MRLiveEngine(MRModel(minute=1530), ML.MRLiveSettings(feed="sip_delayed"), tempfile.mkdtemp(), log=lambda *a: None, broker=br, clock_fn=lambda: br.now)
        self.assertEqual(eng.model_problems(), [])
        tm = eng.times(pd.Timestamp("2024-03-01"))
        self.assertEqual(tm["decision"].strftime("%H:%M"), "15:46")
        self.assertEqual(tm["cutoff"].strftime("%H:%M:%S"), "15:49:30")
        self.assertTrue(any("leverage" in p for p in ML.MRLiveEngine(MRModel(minute=1530, max_gross=1.5), ML.MRLiveSettings(), tempfile.mkdtemp(), broker=br,
                                                                       clock_fn=lambda: br.now).model_problems()))

    def test_dry_run_sends_nothing(self):
        m = MRModel(name="dry", universe="list:" + ",".join(SYMS), minute=1530, entry_signal="ret1", entry_op="<", entry_value=10.0, slots=3)
        br = FakeBroker(MK, 100_000, 1.0)
        eng = ML.MRLiveEngine(m, ML.MRLiveSettings(armed=False, history_days=4000), tempfile.mkdtemp(), log=lambda *a: None, broker=br, clock_fn=lambda: br.now)
        d = MK["days"][200]
        br.now = d + pd.Timedelta(hours=8)
        eng.connect()
        br.now = eng.times(d)["decision"] + pd.Timedelta(seconds=2)
        self.assertEqual(eng.tick(), "decision")
        self.assertEqual(br.ords, [])
        self.assertEqual(len(eng.last_plan.orders), 3)
        with self.assertRaises(RuntimeError):
            eng.submit(eng.last_plan)

    def test_past_cutoff_refused(self):
        m = MRModel(name="late", universe="list:" + ",".join(SYMS), minute=1530, entry_signal="ret1", entry_op="<", entry_value=10.0, slots=3)
        br = FakeBroker(MK, 100_000, 1.0)
        eng = ML.MRLiveEngine(m, ML.MRLiveSettings(armed=True, history_days=4000), tempfile.mkdtemp(), log=lambda *a: None, broker=br, clock_fn=lambda: br.now)
        d = MK["days"][201]
        br.now = d + pd.Timedelta(hours=8)
        eng.connect()
        br.now = eng.times(d)["decision"] + pd.Timedelta(seconds=2)
        plan = eng.plan()
        br.now = eng.times(d)["cutoff"] + pd.Timedelta(seconds=1)
        with self.assertRaises(RuntimeError):
            eng.submit(plan)
        self.assertEqual(br.ords, [])
        self.assertEqual(eng.tick(), "missed")


if __name__ == "__main__":
    unittest.main()
