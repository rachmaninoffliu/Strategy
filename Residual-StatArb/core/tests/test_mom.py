"""
Tests for the time-shared momentum book (resarb.mom). FAKES ONLY: synthetic daily, 15-minute SIP and IEX bars held in memory (plus the daily file in a temp folder), a
simulated broker; no network, no keys, no orders.

The central tests replay a backtest session by session through the LIVE engine (morning step, opening auction, every intraday check, the overnight decision, the closing
auction, the after-close step) and require every fill to equal a backtest trade - overnight and intraday - and the final equity to match. One of them ranks with the ML
forecast (a small model trained on the synthetic data and loaded from disk by both).

  python -m unittest tests.test_mom -v        (from Residual-StatArb/core)
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import synth  # noqa: E402,F401  (must come before resarb: it points the data dir at a temp folder)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from resarb import data as D  # noqa: E402
from resarb.mom import engine as E  # noqa: E402
from resarb.mom import live as ML  # noqa: E402
from resarb.mom import ml  # noqa: E402
from resarb.mom import presets as PR  # noqa: E402
from resarb.mom.features import feature_frame, member_mask, noise_sigma, rankable  # noqa: E402
from resarb.mom.market import MomMarket, lead_bars, slot_arrays  # noqa: E402
from resarb.mom.models import MomModel, check_slots, decide_overnight, intraday_shares, noise_action, pct_rank  # noqa: E402
from resarb.mr.market import DAILY_FILE, session_closes  # noqa: E402

STOCKS = [f"MB{c}" for c in "ABCDEFGHIJKLMNOP"]
SYMS = ["SPY", "QQQ"] + STOCKS
UNI = "list:" + ",".join(SYMS)
FINAL = ML.FINAL


def make_mom_market(start: str = "2022-01-03", end: str = "2024-04-30", seed: int = 11) -> dict:
    """Daily closes with overnight momentum (tonight's gap follows today's move) and Brownian-bridge 15-minute paths; QQQ trends intraday on some days (noise-area breakouts)
    and has IEX bars slightly off the SIP ones. Early closes end at 13:00. SPY has a down-trend in the middle (the trend filter must switch the overnight leg off)."""
    rng = np.random.default_rng(seed)
    days = D.sessions(start, end)
    closes = session_closes(days[0], days[-1])
    daily, sip, iex = [], {}, {}
    n = len(days)
    i_ = np.arange(n)                                                                           # session 501 = 2024-01-02
    spy_drift = np.select([(i_ >= 470) & (i_ < 505), (i_ >= 505) & (i_ < 545)], [-0.005, 0.005], 0.0009)
    mkt = rng.normal(0, 0.009, n) + spy_drift
    for s in SYMS:
        p = 50 + 150 * rng.random()
        prev_id = 0.0
        rows, irows = [], []
        beta = 1.0 if s in ("SPY", "QQQ") else 0.6 + 0.8 * rng.random()
        for i, d in enumerate(days):
            ct = closes.get(d, d + pd.Timedelta(hours=16))
            nb = int((ct - (d + pd.Timedelta(hours=9, minutes=30))) / pd.Timedelta(minutes=15))
            gap = 0.35 * prev_id + rng.normal(0, 0.004) + (0.3 * mkt[i] if s != "SPY" else 0.3 * mkt[i])
            o0 = p * np.exp(gap)
            trend = rng.normal(0, 0.012) + (0.01 * rng.choice([-1, 1]) if s == "QQQ" and rng.random() < 0.35 else 0.0)
            idr = beta * 0.7 * mkt[i] + trend
            target = o0 * np.exp(idr)
            w = np.cumsum(rng.normal(0, 0.0025, nb))
            w -= np.linspace(1 / nb, 1, nb) * w[-1]
            path = np.r_[o0, o0 * np.exp(np.linspace(0, np.log(target / o0), nb + 1)[1:] + w)]
            hi = np.maximum(path[:-1], path[1:]) * (1 + np.abs(rng.normal(0, 0.0008, nb)))
            lo = np.minimum(path[:-1], path[1:]) * (1 - np.abs(rng.normal(0, 0.0008, nb)))
            idx = d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15) * np.arange(nb)
            v = rng.integers(20_000, 90_000, nb).astype(float)
            f = pd.DataFrame({"open": path[:-1], "high": hi, "low": lo, "close": path[1:], "volume": v, "vwap": (path[:-1] + path[1:]) / 2, "trades": v / 50}, index=idx)
            rows.append(f)
            if s == "QQQ" and d >= pd.Timestamp("2023-06-01"):                                   # IEX history starts later (as the real one: 2024)
                jit = 1 + rng.normal(0, 0.0003, nb)
                g = f.copy()
                g[["open", "close", "vwap"]] = g[["open", "close", "vwap"]].mul(jit, axis=0)
                g["volume"] = (v * 0.03).round()
                irows.append(g)
            daily.append(dict(symbol=s, date=d, open=path[0], high=hi.max(), low=lo.min(), close=path[-1], volume=v.sum()))
            prev_id = path[-1] / path[0] - 1
            p = path[-1]
        sip[s] = pd.concat(rows)
        if irows:
            iex[s] = pd.concat(irows)
    dd = pd.DataFrame(daily)
    old = pd.read_parquet(DAILY_FILE) if os.path.exists(DAILY_FILE) else None                  # other test modules share the temp data dir
    keep = old[~old.symbol.isin(SYMS)] if old is not None else None
    (pd.concat([keep, dd], ignore_index=True) if keep is not None and len(keep) else dd).to_parquet(DAILY_FILE)
    return dict(days=days, daily=dd, sip=sip, iex=iex, closes=closes)


MK = make_mom_market()
EMPTY = pd.DataFrame(columns=D.BAR_COLS, dtype="float64")


def bars_fn(s: str, feed: str) -> pd.DataFrame:
    return (MK["iex"] if feed == "iex" else MK["sip"]).get(s, EMPTY)


def market(m: MomModel, start=None) -> MomMarket:
    return MomMarket(SYMS, minute=m.minute, start=start or str(MK["days"][0].date()), daily=MK["daily"], bars_fn=bars_fn, intraday_symbol=m.intraday_symbol,
                     intraday_feed=m.intraday_feed)


FOLDER = tempfile.mkdtemp(prefix="mom_models_")
SMALL = dict(max_iter=25, min_samples_leaf=40, learning_rate=0.1)


def ensure_models(m: MomModel) -> None:
    if ml.available_years(FOLDER, m) == [2023, 2024]:
        return
    df = E.training_frame(m, market(m))
    ml.train(FOLDER, m, df, years=[2023, 2024], params=SMALL, min_rows=500, log=lambda s: None)


class FakeBroker:
    """A simulated paper account: OPG / CLS orders fill in the auctions at the official open / close (auction cost per side); DAY market orders fill at once at the close of
    the newest complete bar of the intraday feed, plus / minus the half spread - the backtest's fill model. ``emu`` = market DAY orders in place of the auctions: those in
    the 15 minutes before (queued) to 20 minutes after the open / the last two minutes before the close fill at the official open / close plus the auction cost - as the
    backtest prices the model, so live must reproduce it exactly with either order type."""
    keys = "own"

    def __init__(self, cash: float, m: MomModel):
        self.dix = MK["daily"].set_index(["symbol", "date"]).sort_index()
        self.cash = float(cash)
        self.c_auc, self.h = m.cost_auction_bp / 1e4, m.half_spread_bp / 1e4
        self.isym = m.intraday_symbol
        self.m = m
        self.feed = m.intraday_feed
        self.pos: dict = {}
        self.ords: list = []
        self.hist: list = []                                                                    # filled orders from before the test (in the history, not in ords)
        self.fills: list = []
        self.now = None
        self.n = 0
        self.emu = False

    def _bar_px(self, s: str, feed: str = "sip") -> float:
        f = bars_fn(s, feed)
        day = self.now.normalize()
        x = f[(f.index >= day) & (f.index + pd.Timedelta(minutes=15) <= self.now)]
        if len(x):
            return float(x.close.iloc[-1])
        d = MK["daily"]
        d = d[(d.symbol == s) & (d.date < day)]
        return float(d.close.iloc[-1])

    def account(self) -> dict:
        eq = self.cash + sum(q * self._bar_px(s) for s, (q, _) in self.pos.items() if q)
        return dict(number="PA-MB-TEST", status="ACTIVE", equity=eq, last_equity=eq, cash=self.cash, buying_power=2 * eq, multiplier=2.0, shorting=True, blocked=False,
                    pdt=False, daytrade_count=0)

    def clock(self) -> dict:
        day = self.now.normalize()
        c = MK["closes"].get(day)
        op = day + pd.Timedelta(hours=9, minutes=30)
        return dict(is_open=c is not None and op <= self.now < c, next_open=op, next_close=c if c is not None else op, ts=self.now)

    def positions(self) -> dict:
        return {s: dict(qty=q, avg_px=a, mv=q * self._bar_px(s)) for s, (q, a) in self.pos.items() if q}

    def _fill(self, o: dict, px: float, fee: float) -> None:
        q = int(o["qty"]) * (1 if o["side"] == "buy" else -1)
        self.cash -= q * px + fee
        old_q, old_a = self.pos.get(o["symbol"], (0, np.nan))
        self.pos[o["symbol"]] = (old_q + q, px if old_q == 0 else old_a)
        o.update(status="filled", filled_qty=float(abs(q)), filled_avg_price=px, filled_at=self.now)
        self.fills.append(dict(day=self.now.normalize(), at=self.now, symbol=o["symbol"], side=o["side"], qty=abs(q), px=px, tif=o["tif"], cid=o["cid"]))

    def submit(self, symbol, qty, side, tif, cid) -> dict:
        assert not any(o["cid"] == cid for o in self.ords), f"duplicate client id {cid}"
        assert qty > 0
        self.n += 1
        o = dict(id=str(self.n), cid=cid, symbol=symbol, side=side, qty=float(qty), status="accepted", filled_qty=0.0, filled_avg_price=np.nan, submitted_at=self.now,
                 filled_at=None, tif=tif)
        self.ords.append(o)
        if tif == "day" and not self.clock()["is_open"] and not self._emulated_auction():
            pass                                                                                # outside the session a day order waits for the next open (Alpaca)
        elif tif == "day":
            k = self._emulated_auction()
            if k:
                px = float(self.dix.loc[(symbol, self.now.normalize()), k])
                self._fill(o, px, int(qty) * px * self.c_auc)
            else:
                p = self._bar_px(symbol, self.feed if symbol == self.isym else "sip")           # only the intraday symbol has IEX bars
                self._fill(o, p * (1 + self.h) if side == "buy" else p * (1 - self.h), 0.0)
        return dict(id=o["id"], status=o["status"], filled_qty=o["filled_qty"], filled_avg_price=o["filled_avg_price"])

    def _emulated_auction(self):
        if not self.emu:
            return None
        day = self.now.normalize()
        c = MK["closes"].get(day)
        op = day + pd.Timedelta(hours=9, minutes=30)
        if c is not None and c - pd.Timedelta(minutes=2) <= self.now < c:
            return "close"
        return "open" if op - pd.Timedelta(minutes=15) <= self.now < op + pd.Timedelta(minutes=20) else None   # sent before the open: queued for it

    def orders(self, after, status: str = "all") -> list:
        out = [dict(o) for o in self.hist + self.ords if o["submitted_at"] >= after]
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
        out = {}
        for s in symbols:
            f = bars_fn(s, feed)
            out[s] = f[(f.index >= start) & (f.index < end)]
        return out

    def daily(self, symbols, start) -> pd.DataFrame:
        d = MK["daily"]
        return d[d.symbol.isin(symbols) & (d.date >= start) & (d.date < self.now.normalize())].copy()

    def auction(self, tif: str, day: pd.Timestamp) -> None:
        for o in self.ords:
            if o["tif"] != tif or o["status"] in FINAL:
                continue
            px = float(self.dix.loc[(o["symbol"], day), "open" if tif == "opg" else "close"])
            self._fill(o, px, int(o["qty"]) * px * self.c_auc)


def replay(m: MomModel, start: str = "2024-01-02", capital: float = 100_000.0, shadow: MomModel = None, auction_orders: str = "auction"):
    """Backtest the book, then drive the live engine through the same sessions with the simulated broker (four months with IEX bars - live always has them - that include
    an SPY down-leg). ``auction_orders``: the live setting ('auction' = MOC / MOO orders, filled in the simulated auctions; 'market' = market DAY orders in their place).
    Returns (backtest result, broker, engine)."""
    days = MK["days"]
    bt = E.run(m, market(m), FOLDER, start=start, capital=capital)
    br = FakeBroker(capital, m)
    folder = tempfile.mkdtemp(prefix="mom_live_")
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)
    os.makedirs(os.path.join(FOLDER, "models"), exist_ok=True)
    for f in os.listdir(os.path.join(FOLDER, "models")):                                       # the live engine reads the same model files
        with open(os.path.join(FOLDER, "models", f), "rb") as a, open(os.path.join(folder, "models", f), "wb") as b:
            b.write(a.read())
    feed = "sip_delayed" if m.minute <= 1530 else "sip"
    eng = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=True, feed=feed, refresh_data=False, auction_orders=auction_orders), folder, log=lambda *a: None, broker=br,
                           clock_fn=lambda: br.now, bars_fn=bars_fn, shadow=shadow)
    br.emu = eng.emulate                                                                        # market orders in place of the auctions: filled at the auction prices
    s0 = days[days >= pd.Timestamp(start)]
    br.now = s0[0] + pd.Timedelta(hours=8)
    eng.connect()
    for d in s0:
        tm = eng.times(d)
        br.now = d + pd.Timedelta(hours=8, minutes=50)
        assert eng.tick() == "morning"
        if br.emu:
            br.now = tm["emu_open"] + pd.Timedelta(seconds=1)
            assert eng.tick() == "open"
        else:
            br.now = tm["open"]
            br.auction("opg", d)
            br.now = tm["open"] + pd.Timedelta(seconds=61)                                      # the market sale of what the opening auction left (nothing here)
            assert eng.tick() == "open"
        for slot, end, due in tm["checks"]:
            br.now = due + pd.Timedelta(seconds=1)
            assert eng.tick() == "check", f"no check {end} on {d:%Y-%m-%d}"
        br.now = tm["prefetch"] + pd.Timedelta(seconds=1)
        if m.overnight:
            assert eng.tick() == "prefetch"
        br.now = tm["decision"] + pd.Timedelta(seconds=1)
        assert eng.tick() == "decision", f"no decision on {d:%Y-%m-%d}"
        if br.emu:
            br.now = tm["emu_close"] + pd.Timedelta(seconds=1)
            assert eng.tick() == "close orders", f"no closing orders on {d:%Y-%m-%d}"
        else:
            br.now = tm["close"]
            br.auction("cls", d)
        br.now = tm["close"] + pd.Timedelta(minutes=6)
        assert eng.tick() == "reconciled"
    eng.flock.release()
    return bt, br, eng


def trade_sets(bt, br) -> dict:
    tr = bt.trades
    on, it = tr[tr.leg == "OVERNIGHT"], tr[tr.leg == "INTRADAY"]
    f = pd.DataFrame(br.fills)
    suf = f.cid.str.split("-").str[-1]
    leg = suf.str[:1]
    when = np.where(f.tif == "opg", "open", np.where(f.tif == "cls", "close", np.where(leg == "o", "open", "close")))   # day orders: o = the open's sale, x = the close's
    out = {}
    out["on_in"] = ({(pd.Timestamp(r.entry_date), r.symbol, int(r.shares)) for r in on.itertuples()},
                    {(r.day, r.symbol, int(r.qty)) for r in f[leg == "e"].itertuples()})
    out["on_out"] = ({(pd.Timestamp(r.exit_date), r.symbol, int(r.shares), "open" if r.exit_how in ("opening auction", "market at the open") else "close")
                      for r in on.itertuples() if pd.notna(r.exit_date)},
                     {(r.day, r.symbol, int(r.qty), w) for r, w in zip(f.itertuples(), when) if r.cid.split("-")[-1][:1] in ("o", "x")})
    out["id_in"] = ({(pd.Timestamp(r.entry_time), int(r.shares), round(float(r.entry_px), 8)) for r in it.itertuples()},
                    {(r.at.floor("min"), int(r.qty), round(float(r.px), 8)) for r in f[(leg == "i") & (f.side == "buy")].itertuples()})
    at = lambda r: MK["closes"][r.day] if r.cid.split("-")[-1][:2] == "iC" else r.at.floor("min")         # the sale at the close: the closing auction's time
    out["id_out"] = ({(pd.Timestamp(r.exit_time).floor("min"), int(r.shares), round(float(r.exit_px), 8)) for r in it.itertuples() if pd.notna(r.exit_date)},
                     {(at(r), int(r.qty), round(float(r.px), 8)) for r in f[(leg == "i") & (f.side == "sell")].itertuples()})
    return out


class TestParity(unittest.TestCase):
    def check(self, m: MomModel, **kw):
        bt, br, eng = replay(m, **kw)
        ts = trade_sets(bt, br)
        for k, (b, l_) in ts.items():
            self.assertEqual(sorted(b - l_, key=str), [], f"{k}: in the backtest, not live")
            self.assertEqual(sorted(l_ - b, key=str), [], f"{k}: live, not in the backtest")
        last = MK["days"][-1]
        live_eq = br.cash + sum(q * float(br.dix.loc[(s, last), "close"]) for s, (q, _) in br.pos.items() if q)
        self.assertAlmostEqual(live_eq / bt.equity.iloc[-1], 1.0, places=9)
        return bt, br, eng, ts

    def test_rule_book_iex_intraday(self):
        m = MomModel(name="rule book", universe=UNI, minute=1530, k=4, ranking="rule", intraday_symbol="QQQ", intraday_feed="iex")
        bt, br, eng, ts = self.check(m)
        self.assertGreater(len(ts["on_in"][0]), 100, "the synthetic market should make the overnight leg trade")
        self.assertGreater(len(ts["id_in"][0]), 12, "the synthetic market should make the intraday leg trade")
        self.assertTrue((~bt.daily.trend).any() and bt.daily.trend.any(), "the trend filter should switch the overnight leg off for a while")
        self.assertLessEqual(bt.stats["max_exposure"], 1.0 + 1e-9)
        src = E.run(m, market(m), start="2023-03-01").daily.intraday_src
        self.assertTrue((src == "iex").any() and (src == "sip").any(), "IEX bars where cached, SIP before")
        closed = eng.read_journal(("trade_closed",))
        self.assertEqual(sum(r["leg"] == "INTRADAY" for r in closed), int(bt.trades[(bt.trades.leg == "INTRADAY")].exit_date.notna().sum()))
        cmp = eng.check_against_backtest()                                                      # the GUI's "Check live vs backtest" on the same data: everything identical
        self.assertEqual(set(cmp.leg), {"overnight", "intraday"})
        self.assertTrue((cmp.match == "same").all(), cmp[cmp.match != "same"].head().to_string())

    def test_ml_book_with_shadow(self):
        m = MomModel(name="ml book", universe=UNI, minute=1530, k=4, ranking="rule+ml", intraday_symbol="QQQ", intraday_feed="iex")
        ensure_models(m)
        shadow = MomModel(**{**m.to_dict(), "name": "shadow", "ranking": "rule"})
        bt, br, eng, ts = self.check(m, shadow=shadow)
        self.assertEqual(bt.stats["sessions_without_forecast"], 0)
        dec = eng.read_journal(("decision",))
        self.assertTrue(dec and all(r.get("shadow") is not None for r in dec))
        rule = E.run(shadow, market(shadow), FOLDER, start="2024-01-02")                       # the shadow's picks are the rule book's picks of the same day
        first = rule.trades[(rule.trades.leg == "OVERNIGHT")].groupby("entry_date").symbol.apply(set)
        agree = [set(x["symbol"] for x in r["shadow"]) == first.get(pd.Timestamp(r["day"]).normalize(), set()) for r in dec if r["shadow"]]
        self.assertGreater(np.mean(agree), 0.95)
        rep = eng.shadow_report()
        self.assertGreater(len(rep), 20)
        self.assertTrue(rep.traded_ret.notna().all() and rep.shadow_ret.notna().all())

    def test_sector_cap_book(self):
        """A sector-capped book: live reproduces its backtest exactly (the same sector file), and the cap binds - no night holds more than sector_cap names of a sector."""
        sec = pd.Series({s: ("Tech" if i < 10 else "Fin") for i, s in enumerate(STOCKS)}, name="sector")
        m = MomModel(name="capped", universe=UNI, minute=1530, k=4, ranking="rule", intraday_symbol="QQQ", intraday_feed="iex", sector_cap=2)
        with mock.patch.object(ML.D, "sectors", lambda: sec):
            bt, br, eng, ts = self.check(m)
            plain = E.run(MomModel(**{**m.to_dict(), "sector_cap": 0}), market(m), start="2024-01-02")
        on = bt.trades[bt.trades.leg == "OVERNIGHT"]
        per = on.assign(sec=on.symbol.map(sec)).groupby(["entry_date", "sec"]).size()
        self.assertLessEqual(int(per.max()), 2)
        self.assertGreater(len(on), 100)
        pon = plain.trades[plain.trades.leg == "OVERNIGHT"]
        self.assertGreater(int(pon.assign(sec=pon.symbol.map(sec)).groupby(["entry_date", "sec"]).size().max()), 2, "the cap must bind somewhere")

    def test_overnight_only_1545_realtime_sip(self):
        m = MomModel(name="overnight 1545", universe=UNI, minute=1545, k=3, ranking="rule", intraday_symbol="")
        bt, br, eng = replay(m)[:3]
        ts = trade_sets(bt, br)
        for k in ("on_in", "on_out"):
            self.assertEqual(ts[k][0], ts[k][1], k)

    def test_intraday_only_sip(self):
        m = MomModel(name="intraday sip", universe=UNI, minute=1530, overnight=False, intraday_symbol="QQQ", intraday_feed="sip")
        bt, br, eng, ts = self.check(m)
        self.assertEqual(int((bt.trades.leg == "OVERNIGHT").sum()), 0)

    def test_market_execution(self):
        """Market DAY orders in place of the model's MOC / MOO orders (the live setting 'market' - what the paper account can do): the same model, decisions and sizes,
        sent 5 s before the close (the intraday sale and the exits before the entries) and 3 minutes before the open (queued for it). Filled at the auction prices by the
        simulated broker, live must reproduce the model's backtest exactly - every order, size and the final equity - and only the order type and time differ."""
        m = MomModel(name="rule book", universe=UNI, minute=1530, k=4, ranking="rule", intraday_symbol="QQQ", intraday_feed="iex")
        bt, br, eng, ts = self.check(m, auction_orders="market")
        _, br_auc, _ = replay(m)                                                                # the same book with MOC / MOO orders: the same orders, other types
        key = lambda b: sorted((o["symbol"], o["side"], o["qty"], o["submitted_at"].normalize()) for o in b.ords if o["cid"].split("-")[-1][:1] in ("e", "x"))
        self.assertEqual(key(br), key(br_auc))
        self.assertEqual(set(pd.DataFrame(br_auc.ords).tif), {"cls", "opg", "day"})
        self.assertGreater(len(ts["on_in"][0]), 100)
        o = pd.DataFrame(br.ords)
        self.assertEqual(set(o.tif), {"day"}, "no market-on-close / -open order")
        o["day"] = o.submitted_at.dt.normalize()
        o["suf"] = o.cid.str.split("-").str[-1]
        tms = {d: eng.times(d) for d in o.day.unique()}
        self.assertEqual({tm["emu_close"].strftime("%H:%M:%S") for tm in tms.values()} - {"12:59:55"}, {"15:59:55"})
        self.assertEqual({tm["emu_open"].strftime("%H:%M:%S") for tm in tms.values()}, {"09:27:00"})
        at_close = o[[tms[d]["emu_close"] <= t < tms[d]["close"] for d, t in zip(o.day, o.submitted_at)]]
        self.assertEqual(set(at_close.suf.str[:1]) - {"e", "x", "i"}, set())
        self.assertTrue((at_close.side == "sell").any() and (at_close.side == "buy").any())
        for d, g in at_close.groupby("day"):                                                    # sells before buys at every close
            sides = list(g.sort_values("id", key=lambda s: s.astype(int)).side)
            self.assertEqual(sides, sorted(sides, key=lambda s: s != "sell"), f"{d:%Y-%m-%d}")
        opens = o[o.suf.str[:1] == "o"]
        self.assertGreater(len(opens), 50)
        self.assertTrue(all(t == tms[d]["emu_open"] + pd.Timedelta(seconds=1) < tms[d]["open"] for d, t in zip(opens.day, opens.submitted_at)), "sent at 09:27:01")
        how = {r["exit_how"] for r in eng.read_journal(("trade_closed",)) if r["leg"] == "OVERNIGHT"}
        self.assertEqual(how, {"market at the open"})
        self.assertIn("market at the close", {r["exit_how"] for r in eng.read_journal(("trade_closed",)) if r["leg"] == "INTRADAY"})
        fr = eng.fill_report()                                                                  # every auction-time order filled completely at the official price (+ fee)
        self.assertGreater(len(fr), 200)
        self.assertTrue((fr.filled == fr.qty).all())
        self.assertLess(float(fr.cost_bp.abs().max()), 1e-6)
        v = eng.book_view()                                                                     # positions + trade history from the broker's orders, next to the backtest
        tr, done = v["trades"], bt.trades[bt.trades.exit_date.notna()]
        self.assertEqual((len(tr), int((tr.leg == "INTRADAY").sum())), (len(done), int((done.leg == "INTRADAY").sum())))
        pr = tr[tr.bt_pnl.notna()]
        self.assertGreater(len(pr), 0.9 * len(tr))
        self.assertTrue(np.allclose(pr.entry_px, pr.bt_entry) and np.allclose(pr.exit_px, pr.bt_exit), "every live fill at the backtest's price")
        cost = pr.shares * (pr.bt_entry + pr.bt_exit) * 1e-4                                   # diff = live - backtest: here only the backtest's 1 bp per auction side
        self.assertTrue(((pr["diff"] > -1e-6) & (pr["diff"] <= cost + 1e-6)).all(), "the market orders cost nothing more than the auctions here")
        self.assertEqual(v["notes"], [])
        self.assertEqual(sorted(v["positions"].symbol), sorted(s for s, (q, _) in br.pos.items() if q))
        cmp = eng.check_against_backtest()
        self.assertTrue((cmp.match == "same").all())


class TestModel(unittest.TestCase):
    def test_ranking_tiebreak_and_exclusion(self):
        m = MomModel(k=2, ranking="rule", intraday_symbol="QQQ")
        syms = ["QQQ", "ZZZ", "AAA", "BBB"]
        d = np.array([0.09, 0.05, 0.05, 0.01])
        dec = decide_overnight(m, syms, np.ones(4, bool), True, d, None, np.full(4, 10.0), {}, 10_000.0)
        self.assertEqual([e[0] for e in dec.entries], ["AAA", "ZZZ"])                              # QQQ (intraday symbol) skipped; the tie broken by symbol
        self.assertEqual(dec.entries[0][1], int((0.98 * 10_000 / 2) // 10.0))
        dec = decide_overnight(m, syms, np.ones(4, bool), False, d, None, np.full(4, 10.0), {"BBB": 5}, 10_000.0)
        self.assertEqual(dec.entries, [])
        self.assertEqual(dec.exits[0][0], "BBB")
        dec = decide_overnight(m, syms, np.ones(4, bool), True, d, None, np.full(4, 10.0), {"AAA": 7}, 10_000.0)
        self.assertEqual(dec.keeps, ["AAA"])
        self.assertEqual([e[0] for e in dec.entries], ["ZZZ"])

    def test_sector_cap(self):
        """At most sector_cap names per sector, in score order: a skipped name's place goes to the next-best name of another sector; names without a sector are uncapped;
        a held name picked again counts (and is kept); off (0) or without sectors = the plain top-k."""
        syms = ["T1", "T2", "T3", "T4", "F1", "H1", "E1", "E2"]
        sec = ["Tech", "Tech", "Tech", "Tech", "Fin", "Health", "", ""]
        d = np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
        args = (syms, np.ones(8, bool), True, d, None, np.full(8, 10.0))
        m = MomModel(k=4, ranking="rule", intraday_symbol="", sector_cap=2)
        dec = decide_overnight(m, *args, {}, 10_000.0, sectors=sec)
        self.assertEqual([e[0] for e in dec.entries], ["T1", "T2", "F1", "H1"])
        self.assertIn("skipped T3 (Tech), T4 (Tech)", dec.notes[-1])
        self.assertEqual({c[0] for c in dec.candidates if c[5]}, {"T1", "T2", "F1", "H1"})
        dec = decide_overnight(MomModel(k=6, ranking="rule", intraday_symbol="", sector_cap=1), *args, {}, 10_000.0, sectors=sec)
        self.assertEqual([e[0] for e in dec.entries], ["T1", "F1", "H1", "E1", "E2"])           # ETF-like names without a sector are not capped; fewer than k qualify
        dec = decide_overnight(m, *args, {"T2": 5, "T3": 4}, 10_000.0, sectors=sec)
        self.assertEqual((dec.keeps, [x[0] for x in dec.exits]), (["T2"], ["T3"]))
        self.assertEqual([e[0] for e in dec.entries], ["T1", "F1", "H1"])
        for mm, s in ((MomModel(k=4, ranking="rule", intraday_symbol=""), sec), (m, None)):
            self.assertEqual([e[0] for e in decide_overnight(mm, *args, {}, 10_000.0, sectors=s).entries], ["T1", "T2", "T3", "T4"])

    def test_blend_is_rank_sum(self):
        U = np.array([True, True, True, False])
        r = pct_rank(np.array([3.0, 1.0, 2.0, 9.0]), U)
        self.assertTrue(np.allclose(r[:3], [1.0, 1 / 3, 2 / 3]) and np.isnan(r[3]))
        m = MomModel(k=1, ranking="rule+ml", intraday_symbol="")
        dec = decide_overnight(m, ["A", "B", "C", "D"], U, True, np.array([0.03, 0.02, 0.01, 0.5]), np.array([0.0, 0.0, 1.0, 1.0]), np.full(4, 5.0), {}, 1000.0)
        sc = {c[0]: c[1] for c in dec.candidates}                                                # move ranks 1, 2/3, 1/3; forecast ranks 1/2, 1/2, 1 (A and B tie); D is not rankable
        self.assertAlmostEqual(sc["A"], 1.5)
        self.assertAlmostEqual(sc["B"], 2 / 3 + 0.5)
        self.assertAlmostEqual(sc["C"], 1 / 3 + 1.0)
        self.assertNotIn("D", sc)
        self.assertEqual(dec.entries[0][0], "A")
        self.assertEqual(decide_overnight(m, ["A", "B", "C", "D"], U, True, np.zeros(4), None, np.full(4, 5.0), {}, 1000.0).entries, [])   # no forecast: no entries
        n = 49                                                                                    # equal rank sums are exactly equal: the ticker decides, not float noise
        rng = np.random.default_rng(3)
        p, d = rng.permutation(n).astype(float), rng.permutation(n).astype(float)
        syms = [f"S{i:02d}" for i in range(n)]
        dec = decide_overnight(MomModel(k=n, ranking="rule+ml", intraday_symbol=""), syms, np.ones(n, bool), True, d, p, np.full(n, 5.0), {}, 1e6)
        c = [(x[1], x[0]) for x in dec.candidates]
        self.assertEqual([x[1] for x in c], [x[1] for x in sorted(c, key=lambda z: (-z[0], z[1]))])
        self.assertTrue(len({x[0] for x in c}) < n)                                             # there are ties

    def test_noise_rule(self):
        self.assertEqual(noise_action(False, 101.0, 100.0, 99.0), "buy")
        self.assertIsNone(noise_action(False, 99.0, 100.0, 99.0))
        self.assertEqual(noise_action(True, 100.5, 100.0, 101.0), "sell")                         # below the VWAP, above the band: the trailing stop is max(band, VWAP)
        self.assertIsNone(noise_action(True, 101.5, 100.0, 101.0))
        self.assertIsNone(noise_action(True, 100.5, 100.0, 101.0, trail=False))
        self.assertEqual(noise_action(True, 99.0, 100.0, np.nan), "sell")                         # no VWAP yet: the band alone
        self.assertIsNone(noise_action(False, np.nan, 100.0, 99.0))
        self.assertEqual(check_slots(26, 2), list(range(1, 25, 2)))                               # 10:00 .. 15:30
        self.assertEqual(check_slots(14, 2)[-1], 11)                                              # 13:00 close: last check 12:30
        self.assertEqual(lead_bars(1530), 2)
        self.assertEqual(lead_bars(1545), 1)
        m = MomModel(cash_reserve=0.02, half_spread_bp=0.5)
        self.assertEqual(intraday_shares(m, 100_000, 100_000, 400.0), int(98_000 // (400 * 1.00005)))
        self.assertEqual(intraday_shares(m, 100_000, 10_000, 400.0), int(10_000 // (400 * 1.00005)))   # never more than the cash

    def test_slot_arrays_early_close(self):
        d = pd.Timestamp("2023-11-24")                                                           # NYSE closes at 13:00
        b = MK["sip"]["MBA"]
        b = b[b.index.normalize() == d]
        late = b.iloc[[-1]].copy()
        late.index = [d + pd.Timedelta(hours=14)]                                                  # an after-hours print must be dropped
        a = slot_arrays(pd.concat([b, late]), pd.DatetimeIndex([d]), np.array([14]))
        self.assertEqual(int(np.isfinite(a["cl"][0]).sum()), 14)
        mk = market(MomModel(intraday_symbol=""))
        t = mk.t(d)
        self.assertEqual(int(mk.nb[t]), 14)
        self.assertAlmostEqual(mk.S[t, mk.col["MBA"]], b.loc[d + pd.Timedelta(hours=12, minutes=15), "close"])    # 15:30-equivalent = 30 minutes before the close

    def test_features_short_panel_equal_long_panel(self):
        """Live builds the features from ~460 days of history; the backtest from the whole history. Row by row they must agree (float32)."""
        m = MomModel(universe=UNI, minute=1530, intraday_symbol="")
        long = market(m)
        day = pd.Timestamp("2024-02-14")
        t = long.t(day)
        U = rankable(long, member_mask(long, m))
        a = feature_frame(long, U, rows=[t])[0]
        short = MomMarket(SYMS, minute=1530, start=day - pd.Timedelta(days=460), end=day + pd.Timedelta(days=1), daily=MK["daily"], bars_fn=bars_fn,
                          bars_from=day - pd.Timedelta(days=45), intraday_symbol="")
        ts = len(short.dates) - 1
        b = feature_frame(short, rankable(short, member_mask(short, m)), rows=[ts])[0]
        a = a.assign(symbol=[long.syms[j] for j in a.j]).set_index("symbol").drop(columns=["t", "j", "date"])
        b = b.assign(symbol=[short.syms[j] for j in b.j]).set_index("symbol").drop(columns=["t", "j", "date"])
        self.assertEqual(sorted(a.index), sorted(b.index))
        diff = (a - b.loc[a.index]).abs().to_numpy()
        self.assertTrue(np.nanmax(diff) < 1e-6, f"max difference {np.nanmax(diff)}")
        self.assertTrue((np.isnan(a.to_numpy()) == np.isnan(b.loc[a.index].to_numpy())).all())
        sa = noise_sigma(market(MomModel(intraday_symbol="QQQ", intraday_feed="iex")), 14)[t]
        q = MomMarket(["QQQ"], minute=1530, start=day - pd.Timedelta(days=460), end=day + pd.Timedelta(days=1), daily=MK["daily"], bars_fn=bars_fn,
                      bars_from=day - pd.Timedelta(days=45), intraday_symbol="QQQ", intraday_feed="iex")
        self.assertTrue(np.allclose(sa, noise_sigma(q, 14)[len(q.dates) - 1], equal_nan=True, rtol=1e-12))

    def test_backtest_trades_have_every_field(self):
        m = MomModel(universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")
        mk = market(m)
        r = E.run(m, mk, start="2023-06-01")
        need = {"trade_id", "leg", "symbol", "direction", "shares", "entry_date", "entry_time", "entry_how", "entry_px", "entry_value", "score", "score_value", "rank", "move",
                "decision_px", "exit_date", "exit_time", "exit_how", "exit_px", "exit_reason", "fees", "pnl", "ret"}
        self.assertTrue(need <= set(r.trades.columns))
        self.assertEqual(set(r.trades.leg), {"OVERNIGHT", "INTRADAY"})
        self.assertTrue((r.trades.direction == "LONG").all())
        done = r.trades[r.trades.exit_date.notna()]
        on = done[done.leg == "OVERNIGHT"]
        self.assertTrue(((on.exit_date - on.entry_date).dt.days >= 1).all())
        it = done[done.leg == "INTRADAY"]
        self.assertTrue((it.exit_date == it.entry_date).all())
        last = mk.t(r.equity.index[-1])
        unreal = sum(x.shares * (mk.C[last, mk.col[x.symbol]] - x.entry_px) - x.fees for x in r.trades[r.trades.exit_date.isna()].itertuples())
        self.assertAlmostEqual(r.equity.iloc[-1], 100_000 + done.pnl.sum() + unreal, places=4)          # the trade list accounts for every dollar
        self.assertLessEqual(r.stats["max_exposure"], 1.0 + 1e-9)

    def test_dry_run_sends_nothing(self):
        m = MomModel(name="dry", universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")
        br = FakeBroker(100_000, m)
        eng = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=False, refresh_data=False, auction_orders="auction"), tempfile.mkdtemp(), log=lambda *a: None, broker=br,
                               clock_fn=lambda: br.now, bars_fn=bars_fn)
        d = MK["days"][400]
        br.now = d + pd.Timedelta(hours=8)
        eng.connect()
        tm = eng.times(d)
        br.now = d + pd.Timedelta(hours=8, minutes=50)
        self.assertEqual(eng.tick(), "morning")
        br.now = tm["open"] + pd.Timedelta(seconds=61)
        self.assertEqual(eng.tick(), "open")
        for slot, end, due in tm["checks"]:
            br.now = due + pd.Timedelta(seconds=1)
            self.assertEqual(eng.tick(), "check")
        br.now = tm["decision"] + pd.Timedelta(seconds=2)
        self.assertEqual(eng.tick(), "decision")
        self.assertEqual(br.ords, [])
        self.assertTrue(len(eng.last_plan.orders) in (0, 4))
        with self.assertRaises(RuntimeError):
            eng.submit(eng.last_plan)

    def test_past_cutoff_and_late_check(self):
        m = MomModel(name="late", universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")
        br = FakeBroker(100_000, m)
        eng = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=True, refresh_data=False, auction_orders="auction"), tempfile.mkdtemp(), log=lambda *a: None, broker=br,
                               clock_fn=lambda: br.now, bars_fn=bars_fn)
        d = MK["days"][401]
        br.now = d + pd.Timedelta(hours=8)
        eng.connect()
        tm = eng.times(d)
        br.now = tm["checks"][0][2] + pd.Timedelta(minutes=10)
        self.assertEqual(eng.tick(), "morning")
        self.assertEqual(eng.tick(), "open")
        self.assertEqual(eng.tick(), "check missed")
        br.now = tm["decision"] + pd.Timedelta(seconds=2)
        plan = eng.plan()
        br.now = tm["cutoff"] + pd.Timedelta(seconds=1)
        with self.assertRaises(RuntimeError):
            eng.submit(plan)
        self.assertFalse(any(o["tif"] == "cls" for o in br.ords))
        eng.flock.release()
        e2 = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=False, refresh_data=False), tempfile.mkdtemp(), log=lambda *a: None, broker=br, clock_fn=lambda: br.now,
                              bars_fn=bars_fn)
        br.now = tm["close"] + pd.Timedelta(hours=4)                                              # a loop started in the evening: today's checks are not "missed"
        self.assertIsNone(e2.tick())
        self.assertEqual(e2.read_journal(("check_missed",)), [])

    def test_capital_is_the_books_own_equity(self):
        """Live sizes on the account equity: cash + the book's own positions. A position outside the book is not part of it (buying on top of it would be leverage),
        buying power is never used, and a fixed capital above the equity is capped."""
        m = MomModel(name="cap", universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")
        br = FakeBroker(100_000, m)
        br.cash -= 30_000                                                                        # $30k bought outside the book
        own = br.positions
        br.positions = lambda: {**own(), "ZZZZ": dict(qty=300, avg_px=100.0, mv=30_000.0)}
        eng = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=False, refresh_data=False), tempfile.mkdtemp(), log=lambda *a: None, broker=br, clock_fn=lambda: br.now,
                               bars_fn=bars_fn)
        d = MK["days"][520]
        br.now = d + pd.Timedelta(hours=8)
        eng.connect()
        br.now = eng.times(d)["decision"] + pd.Timedelta(seconds=2)
        p = eng.plan()
        self.assertAlmostEqual(p.capital, 70_000)
        self.assertGreater(len(p.orders), 0)
        self.assertLessEqual(p.orders.notional.sum(), 0.98 * 70_000 + 1e-6)
        self.assertTrue(any("not part of its capital" in n for n in p.notes))
        eng.live.capital = 500_000
        p = eng.plan()
        self.assertAlmostEqual(p.capital, 70_000)
        self.assertTrue(any("capped at the equity" in n for n in p.notes))
        eng.live.capital = 20_000
        self.assertAlmostEqual(eng.plan().capital, 20_000)

    def test_auction_orders_setting(self):
        """The live setting replaces only the model's auction orders - MOC / MOO (default) or market DAY orders 5 s before the close and 3 minutes before the open - on paper
        and real money alike; a model without auction orders has nothing to replace. The model itself has no execution field (old saved models still load)."""
        br = FakeBroker(100_000, MomModel())
        br.now = pd.Timestamp("2024-03-01 10:00")
        mk = lambda m=MomModel(), **kw: ML.MomLiveEngine(m, ML.MomLiveSettings(**kw), tempfile.mkdtemp(), broker=br, clock_fn=lambda: br.now)
        for paper in (True, False):
            self.assertFalse(mk(paper=paper).emulate)                                           # default: MOC / MOO, as the model
            self.assertTrue(mk(paper=paper, auction_orders="market").emulate)
            self.assertTrue(mk(MomModel(overnight=False), paper=paper, auction_orders="market").emulate)   # the QQQ sale at the close is an auction order too
        self.assertFalse(mk(MomModel(overnight=False, intraday_symbol=""), auction_orders="market").emulate)   # no auction order to replace
        self.assertEqual([mk(auction_orders=x).emulate for x in ("model", "auto", "emulate")], [False, False, True])   # names saved by earlier versions
        self.assertTrue(any("auction orders" in p for p in mk(auction_orders="moc").model_problems()))
        pf = {c.name: c for c in mk(auction_orders="market").preflight()}
        self.assertEqual(pf["auction orders"].level, "ok")
        self.assertIn("in place of the model's MOC / MOO", pf["auction orders"].msg)
        self.assertEqual(pf["PC clock"].level, "ok")
        self.assertEqual({c.name: c for c in mk().preflight()}["auction orders"].level, "warn")  # MOC / MOO on paper: it does not run the auctions
        self.assertIn("none", {c.name: c for c in mk(MomModel(overnight=False, intraday_symbol="")).preflight()}["auction orders"].msg)
        tm = mk(auction_orders="market").times(pd.Timestamp("2024-03-01"))
        self.assertEqual((tm["emu_open"].strftime("%H:%M:%S"), tm["emu_close"].strftime("%H:%M:%S")), ("09:27:00", "15:59:55"))
        early = next(d for d, c in MK["closes"].items() if c.hour == 13)
        self.assertEqual(mk(auction_orders="market").times(early)["emu_close"].strftime("%H:%M:%S"), "12:59:55")
        self.assertTrue(any("in place of the auctions" in p for p in mk(mkt_close_s=2).model_problems()))
        self.assertTrue(any("in place of the auctions" in p for p in mk(mkt_close_s=700).model_problems()))
        self.assertTrue(any("in place of the auctions" in p for p in mk(mkt_open_s=-1000).model_problems()))
        self.assertEqual(mk(auction_orders="market").model_problems(), [])
        self.assertEqual(MomModel().auction_costs(), (1e-4, 1e-4, 1e-4))
        self.assertFalse(hasattr(PR.get(PR.TRADED), "execution"))
        old = MomModel.from_dict({**PR.get(PR.TRADED).to_dict(), "execution": "market", "mkt_close_s": 10, "cost_mkt_open_bp": 17.8})
        self.assertEqual(old, PR.get(PR.TRADED))
        self.assertIn(PR.RENAMED["Auction orders: rule + ML forecast, 15:30, MOC / MOO (real-money target)"], PR.PRESETS)
        clk = br.clock
        br.clock = lambda: {**clk(), "ts": br.now - pd.Timedelta(seconds=3)}                    # this PC 3 s ahead of the broker's clock
        try:
            pf = {c.name: c for c in mk(auction_orders="market").preflight()}
            self.assertEqual(pf["PC clock"].level, "warn")
            self.assertIn("early", pf["PC clock"].msg)
        finally:
            br.clock = clk

    def test_emulated_close_queue_kill_dry_run_old_state(self):
        """Emulating the auctions the decision only queues its orders: they go out at the emulated close - not before, once - and never after a KILL, in DRY RUN, or when the
        pending submit was written by an engine that sent MOC orders (no 'sent' flag)."""
        m = MomModel(name="emu", universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")

        def decided(armed=True):
            br = FakeBroker(100_000, m)
            br.emu = True
            eng = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=armed, refresh_data=False, auction_orders="market"), tempfile.mkdtemp(), log=lambda *a: None, broker=br,
                                   clock_fn=lambda: br.now, bars_fn=bars_fn)
            d = MK["days"][520]
            br.now = d + pd.Timedelta(hours=8)
            eng.connect()
            tm = eng.times(d)
            br.now = tm["decision"] + pd.Timedelta(seconds=2)
            self.assertIn("decision", [eng.tick() for _ in range(20)])                         # (today's checks are journaled as missed first)
            return br, eng, tm
        br, eng, tm = decided()
        self.assertEqual(br.ords, [], "queued at the decision, not sent")
        q = eng.load_state()["pending"]
        self.assertFalse(q["sent"])
        self.assertGreater(len(q["orders"]), 0)
        br.now = tm["emu_close"] - pd.Timedelta(seconds=1)
        self.assertIsNone(eng.tick())
        br.now = tm["emu_close"] + pd.Timedelta(seconds=1)
        self.assertEqual(eng.tick(), "close orders")
        self.assertEqual([(o["symbol"], o["qty"], o["tif"]) for o in br.ords], [(o["symbol"], float(o["qty"]), "day") for o in q["orders"]])
        self.assertNotEqual(eng.tick(), "close orders")
        eng.close_orders(tm["close"].normalize())                                               # a second call sends nothing
        self.assertEqual(len(br.ords), len(q["orders"]))
        eng.flock.release()
        for case in ("kill", "dry", "old state", "too late"):
            br, eng, tm = decided(armed=case != "dry")
            if case == "kill":
                self.assertTrue(any("dropped" in str(r["status"]) for r in eng.kill()))
                self.assertFalse(eng.live.armed)
                eng.live.armed = True                                                           # even armed again, the queued orders stay dropped
            elif case == "old state":
                st = eng.load_state()
                st["pending"].pop("sent")
                eng.save_state(st)
            br.now = tm["emu_close"] + pd.Timedelta(seconds=1) if case != "too late" else tm["close"] - pd.Timedelta(seconds=1)
            self.assertEqual(eng.tick(), "close orders")
            self.assertEqual(br.ords, [], case)                                                 # too late: a day order after the close would trade at the next open
            if case == "too late":
                self.assertEqual(len(eng.read_journal(("close_orders_cut",))[0]["not_sent"]), len(q["orders"]))
            eng.flock.release()
        br, eng, tm = decided()                                                                 # queued, then the loop restarted with MOC / MOO orders: still sent
        eng.live.auction_orders = "auction"
        self.assertFalse(eng.emulate)
        br.now = tm["emu_close"] + pd.Timedelta(seconds=1)
        self.assertEqual(eng.tick(), "close orders")
        self.assertEqual([(o["symbol"], o["qty"], o["tif"]) for o in br.ords], [(o["symbol"], float(o["qty"]), "day") for o in q["orders"]])
        eng.flock.release()

    def _eng(self, orders: str, folder: str = None, held: dict = None, day_i: int = 520, armed: bool = True, foreign: dict = None):
        """An engine on a FakeBroker at session ``day_i``; ``held`` = {symbol: shares} the book bought the session before (its filled order, in the book; QQQ: bought
        today at 10:00); ``foreign`` = {symbol: shares} bought outside the book (no order of the book)."""
        m = MomModel(name="safety", universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")
        br = FakeBroker(100_000, m)
        br.emu = orders == "market"
        d, prev = MK["days"][day_i], MK["days"][day_i - 1]
        br.now = d + pd.Timedelta(hours=8)
        eng = ML.MomLiveEngine(m, ML.MomLiveSettings(armed=armed, refresh_data=False, auction_orders=orders), folder or tempfile.mkdtemp(), log=lambda *a: None,
                               broker=br, clock_fn=lambda: br.now, bars_fn=bars_fn)
        eng.connect()
        if held:
            st = eng.load_state()
            for s, q in held.items():
                px = float(br.dix.loc[(s, prev), "close"])
                br.pos[s] = (q, px)
                br.cash -= q * px
                at, cid = ((d + pd.Timedelta(hours=10), f"MB-{d:%Y%m%d}-{s}-iB011") if s == m.intraday_symbol else
                           (prev + pd.Timedelta(hours=15, minutes=46), f"MB-{prev:%Y%m%d}-{s}-e1"))
                br.n += 1
                br.hist.append(dict(id=str(br.n), cid=cid, symbol=s, side="buy", qty=float(q), status="filled", filled_qty=float(q), filled_avg_price=px, submitted_at=at,
                                    filled_at=at, tif="day" if s == m.intraday_symbol else "cls"))
                if s != m.intraday_symbol:
                    st["book"][s] = dict(entry_date=str(prev.date()), entry_px=px, shares=q, trade_id=1)
            eng.save_state(st)
        for s, q in (foreign or {}).items():
            px = float(br.dix.loc[(s, prev), "close"])
            old_q, old_a = br.pos.get(s, (0, np.nan))
            br.pos[s] = (old_q + q, px if not old_q else old_a)
            br.cash -= q * px
        return eng, br, d, eng.times(d)

    def _to_decision(self, eng, br, tm):
        br.now = tm["decision"] + pd.Timedelta(seconds=2)
        self.assertIn("decision", [eng.tick() for _ in range(25)])

    def test_only_the_books_own_shares_are_sold(self):
        """Shares in the book's symbols that its own orders did not buy - bought by hand, or extra shares on top of the book's - are never sold (morning sale, decision
        exits, KILL) and are not part of its capital. No position: no order."""
        for orders in ("market", "auction"):
            eng, br, d, tm = self._eng(orders, held={"MBA": 50}, foreign={"MBA": 30, "MBB": 25, "QQQ": 40})
            ours, iq, foreign = eng.held_positions([])
            self.assertEqual(({s: p["qty"] for s, p in ours.items()}, iq, {s: p["qty"] for s, p in foreign.items()}), ({"MBA": 50}, 0, {"MBA": 30, "MBB": 25, "QQQ": 40}))
            br.now = d + pd.Timedelta(hours=8, minutes=50)
            self.assertEqual(eng.tick(), "morning")
            br.now = (tm["emu_open"] if orders == "market" else tm["open"] + pd.Timedelta(seconds=60)) + pd.Timedelta(seconds=1)
            if orders == "auction":
                br.auction("opg", d)
            self.assertEqual(eng.tick(), "open")
            self.assertEqual([(o["symbol"], o["side"], o["qty"]) for o in br.ords if o["cid"].split("-")[-1][:1] == "o"], [("MBA", "sell", 50.0)], orders)
            self.assertEqual({s: q for s, (q, _) in br.pos.items() if q}, {"MBA": 30, "MBB": 25, "QQQ": 40}, orders)   # the foreign shares are all still there
            self.assertEqual(eng.held_positions([])[:2], ({}, 0))
            n = len(br.ords)
            res = {r["symbol"]: r["status"] for r in eng.kill()}
            self.assertEqual(len(br.ords), n, "KILL sends nothing: nothing is the book's")
            self.assertTrue(all("not this book's" in v for v in res.values()), res)
            eng.flock.release()
        eng, br, d, tm = self._eng("market", foreign={"MBA": 30})                               # no position of the book: the open step sends nothing
        br.now = tm["emu_open"] + pd.Timedelta(seconds=1)
        self.assertEqual(eng.tick(), "morning")
        self.assertEqual(eng.tick(), "open")
        self.assertEqual(br.ords, [])
        br.now = tm["decision"] + pd.Timedelta(seconds=2)
        p = eng.plan()
        self.assertNotIn("MBA", set(p.orders[p.orders.kind == "exit"].symbol))
        self.assertAlmostEqual(p.capital, 100_000 - 30 * float(br.dix.loc[("MBA", MK["days"][519]), "close"]))
        eng.flock.release()

    def test_failed_closing_sale_blocks_the_buys(self):
        """If the intraday position's closing sale does not go out (exception, rejected, unknown), no overnight buy goes out: its value would be bought on top of it (up to
        2x gross). A failed exit takes the lowest-ranked buys worth it out. Market-order and MOC execution alike."""
        for orders in ("market", "auction"):
            eng, br, d, tm = self._eng(orders, held={"QQQ": 100})
            sub = br.submit
            br.submit = lambda s, *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 503")) if s == "QQQ" else sub(s, *a, **k)
            self._to_decision(eng, br, tm)
            if orders == "market":
                br.now = tm["emu_close"] + pd.Timedelta(seconds=1)
                self.assertEqual(eng.tick(), "close orders")
            self.assertEqual([o for o in br.ords if o["side"] == "buy"], [], orders)
            self.assertTrue(eng.read_journal(("entries_dropped",)), orders)
            eng.flock.release()
        eng, br, d, tm = self._eng("market")
        e = [dict(symbol=s, qty=10, ref_px=100.0) for s in "ABC"]
        self.assertEqual([x["symbol"] for x in eng._trim_entries(e, [dict(symbol="X", qty=15, ref_px=100.0, status="rejected")], False)], ["A"])
        self.assertEqual(eng._trim_entries(e, [dict(symbol="X", qty=15, ref_px=100.0, status="accepted")], False), e)
        self.assertEqual(eng._trim_entries(e, [], True), [])

    def test_kill_disarms_blocks_the_day_and_never_double_sells(self):
        eng, br, d, tm = self._eng("market", held={"MBA": 50, "MBB": 40})
        self._to_decision(eng, br, tm)
        br.ords.append(dict(id="x1", cid=f"MB-{d:%Y%m%d}-MBB-o9", symbol="MBB", side="sell", qty=40.0, status="accepted", filled_qty=0.0, filled_avg_price=np.nan,
                            submitted_at=br.now, filled_at=None, tif="day"))
        cancel = br.cancel
        br.cancel = lambda oid: None if oid == "x1" else cancel(oid)                            # an order that cannot be cancelled
        res = {r["symbol"]: r["status"] for r in eng.kill()}
        self.assertFalse(eng.live.armed)
        self.assertTrue(eng.load_state()["killed"].startswith(str(d.date())))
        self.assertTrue(any("KILL is active" in p for p in eng.arm_problems()))                # cannot be armed again until a person clears it
        self.assertEqual(res["MBA"], "close sent (market)")
        self.assertIn("NOT closed", res["MBB"])                                                # its open sell and a KILL sell could both fill: a short
        n = len(br.ords)
        eng.live.armed = True                                                                   # even re-armed: nothing more today
        br.now = tm["emu_close"] + pd.Timedelta(seconds=1)
        self.assertEqual(eng.tick(), "close orders")
        self.assertEqual(len(br.ords), n)
        with self.assertRaises(RuntimeError):
            eng._guard()
        for o in br.ords:                                                                       # the uncancellable order is done the next day
            o["status"] = "filled" if o["status"] not in FINAL else o["status"]
        br.now = MK["days"][list(MK["days"]).index(d) + 1] + pd.Timedelta(hours=11)             # KILL still blocks the next day
        with self.assertRaises(RuntimeError):
            eng._guard()
        eng.reset_drawdown()                                                                    # until a person clears it
        self.assertFalse(any("KILL" in p for p in eng.arm_problems()))
        eng._guard()
        eng.flock.release()

    def test_exits_are_resized_when_sent(self):
        """A decision's exit is sized when it is sent: the position now minus the book's open orders on it - a sale that filled late is never sold again."""
        eng, br, d, tm = self._eng("market", held={"MBA": 50, "MBB": 40})
        br.now = tm["decision"] + pd.Timedelta(seconds=2)
        br.ords.append(dict(id="z1", cid=f"MB-{d:%Y%m%d}-MBA-o9", symbol="MBA", side="sell", qty=30.0, status="accepted", filled_qty=0.0, filled_avg_price=np.nan,
                            submitted_at=br.now, filled_at=None, tif="day"))
        br.pos["MBB"] = (0, br.pos["MBB"][1])                                                   # sold late, after the decision read it
        rows = [dict(symbol="MBA", qty=50, side="sell", kind="exit"), dict(symbol="MBB", qty=40, side="sell", kind="exit")]
        self.assertEqual([(r["symbol"], r["qty"]) for r in eng._sellable_exits(d, rows)], [("MBA", 20)])
        eng.flock.release()

    def test_kill_after_the_close_is_not_sold_twice_next_morning(self):
        """KILL after the close queues its sales for the next open. The next morning nothing sells those names again (the open orders are netted) and no order goes out
        until the KILL orders are done."""
        eng, br, d, tm = self._eng("market", held={"MBA": 50})
        br.now = tm["close"] + pd.Timedelta(hours=1)
        res = {r["symbol"]: r["status"] for r in eng.kill()}
        self.assertEqual(res["MBA"], "close sent (market)")
        self.assertEqual([(o["symbol"], o["status"]) for o in br.ords], [("MBA", "accepted")])  # queued for the next open
        nxt = MK["days"][list(MK["days"]).index(d) + 1]
        eng.live.armed = True                                                                   # a runner armed again the next day
        br.now = eng.times(nxt)["emu_open"] + pd.Timedelta(seconds=1)
        self.assertEqual(eng.overnight_exits(br.now, late_ok=True), [])
        self.assertEqual(len(br.ords), 1)
        with self.assertRaises(RuntimeError):
            eng._guard()
        eng.flock.release()

    def test_dry_pass_does_not_consume_the_armed_steps(self):
        """A DRY loop run earlier the same day keeps its own step flags: the armed loop started afterwards still sells, checks and decides."""
        folder = tempfile.mkdtemp()
        dry, br, d, tm = self._eng("market", folder=folder, armed=False, held={"MBA": 50})
        br.now = d + pd.Timedelta(hours=8, minutes=50)
        self.assertEqual(dry.tick(), "morning")
        br.now = tm["emu_open"] + pd.Timedelta(seconds=1)
        self.assertEqual(dry.tick(), "open")
        self._to_decision(dry, br, tm)
        self.assertEqual(br.ords, [])
        arm = ML.MomLiveEngine(dry.m, ML.MomLiveSettings(armed=True, refresh_data=False, auction_orders="market"), folder, log=lambda *a: None, broker=br,
                               clock_fn=lambda: br.now, bars_fn=bars_fn)
        arm.connect()
        br.now = tm["emu_open"] + pd.Timedelta(seconds=30)
        self.assertEqual(arm.tick(), "morning")
        self.assertEqual(arm.tick(), "open")                                                   # the dry pass did not mark it done
        self.assertEqual([(o["symbol"], o["side"]) for o in br.ords], [("MBA", "sell")])
        self._to_decision(arm, br, tm)
        self.assertTrue(arm.load_state()["pending"]["orders"])
        arm.flock.release()
        arm.live.armed = False                                                                  # every order re-checks: disarmed half-way, the rest is not sent
        self.assertEqual(arm._send("MBB", 5, "buy", "day", "MB-x-MBB-e9")["status"], "disarmed")

    def test_every_loop_holds_the_lock(self):
        """A DRY-RUN loop next to an armed one would mark the day's steps done in their shared state file: every loop takes the account's lock."""
        folder = tempfile.mkdtemp()
        e1, br, d, tm = self._eng("market", folder=folder)
        self.assertTrue(e1.flock.acquire())
        e2, _, _, _ = self._eng("market", folder=folder, armed=False)
        stop = threading.Event()
        stop.set()
        with self.assertRaises(RuntimeError):
            e2.run(stop)
        e1.flock.release()

    def test_opening_sale_retried_and_missed_decision(self):
        """MOC / MOO execution: what the opening auction left is sold with a market order at 09:31. A loop started after the decision window sells last night's book
        before the close (the backtest sold it at the open) instead of carrying it another session."""
        for opg_left in ("expired", "still open"):
            eng, br, d, tm = self._eng("auction", held={"MBA": 50})
            br.now = d + pd.Timedelta(hours=8, minutes=50)
            self.assertEqual(eng.tick(), "morning")
            self.assertEqual([(o["symbol"], o["tif"]) for o in br.ords], [("MBA", "opg")])
            if opg_left == "expired":
                br.ords[0]["status"] = "expired"                                                # paper expired the unfilled OPG order
            br.now = tm["open"] + pd.Timedelta(seconds=61)
            self.assertEqual(eng.tick(), "open")
            if opg_left == "expired":                                                           # what the opening auction left is sold
                self.assertEqual([(o["symbol"], o["side"], o["tif"], o["status"]) for o in br.ords[1:]], [("MBA", "sell", "day", "filled")])
                self.assertEqual(br.pos["MBA"][0], 0)
            else:                                                                               # a delayed opening auction still running: not sold twice
                self.assertEqual(len(br.ords), 1)
                self.assertNotEqual(br.ords[0]["status"], "canceled")
            eng.flock.release()
        eng, br, d, tm = self._eng("market", held={"MBA": 50})
        br.now = tm["cutoff"] + pd.Timedelta(minutes=2)
        self.assertEqual(eng.tick(), "open")
        self.assertEqual([(o["symbol"], o["side"], o["tif"]) for o in br.ords], [("MBA", "sell", "day")])
        self.assertIn("late", ML._how(dict(br.ords[0], cid=br.ords[0]["cid"])))
        eng.flock.release()

    def test_live_account_is_view_only_while_real_money_is_disabled(self):
        """A real-money account can be connected and viewed (positions, orders, trade history), but arming, every order path and KILL refuse it unless
        RESARB_ALLOW_LIVE=1."""
        m = MomModel(name="live view", universe=UNI, minute=1530, k=4, ranking="rule", intraday_feed="iex")
        br = FakeBroker(100_000, m)
        br.now = MK["days"][520] + pd.Timedelta(hours=10)
        old = os.environ.pop("RESARB_ALLOW_LIVE", None)
        try:
            eng = ML.MomLiveEngine(m, ML.MomLiveSettings(paper=False, armed=True, refresh_data=False), tempfile.mkdtemp(), log=lambda *a: None, broker=br,
                                   clock_fn=lambda: br.now, bars_fn=bars_fn)
            self.assertEqual(eng.connect()["number"], "PA-MB-TEST")                            # viewing works
            v = eng.book_view()
            self.assertEqual((len(v["positions"]), len(v["orders"])), (0, 0))
            self.assertIn("real money is disabled", eng.arm_problems())
            with self.assertRaises(RuntimeError):
                eng.kill()
            with self.assertRaises(RuntimeError):
                eng._guard()
            self.assertEqual(br.ords, [])
            self.assertTrue(eng.path("journal", "jsonl").endswith("mb_journal_live.jsonl"))     # its own journal and state, apart from paper's
        finally:
            if old is not None:
                os.environ["RESARB_ALLOW_LIVE"] = old

    def test_problems(self):
        br = FakeBroker(100_000, MomModel())
        br.now = pd.Timestamp("2024-03-01 10:00")
        mk = lambda m, **kw: ML.MomLiveEngine(m, ML.MomLiveSettings(**kw), tempfile.mkdtemp(), broker=br, clock_fn=lambda: br.now)
        self.assertTrue(any("cut-off" in p for p in mk(MomModel(minute=1545), feed="sip_delayed").model_problems()))
        self.assertEqual(mk(MomModel(minute=1545), feed="sip").model_problems(), [])
        self.assertEqual(mk(MomModel(minute=1530), feed="sip_delayed").model_problems(), [])
        e = mk(MomModel(minute=1530, ranking="rule+ml"))
        e.connect()
        self.assertTrue(any("no ML model" in p for p in e.arm_problems()))
        tm = e.times(pd.Timestamp("2024-03-01"))
        self.assertEqual(tm["decision"].strftime("%H:%M:%S"), "15:46:06")
        self.assertEqual(tm["cutoff"].strftime("%H:%M:%S"), "15:49:30")
        self.assertEqual(tm["checks"][0][2].strftime("%H:%M:%S"), "10:00:05")
        self.assertEqual(tm["checks"][-1][1].strftime("%H:%M"), "15:30")


if __name__ == "__main__":
    unittest.main()
