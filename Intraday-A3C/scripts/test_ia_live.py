"""
Tests for the dry-run runner (ia_live.py). Plain script because pytest is not installed:  python test_ia_live.py

  parity    live features == ia_study features; live-trained models == the cached walk-forward edges; the engine's position path and settled P&L == ia_audit's backtest rule
  behaviour 40-lot cap, no risk before 10:00, loss halt, kill, stale feed, restart/idempotency, settlement arithmetic, feed alignment
  safety    the module has no order endpoint, uses the paper client only and reads nothing but the clock, the calendar and the account
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_audit as A
import ia_data as D
import ia_live as L
import ia_portfolio as P
import ia_study as S

SYM = "TSLA"
_cache: dict = {}
REAL_HISTORY = L.History                                       # the simulated-day test replaces L.History; the fixtures must keep the real one


def full(sym=SYM):
    if sym not in _cache:
        _cache[sym] = (S.build_days(sym), REAL_HISTORY(sym, "sip", top_up=False))
    return _cache[sym]


def day_raw(sym, d):
    b = D.load_bars(sym)
    return b[(b.index >= d) & (b.index < d + pd.Timedelta(days=1))].copy()


class FakeStore(L.Store):
    """Store in a temp dir whose kill switch is a flag (the real one is a file in live/)."""

    def __init__(self):
        super().__init__(tempfile.mkdtemp(prefix="ia_live_test_"))
        self.kill = False

    def killed(self) -> bool:
        return self.kill


def const_predict(v):
    return lambda sym, x: (v, v)


# ── parity ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_feature_parity():
    dd, h = full()
    bars = D.load_bars(SYM)
    rng = np.random.default_rng(0)
    rows = rng.choice(np.where(dd["days"].year >= 2022)[0], 25, replace=False)
    js = sorted(set([0, 1, 2, 3, 5, 6, 11, 12, 13, 35, 70, 71] + list(rng.choice(L.NDEC, 8, replace=False))))
    worst = 0.0
    for i in rows:
        d = dd["days"][i]
        ctx = h.ctx_for(d)
        raw = bars[(bars.index >= d) & (bars.index < d + pd.Timedelta(days=1))]
        for j in js:
            b, miss = L.complete_bars(raw[raw.index < L.decision_time(d, j)], d, j, ctx.last_close)
            assert not miss.any()
            x = L.features_at(b, j, d, ctx)
            worst = max(worst, float(np.abs(x - dd["X"][i, j]).max()))
            assert np.allclose(x, dd["X"][i, j], rtol=1e-9, atol=1e-9), (d, j, dict(zip(S.FEATS, x - dd["X"][i, j])))
    print(f"  feature parity: {len(rows)} sessions x {len(js)} decisions, max abs diff {worst:.2e}")


def test_model_parity_and_pickle():
    import joblib
    dd, _ = full()
    t = time.time()
    m = L.fit_models(SYM, 2025)
    yrs = dd["days"].year.values
    te = np.where(yrs == 2025)[0][[3, 100, 200]]
    ridge, gbm = np.load(os.path.join(D.DATA, f"edges_{SYM}_ridge.npy")), np.load(os.path.join(D.DATA, f"edges_{SYM}_gbm.npy"))
    tmp = os.path.join(tempfile.mkdtemp(prefix="ia_model_"), "m.joblib")
    joblib.dump(m, tmp)
    m2 = joblib.load(tmp)
    for i in te:
        for name, ref in (("ridge", ridge), ("gbm", gbm)):
            got = m[name](dd["X"][i])
            assert np.allclose(got, ref[i], atol=1e-6), (name, i, np.abs(got - ref[i]).max())
            assert np.allclose(m2[name](dd["X"][i]), got, atol=1e-12)
    print(f"  model parity: live-trained ridge and gbm (years < 2025) reproduce the cached walk-forward edges, pickle round-trip exact ({time.time() - t:.0f}s)")


def test_engine_matches_backtest():
    cfg = L.Cfg(feed="sip")
    costs = L.load_costs(cfg)
    syms = cfg.symbols
    dds, hist, E = {}, {}, {}
    for s in syms:
        dds[s], hist[s] = full(s)
        E[s] = (np.load(os.path.join(D.DATA, f"edges_{s}_ridge.npy")) + np.load(os.path.join(D.DATA, f"edges_{s}_gbm.npy"))) / 2         # = ia_portfolio.edges_for(s)["ens"]
    pick = {s: np.where(dds[s]["days"].year.values >= 2025)[0] for s in syms}
    common = sorted(set.intersection(*[set(dds[s]["days"][pick[s]]) for s in syms]))
    days = [common[i] for i in (2, 40, 90, 140, 200)]
    cur = {"date": None}

    def predict(sym, x):
        i = dds[sym]["days"].get_loc(cur["date"])
        e = float(E[sym][i, int(round(x[0]))])
        return e, e

    for d in days:
        cur["date"] = d
        out = L.replay_day(d, cfg, {}, costs, 100_000.0, None, "parquet", root=tempfile.mkdtemp(prefix="ia_replay_"), predict=predict, hist=hist)
        rows = out["store"].read_rows(d)
        for s in syms:
            i = dds[s]["days"].get_loc(d)
            c = costs[s]
            e2 = E[s][i:i + 1].copy()
            e2[:, :5] = 0.0
            pos = A.path(e2, c, 3.0)
            got = L.position_paths(rows, s)
            assert np.array_equal(got, pos[0]), (d, s, np.where(got != pos[0])[0])
            ref = A.net_j(pos, dds[s]["nxt"][i:i + 1], np.full(72, c))[0]
            assert abs(out["res"][s]["net_bp"] - ref) < 1e-6, (d, s, out["res"][s]["net_bp"], ref)
    print(f"  engine parity: positions and settled net bp identical to the backtest rule on {len(days)} sessions x {len(syms)} products")


# ── behaviour ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def one_line_engine(predict, store=None, equity=100_000.0, day_index=-30, mutate=None):
    dd, h = full()
    d = dd["days"][day_index]
    cfg = L.Cfg(symbols=(SYM,), feed="sip")
    store = store or FakeStore()
    eng = L.Engine(cfg, d, equity, {SYM: h}, predict, store, L.load_costs(cfg))
    raw = day_raw(SYM, d)
    if mutate:
        raw = mutate(raw)
    return eng, raw, d, store


def run_steps(eng, raw, d, j0=0, j1=L.NDEC - 1):
    out = []
    for j in range(j0, j1 + 1):
        out += eng.step(j, {SYM: raw[raw.index < L.decision_time(d, j)]}, wall="t")
    return out


def test_cap_and_limits():
    flip = {"n": 0}

    def predict(sym, x):
        flip["n"] += 1
        v = 1e4 if flip["n"] % 2 else -1e4                                   # +3 lots, then -3 lots, ... 6 lots traded a step: the 40-lot cap must bind
        return v, v
    eng, raw, d, _ = one_line_engine(predict)
    rows = run_steps(eng, raw, d)
    ln = eng.lines[SYM]
    pos = np.array([r["pos_after"] for r in rows])
    assert ln.traded <= 40 and sum(abs(r["trade"]) for r in rows) <= 40, ln.traded
    assert np.abs(pos).max() <= 3
    assert ln.traded == 40, "the cap should have been reached with +-1e4 bp edges"
    print(f"  cap: lots traded {ln.traded} (<= 40), max |position| {np.abs(pos).max()}")


def test_no_risk_before_ten():
    eng, raw, d, _ = one_line_engine(const_predict(1e4))
    rows = run_steps(eng, raw, d, 0, 8)
    assert [r["pos_after"] for r in rows[:5]] == [0] * 5 and rows[5]["pos_after"] == 3 and rows[5]["t"] == "10:00", [(r["t"], r["pos_after"]) for r in rows]
    print("  no risk before 10:00: first position at the 10:00 decision")


def test_actions_say_buy_or_sell_and_the_position_says_long_or_short():
    assert L.pos_text(0, 35, "TSLA") == "FLAT"
    assert L.pos_text(2, 35, "TSLA") == "LONG 2 lots = 70 sh TSLA"
    assert L.pos_text(-1, 35, "TSLA") == "SHORT 1 lot = 35 sh TSLA"
    for sign in (1, -1):
        eng, raw, d, _ = one_line_engine(const_predict(1e4 * sign))
        rows = run_steps(eng, raw, d, 0, 8)
        spl = eng.lines[SYM].spl
        first = next(r for r in rows if r["trade"])
        assert first["action"].startswith(f"{'BUY' if sign > 0 else 'SELL'} 3 lots = {3 * spl} sh {SYM} @ "), first["action"]
        assert first["position"] == L.pos_text(3 * sign, spl, SYM) and first["position"].startswith("LONG" if sign > 0 else "SHORT"), first["position"]
        assert all(r.get("action", "") == "" for r in rows if not r["trade"]) and all(r["position"] == "FLAT" for r in rows if r["pos_after"] == 0 and "position" in r)
    print("  readable output: BUY/SELL n lots = x sh SYM @ price, and LONG/SHORT/FLAT position text")


def test_loss_halt_and_kill():
    def crash(raw):
        k = raw.index >= L.bar_start(raw.index[0].normalize(), 20)
        for c in ("open", "high", "low", "close", "vwap"):
            raw.loc[k, c] = raw.loc[k, c] * 0.90
        return raw
    eng, raw, d, _ = one_line_engine(const_predict(1e4), mutate=crash)
    rows = run_steps(eng, raw, d)
    assert eng.halted, "a 10% drop on a 3-lot line (40% of equity) is -4% of equity, past the 1% halt"
    first = next(i for i, r in enumerate(rows) if "loss_halt" in str(r["note"]))
    assert all(r["pos_after"] == 0 for r in rows[first:]), "flat for the rest of the day despite a +1e4 bp edge"
    # kill switch flattens even after the 40-lot cap is exhausted
    eng2, raw2, d2, st = one_line_engine(const_predict(1e4))
    run_steps(eng2, raw2, d2, 0, 10)
    eng2.lines[SYM].traded = 40
    st.kill = True
    r = eng2.step(11, {SYM: raw2[raw2.index < L.decision_time(d2, 11)]}, wall="t")
    assert r[0]["pos_after"] == 0 and r[0]["note"] == "kill", r[0]
    print(f"  loss halt at {rows[first]['t']} (flat afterwards), kill switch flattens through the exhausted cap")


def test_stale_and_missing_bars():
    eng, raw, d, _ = one_line_engine(const_predict(1e4))
    run_steps(eng, raw, d, 0, 10)
    pos = eng.lines[SYM].pos
    assert pos == 3
    holes = raw[raw.index < L.decision_time(d, 11)]
    one_missing = holes.drop(holes.index[-1])
    r = eng.step(11, {SYM: one_missing}, wall="t")[0]
    assert r["missing_now"] == 1 and r["missing_tail"] == 1 and r["note"] == "" and r["pos_after"] == 3, r
    three_missing = raw[raw.index < L.decision_time(d, 14)].iloc[:-3]
    r = eng.step(14, {SYM: three_missing}, wall="t")[0]
    assert r["note"] == "stale_feed_hold" and r["trade"] == 0 and r["pos_after"] == 3, r
    print("  one missing bar is carried forward and traded on, three in a row holds the position")


def test_restart_and_idempotency():
    def predict(sym, x):
        v = 40.0 * np.sin(x[0] / 3.0)                                        # a function of the decision index only, so a restarted run must reproduce the uninterrupted one
        return v, v
    eng, raw, d, st = one_line_engine(predict, day_index=-10)
    rows_a = run_steps(eng, raw, d)
    again = eng.step(20, {SYM: raw[raw.index < L.decision_time(d, 20)]}, wall="t")
    assert again == [], "a decision is never repeated"
    full_pos = [r["pos_after"] for r in rows_a]
    assert len(set(full_pos)) > 2 and sum(abs(r["trade"]) for r in rows_a) > 10
    # same day, process killed after j=30 and restarted on the same store
    st2 = FakeStore()
    eng1, raw1, d1, _ = one_line_engine(predict, store=st2, day_index=-10)
    run_steps(eng1, raw1, d1, 0, 30)
    eng2 = L.Engine(L.Cfg(symbols=(SYM,), feed="sip"), d1, 100_000.0, {SYM: full()[1]}, predict, st2, L.load_costs(L.Cfg(symbols=(SYM,), feed="sip")))
    assert eng2.lines[SYM].pos == eng1.lines[SYM].pos and eng2.lines[SYM].traded == eng1.lines[SYM].traded and eng2.lines[SYM].last_j == 30
    rest = run_steps(eng2, raw1, d1, 25, L.NDEC - 1)                          # steps 25..30 are skipped, 31.. continue
    assert [r["j"] for r in rest] == list(range(31, L.NDEC))
    final = L.position_paths(st2.read_rows(d1), SYM)
    assert np.array_equal(final, np.array(full_pos)), np.where(final != np.array(full_pos))[0]
    print("  restart resumes positions and counters; decisions are never repeated")


def test_settlement_arithmetic():
    P_ = np.full(L.NDEC, 100.0)
    pos = np.array([0.0] * 10 + [2.0] * 62)
    m = L.mark(pos, P_, 101.0, cost_bp=1.0, spl=7)
    assert abs(m["gross_bp"] - 200.0) < 1e-9 and abs(m["lots"] - 2) < 1e-9 and abs(m["net_bp"] - 198.0) < 1e-9
    assert abs(m["net_usd"] - (2 * 7 * 1.0 - 2 * 7 * 100.0 * 1.0 / 1e4)) < 1e-9, m
    print("  settlement arithmetic: gross, lots, net bp and USD on a hand-checked path")


def test_grid_alignment():
    idx = L.day_grid(pd.Timestamp("2025-03-03"), 4)
    sip = pd.DataFrame({"close": [1.0, 2, 3, 4]}, index=idx)
    fb = pd.DataFrame({"open": [10.0, 30], "high": [10.0, 30], "low": [10.0, 30], "close": [10.0, 30], "volume": [5.0, 6], "vwap": [10.0, 30], "trade_count": [1.0, 1]}, index=idx[[0, 3]])
    a = L.align_to_grid(fb, sip)
    assert list(a.close) == [10.0, 10.0, 10.0, 30.0] and list(a.volume) == [5.0, 0.0, 0.0, 6.0] and list(a.vwap) == [10.0, 10.0, 10.0, 30.0]
    b, miss = L.complete_bars(fb.iloc[:1].drop(fb.index[0]), pd.Timestamp("2025-03-03"), 2, last_close=9.0)
    assert list(b.close) == [9.0, 9.0, 9.0] and miss.all() and list(b.volume) == [0.0, 0.0, 0.0]
    print("  missing bars are carried forward with zero volume, also at the open")


def test_weights():
    w = L.weights(L.SYMBOLS)
    assert abs(sum(w.values()) - 1) < 1e-12 and w["SMH"] > w["NVDA"] > w["TSLA"]
    print("  weights:", {k: round(v, 3) for k, v in w.items()})



def test_run_day_simulated():
    """The live loop end to end on a fake clock: waits, polls bars that exist at that moment, decides, logs the MOC intents, settles - and reproduces the replay."""
    import types
    dd, _ = full()
    d = dd["days"][-6]
    cfg = L.Cfg(feed="sip")
    names = ("now_et", "wait_until", "trading_client", "session", "previous_session_date", "Data", "History", "Store", "settle_cmd", "time")
    saved = {k: getattr(L, k) for k in names}
    clock = [d + pd.Timedelta(hours=9)]
    settled, prior = [], {"date": None}

    class FakeData:
        def day_bars(self, sym, date, feed, end=None):
            raw = day_raw(sym, pd.Timestamp(date))
            return raw[raw.index + pd.Timedelta(minutes=5) <= clock[0]]            # a bar exists once its five minutes are over

        def last_quote(self, sym, feed):
            p = float(self.day_bars(sym, d, feed).close.iloc[-1])
            return p - 0.01, p + 0.01

    class FakeTC:
        def get_account(self):
            return types.SimpleNamespace(equity="100000")

    def start(at, store):
        clock[0] = d + pd.Timedelta(hours=at[0], minutes=at[1])
        L.Store = lambda *a, **k: store if not a else saved["Store"](*a, **k)                  # run_day calls Store() (the fake); replay_day calls Store(path) (the real one)
        settled.clear()
        L.run_day(cfg, d)

    try:
        L.now_et = lambda: clock[0]
        L.wait_until = lambda ts: clock.__setitem__(0, max(clock[0], pd.Timestamp(ts)))
        L.trading_client = lambda: FakeTC()
        L.session = lambda tc, date: (d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=16))
        L.previous_session_date = lambda tc, date: prior["date"]
        L.Data = FakeData
        L.History = lambda s, feed, until=None, top_up=True, data=None: full(s)[1]
        L.settle_cmd = lambda c, date: settled.append((str(pd.Timestamp(date).date()), clock[0]))
        L.time = types.SimpleNamespace(time=time.time, sleep=lambda sec: clock.__setitem__(0, clock[0] + pd.Timedelta(seconds=sec)))
        prior["date"] = full(SYM)[1].ctx_for(d).prior_date
        # 1. a full day from 09:00
        st = FakeStore()
        start((9, 0), st)
        costs, models = L.load_costs(cfg), {s: L.ensure_models(s, d.year) for s in cfg.symbols}
        ref = L.replay_day(d, cfg, models, costs, 100_000.0, None, "parquet", root=tempfile.mkdtemp(prefix="ia_replay_"), hist={s: full(s)[1] for s in cfg.symbols})
        a, b = st.read_rows(d), ref["store"].read_rows(d)
        for s in cfg.symbols:
            assert np.array_equal(L.position_paths(a, s), L.position_paths(b, s)), s
        assert sorted(a.j.unique()) == list(range(L.NDEC)), "one decision per bar, 09:35 .. 15:30"
        intents = [__import__("json").loads(x) for x in open(os.path.join(st.day_dir(d), "intents.jsonl"), encoding="utf-8")]
        moc = [i for i in intents if i["type"] == "MOC"]
        assert moc and all(i["mode"] == "DRY_RUN" and i["ts"] < "15:50:00" for i in intents), "MOC intents exist and come before the 15:50 deadline"
        for s in cfg.symbols:                                                       # the MOC quantity offsets what is held after the last decision
            held = int(a[(a.sym == s) & (a.j == L.NDEC - 1)].pos_after.iloc[0])
            q = sum(i["qty"] * (1 if i["side"] == "BUY" else -1) for i in moc if i["sym"] == s)
            spl = int(a[a.sym == s].spl.iloc[0])
            assert q == -held * spl, (s, q, held, spl)
        assert settled and settled[0][1] >= d + pd.Timedelta(hours=16, minutes=20), settled
        # 2. started mid-session: only the latest completed bar and later are decided
        st2 = FakeStore()
        start((12, 7), st2)
        rows = st2.read_rows(d)
        assert rows.j.min() == 30 and rows.j.max() == L.NDEC - 1, (rows.j.min(), rows.j.max())
        # 3. history that does not end at the previous session is refused
        prior["date"] = d
        try:
            start((9, 0), FakeStore())
            raise AssertionError("stale prior closes must stop the run")
        except RuntimeError as e:
            assert "stale" in str(e)
    finally:
        for k, v in saved.items():
            setattr(L, k, v)
    print(f"  simulated live day: same positions as the replay, {len(moc)} MOC intents before 15:50, settle after 16:20, mid-day start resumes at 12:05, stale history refused")


# ── safety ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_no_order_endpoints():
    src = open(L.__file__, encoding="utf-8").read()
    forbidden = ["submit_order", "OrderRequest", "close_all_positions", "close_position", "cancel_order", "replace_order", "place_stock_order", "place_order", "paper=False", "api.alpaca.markets",
                 "update_account_config"]
    hits = [t for t in forbidden if t in src]
    assert not hits, hits
    ctors = re.findall(r"TradingClient\(([^)]*)\)", src)
    assert ctors and all("paper=True" in c for c in ctors), ctors                   # every trading client is the paper one
    called = set(re.findall(r"\btc\.(\w+)\(", src))
    assert called <= {"get_clock", "get_account", "get_calendar"}, called
    assert "arm" not in [c for c in re.findall(r"choices=\[([^\]]*)\]", src)[0].replace('"', "").replace(" ", "").split(",")]
    print(f"  no order endpoint in ia_live.py; trading client is paper-only and calls only {sorted(called)}")


TESTS = [test_weights, test_settlement_arithmetic, test_grid_alignment, test_no_order_endpoints, test_no_risk_before_ten, test_actions_say_buy_or_sell_and_the_position_says_long_or_short, test_cap_and_limits, test_loss_halt_and_kill, test_stale_and_missing_bars,
         test_restart_and_idempotency, test_run_day_simulated, test_feature_parity, test_model_parity_and_pickle, test_engine_matches_backtest]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t0 = time.time()
            print(f"{t.__name__} ...", flush=True)
            t()
        except Exception as e:                                                       # noqa: BLE001
            import traceback
            bad += 1
            print(f"  FAILED {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=4)
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    for d in [p for p in os.listdir(tempfile.gettempdir()) if p.startswith(("ia_live_test_", "ia_replay_", "ia_model_"))]:
        shutil.rmtree(os.path.join(tempfile.gettempdir(), d), ignore_errors=True)
    sys.exit(1 if bad else 0)
