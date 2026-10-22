"""Does the live Engine trade the backtest's exits and modes?  (ADX dynamic trail + Auto-Switch via Vol)

Headless, no Qt, no network, no account: the REAL supertrend_gui.Engine runs with its clock frozen, its state / log files in a throw-away folder and no broker client at all
(DRY RUN: nothing is ever sent). Data = the cached bars (data/SOXL_15m_2016_split.parquet).   Usage:  python test_supertrend_engine_parity.py

  A. unit checks     the formulas, the pass-through of every setting, the reference settings untouched, simulate_session() really uses the user's inputs
  B. modes           every entry of the 10-year backtest (~1.6k): the mode the Engine freezes (record_entry on the 900-bar window the live loop sees) == the backtest's
  C. exit replay     the backtest runs on the newest WINDOW bars with a spy that records its trail factor and entry state; each trade is then replayed bar by bar through the
                     Engine (signal -> record_entry -> manage_position, one call per closed bar). Compared: the mode, the run-up target, the ADX trail factor on every bar,
                     and that the Engine fires its take-profit / trailing stop on exactly the bar the backtest exits, at the backtest's price. Five settings, incl. the reference.

What C isolates: the two settings above. It enters each trade at the BACKTEST's entry reference (entry_ref / frozen Supertrend) - how the Engine picks those is a separate
matter (record_entry freezes the signal bar's open, the backtest the fill bar's) and is reported, not asserted.
Environment: SUPERTREND_PARITY_STRIDE=n replays every n-th trade (default 2), SUPERTREND_PARITY_WINDOW=bars (default 6000), SUPERTREND_PARITY_REPLAY_ONLY=1 skips section A."""
import dataclasses
import os
import queue
import sys
import tempfile
import time
from datetime import datetime, timezone
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["SUPERTREND_NO_AUTO_UPDATE"] = "1"

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import talib                                                  # noqa: E402

import supertrend_gui as GU                                   # noqa: E402
import supertrend_strategy as S                               # noqa: E402
from supertrend_ui import data as D                           # noqa: E402
from supertrend_ui import params as P                         # noqa: E402

STRIDE = int(os.environ.get("SUPERTREND_PARITY_STRIDE", "2"))
WINDOW = int(os.environ.get("SUPERTREND_PARITY_WINDOW", "6000"))
REPLAY_ONLY = os.environ.get("SUPERTREND_PARITY_REPLAY_ONLY") == "1"
S_FILE = os.path.abspath(S.__file__)
INITIAL_CFG = GU.CFG                                          # the module default, before anything assigns CFG

tmp = tempfile.mkdtemp(prefix="st_parity_")
GU.STATE_FILE, GU.LOG_FILE, GU.KILL_FILE = (os.path.join(tmp, n) for n in ("gui_state.json", "live_log.jsonl", "KILL"))
BARS = D.read_cache("SOXL", 15)
REAL_DT = GU.datetime


class FrozenDT(datetime):
    """The Engine reads the wall clock to tell a forming bar from a closed one; in 2099 every cached bar is closed."""
    @classmethod
    def now(cls, tz=None):
        t = datetime(2099, 1, 1, 12, 0, tzinfo=timezone.utc)
        return t.astimezone(tz) if tz is not None else t.replace(tzinfo=None)


def make_engine():
    """A real Engine with no broker: events and log lines are collected, never queued or written."""
    eng = GU.Engine(queue.Queue(), True, 100_000.0, 1.0, "2026-01-01", False)
    eng.events, eng.lines = [], []
    eng.emit = lambda kind, **kw: eng.events.append((kind, kw))
    eng.log = lambda msg, level="info": eng.lines.append(msg)
    return eng


def cfg_for(**over) -> S.Config:
    return P.to_config({**P.DEFAULTS, **over})


# ══ A. unit checks ═══════════════════════════════════════════════════════════════════
def section_a() -> None:
    rng = np.random.default_rng(1)
    # the helper == the backtest's own _trail_factor, for every combination
    n = 0
    for dyn in (False, True):
        for tfac in (0.0, 0.1, 0.3, 0.7, 1.0):
            for base in (10.0, 25.0, 40.0):
                c = dataclasses.replace(cfg_for(), useDynamic=dyn, trailFactor=tfac, adxBase=base)
                for a in [None, float("nan"), 0.0, 12.5, 25.0, 33.3, 61.0, 100.0] + list(rng.uniform(0, 80, 6)):
                    assert GU.dynamic_trail_factor(c, a) == S.SupertrendStrategy(c)._trail_factor(float("nan") if a is None else a), (dyn, tfac, base, a)
                    n += 1
    # Engine.trail_factor / trail_level take the ADX of the last closed bar
    eng = make_engine()
    GU.CFG = dataclasses.replace(cfg_for(), useDynamic=True, trailFactor=0.30, adxBase=25.0)
    assert abs(eng.trail_factor(40.0) - 0.15) < 1e-12 and abs(eng.trail_factor(10.0) - 0.45) < 1e-12 and eng.trail_factor(None) == 0.30 and eng.trail_factor(float("nan")) == 0.30
    long_state = {"side": "long", "entry_ref": 100.0, "target": 104.5, "max_high": 110.0}
    short_state = {"side": "short", "entry_ref": 100.0, "target": 95.5, "min_low": 90.0}
    assert abs(eng.trail_level(long_state, 40.0) - 101.5) < 1e-9 and abs(eng.trail_level(long_state) - 103.0) < 1e-9 and abs(eng.trail_level(long_state, 10.0) - 104.5) < 1e-9
    assert abs(eng.trail_level(short_state, 40.0) - 98.5) < 1e-9 and abs(eng.trail_level(short_state) - 97.0) < 1e-9
    assert eng.trail_level({**long_state, "max_high": 104.0}, 40.0) is None                       # not armed below the run-up target
    # record_entry: the mode follows the volatility regime in Auto-Switch only
    sig0 = {"bar": pd.Timestamp("2026-01-05 10:00"), "open": 100.0, "st": 96.0, "adx": 30.0}
    for mode, hv, want_fixed in (("Always Fixed", True, True), ("Always Fixed", False, True), ("Always Dynamic", True, False), ("Always Dynamic", False, False),
                                 ("Auto-Switch via Vol", True, True), ("Auto-Switch via Vol", False, False), ("Auto-Switch via Vol", None, False)):
        GU.CFG = dataclasses.replace(cfg_for(), strategyMode=mode, riskfactor=1.5, limitfactor=1.2)
        st = eng.record_entry({**sig0, "high_vol": hv}, "SOXL", "long", 10)
        risk = abs(100.0 - 96.0) / 100.0 * 1.5
        assert st["active_fixed"] is want_fixed, (mode, hv)
        assert abs(st["target"] - (100.0 * (1 + (GU.CFG.riskReward if want_fixed else risk)))) < 1e-9, (mode, hv)
        assert abs(eng.tp_level(st) - (100.0 * (1 + GU.CFG.price_perc) if want_fixed else 100.0 * (1 + (100.0 - 96.0) / 100.0 * 1.2))) < 1e-9
    # signal(): ADX / regime computed on the signal bar exactly as the backtest does; nothing extra for the reference settings
    win = BARS.tail(900)
    h, l, c = (win[k].to_numpy(float) for k in ("high", "low", "close"))
    GU.datetime = FrozenDT
    GU.CFG = cfg_for()
    real_adx = talib.ADX
    talib.ADX = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ADX must not be computed with the dynamic trail off"))
    try:
        sig = eng.signal(win)
    finally:
        talib.ADX = real_adx
    assert sig["adx"] is None and sig["high_vol"] is None
    GU.CFG = cfg_for(useDynamic=True, strategyMode="Auto-Switch via Vol", adxLength=21, volLength=30, volSmaLength=80, volThreshold=1.1)
    sig = eng.signal(win)
    ap = talib.ATR(h, l, c, timeperiod=30) / c
    vb = talib.SMA(ap, timeperiod=80)
    assert sig["adx"] == float(talib.ADX(h, l, c, timeperiod=21)[-1]) and sig["high_vol"] == bool(ap[-1] > vb[-1] * 1.1)
    GU.datetime = REAL_DT
    # simulate_session(): every strategy input is the user's; only the session sizing is its own; the reference result is unchanged
    class FakeData:
        def get_stock_bars(self, req):
            idx = BARS.index.tz_localize("America/New_York").tz_convert("UTC")
            st = pd.Timestamp(req.start)
            keep = idx >= (st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC"))
            df = BARS[keep].copy()
            df.index = idx[keep]
            df.index.name = "timestamp"
            df["symbol"] = "SOXL"
            return SimpleNamespace(df=df.set_index("symbol", append=True).swaplevel())

    captured = {}

    class Capture(S.SupertrendStrategy):
        def __init__(self, cfg):
            super().__init__(cfg)
            captured["cfg"] = cfg

    def simulate(cfg):
        GU.CFG, GU.SHORT_EQUITY_MULT = cfg, 0.5                       # the app sets the two separately (SHORT_EQUITY_MULT is not read from CFG)
        e = GU.Engine(queue.Queue(), True, 50_000.0, 0.6, (BARS.index[-1] - pd.Timedelta(days=20)).strftime("%Y-%m-%d"), False)
        e.dclient = FakeData()
        real = S.SupertrendStrategy
        S.SupertrendStrategy = Capture
        try:
            e.simulate_session()
        finally:
            S.SupertrendStrategy = real
        return captured["cfg"]

    user = cfg_for(strategyMode="Auto-Switch via Vol", useDynamic=True, riskfactor=1.7, limitfactor=0.8, adxLength=9, adxBase=30.0, volLength=15, volSmaLength=60,
                   volThreshold=1.15, pessimistic=True, commission_bps=22.0, trailFactor=0.2, touch_fill=True, capacity_usd=5e6)
    got = simulate(user)
    for f in ("strategyMode", "useDynamic", "riskfactor", "limitfactor", "adxLength", "adxBase", "volLength", "volSmaLength", "volThreshold", "pessimistic", "commission_bps",
              "trailFactor", "atrPeriod", "factor", "riskReward", "buyback", "price_perc"):
        assert getattr(got, f) == getattr(user, f), f
    assert got.initial_capital == 50_000.0 and got.equity_frac == 0.6 and got.short_mult == 0.5 and got.touch_fill is False and got.capacity_usd == 1e18 and got.useDateFilter is False and got.strict_entries is True
    # reference settings (and the module default CFG): the Config is what the old hard-coded one was
    for src, label in ((cfg_for(), "reference"), (INITIAL_CFG, "module default")):
        old = S.Config(strategyMode=src.strategyMode, atrPeriod=src.atrPeriod, factor=src.factor, riskReward=src.riskReward, buyback=src.buyback, trailFactor=src.trailFactor,
                       price_perc=src.price_perc, useDynamic=src.useDynamic, useDateFilter=False, initial_capital=50_000.0, equity_frac=0.6, short_mult=0.5,
                       commission_bps=5.0, touch_fill=False, capacity_usd=1e18, strict_entries=True)       # the simulated history is the live-accurate backtest
        assert dataclasses.asdict(simulate(src)) == dataclasses.asdict(old), label
    GU.SHORT_EQUITY_MULT = 0.5
    print(f"A. unit checks OK ({n} trail-factor combinations, 7 mode cases, simulate_session pass-through)")


# ══ spy on the backtest ═══════════════════════════════════════════════════════════════
class SpyStrategy(S.SupertrendStrategy):
    """The backtest, unchanged, plus a record of the trail factor it used on every bar of every open position."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tf, self._ts = {}, None

    def _in_date_range(self, ts):                                     # called once per bar, before the exit logic: tells us which bar the next _trail_factor call is for
        self._ts = ts
        return super()._in_date_range(ts)

    def _trail_factor(self, adx_val):
        tf = super()._trail_factor(adx_val)
        self.tf[self._ts] = (adx_val, tf)
        return tf


def run_spy(cfg: S.Config, bars: pd.DataFrame):
    """(spy, trades, entries): entries[k] = the state the backtest froze when it opened trade k (read from its open_position() call, no change to the strategy)."""
    spy, entries = SpyStrategy(cfg), []

    def prof(frame, event, _arg):
        if frame.f_code.co_name != "open_position" or not frame.f_code.co_filename.endswith("supertrend_strategy.py"):
            return
        if event == "call":
            a, outer = frame.f_locals, frame.f_back.f_locals
            i = a["i"]
            entries.append(dict(i=i, side=a["side"], kind="reversal" if a["qty_fixed"] is not None else "reentry",
                                ref=float(outer["o"][i] if a["ref"] is None else a["ref"]), st_ref=float(outer["st"][i] if a["st_ref"] is None else a["st_ref"]),
                                active_fixed=bool(outer["active_fixed"])))
        elif event == "return":
            loc = frame.f_locals
            if loc["open_rec"] is None:                                # q <= 0: nothing was opened
                entries.pop()
            else:
                entries[-1]["target"] = float(loc["target"])

    sys.setprofile(prof)
    try:
        _eq, trades = spy.run(bars)
    finally:
        sys.setprofile(None)
    return spy, trades.reset_index(drop=True), entries


# ══ B. the mode of every entry ═══════════════════════════════════════════════════════
def section_b() -> None:
    GU.datetime = FrozenDT
    eng = make_engine()
    GU.CFG = cfg = cfg_for(strategyMode="Auto-Switch via Vol", volThreshold=1.1, useDynamic=True)
    spy, trades, ents = run_spy(cfg, BARS)
    assert len(ents) - len(trades) in (0, 1) and all(BARS.index[e["i"]] == t for e, t in zip(ents, trades.entry_time)), (len(ents), len(trades))
    adx_full = talib.ADX(*(BARS[k].to_numpy(float) for k in ("high", "low", "close")), timeperiod=cfg.adxLength)
    bad, n, fixed_n, by_kind = [], 0, 0, {}
    for e in ents:
        i = e["i"]
        win = BARS.iloc[max(0, i + 1 - GU.WARMUP_BARS): i + 1]           # exactly what the live loop would have fetched at that bar
        sig = eng.signal(win)
        if sig is None:
            continue
        st = eng.record_entry({**sig, "open": e["ref"], "st": e["st_ref"]}, "SOXL", e["side"], 10)
        eng.trade_state_save({})
        n += 1
        fixed_n += int(e["active_fixed"])
        k = by_kind.setdefault(e["kind"], [0, 0])
        k[0] += 1
        k[1] += int(st["active_fixed"] == e["active_fixed"])
        if st["active_fixed"] != e["active_fixed"] or abs(st["target"] - e["target"]) > 1e-9 * e["target"]:
            bad.append((BARS.index[i], e["kind"], e["active_fixed"], st["active_fixed"]))
        a = sig.get("adx")                                                 # the ADX of a 900-bar window == the backtest's full-history ADX at that bar
        if a is None or not abs(a - adx_full[i]) <= 1e-9:
            bad.append((BARS.index[i], "adx", float(adx_full[i]), a))
    GU.datetime = REAL_DT
    assert n > 1000 and 0.1 < fixed_n / n < 0.9, (n, fixed_n)              # both modes really occur
    assert not bad, f"{len(bad)} of {n} entries differ from the backtest (mode / target / ADX): {bad[:5]}"
    print(f"B. modes OK: {n} entries on live-sized 900-bar windows, Fixed {fixed_n} / Dynamic {n - fixed_n}; agree {by_kind}; ADX identical to the full-history ADX")


# ══ C. bar-by-bar exit replay ════════════════════════════════════════════════════════
SCENARIOS = {
    "Auto-Switch + ADX trail": dict(strategyMode="Auto-Switch via Vol", useDynamic=True, trailFactor=0.30, riskfactor=1.5, limitfactor=1.2, volThreshold=1.1),
    "Always Dynamic + ADX trail": dict(strategyMode="Always Dynamic", useDynamic=True, trailFactor=0.30, adxBase=20.0),
    "Always Fixed + ADX trail": dict(strategyMode="Always Fixed", useDynamic=True, trailFactor=0.30, adxLength=10),
    "Auto-Switch, ADX off": dict(strategyMode="Auto-Switch via Vol", volThreshold=1.1, riskfactor=1.5),
    "Reference (Always Fixed, ADX off)": dict(),
}


def replay_trade(eng, bars, pos_of, e, trade, spy, stats, problems, tag):
    i, side = e["i"], e["side"]
    sig_i = eng.signal(bars.iloc[: i + 1])
    st = eng.record_entry({**sig_i, "open": e["ref"], "st": e["st_ref"]}, "SOXL", side, 100)
    if st["active_fixed"] != e["active_fixed"]:
        problems.append((tag, bars.index[i], "mode", f"backtest {'fixed' if e['active_fixed'] else 'dynamic'}, Engine {'fixed' if st['active_fixed'] else 'dynamic'}"))
    if abs(st["target"] - e["target"]) > 1e-9 * e["target"]:
        problems.append((tag, bars.index[i], "target", f"backtest {e['target']:.4f}, Engine {st['target']:.4f}"))
    stats["fixed" if e["active_fixed"] else "dynamic"] += 1
    held = {"SOXL": {"qty": 100.0 if side == "long" else -100.0}}
    j_exit = pos_of[trade["exit_time"]] if trade is not None else len(bars) - 1
    reason = trade["exit_reason"] if trade is not None else "open"
    for j in range(i + 1, j_exit + 1):
        sig = eng.signal(bars.iloc[: j + 1])
        ts = bars.index[j]
        if ts in spy.tf:                                                  # the backtest tested this bar's exits
            a_bt, tf_bt = spy.tf[ts]
            tf_eng = eng.trail_factor(sig.get("adx"))
            stats["bars"] += 1
            stats["adx_moves"] += int(abs(tf_bt - S.SupertrendStrategy._trail_factor(spy, float("nan"))) > 1e-12)       # vs the factor with no ADX information (what the old Engine used)
            if abs(tf_eng - tf_bt) > 1e-12:
                problems.append((tag, ts, "trail factor", f"backtest {tf_bt:.4f} (ADX {a_bt:.2f}), Engine {tf_eng:.4f} (ADX {sig.get('adx')})"))
                break
        n_ev = len(eng.events)
        fired = eng.manage_position(sig, held)
        lv = [kw for k, kw in eng.events[n_ev:] if k == "levels"]
        flip = (reason == "reversal" and j == j_exit)
        if flip:                                                           # the backtest skips exits on the bar the trend flips (it reverses instead); the Engine does not
            stats["flip_fires"] += int(fired)
            break
        if j < j_exit:
            if fired:
                problems.append((tag, ts, "early exit", f"Engine exits at {ts}, backtest holds until {bars.index[j_exit]} ({reason})"))
                break
            continue
        # j == j_exit, an exit by take-profit / trailing stop (or the end of the window for a still-open trade)
        if trade is None:
            if fired:
                problems.append((tag, ts, "early exit", "Engine exits, the backtest still holds at the end of the window"))
            break
        if not fired:
            problems.append((tag, ts, "missed exit", f"backtest exits at {trade['exit_price']:.4f} ({reason}), Engine holds"))
            break
        msg = next(m for m in reversed(eng.lines) if m.startswith("EXIT "))
        is_tp = "take-profit" in msg
        tp, trail = lv[-1]["tp"], lv[-1]["trail"]
        lvl = tp if is_tp else trail
        o_j = float(bars.open.iloc[j])
        want_px = (max(o_j, lvl) if side == "long" else min(o_j, lvl)) if is_tp else (min(o_j, lvl) if side == "long" else max(o_j, lvl))
        stats["exits"] += 1
        if abs(want_px - trade["exit_price"]) > 1e-9 * trade["exit_price"]:
            problems.append((tag, ts, "exit level", f"backtest {trade['exit_price']:.4f}, Engine {want_px:.4f} ({'take-profit' if is_tp else 'trailing stop'} {lvl:.4f})"))
    eng.trade_state_save({})


def section_c() -> None:
    bars = BARS.iloc[-WINDOW:]
    pos_of = {t: k for k, t in enumerate(bars.index)}
    problems, vacuous, t0 = [], [], time.time()
    GU.datetime = FrozenDT
    for tag, over in SCENARIOS.items():
        GU.CFG = cfg = cfg_for(**over)
        spy, trades, ents = run_spy(cfg, bars)
        assert len(ents) - len(trades) in (0, 1) and all(bars.index[e["i"]] == t for e, t in zip(ents, trades.entry_time)), (tag, len(ents), len(trades))
        eng = make_engine()
        stats = dict(fixed=0, dynamic=0, bars=0, adx_moves=0, exits=0, flip_fires=0)
        before = len(problems)
        picked = list(range(0, len(ents), STRIDE))
        for k in picked:
            replay_trade(eng, bars, pos_of, ents[k], trades.iloc[k] if k < len(trades) else None, spy, stats, problems, tag)
        if cfg.useDynamic and stats["adx_moves"] <= 20:                    # the ADX really moves the trail in the compared bars, both modes occur, exits are compared: not vacuous
            vacuous.append(f"{tag}: the ADX moved the trail on only {stats['adx_moves']} compared bars")
        if cfg.strategyMode == "Auto-Switch via Vol" and not (stats["fixed"] > 3 and stats["dynamic"] > 3):
            vacuous.append(f"{tag}: only one mode occurred ({stats['fixed']} fixed / {stats['dynamic']} dynamic)")
        if stats["exits"] <= 10 and len(problems) == before:
            vacuous.append(f"{tag}: only {stats['exits']} take-profit / stop exits were compared")
        print(f"   {tag:36s} backtest {len(trades):3d} trades, replayed {len(picked):3d} (fixed {stats['fixed']}/dynamic {stats['dynamic']}); {stats['bars']:5d} bars, ADX moved the trail on "
              f"{stats['adx_moves']}; {stats['exits']} TP/stop exits compared; Engine would also fire on {stats['flip_fires']} flip bars; mismatches {len(problems) - before}")
    GU.datetime = REAL_DT
    if problems:
        by = {}
        for tag, _ts, kind, _msg in problems:
            by.setdefault((tag, kind), 0)
            by[(tag, kind)] += 1
        for tag, ts, kind, msg in problems[:12]:
            print(f"   MISMATCH [{tag}] {ts} {kind}: {msg}")
        raise AssertionError(f"C. the Engine differs from the backtest: {by}")
    assert not vacuous, f"C. the replay compared too little to mean anything: {vacuous}"
    print(f"C. exit replay OK ({len(SCENARIOS)} settings, window {len(bars)} bars, stride {STRIDE}, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    failed = []
    for name, fn in (("A", section_a), ("B", section_b), ("C", section_c)):
        if REPLAY_ONLY and name == "A":
            continue
        try:
            fn()
        except Exception as exc:                                       # noqa: BLE001  (every section reports, so one failure does not hide the others)
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:900]}")
    if failed:
        print(f"ENGINE PARITY FAILED in section(s) {', '.join(failed)}")
        sys.exit(1)
    print("ENGINE PARITY OK - exits and modes equal the backtest's on the cached bars (this is NOT Valid's sign-off)")
