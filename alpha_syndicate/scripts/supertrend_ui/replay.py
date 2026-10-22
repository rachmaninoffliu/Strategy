"""Replay the REAL live Engine over history against a simulated broker and compare every trade with the backtest ("does live trade what the backtest trades?").

How it works: the Engine's own poll (supertrend_gui.Engine.poll_once) is called twice per 15-minute bar - just after the bar opens (the forming bar shows only its open) and just
before it closes (the forming bar shows its full range) - with the Engine's clock, broker and data replaced by simulations. The simulated broker fills market orders at the
current price and rests limit / stop orders, which fill when a bar's range crosses them (the same pierce rules as the backtest). Nothing is sent anywhere.

IMPORTANT: this patches module-level state of supertrend_gui (clock, state file, config). Never run it in the same process as a live session: the GUI runs it in a child process
(python -m supertrend_ui.replay), the tests call it directly."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import pandas as pd

import supertrend_gui as GU
import supertrend_strategy as S

from . import data as D
from . import params as P

ET = "America/New_York"
SOXS_K = 5000.0                      # synthetic SOXS = K / SOXL (a stand-in inverse: the replay checks timing and levels, not SOXS P&L)


class SimDT(datetime):
    """datetime whose now() is the replay clock (the Engine reads the wall clock to tell a forming bar from a closed one)."""
    sim_utc: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        t = cls.sim_utc
        return t.astimezone(tz) if tz is not None else t.astimezone(timezone.utc).replace(tzinfo=None)


class NullQueue:
    def put(self, _item) -> None:
        pass


def _utc(ts: pd.Timestamp) -> datetime:
    return pd.Timestamp(ts).tz_localize(ET).tz_convert("UTC").to_pydatetime()


@dataclass
class Fill:
    bar: int
    phase: str
    symbol: str
    side: str
    qty: float
    price: float
    soxl: float
    oid: str


class SimBroker:
    """Stands in for TradingClient AND the data client. Positions, resting orders and fills follow the backtest's bar rules."""

    def __init__(self, capital: float, bars: pd.DataFrame, shortable: bool = True):
        self.cash = float(capital)
        self.capital = float(capital)
        self.pos: dict = {}                  # symbol -> [qty, avg price]
        self.orders: list = []
        self.fills: list[Fill] = []
        self.shortable = shortable
        self.all_bars = bars
        self.view = bars.iloc[:1]
        self.px = float(bars.close.iloc[0])
        self.bar_i = 0
        self.phase = "open"
        self.now_utc = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self._n = 0
        self._paper = True
        self._base_url = "BaseURL.TRADING_PAPER"

    # ── prices / equity ──
    def price(self, sym: str) -> float:
        return self.px if sym == GU.SYMBOL_LONG else SOXS_K / self.px

    def equity(self) -> float:
        return self.cash + sum(q * self.price(s) for s, (q, _a) in self.pos.items())

    # ── TradingClient surface used by the Engine ──
    def get_account(self):
        return SimpleNamespace(account_number="SIM000000", status="ACTIVE", equity=str(self.equity()), buying_power=str(self.equity() * 4), shorting_enabled=True,
                               pattern_day_trader=False, daytrade_count=0, trading_blocked=False, account_blocked=False)

    def get_clock(self):
        return SimpleNamespace(is_open=True, next_open="", next_close="")

    def get_asset(self, sym):
        return SimpleNamespace(symbol=sym, tradable=True, shortable=self.shortable if sym == GU.SYMBOL_LONG else True, easy_to_borrow=True)

    def get_all_positions(self):
        out = []
        for s, (q, a) in self.pos.items():
            if abs(q) < 1e-9:
                continue
            p = self.price(s)
            out.append(SimpleNamespace(symbol=s, qty=str(q), side=SimpleNamespace(value="long" if q > 0 else "short"), avg_entry_price=str(a), market_value=str(q * p),
                                       unrealized_pl=str(q * (p - a)), unrealized_plpc=str((p / a - 1) * (1 if q > 0 else -1)), current_price=str(p)))
        return out

    def get_orders(self, req=None):
        st = str(getattr(req, "status", "")).upper()
        if "OPEN" in st:
            return [o for o in self.orders if o.open]
        return [o for o in self.orders if not o.open and float(o.filled_qty) > 0]

    def get_order_by_id(self, oid):
        return next(o for o in self.orders if o.id == oid)

    def _new(self, req, kind: str, parent=None) -> SimpleNamespace:
        self._n += 1
        o = SimpleNamespace(id=f"sim-{self._n}", symbol=req.symbol, side=req.side, qty=float(req.qty), kind=kind, limit_price=None, stop_price=None, open=True, status="new",
                            filled_qty="0", filled_avg_price=None, filled_at=None, legs=[], parent=parent, created=self.now_utc)
        self.orders.append(o)
        return o

    def submit_order(self, req):
        lp = getattr(req, "limit_price", None)
        if lp is None:
            o = self._new(req, "market")
            self._fill(o, self.price(o.symbol))
            return o
        o = self._new(req, "limit")
        o.limit_price = float(lp)
        oco = "OCO" in str(getattr(req, "order_class", "")).upper() and getattr(req, "stop_loss", None) is not None
        if oco:
            leg = self._new(SimpleNamespace(symbol=req.symbol, qty=req.qty, side=req.side), "stop", parent=o)
            leg.stop_price = float(req.stop_loss.stop_price)
            o.legs.append(leg)
        if self._marketable(o, self.price(o.symbol)):                     # a limit through the market fills at the market (price improvement)
            self._fill(o, self.price(o.symbol))
        return o

    def cancel_order_by_id(self, oid):
        for o in self.orders:
            if o.id == oid and o.open:
                o.open, o.status = False, "canceled"

    # ── data client surface ──
    def get_stock_latest_quote(self, req):
        return {s: SimpleNamespace(ask_price=self.price(s), bid_price=self.price(s)) for s in req.symbol_or_symbols}

    # ── fills ──
    @staticmethod
    def _buy(o) -> bool:
        return "BUY" in str(o.side).upper()

    def _marketable(self, o, px: float) -> bool:
        if o.kind == "limit":
            return px <= o.limit_price if self._buy(o) else px >= o.limit_price
        return px <= o.stop_price if not self._buy(o) else px >= o.stop_price

    def _fill(self, o, price: float) -> None:
        q = o.qty if self._buy(o) else -o.qty
        cur, avg = self.pos.get(o.symbol, [0.0, 0.0])
        new = cur + q
        if abs(new) < 1e-9:
            self.pos.pop(o.symbol, None)
        else:
            same = cur == 0 or (cur > 0) == (q > 0)
            self.pos[o.symbol] = [new, (avg * abs(cur) + price * abs(q)) / abs(new) if same else (avg if (new > 0) == (cur > 0) else price)]
        self.cash -= q * price
        o.open, o.status, o.filled_qty, o.filled_avg_price = False, "filled", str(o.qty), str(price)
        o.filled_at = self.now_utc
        self.fills.append(Fill(self.bar_i, self.phase, o.symbol, "BUY" if self._buy(o) else "SELL", o.qty, price, self.px, o.id))
        sib = o.parent if o.parent is not None else (o.legs[0] if o.legs else None)
        if sib is not None and sib.open:
            sib.open, sib.status = False, "canceled"

    def at_open(self, o_px: float) -> None:
        """New bar opens at ``o_px``: orders the open gaps through fill at the open."""
        self.px = o_px
        for o in [x for x in self.orders if x.open]:
            if o.open and self._marketable(o, self.price(o.symbol)):
                self._fill(o, self.price(o.symbol))

    def through_range(self, hi: float, lo: float, close: float) -> None:
        """The bar trades through its range: resting take-profits first, then stops, then entry limits (the backtest's order), at their own prices."""
        def pass_(kind_filter):
            for o in [x for x in self.orders if x.open and kind_filter(x)]:
                if not o.open:
                    continue
                lvl = o.limit_price if o.kind == "limit" else o.stop_price
                soxs = o.symbol != GU.SYMBOL_LONG                              # SOXS orders are never resting here; guard anyway
                if soxs:
                    continue
                if o.kind == "limit":
                    hit = (lo <= lvl) if self._buy(o) else (hi >= lvl)
                else:
                    hit = (hi >= lvl) if self._buy(o) else (lo <= lvl)
                if hit:
                    self.px = lvl
                    self._fill(o, lvl)
        held = lambda o: (self.pos.get(o.symbol, [0])[0] > 0) != self._buy(o) and self.pos.get(o.symbol, [0])[0] != 0        # noqa: E731  (reduces a position)
        pass_(lambda o: o.kind == "limit" and held(o))
        pass_(lambda o: o.kind == "stop")
        pass_(lambda o: o.kind == "limit" and not held(o))
        self.px = close


@dataclass
class ReplayResult:
    bt: pd.DataFrame
    live: pd.DataFrame
    pairs: list = field(default_factory=list)            # (bt index | None, live index | None, note)
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0
    log_tail: list = field(default_factory=list)

    @property
    def identical(self) -> bool:
        s = self.stats
        return s["bt_trades"] == s["live_trades"] == s["entries_matched"] and s["exits_matched"] == s["bt_trades"]

    def rows(self) -> list[dict]:
        """One plain dict per paired / unpaired trade (JSON-friendly: the GUI gets these from the child process)."""
        def ts(x):
            return "" if x is None or pd.isna(x) else f"{pd.Timestamp(x):%Y-%m-%d %H:%M}"
        out = []
        for k, j, note in self.pairs:
            b = self.bt.iloc[k] if k is not None else None
            l = self.live.iloc[j] if j is not None else None
            out.append(dict(note=note,
                            bt_side="" if b is None else str(b["side"]), bt_in=ts(None if b is None else b["entry_time"]), bt_px=None if b is None else float(b["price"]),
                            bt_out=ts(None if b is None else b["exit_time"]), bt_exit_px=None if b is None else float(b["exit_price"]), bt_why="" if b is None else str(b.get("exit_reason", "")),
                            live_side="" if l is None else str(l["side"]), live_in=ts(None if l is None else l["entry_time"]), live_px=None if l is None else float(l["entry_px"]),
                            live_out=ts(None if l is None else l["exit_time"]), live_exit_px=None if l is None or pd.isna(l["exit_px"]) else float(l["exit_px"])))
        return out


def _episodes(fills: list[Fill], times: pd.DatetimeIndex) -> pd.DataFrame:
    """Round trips of the STRATEGY side (long SOXL, or short = short SOXL / long SOXS) from the broker's fills."""
    cur, rows = None, []
    hold = {GU.SYMBOL_LONG: 0.0, GU.SYMBOL_SHORT: 0.0}               # shares per instrument (SOXL and SOXS shares are not comparable: only the sign of the strategy side matters)
    for f in fills:
        hold[f.symbol] += f.qty if f.side == "BUY" else -f.qty
        s_long, s_short = hold[GU.SYMBOL_LONG], hold[GU.SYMBOL_SHORT]
        side = 1 if s_long > 1e-6 else -1 if (s_long < -1e-6 or s_short > 1e-6) else 0
        if cur is not None and side != cur["side"]:
            cur.update(exit_bar=f.bar, exit_time=times[f.bar], exit_px=f.soxl, exit_phase=f.phase)
            rows.append(cur); cur = None
        if side != 0 and cur is None:
            cur = {"side": side, "entry_bar": f.bar, "entry_time": times[f.bar], "entry_px": f.soxl, "entry_phase": f.phase, "symbol": f.symbol, "qty": f.qty}
    if cur is not None:
        cur.update(exit_bar=None, exit_time=pd.NaT, exit_px=np.nan, exit_phase="")
        rows.append(cur)
    df = pd.DataFrame(rows)
    if len(df):
        df["side"] = np.where(df["side"] > 0, "long", "short")
    return df


def compare(bt: pd.DataFrame, live: pd.DataFrame, times: pd.DatetimeIndex, tol_bars: int = 1) -> tuple[list, dict]:
    """Pair each backtest trade with the live trade of the same side whose entry is within ``tol_bars`` bars (a reversal is decided on bar i's close and filled at bar i+1's open,
    so it can be one bar apart); then compare the exits the same way. Entry-price differences are reported, not judged (a polled SOXS entry cannot match a limit price)."""
    pos = {t: i for i, t in enumerate(times)}
    bt_i = [pos.get(pd.Timestamp(t), -1) for t in bt["entry_time"]] if len(bt) else []
    used, pairs, em, xm, pdiff = set(), [], 0, 0, []
    bt_x = [pos.get(pd.Timestamp(t), -1) if not pd.isna(t) else None for t in bt["exit_time"]] if len(bt) else []     # None = still open
    for k in range(len(bt)):
        best = None
        for j in range(len(live)):
            if j in used or live.at[j, "side"] != bt.at[k, "side"]:
                continue
            d = abs(int(live.at[j, "entry_bar"]) - bt_i[k])
            if d <= tol_bars and (best is None or d < best[0]):
                best = (d, j)
        if best is None:
            pairs.append((k, None, "backtest only"))
            continue
        j = best[1]
        used.add(j)
        em += 1
        live_open = live.at[j, "exit_bar"] is None or pd.isna(live.at[j, "exit_bar"])
        ex_ok = live_open if bt_x[k] is None else (not live_open and abs(int(live.at[j, "exit_bar"]) - bt_x[k]) <= tol_bars)
        pdiff.append(abs(live.at[j, "entry_px"] / bt.at[k, "price"] - 1))
        xm += int(ex_ok)
        pairs.append((k, j, "matched" if ex_ok else "entry matched; exit differs"))
    for j in range(len(live)):
        if j not in used:
            pairs.append((None, j, "live only"))
    st = dict(bt_trades=len(bt), live_trades=len(live), entries_matched=em, exits_matched=xm,
              full_matches=sum(1 for p in pairs if p[2] == "matched"), backtest_only=sum(1 for p in pairs if p[2] == "backtest only"), live_only=sum(1 for p in pairs if p[2] == "live only"),
              entry_px_diff_mean=float(np.mean(pdiff)) if pdiff else 0.0, entry_px_diff_max=float(np.max(pdiff)) if pdiff else 0.0)
    return pairs, st


def replay_live(bars: pd.DataFrame, values: dict, start: str, end: str, capital: float = 100_000.0, shortable: bool = True, log: Callable = lambda m: None,
                progress: Callable = lambda i, n: None, cancelled: Callable = lambda: False, strict: bool = True, engine_bars: Optional[int] = None) -> ReplayResult:
    """Run the Engine over [start, end] of ``bars`` and the backtest over the same window with the same settings (zero commission: a simulated broker has none).
    ``strict``: compare against the backtest WITHOUT look-ahead (the only one a live engine can match); strict=False compares with the original backtest.
    ``engine_bars``: how many bars of history the Engine loads on each poll (default: the Engine's own WARMUP_BARS)."""
    t0 = time.time()
    n_tail = int(engine_bars or GU.WARMUP_BARS)
    warm = max(D.warmup_bars(values), 520, n_tail)
    sub, first = D.window(bars, start, end, warm)
    if first is None or len(sub) < 80:
        raise ValueError(f"no bars between {start} and {end}")
    i0 = int(sub.index.get_loc(first))
    vals = {**values, "commission_bps": 0.0, "touch_fill": False, "pessimistic": False, "capacity_usd": 0.0}
    # the backtest side uses strict_entries: a live order can only know the trend / regime as of the PREVIOUS bar's close (see SupertrendStrategy / Config.strict_entries)
    bt_cfg = P.to_config(vals, initial_capital=float(capital), useDateFilter=True, startDate=str(first), endDate="2099-12-31", strict_entries=strict)
    strat = S.SupertrendStrategy(bt_cfg)
    _eq, tr = strat.run(sub)
    bt = tr.reset_index(drop=True) if len(tr) else pd.DataFrame(columns=["side", "entry_time", "price", "exit_time", "exit_price", "exit_reason"])
    bt_open = strat.open_trade                       # the position the backtest still holds after the last bar: it must be compared too
    if bt_open is not None:
        bt = pd.concat([bt, pd.DataFrame([{"side": bt_open["side"], "entry_time": bt_open["entry_time"], "price": bt_open["price"], "exit_time": pd.NaT, "exit_price": np.nan,
                                           "exit_reason": "open"}])], ignore_index=True)

    live_cfg = P.to_config(vals, initial_capital=float(capital))
    tmp = tempfile.mkdtemp(prefix="st_replay_")
    saved = {k: getattr(GU, k) for k in ("CFG", "SHORT_EQUITY_MULT", "MAX_TRADES_PER_DAY", "DAILY_LOSS_LIMIT_PCT", "PREFER_SHORT_SOXL", "STATE_FILE", "LOG_FILE", "KILL_FILE",
                                         "REQUIRE_MARKET_OPEN", "datetime", "WARMUP_BARS")}
    GU.WARMUP_BARS = n_tail
    GU.CFG, GU.SHORT_EQUITY_MULT = live_cfg, live_cfg.short_mult
    GU.MAX_TRADES_PER_DAY, GU.DAILY_LOSS_LIMIT_PCT, GU.PREFER_SHORT_SOXL = 10 ** 9, -1e9, True
    GU.STATE_FILE, GU.LOG_FILE, GU.KILL_FILE = (os.path.join(tmp, n) for n in ("state.json", "log.jsonl", "KILL"))
    GU.datetime = SimDT
    broker = SimBroker(capital, sub, shortable)
    eng = GU.Engine(NullQueue(), True, float(capital), live_cfg.equity_frac, str(first)[:10], True)
    eng.tclient = eng.dclient = broker
    eng.report_open = False              # the replay compares the open position itself, once, at the end
    msgs: list = []
    eng.log = lambda m, level="info": msgs.append(f"{SimDT.sim_utc.astimezone(timezone.utc):%Y-%m-%d %H:%M} {level}: {m}")
    eng.fetch_bars = lambda: broker.view.tail(n_tail)
    eng.session_equity = lambda equity, use_cache=False: dict(baseline=float(capital), realised=broker.equity() - capital, unrealised=0.0, equity=broker.equity(),
                                                               source="replay", n_trades=0, trades=[])
    n = len(sub)
    try:
        for j in range(i0, n):
            if cancelled():
                break
            ts = sub.index[j]
            o_, h_, l_, c_ = (float(sub[k].iloc[j]) for k in ("open", "high", "low", "close"))
            closed = sub.iloc[:j]
            broker.bar_i = j
            # 1) the bar opens: gaps fill resting orders at the open, then the Engine polls and sees only the open
            broker.phase = "open"
            SimDT.sim_utc = _utc(ts) + timedelta(seconds=20)
            broker.now_utc = SimDT.sim_utc
            broker.at_open(o_)
            forming = pd.DataFrame({"open": [o_], "high": [o_], "low": [o_], "close": [o_]}, index=[ts])
            broker.view = pd.concat([closed, forming])
            _poll(eng, msgs)
            # 2) the bar trades through its range (resting orders fill), then the Engine polls again just before the close
            broker.phase = "late"
            SimDT.sim_utc = _utc(ts) + timedelta(minutes=14, seconds=40)
            broker.now_utc = _utc(ts) + timedelta(minutes=7)
            broker.through_range(h_, l_, c_)
            broker.now_utc = SimDT.sim_utc
            broker.view = sub.iloc[:j + 1]
            _poll(eng, msgs)
            if (j - i0) % 200 == 0:
                progress(j - i0, n - i0)
        open_check = None                                         # the position the Engine holds at the end vs the one the strict backtest holds on the bars it can see (levels included)
        try:
            bars_end = broker.view.tail(n_tail)
            sig_end = eng.signal(bars_end)
            live_end, exp_end = eng.live_open(sig_end, eng.positions()), eng.expected_open(bars_end, sig_end)
            open_check = {**GU.compare_open(live_end, exp_end, forming=bool(sig_end.get("forming"))), "live": live_end, "expected": exp_end}
        except Exception as exc:                                  # noqa: BLE001
            open_check = {"status": "error", "text": f"{type(exc).__name__}: {exc}", "diffs": []}
    finally:
        for k, v in saved.items():
            setattr(GU, k, v)
    live = _episodes(broker.fills, sub.index)
    pairs, st = compare(bt, live, sub.index)
    st.update(open_check=open_check, bars=n - i0, polls=2 * (n - i0), live_final_equity=broker.equity(), bt_final_equity=float(_eq.iloc[-1]) if len(_eq) else float(capital),
              bt_open=None if bt_open is None else {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in bt_open.items()},
              live_open=None if not len(live) or not pd.isna(live["exit_time"].iloc[-1]) else {"side": live["side"].iloc[-1], "entry_time": str(live["entry_time"].iloc[-1]),
                                                                                          "price": float(live["entry_px"].iloc[-1])})
    return ReplayResult(bt=bt, live=live, pairs=pairs, stats=st, seconds=time.time() - t0, log_tail=msgs[-40:])


def _poll(eng, msgs: list) -> None:
    try:
        eng.poll_once()
    except Exception as exc:                                      # noqa: BLE001  (the live loop logs and carries on, so does the replay)
        msgs.append(f"POLL ERROR {type(exc).__name__}: {exc}")


def describe(res: ReplayResult, times: Optional[pd.DatetimeIndex] = None, limit: int = 12) -> str:
    s = res.stats
    head = (f"{'IDENTICAL' if res.identical else 'DIFFERENT'}: backtest {s['bt_trades']} trades, live {s['live_trades']}; entries matched {s['entries_matched']}, exits matched {s['exits_matched']}, "
            f"fully matched {s['full_matches']}; backtest-only {s['backtest_only']}, live-only {s['live_only']}; entry price vs backtest: mean {s['entry_px_diff_mean'] * 100:.2f}%, "
            f"max {s['entry_px_diff_max'] * 100:.2f}%  ({s['bars']:,} bars, {res.seconds:.0f} s)")
    lines = [head]
    oc = s.get("open_check")
    if oc:
        lines.append(f"  open position at the end ({oc['status'].upper()}): {oc['text']}")
    n = 0
    for k, j, note in res.pairs:
        if note == "matched":
            continue
        n += 1
        if n > limit:
            lines.append("  ...")
            break
        b = res.bt.iloc[k] if k is not None else None
        l = res.live.iloc[j] if j is not None else None
        lines.append(f"  {note:30s} " + (f"BT {b['side']:5s} in {pd.Timestamp(b['entry_time']):%m-%d %H:%M} @ {b['price']:.2f} out {pd.Timestamp(b['exit_time']):%m-%d %H:%M} ({b.get('exit_reason', '')})" if b is not None else " " * 52) +
                     "   |   " + (f"LIVE {l['side']:5s} in {pd.Timestamp(l['entry_time']):%m-%d %H:%M} @ {l['entry_px']:.2f} out " + (f"{pd.Timestamp(l['exit_time']):%m-%d %H:%M} @ {l['exit_px']:.2f}" if not pd.isna(l['exit_time']) else "open") if l is not None else ""))
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Replay the live Engine against the backtest")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--months", type=int, default=3, help="window length ending at the newest cached bar (when --start is not given)")
    ap.add_argument("--no-shortable", action="store_true", help="SOXL cannot be shorted: shorts go to SOXS")
    ap.add_argument("--settings", default=None, help="json file with the strategy values (default: the reference settings)")
    ap.add_argument("--lookahead", action="store_true", help="compare with the ORIGINAL backtest (which lets a flat re-entry use the entry bar's own close) instead of the strict one")
    ap.add_argument("--engine-bars", type=int, default=None, help="bars of history the Engine loads per poll (default: its WARMUP_BARS)")
    ap.add_argument("--json", action="store_true", help="print one JSON result line (used by the GUI)")
    a = ap.parse_args(argv)
    bars = D.read_cache("SOXL", 15)
    end = a.end or f"{bars.index[-1]:%Y-%m-%d}"
    start = a.start or f"{(pd.Timestamp(end) - pd.DateOffset(months=a.months)):%Y-%m-%d}"
    values = dict(P.DEFAULTS)
    if a.settings:
        values.update(json.load(open(a.settings, encoding="utf-8")))

    def prog(i, n):
        print(f"PROGRESS {i} {n}", flush=True)
    res = replay_live(bars, values, start, end, shortable=not a.no_shortable, progress=prog, strict=not a.lookahead, engine_bars=a.engine_bars)
    if a.json:
        print("RESULT " + json.dumps({"stats": res.stats, "identical": res.identical, "text": describe(res), "seconds": res.seconds, "log_tail": res.log_tail[-15:], "rows": res.rows(),
                                      "start": start, "end": end, "shortable": not a.no_shortable, "strict": not a.lookahead}, default=str), flush=True)
    else:
        print(describe(res, limit=40))
        for m in res.log_tail[-12:]:
            print("   ", m)
    return 0 if res.identical else 1


if __name__ == "__main__":
    sys.exit(main())
