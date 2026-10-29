"""
DRY-RUN paper runner for the intraday session-convergence basket (TSLA / SMH / NVDA, see DESIGN.md).

* Decides every 5 minutes (bar ends 09:35 .. 15:30 ET, first trade at 10:00; what is held after the 15:30 decision goes straight into a market-on-close order, 20 minutes before the deadline)
  with the same features, models, cost rule and caps as the backtest (ia_study / ia_audit), keeps a SHADOW book, and writes what it would have sent.
* It never places, replaces or cancels an order. The only Alpaca calls are market-data reads plus the paper account's clock, calendar and account balance
  (TradingClient(paper=True)); the intents it writes to live/<date>/intents.jsonl are files, nothing else. test_ia_live.py checks the source for order endpoints.
* This account may not query recent SIP data (only IEX in real time), so live signals use IEX bars with the SIP-trained models (ia_feed_test.py: edge correlation
  0.98-0.99, basket Sharpe 1.28 vs 1.35 on 2022-26). Settlement after the close re-marks everything on the consolidated tape and the official closing price.

  python ia_live.py status | train | run | replay --date YYYY-MM-DD [--feed iex|sip] [--source api|parquet] | settle --date YYYY-MM-DD | report | refresh
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_data as D
import ia_study as S

LIVE = os.path.join(D.ROOT, "live")
MODEL_DIR = os.path.join(D.DATA, "models")
SYMBOLS = ("TSLA", "SMH", "NVDA")
LINE_VOL = {"TSLA": 2.46, "SMH": 1.18, "NVDA": 2.00}         # daily P&L vol of a line run at 3 lots, % of its capital (reports/audit.log, section A): inverse-vol weights
NB, NDEC, BAR_MIN = S.NB, S.NDEC, 5


def log(msg: str):
    print(f"{pd.Timestamp.now(tz=D.NY):%H:%M:%S} {msg}", flush=True)


# ── configuration ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    symbols: tuple = SYMBOLS
    feed: str = "iex"
    k: float = 3.0                         # position = floor(edge / (k * cost)), the ens_k3 configuration
    slip_bp: float = S.SLIP_BP
    vmax: int = S.VMAX
    qhigh: int = S.QHIGH
    first_j: int = 5                       # no new risk before 10:00: the first decision that may trade is the one at the 10:00 bar end
    last_j: int = NDEC - 1                 # the 15:30 decision is the last; the position it leaves is exited by a market-on-close order
    deploy_frac: float = 0.40              # share of equity at full lots; about an 8% annual vol target (DESIGN.md section 5)
    loss_halt: float = 0.010               # flatten and stop for the day when the shadow day P&L reaches -1.0% of equity
    dd_stop: float = 0.06                  # no new risk while the strategy's shadow drawdown is >= 6% of equity
    equity: float = 100_000.0              # used only when the paper account cannot be read
    stale_bars: int = 3                    # this many consecutive missing bars at the decision time = feed outage, hold
    decision_delay_s: float = 4.0          # wait this long after a bar end for the bar to be published
    moc_deadline: str = "15:50:00"         # market-on-close entry deadline (NYSE and Nasdaq)
    settle_time: str = "16:20:00"          # consolidated (SIP) history is only queryable 15 minutes after the fact


def pos_text(lots: int, spl: int, sym: str) -> str:
    """'LONG 2 lots = 70 sh TSLA' / 'SHORT 1 lot = 35 sh TSLA' / 'FLAT'."""
    if not lots:
        return "FLAT"
    return f"{'LONG' if lots > 0 else 'SHORT'} {abs(lots)} lot{'s' if abs(lots) > 1 else ''} = {abs(lots) * spl} sh {sym}"


def weights(symbols) -> dict:
    inv = {s: 1.0 / LINE_VOL[s] for s in symbols}
    tot = sum(inv.values())
    return {s: v / tot for s, v in inv.items()}


def load_costs(cfg: Cfg) -> dict:
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    return {s: float(sp.spread_bp[s]) / 2.0 + cfg.slip_bp for s in cfg.symbols}         # half the sampled SIP quoted spread + slippage, bp per lot (the backtest rule)


# ── time grid ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def bar_start(date, j: int) -> pd.Timestamp:
    return pd.Timestamp(date).normalize() + pd.Timedelta(hours=9, minutes=30 + BAR_MIN * j)


def decision_time(date, j: int) -> pd.Timestamp:
    return bar_start(date, j + 1)


def day_grid(date, n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([bar_start(date, i) for i in range(n)])


def to_utc(ts_et: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts_et).tz_localize(D.NY).tz_convert("UTC")


# ── data access (reads only) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Data:
    def __init__(self):
        self.dc = D.client()

    def _bars(self, sym, tf, start, end, feed):
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=start.to_pydatetime(), end=end.to_pydatetime(), adjustment=Adjustment.SPLIT,
                               feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)
        for attempt in range(2):
            try:
                df = self.dc.get_stock_bars(req).df
                return df.reset_index(level=0, drop=True) if len(df) else pd.DataFrame()
            except KeyError:
                return pd.DataFrame()
            except Exception as e:                                                      # noqa: BLE001
                if attempt == 1:
                    raise
                time.sleep(1.5)

    def day_bars(self, sym: str, date, feed: str, end: pd.Timestamp | None = None) -> pd.DataFrame:
        """Regular-hours 5-minute bars of one session, index = bar start in ET (naive). `end` (UTC) cuts the day short for the live loop."""
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        t0, t1 = to_utc(bar_start(date, 0)), to_utc(bar_start(date, NB))
        df = self._bars(sym, TimeFrame(BAR_MIN, TimeFrameUnit.Minute), t0, min(t1, end) if end is not None else t1, feed)
        if not len(df):
            return df
        df.index = df.index.tz_convert(D.NY).tz_localize(None)
        return df[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]

    def daily(self, sym: str, start, end) -> pd.DataFrame:
        """Official (consolidated, split-adjusted) daily bars [start, end], same layout as daily_closes.parquet."""
        from alpaca.data.timeframe import TimeFrame
        df = self._bars(sym, TimeFrame.Day, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1), "sip")
        if not len(df):
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        out = df.reset_index()
        out["date"] = pd.to_datetime(out["timestamp"]).dt.tz_convert(D.NY).dt.tz_localize(None).dt.normalize()
        out["symbol"] = sym
        return out[["symbol", "date", "open", "high", "low", "close", "volume"]]

    def last_quote(self, sym: str, feed: str):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self.dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym, feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP))[sym]
        return float(q.bid_price), float(q.ask_price)

    def quote_window(self, sym: str, ts_et: pd.Timestamp, seconds: float = 5.0):
        """Median consolidated bid and ask over the `seconds` after ts (historical SIP quotes, needs them to be older than 15 minutes). A single quote is too noisy: the first one after a bar
        end can sit several cents away from the typical quote of the next seconds."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest
        t0 = to_utc(ts_et)
        df = self.dc.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=t0.to_pydatetime(), end=(t0 + pd.Timedelta(seconds=seconds)).to_pydatetime(), limit=300, feed=DataFeed.SIP)).df
        if not len(df):
            return None
        df = df[(df.bid_price > 0) & (df.ask_price > df.bid_price)]
        return (float(df.bid_price.median()), float(df.ask_price.median())) if len(df) >= 3 else None


def trading_client():
    from alpaca.trading.client import TradingClient

    import alpaca_keys
    k, s = alpaca_keys.creds(True)
    return TradingClient(k, s, paper=True)                                                      # paper endpoint only; used for clock, calendar and balance reads


def session(tc, date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(open, close) of the session in ET, None when the market is closed that day."""
    from alpaca.trading.requests import GetCalendarRequest
    d = pd.Timestamp(date).date()
    cal = tc.get_calendar(GetCalendarRequest(start=d, end=d))
    return (pd.Timestamp(cal[0].open), pd.Timestamp(cal[0].close)) if cal and cal[0].date == d else None


def previous_session_date(tc, date) -> pd.Timestamp:
    from alpaca.trading.requests import GetCalendarRequest
    d = pd.Timestamp(date).date()
    cal = [c for c in tc.get_calendar(GetCalendarRequest(start=d - pd.Timedelta(days=12), end=d - pd.Timedelta(days=1)))]
    return pd.Timestamp(cal[-1].date)


# ── history: prior closes and the volume-pace reference, point in time ────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Ctx:
    date: pd.Timestamp
    prevC: float
    prev2C: float
    pace_ref: np.ndarray
    last_close: float            # last bar close of the previous session on the signal feed (fills a missing first bar)
    prior_date: pd.Timestamp     # date of prevC


def align_to_grid(fb: pd.DataFrame, sip: pd.DataFrame) -> pd.DataFrame:
    """Feed bars on the SIP 78-bar grid; a bar with no trade is carried forward (close), volume 0 - the rule used in ia_feed_test and by the runner."""
    grid = sip.index[sip.index >= fb.index[0].normalize()]
    a = fb.reindex(grid)
    a["close"] = a.close.ffill()
    for c in ("open", "high", "low", "vwap"):
        a[c] = a[c].fillna(a.close)
    a["volume"] = a.volume.fillna(0.0)
    a["trade_count"] = a.trade_count.fillna(0.0)
    return a


class History:
    """Everything the features need from before the trading day. `until` (exclusive) makes it point in time for replays; with top_up the research parquets are extended
    in memory up to `until` from the API (nothing is written)."""

    def __init__(self, sym: str, feed: str, until=None, top_up: bool = True, data: Data | None = None):
        sip = D.load_bars(sym)
        daily = D.load_daily()
        daily = daily[daily.symbol == sym]
        fb = sip if feed == "sip" else D.load_bars(sym, feed)
        if top_up and until is not None:
            data = data or Data()
            sip, fb, daily = self._top_up(data, sym, feed, sip, fb, daily, pd.Timestamp(until).normalize())
        if until is not None:
            u = pd.Timestamp(until).normalize()
            sip, fb, daily = sip[sip.index < u], fb[fb.index < u], daily[daily.date < u]
        self.sym, self.feed = sym, feed
        self.bars = fb if feed == "sip" else align_to_grid(fb, sip)
        self.dc = daily.set_index("date").close.sort_index()
        dd = S.build_days(sym, self.bars, daily)
        self.days, self.cumv = dd["days"], dd["cumv"]

    @staticmethod
    def _top_up(data, sym, feed, sip, fb, daily, until):
        end = min(to_utc(until), pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))
        def more(frame, which):
            start = frame.index.max().normalize() + pd.Timedelta(days=1)
            if to_utc(until) - pd.Timestamp(start, tz="UTC") < pd.Timedelta(hours=1):
                return frame
            add = D.fetch_bars(sym, BAR_MIN, log, feed=which, start=start, end=end)
            return pd.concat([frame, add]) if len(add) else frame
        sip2 = more(sip, "sip")
        fb2 = sip2 if feed == "sip" else more(fb, feed)
        last = daily.date.max()
        if (until - last).days > 1:
            add = data.daily(sym, last + pd.Timedelta(days=1), until - pd.Timedelta(days=1))
            if len(add):
                daily = pd.concat([daily, add], ignore_index=True)
        return sip2, fb2, daily

    def ctx_for(self, date) -> Ctx:
        date = pd.Timestamp(date).normalize()
        n = int(self.days.searchsorted(date))                                         # kept sessions strictly before the trading day
        lo = max(0, n - 20)
        pace = self.cumv[lo:n].mean(axis=0) if n - lo >= 5 else np.full(NB, np.nan)  # rolling(20, min_periods=5).mean().shift(1) of ia_study.build_days
        prior = self.dc[self.dc.index < date]
        before = self.bars.close[self.bars.index < date]
        return Ctx(date, float(prior.iloc[-1]), float(prior.iloc[-2]), pace, float(before.iloc[-1]), prior.index[-1])


# ── features: the 16 inputs of ia_study.build_days, from the bars of ONE session up to the decision ─────────────────────────────────────────────────────────
def complete_bars(raw: pd.DataFrame | None, date, j: int, last_close: float) -> tuple[pd.DataFrame, np.ndarray]:
    """The session's bars 0..j on the full grid. A missing bar (no trade in the 5 minutes, or not published yet) is carried forward with volume 0. Also returns the missing mask."""
    g = day_grid(date, j + 1)
    a = raw.reindex(g) if raw is not None and len(raw) else pd.DataFrame(index=g, columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"], dtype=float)
    miss = a.close.isna().to_numpy()
    if miss[0]:
        a.iloc[0, a.columns.get_loc("close")] = last_close
    a["close"] = a.close.ffill()
    for c in ("open", "high", "low", "vwap"):
        a[c] = a[c].fillna(a.close)
    a["volume"] = a.volume.fillna(0.0)
    return a, miss


def features_at(b: pd.DataFrame, j: int, date, ctx: Ctx) -> np.ndarray:
    """Same arithmetic as ia_study.build_days for decision j; b = complete_bars(...) output with j+1 rows."""
    P, V, VW, H, L = (b[c].to_numpy(float) for c in ("close", "volume", "vwap", "high", "low"))
    O0 = float(b.open.iloc[0])
    cumv = np.cumsum(V)
    vwap = np.cumsum(VW * V) / np.maximum(cumv, 1)
    hi, lo = H.max(), L.min()
    def lag(k):
        return (P[j] / P[j - k] - 1) * 1e4 if j >= k else 0.0
    r1 = np.zeros(j + 1)
    r1[1:] = (P[1:] / P[:-1] - 1) * 1e4
    rv = pd.Series(r1).rolling(12, min_periods=3).std().iloc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = cumv[j] / ctx.pace_ref[j]
    vp = 1.0 if (not np.isfinite(vp)) else vp                                         # nan_to_num(nan=1.0, posinf=1.0) in build_days
    f = {
        "tod": float(j), "vwap_dist": (P[j] - vwap[j]) / P[j] * 1e4, "from_open": (P[j] / O0 - 1) * 1e4, "from_prev": (P[j] / ctx.prevC - 1) * 1e4, "gap": (O0 / ctx.prevC - 1) * 1e4,
        "ret1": lag(1), "ret3": lag(3), "ret6": lag(6), "ret12": lag(12), "rng_pos": (P[j] - lo) / max(hi - lo, 1e-9), "range_bp": (hi - lo) / P[j] * 1e4,
        "rvol12": 0.0 if np.isnan(rv) else float(rv), "volpace": float(vp), "r30": (P[min(5, j)] / O0 - 1) * 1e4, "prev_ret": (ctx.prevC / ctx.prev2C - 1) * 1e4,
        "dow": float(pd.Timestamp(date).dayofweek),
    }
    return np.nan_to_num(np.array([f[k] for k in S.FEATS], float))


# ── models: ridge + boosted trees on all SIP history before the current year, refit each January ──────────────────────────────────────────────────────────
def fit_models(sym: str, year: int, stride: int = 2) -> dict:
    dd = S.build_days(sym)
    tr = np.where(dd["days"].year.values < year)[0]
    nf = dd["X"].shape[2]
    Xtr, ytr = dd["X"][tr][:, ::stride].reshape(-1, nf), dd["y"][tr][:, ::stride].reshape(-1)
    return dict(sym=sym, year=year, ridge=S.fit_model(Xtr, ytr, "ridge"), gbm=S.fit_model(Xtr, ytr, "gbm"), n_days=len(tr), last_day=str(dd["days"][tr[-1]].date()))


def ensure_models(sym: str, year: int, force: bool = False) -> dict:
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    f = os.path.join(MODEL_DIR, f"{sym}_{year}.joblib")
    if os.path.exists(f) and not force:
        return joblib.load(f)
    m = fit_models(sym, year)
    joblib.dump(m, f)
    return m


def model_predict(models: dict):
    def predict(sym: str, x: np.ndarray) -> tuple[float, float]:
        m = models[sym]
        return float(m["ridge"](x[None])[0]), float(m["gbm"](x[None])[0])
    return predict


# ── persistence ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Store:
    def __init__(self, root: str = LIVE):
        self.root = root

    def day_dir(self, date) -> str:
        d = os.path.join(self.root, f"{pd.Timestamp(date):%Y-%m-%d}")
        os.makedirs(d, exist_ok=True)
        return d

    def killed(self) -> bool:
        return os.path.exists(os.path.join(LIVE, "KILL"))

    def append_rows(self, date, rows: list[dict]):
        if rows:
            f = os.path.join(self.day_dir(date), "decisions.csv")
            pd.DataFrame(rows).to_csv(f, mode="a", header=not os.path.exists(f), index=False)

    def read_rows(self, date) -> pd.DataFrame:
        f = os.path.join(self.day_dir(date), "decisions.csv")
        return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()

    def intent(self, date, rec: dict):
        with open(os.path.join(self.day_dir(date), "intents.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def save_state(self, date, st: dict):
        f = os.path.join(self.day_dir(date), "state.json")
        with open(f + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1, default=str)
        os.replace(f + ".tmp", f)

    def load_state(self, date) -> dict | None:
        f = os.path.join(self.day_dir(date), "state.json")
        return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


# ── the engine: one decision step per bar, shared by the live loop, replay and the tests ────────────────────────────────────────────────────────────────────
@dataclass
class Line:
    sym: str
    ctx: Ctx
    spl: int                                  # shares per lot, fixed for the day
    cost_bp: float
    pos: int = 0                              # lots held after the latest decision
    traded: int = 0                           # lots traded today
    last_j: int = -1
    px_prev: float = float("nan")
    pnl_est: float = 0.0                      # shadow day P&L in USD on the signal feed's own prices (the loss halt uses it; settle re-marks on the SIP tape)
    ffilled: int = 0

    def state(self) -> dict:
        return dict(pos=self.pos, traded=self.traded, last_j=self.last_j, px_prev=self.px_prev, pnl_est=self.pnl_est, ffilled=self.ffilled)


class Engine:
    def __init__(self, cfg: Cfg, date, equity: float, hist: dict, predict, store: Store, costs: dict, blocked: bool = False):
        self.cfg, self.date, self.equity, self.store, self.blocked = cfg, pd.Timestamp(date).normalize(), float(equity), store, blocked
        self.predict, self.halted, self.closed_out = predict, False, False
        w = weights(cfg.symbols)
        self.lines: dict[str, Line] = {}
        for s in cfg.symbols:
            ctx = hist[s].ctx_for(self.date)
            spl = max(1, int(equity * cfg.deploy_frac * w[s] / cfg.vmax / ctx.prevC))
            self.lines[s] = Line(s, ctx, spl, costs[s])
        st = store.load_state(self.date)
        if st and abs(st["equity"] - equity) < 1e-6:                                    # resume after a restart: positions and the 40-lot counters carry over
            self.halted, self.closed_out = st["halted"], st["closed_out"]
            for s, v in st["lines"].items():
                if s in self.lines:
                    for k, x in v.items():
                        setattr(self.lines[s], k, x)

    def _save(self):
        self.store.save_state(self.date, dict(date=str(self.date.date()), equity=self.equity, halted=self.halted, closed_out=self.closed_out, lines={s: ln.state() for s, ln in self.lines.items()}))

    def _trade(self, ln: Line, tgt: int, P: float, j: int, row: dict, why: str = "", force: bool = False):
        step = tgt - ln.pos
        room = abs(step) if force else max(self.cfg.qhigh - ln.traded, 0)                 # the 40-lot cap limits risk-taking, never a flatten
        step = int(np.sign(step) * min(abs(step), room))
        row.update(target=tgt, trade=step, pos_before=ln.pos, pos_after=ln.pos + step, note=why or row.get("note", ""))
        row["action"] = f"{'BUY' if step > 0 else 'SELL'} {abs(step)} lot{'s' if abs(step) > 1 else ''} = {abs(step) * ln.spl} sh {ln.sym} @ {P:,.2f}" if step else ""
        row["position"] = pos_text(ln.pos + step, ln.spl, ln.sym)
        if step:
            ln.pos += step
            ln.traded += abs(step)
            ln.pnl_est -= abs(step) * ln.spl * P * ln.cost_bp / 1e4
            self.store.intent(self.date, dict(ts=row["wall"], j=j, sym=ln.sym, side="BUY" if step > 0 else "SELL", lots=abs(step), qty=abs(step) * ln.spl, ref_price=P, type="MARKETABLE_LIMIT",
                                              mode="DRY_RUN", target=tgt, edge=row.get("edge"), note=why))

    def step(self, j: int, raws: dict, wall: str = "", quotes: dict | None = None) -> list[dict]:
        cfg, rows = self.cfg, []
        flatten = self.halted or self.store.killed()
        for s, ln in self.lines.items():
            if j <= ln.last_j or self.closed_out:
                continue
            row = dict(date=str(self.date.date()), j=j, t=f"{decision_time(self.date, j):%H:%M}", sym=s, wall=wall or f"{now_et():%H:%M:%S}", spl=ln.spl, cost_bp=ln.cost_bp)
            try:
                b, miss = complete_bars(raws.get(s), self.date, j, ln.ctx.last_close)
                P = float(b.close.iloc[-1])
                tail = 0
                while tail < len(miss) and miss[len(miss) - 1 - tail]:
                    tail += 1
                ln.ffilled = int(miss.sum())
                if j > 0 and np.isfinite(ln.px_prev):
                    ln.pnl_est += ln.pos * ln.spl * (P - ln.px_prev)                    # the position held through the bar that just ended
                ln.px_prev = P
                x = features_at(b, j, self.date, ln.ctx)
                er, eg = self.predict(s, x)
                edge = (er + eg) / 2.0
                row.update(price=P, edge_ridge=er, edge_gbm=eg, edge=edge, missing_now=int(miss[-1]), missing_tail=tail)
                if quotes and s in quotes and quotes[s]:
                    row.update(feed_bid=quotes[s][0], feed_ask=quotes[s][1])
                if tail >= cfg.stale_bars:
                    self._trade(ln, ln.pos, P, j, row, "stale_feed_hold")
                elif flatten or self.blocked or j < cfg.first_j:
                    self._trade(ln, 0, P, j, row, "kill" if self.store.killed() else "loss_halt" if self.halted else "dd_stop" if self.blocked else "pre_open_no_risk", force=True)
                else:
                    tgt = int(np.clip(np.sign(edge) * np.floor(abs(edge) / max(cfg.k * ln.cost_bp, 1e-9)), -cfg.vmax, cfg.vmax))
                    self._trade(ln, tgt, P, j, row)
            except Exception as e:                                                      # noqa: BLE001 - one product failing must not stop the others; it holds its position
                row.update(target=ln.pos, trade=0, pos_before=ln.pos, pos_after=ln.pos, note=f"error {type(e).__name__}: {str(e)[:80]}")
            ln.last_j = j
            rows.append(row)
        total = sum(ln.pnl_est for ln in self.lines.values())
        if not self.halted and total <= -cfg.loss_halt * self.equity:
            self.halted = True                                                          # flatten now at this step's prices; stays flat for the day
            for s, ln in self.lines.items():
                r = next((r for r in rows if r["sym"] == s), None)
                if r is not None and ln.pos != 0 and np.isfinite(r.get("price", np.nan)):
                    self._trade(ln, 0, r["price"], j, r, f"loss_halt day P&L {total:.0f}", force=True)
        self.store.append_rows(self.date, rows)
        self._save()
        return rows

    def close_out(self, wall: str = "") -> list[dict]:
        """After the last (15:30) decision: whatever is held goes into a market-on-close order (intent only)."""
        if self.closed_out:
            return []
        for s, ln in self.lines.items():
            if ln.pos:
                self.store.intent(self.date, dict(ts=wall, j=NDEC - 1, sym=s, side="SELL" if ln.pos > 0 else "BUY", lots=abs(ln.pos), qty=abs(ln.pos) * ln.spl, type="MOC", mode="DRY_RUN",
                                                  note="flatten at the official close; entry deadline 15:50 ET"))
        self.closed_out = True
        self._save()
        return []


# ── settlement: re-mark the day on the consolidated tape and the official close ────────────────────────────────────────────────────────────────────────────
def position_paths(rows: pd.DataFrame, sym: str) -> np.ndarray:
    r = rows[rows.sym == sym].drop_duplicates("j", keep="last").set_index("j").pos_after.reindex(range(NDEC)).ffill().fillna(0.0)
    return r.to_numpy(float)


def mark(pos: np.ndarray, P: np.ndarray, C: float, cost_bp: float, spl: int) -> dict:
    """Backtest accounting (ia_audit.net_j) of a position path: P = SIP bar closes at the 72 decisions, C = official close."""
    nxt = np.zeros(NDEC)
    nxt[:-1] = (P[1:] / P[:-1] - 1) * 1e4
    nxt[-1] = (C / P[-1] - 1) * 1e4
    prev = np.concatenate([[0.0], pos[:-1]])
    lots = np.abs(pos - prev).sum()
    gross = float((pos * nxt).sum())
    nxt_px = np.append(P[1:], C)
    usd = float((pos * spl * (nxt_px - P)).sum() - (np.abs(pos - prev) * spl * P * cost_bp / 1e4).sum())
    return dict(gross_bp=gross, net_bp=gross - lots * cost_bp, lots=float(lots), net_usd=usd)


def truth_day(date, syms, source: str, data: Data | None) -> dict:
    """SIP bar closes at the 72 decision points and the official close, per product; None when incomplete."""
    out = {}
    d = pd.Timestamp(date).normalize()
    for s in syms:
        if source == "parquet":
            bars = D.load_bars(s)
            raw = bars[(bars.index >= d) & (bars.index < d + pd.Timedelta(days=1))]
            dc = D.load_daily()
            dc = dc[(dc.symbol == s) & (dc.date == d)]
            C = float(dc.close.iloc[0]) if len(dc) else np.nan
        else:
            raw = data.day_bars(s, d, "sip")
            dd = data.daily(s, d, d)
            C = float(dd[dd.date == d].close.iloc[0]) if len(dd[dd.date == d]) else np.nan
        P = raw.reindex(day_grid(d, NB)).close
        out[s] = None if (P.isna().any() or not np.isfinite(C)) else (P.to_numpy(float)[:NDEC], C, raw)
    return out


def settle_day(date, store: Store, cfg: Cfg, costs: dict, source: str = "api", data: Data | None = None, with_quotes: bool = True) -> dict | None:
    rows = store.read_rows(date)
    if rows.empty:
        log(f"settle {date}: no decisions logged")
        return None
    truth = truth_day(date, cfg.symbols, source, data)
    res = {}
    for s in cfg.symbols:
        if truth[s] is None:
            log(f"settle {s}: consolidated bars or official close incomplete, skipped")
            continue
        P, C, _ = truth[s]
        sub = rows[rows.sym == s]
        spl = int(sub.spl.iloc[0])
        m = mark(position_paths(rows, s), P, C, costs[s], spl)
        dec = sub[(sub.j < NDEC - 1) & sub.edge.notna() & (sub.target != 0)]
        if len(dec):
            y = (C / P[dec.j.to_numpy(int)] - 1) * 1e4
            m["sign_agree"] = float((np.sign(dec.edge.to_numpy(float)) == np.sign(y)).mean())
            m["n_dec"] = int(len(dec))
        tr = sub[sub.trade != 0]
        if with_quotes and data is not None and len(tr):
            real, half, w = 0.0, 0.0, 0.0
            for _, r in tr.iterrows():
                q = data.quote_window(s, decision_time(date, int(r.j)) + pd.Timedelta(seconds=cfg.decision_delay_s + 2))
                if q:
                    ref = P[int(r.j)]
                    slip = (q[1] - ref) / ref * 1e4 if r.trade > 0 else (ref - q[0]) / ref * 1e4        # buy at the offer, sell at the bid, vs the bar close the backtest fills at
                    real += slip * abs(r.trade)
                    half += (q[1] - q[0]) / (q[0] + q[1]) * 1e4 * abs(r.trade)                         # half the quoted spread
                    w += abs(r.trade)
            if w:
                m["realised_cost_bp"], m["half_spread_bp"] = real / w, half / w
        m["model_cost_bp"] = costs[s]
        res[s] = m
    return res


def sip_counterfactual(date, cfg: Cfg, models: dict, costs: dict, equity: float, data: Data | None, source: str) -> dict:
    """What the consolidated-tape (SIP) signals would have done on the same day: the research version of the strategy, replayed through the same engine."""
    return replay_day(date, dataclasses.replace(cfg, feed="sip"), models, costs, equity, data, source, root=os.path.join(LIVE, "replay"), settle_quotes=False)


# ── replay: a past session, bar by bar, through exactly the live code path ────────────────────────────────────────────────────────────────────────────────
def replay_day(date, cfg: Cfg, models: dict, costs: dict, equity: float, data: Data | None, source: str = "api", root: str | None = None, settle_quotes: bool = False,
               predict=None, hist: dict | None = None) -> dict:
    d = pd.Timestamp(date).normalize()
    store = Store(os.path.join(root or os.path.join(LIVE, "replay"), f"{cfg.feed}-{source}"))
    day_dir = store.day_dir(d)
    for f in ("decisions.csv", "intents.jsonl", "state.json"):                           # a replay always starts clean
        if os.path.exists(os.path.join(day_dir, f)):
            os.remove(os.path.join(day_dir, f))
    hist = hist or {s: History(s, cfg.feed, until=d, top_up=(source == "api"), data=data) for s in cfg.symbols}
    raw = {}
    for s in cfg.symbols:
        if source == "parquet":
            fb = D.load_bars(s, cfg.feed)
            raw[s] = fb[(fb.index >= d) & (fb.index < d + pd.Timedelta(days=1))]
        else:
            raw[s] = data.day_bars(s, d, cfg.feed)
    eng = Engine(cfg, d, equity, hist, predict or model_predict(models), store, costs)
    for j in range(cfg.last_j + 1):
        cut = {s: r[r.index < decision_time(d, j)] for s, r in raw.items()}               # only bars that had ended by the decision time
        eng.step(j, cut, wall=f"{decision_time(d, j) + pd.Timedelta(seconds=cfg.decision_delay_s):%H:%M:%S}")
    eng.close_out(wall="15:30:05")
    return dict(store=store, res=settle_day(d, store, cfg, costs, source, data, with_quotes=settle_quotes), engine=eng)


# ── live loop ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=D.NY).tz_localize(None)


def wait_until(ts_et: pd.Timestamp):
    while True:
        dt = (pd.Timestamp(ts_et) - now_et()).total_seconds()
        if dt <= 0:
            return
        time.sleep(min(dt, 30))


def strategy_blocked(cfg: Cfg) -> bool:
    f = os.path.join(LIVE, "daily_summary.csv")
    if not os.path.exists(f):
        return False
    d = pd.read_csv(f)
    tot = d.groupby("date").net_usd.sum().cumsum()
    return bool(len(tot) and (tot.cummax() - tot).max() >= cfg.dd_stop * cfg.equity)


def run_day(cfg: Cfg, date=None):
    tc, data, store = trading_client(), Data(), Store()
    d = pd.Timestamp(date or now_et()).normalize()
    ses = session(tc, d)
    if ses is None:
        clk = tc.get_clock()
        log(f"{d.date()} is not a trading day; next open {clk.next_open}")
        return
    if ses[1].hour != 16:
        log(f"{d.date()} closes early ({ses[1]:%H:%M}); the 78-bar models do not apply, no decisions today")
        return
    if now_et() > ses[1]:
        log("the session is over; use replay/settle")
        return
    try:
        equity = float(tc.get_account().equity)
    except Exception as e:                                                              # noqa: BLE001
        equity = cfg.equity
        log(f"account read failed ({type(e).__name__}), sizing on {equity:,.0f}")
    log(f"DRY RUN on the paper account, equity {equity:,.0f}, feed {cfg.feed}; no order will be sent")
    models = {s: ensure_models(s, d.year) for s in cfg.symbols}
    hist = {s: History(s, cfg.feed, until=d, data=data) for s in cfg.symbols}
    prev = previous_session_date(tc, d)
    for s, h in hist.items():
        if h.ctx_for(d).prior_date != prev:
            raise RuntimeError(f"{s}: history ends {h.ctx_for(d).prior_date.date()} but the previous session was {prev.date()}; refusing to trade on stale prior closes")
    costs = load_costs(cfg)
    eng = Engine(cfg, d, equity, hist, model_predict(models), store, costs, blocked=strategy_blocked(cfg))
    if eng.blocked:
        log("drawdown stop active: no new risk")
    j_now = int((now_et() - bar_start(d, 0)).total_seconds() // (BAR_MIN * 60)) - 1
    for j in range(max(eng.lines[cfg.symbols[0]].last_j + 1, j_now, 0), cfg.last_j + 1):
        wait_until(decision_time(d, j) + pd.Timedelta(seconds=cfg.decision_delay_s))
        raws, quotes = {}, {}
        t_poll = time.time()
        while True:                                                                     # the bar that just ended may take a few seconds to appear
            for s in cfg.symbols:
                try:
                    raws[s] = data.day_bars(s, d, cfg.feed, end=pd.Timestamp.now(tz="UTC"))
                except Exception as e:                                                  # noqa: BLE001
                    log(f"{s} bars: {type(e).__name__} {str(e)[:80]}")
                    raws.setdefault(s, pd.DataFrame())
            if all(len(raws[s]) and raws[s].index[-1] >= bar_start(d, j) for s in cfg.symbols) or time.time() - t_poll > 25:
                break
            time.sleep(2)
        for s in cfg.symbols:
            try:
                quotes[s] = data.last_quote(s, cfg.feed)
            except Exception:                                                           # noqa: BLE001
                quotes[s] = None
        rows = eng.step(j, raws, quotes=quotes)
        log(f"j={j:2d} {decision_time(d, j):%H:%M} " + " | ".join(f"{r['sym']} {r.get('price', float('nan')):8.2f} edge {r.get('edge', float('nan')):+6.1f} pos {r['pos_after']:+d} {r['note']}".rstrip() for r in rows))
    eng.close_out(wall=f"{now_et():%H:%M:%S}")
    late = now_et() > pd.Timestamp(f"{d:%Y-%m-%d} {cfg.moc_deadline}")
    log("MOC intents logged (dry run)" + (" - AFTER the 15:50 deadline, a real MOC would have been rejected" if late else ""))
    wait_until(pd.Timestamp(f"{d:%Y-%m-%d} {cfg.settle_time}"))
    settle_cmd(cfg, d)


def edge_corr(date, live: Store, cf: Store, sym: str) -> float:
    a, b = live.read_rows(date), cf.read_rows(date)
    if a.empty or b.empty:
        return float("nan")
    m = a[(a.sym == sym) & a.edge.notna()][["j", "edge"]].merge(b[(b.sym == sym) & b.edge.notna()][["j", "edge"]], on="j", suffixes=("_live", "_sip"))
    return float(np.corrcoef(m.edge_live, m.edge_sip)[0, 1]) if len(m) > 5 else float("nan")


def append_summary(date, res: dict, cf: dict | None, ec: dict | None = None):
    f = os.path.join(LIVE, "daily_summary.csv")
    rows = []
    for s, m in res.items():
        r = dict(date=f"{pd.Timestamp(date):%Y-%m-%d}", sym=s, **{k: m.get(k) for k in ("net_bp", "gross_bp", "lots", "net_usd", "model_cost_bp", "realised_cost_bp", "half_spread_bp", "sign_agree", "n_dec")})
        if cf and s in cf:
            r["sip_net_bp"], r["sip_net_usd"] = cf[s]["net_bp"], cf[s]["net_usd"]
        if ec:
            r["edge_corr"] = ec.get(s)
        rows.append(r)
    new = pd.DataFrame(rows)
    if os.path.exists(f):
        old = pd.read_csv(f)
        new = pd.concat([old[old.date != new.date.iloc[0]], new], ignore_index=True)
    new.to_csv(f, index=False)


def settle_cmd(cfg: Cfg, date):
    d = pd.Timestamp(date).normalize()
    data, store, costs = Data(), Store(), load_costs(cfg)
    res = settle_day(d, store, cfg, costs, "api", data)
    if not res:
        return
    equity = (store.load_state(d) or {}).get("equity", cfg.equity)
    models = {s: ensure_models(s, d.year) for s in cfg.symbols}
    cf, ec = None, None
    try:
        out = sip_counterfactual(d, cfg, models, costs, equity, data, "api")
        cf = out["res"]
        ec = {s: edge_corr(d, store, out["store"], s) for s in cfg.symbols}
    except Exception as e:                                                              # noqa: BLE001
        log(f"SIP counterfactual failed: {type(e).__name__} {str(e)[:100]}")
    append_summary(d, res, cf, ec)
    for s, m in res.items():
        log(f"settled {s}: net {m['net_bp']:+.1f} bp/lot-day, ${m['net_usd']:+,.0f}, lots {m['lots']:.0f}, realised cost {m.get('realised_cost_bp', float('nan')):.2f} bp vs model {m['model_cost_bp']:.2f}"
            + (f", SIP-signal net {cf[s]['net_bp']:+.1f} bp" if cf and s in cf else ""))


# ── report: the paper-test gate ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def report_cmd(cfg: Cfg):
    f = os.path.join(LIVE, "daily_summary.csv")
    if not os.path.exists(f):
        print("no settled days yet")
        return
    d = pd.read_csv(f)
    day = d.groupby("date").agg(net_usd=("net_usd", "sum"), net_bp=("net_bp", "mean"))
    n = len(day)
    sh = day.net_usd.mean() / day.net_usd.std(ddof=1) * np.sqrt(252) if n > 2 and day.net_usd.std() > 0 else float("nan")
    ratio = (d.realised_cost_bp * d.lots).sum() / (d.model_cost_bp * d.lots).sum() if d.realised_cost_bp.notna().any() else float("nan")
    cum = day.net_usd.cumsum()
    print(f"settled days {n} | net ${day.net_usd.sum():+,.0f} ({day.net_usd.sum() / cfg.equity:+.2%} of {cfg.equity:,.0f}) | shadow Sharpe {sh:.2f} | max drawdown {(cum.cummax() - cum).max() / cfg.equity:.2%}")
    print(f"live-feed edge vs SIP edge correlation {d.edge_corr.mean() if 'edge_corr' in d else float('nan'):.2f} (gate >= 0.90; backtest 0.98-0.99)")
    print(f"realised / model cost per lot {ratio:.2f} (gate: <= 1.5, stop at 2.5) | SIP-signal net ${d.sip_net_usd.sum() if 'sip_net_usd' in d else float('nan'):+,.0f} vs live-feed ${day.net_usd.sum():+,.0f}")
    verdict = "CONTINUE (fewer than 60 settled days)"
    if (n >= 40 and sh < -0.5) or (ratio == ratio and ratio > 2.5):
        verdict = "FAIL / STOP"
    elif n >= 60:
        ec = d.edge_corr.mean() if "edge_corr" in d else float("nan")
        verdict = "PASS" if (sh > 0.5 and ratio == ratio and ratio <= 1.5 and ec == ec and ec >= 0.9) else "NOT PASSED"
    print("gate:", verdict)


# ── commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def status_cmd(cfg: Cfg):
    tc = trading_client()
    clk, acct = tc.get_clock(), tc.get_account()
    print(f"paper account {acct.status}, equity {float(acct.equity):,.2f}, trading_blocked {acct.trading_blocked}")
    print(f"market open {clk.is_open}; next open {clk.next_open}; next close {clk.next_close}")
    data = Data()
    for feed in ("sip", "iex"):
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestQuoteRequest
            data.dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols="TSLA", feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX))
            print(f"real-time {feed.upper()} data: permitted")
        except Exception as e:                                                          # noqa: BLE001
            print(f"real-time {feed.upper()} data: NOT permitted ({str(e)[:70].strip()})")
    year = pd.Timestamp.now().year
    for s in cfg.symbols:
        f = os.path.join(MODEL_DIR, f"{s}_{year}.joblib")
        print(f"{s}: model {year} {'cached' if os.path.exists(f) else 'MISSING (run train)'}; research bars end {D.load_bars(s).index.max():%Y-%m-%d}, {cfg.feed} bars end {D.load_bars(s, cfg.feed).index.max():%Y-%m-%d}")
    print("kill file:", "PRESENT (everything flattens)" if Store().killed() else "none", "| strategy drawdown stop:", strategy_blocked(cfg))


def train_cmd(cfg: Cfg, force: bool):
    year = pd.Timestamp.now().year
    for s in cfg.symbols:
        t = time.time()
        m = ensure_models(s, year, force)
        print(f"{s}: model {year} trained on {m['n_days']} sessions through {m['last_day']} ({time.time() - t:.0f}s)")


def refresh_cmd(cfg: Cfg):
    """Extend the research parquets (SIP bars, IEX bars, official closes) to yesterday - the only command that writes to data/."""
    data, today = Data(), pd.Timestamp.now(tz=D.NY).tz_localize(None).normalize()
    daily = D.load_daily()
    for s in cfg.symbols:
        for feed in ("sip", cfg.feed):
            fb = D.load_bars(s, feed)
            add = D.fetch_bars(s, BAR_MIN, log, feed=feed, start=fb.index.max().normalize() + pd.Timedelta(days=1), end=min(to_utc(today), pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)))
            if len(add):
                pd.concat([fb, add]).to_parquet(D.bars_path(s, feed))
            log(f"{s} {feed}: +{len(add)} bars")
        last = daily[daily.symbol == s].date.max()
        add = data.daily(s, last + pd.Timedelta(days=1), today - pd.Timedelta(days=1))
        if len(add):
            daily = pd.concat([daily, add], ignore_index=True)
        log(f"{s} daily: +{len(add)} rows")
    daily.to_parquet(D.daily_path())


def replay_cmd(cfg: Cfg, date, source: str):
    data = Data() if source == "api" else None
    models = {s: ensure_models(s, pd.Timestamp(date).year) for s in cfg.symbols}
    costs = load_costs(cfg)
    out = replay_day(date, cfg, models, costs, cfg.equity, data, source, settle_quotes=(source == "api"))
    rows = out["store"].read_rows(date)
    print(rows[rows.trade != 0][["t", "sym", "price", "edge", "action", "position", "note"]].to_string(index=False))
    print("\nsettled on the consolidated tape:", {s: {k: round(v, 2) for k, v in m.items()} for s, m in (out["res"] or {}).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "train", "refresh", "run", "replay", "settle", "report"])
    ap.add_argument("--date")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    ap.add_argument("--source", default="api", choices=["api", "parquet"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = Cfg(feed=a.feed)
    if a.cmd == "status":
        status_cmd(cfg)
    elif a.cmd == "train":
        train_cmd(cfg, a.force)
    elif a.cmd == "refresh":
        refresh_cmd(cfg)
    elif a.cmd == "run":
        run_day(cfg, a.date)
    elif a.cmd == "replay":
        replay_cmd(cfg, a.date, a.source)
    elif a.cmd == "settle":
        settle_cmd(cfg, a.date)
    elif a.cmd == "report":
        report_cmd(cfg)


if __name__ == "__main__":
    main()
