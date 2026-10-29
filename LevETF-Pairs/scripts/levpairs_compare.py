"""
Live paper trading against the backtest: what the account really did, and what the research code says it should have done on the same session.  No Qt, read-only (it never sends an order).

  python levpairs_compare.py --date 2026-10-27        one session (after the close; SIP history is available 16 minutes late)
  python levpairs_compare.py --all                     every session found in the order journal
  python levpairs_compare.py --month                   rebuild reports/live_compare/MONTH_TO_DATE.md from the daily summaries
  options: --mode paper|live|dry   --no-tape (skip the SIP quotes: no NBBO slippage)   --update-data (top up the cached 1-minute SIP and IEX bars first)

For each session it builds
  * the live round trips from the order journal (reports/levpairs_fills_<mode>.jsonl) and, when it has been synced, the broker's own record (levpairs_broker_orders_<mode>_<date>.csv): fixed-share P&L from the
    real fill prices, which product was bought and which sold, every timestamp;
  * two backtest twins of the same session with the SAME configuration (reports/levpairs_run_<mode>_<date>.json): one on the final SIP bars (what the research says) and one on the final IEX bars (what the
    live engine was fed), and matches the live trips to them;
  * the fills against four references: the signal bar's close, the twin's next-open price, the SIP NBBO at the moment the order reached the broker (mid and touch) and the first SIP trade after it;
  * the decisions: the live s-score and position path against both twins, bar by bar;
  * the latency stack: bar end -> decision -> order submit -> broker accepted -> filled.
Outputs go to reports/live_compare/: <date>_trips.csv, <date>_legs.csv, <date>_decisions.csv, REPORT_<date>.md, summary.csv (one row per session) and MONTH_TO_DATE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields, replace
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levpairs_core as C
import levpairs_data as D
import levpairs_sim as S

NY = D.NY
ORDER_ACTIONS = ("ENTER", "EXIT", "UNWIND")                                                # every other journal action (KILL, end of day, session end, watchdog, ORPHAN ...) is a flatten's close order


# ── reading what the live engine wrote ─────────────────────────────────────────────────────────────────────────────────────
def read_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except ValueError:                                                       # a torn last line after a crash
                    continue
    return pd.DataFrame(rows)


def ts(x) -> pd.Timestamp:
    """Any timestamp text the broker or the engine wrote -> tz-aware New York time (naive text is read as UTC); NaT if there is none."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x) in ("", "None", "nan", "NaT"):
        return pd.NaT
    t = pd.Timestamp(x)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(NY)


def tsv(values) -> pd.Series:
    """A column of timestamps (any text the engine / broker wrote, or NaT) as ONE tz-aware dtype, so that columns can be subtracted even when one of them is empty."""
    return pd.to_datetime(pd.Series([ts(x) for x in values], dtype="object"), utc=True).dt.tz_convert(NY)


def cfg_from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


class DayInputs:
    def __init__(self, date: str, mode: str, journal: pd.DataFrame, decisions: pd.DataFrame, broker: pd.DataFrame, run: dict):
        self.date, self.mode, self.journal, self.decisions, self.broker, self.run = date, mode, journal, decisions, broker, run


def load_day(date, mode: str = "paper", reports: Optional[str] = None) -> DayInputs:
    reports = reports or D.REPORTS
    d = str(pd.Timestamp(date).date())
    key = pd.Timestamp(d).strftime("%Y%m%d")

    def day(df):
        return df[df.date.astype(str) == d].reset_index(drop=True) if len(df) and "date" in df else df
    journal = day(read_jsonl(os.path.join(reports, f"levpairs_fills_{mode}.jsonl")))
    decisions = day(read_jsonl(os.path.join(reports, f"levpairs_decisions_{mode}.jsonl")))
    fb = os.path.join(reports, f"levpairs_broker_orders_{mode}_{key}.csv")
    broker = pd.read_csv(fb) if os.path.exists(fb) and os.path.getsize(fb) > 2 else pd.DataFrame()
    fr = os.path.join(reports, f"levpairs_run_{mode}_{key}.json")
    run = json.load(open(fr, encoding="utf-8")) if os.path.exists(fr) else {}
    return DayInputs(d, mode, journal, decisions, broker, run)


def configs(day: DayInputs):
    """(SigCfg, SimCfg) the session ran with; the default preset if there is no run record."""
    if day.run.get("sc") and day.run.get("c"):
        return cfg_from_dict(C.SigCfg, day.run["sc"]), cfg_from_dict(S.SimCfg, day.run["c"])
    from levpairs_ui.presets import DEFAULT_PRESET, cfgs
    return cfgs(DEFAULT_PRESET)


# ── the orders of the day, one row per order, the broker's record laid over the journal ──────────────────────────────────
def legs_table(day: DayInputs) -> pd.DataFrame:
    j = day.journal.copy()
    if not len(j):
        return pd.DataFrame()
    for col in ("trip", "action", "px", "qty", "ref", "t_submit_ns", "t_ack_ns", "t_seen_ns", "client_order_id", "decided_at", "submitted_at", "filled_at", "status"):
        if col not in j:
            j[col] = np.nan
    b = day.broker.set_index("client_order_id") if len(day.broker) and "client_order_id" in day.broker else pd.DataFrame()
    j["broker_status"] = j.client_order_id.map(b.status) if len(b) else np.nan
    j["fill_px"] = pd.to_numeric(j.client_order_id.map(b.filled_avg_price), errors="coerce") if len(b) else np.nan
    j["fill_px"] = j.fill_px.fillna(pd.to_numeric(j.px, errors="coerce"))
    j["fill_qty"] = pd.to_numeric(j.client_order_id.map(b.filled_qty), errors="coerce") if len(b) else np.nan
    j["fill_qty"] = j.fill_qty.fillna(pd.to_numeric(j.qty, errors="coerce"))
    j["submitted_ts"] = tsv(j.client_order_id.map(b.submitted_at) if len(b) else j.submitted_at)
    j["filled_ts"] = tsv(j.client_order_id.map(b.filled_at) if len(b) else j.filled_at)
    j["local_submit_ts"] = pd.to_datetime(pd.to_numeric(j.t_submit_ns, errors="coerce"), unit="ns", utc=True).dt.tz_convert(NY)
    j["bar_end"] = pd.to_datetime(pd.Series([f"{day.date} {t}" if isinstance(t, str) and len(t) <= 5 else None for t in j.time]), errors="coerce").dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    j["decided_ts"] = tsv(j.decided_at)
    sec = lambda a, b_: (a - b_).dt.total_seconds()
    j["lat_bar_to_decision_s"] = sec(j.decided_ts, j.bar_end)
    j["lat_decision_to_submit_s"] = sec(j.local_submit_ts, j.decided_ts)
    j["lat_submit_to_fill_s"] = sec(j.filled_ts, j.submitted_ts.fillna(j.local_submit_ts))
    j["lat_local_to_broker_ms"] = sec(j.submitted_ts, j.local_submit_ts) * 1e3
    j["is_buy"] = j.side.astype(str).str.upper().str.startswith("B")
    return j


def _sgn(is_buy) -> int:
    return 1 if is_buy else -1


def _agg_fills(g: pd.DataFrame) -> dict:
    """symbol -> quantity, quantity-weighted price, last fill time, bar label, side of the FILLED orders of one side of a trip (an exit split over several orders counts all of them)."""
    out = {}
    g = g[g.fill_px.notna() & (g.fill_qty > 0)]
    for sym, x in g.groupby("symbol", sort=False):
        out[sym] = dict(qty=float(x.fill_qty.sum()), px=float(np.average(x.fill_px, weights=x.fill_qty)), ts=x.filled_ts.max() if x.filled_ts.notna().any() else x.local_submit_ts.max(), time=x.time.iloc[0],
                        is_buy=bool(x.is_buy.iloc[0]), s=x.s.iloc[0] if "s" in x else np.nan, bars=set(x.time.astype(str)))
    return out


def live_trips(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per live round trip with its real fixed-share P&L (from the quantities actually exited). A trip whose entry was undone (a failed leg or a KILL in flight) is listed with kind 'unwound'; one that
    ended in a flatten (KILL / end of day / session end / watchdog / orphan close) takes its exit from those orders (needs the broker's record for the fill price)."""
    if not len(legs):
        return pd.DataFrame()
    rows = []
    closes = legs[legs.trip.isna() & ~legs.action.isin(ORDER_ACTIONS)]
    used_closes: set = set()                                                             # a flatten's close order belongs to ONE trip
    starts: dict = {}                                                                    # when each pair's entries were sent: a close belongs to the trip it follows, not to an earlier one
    for trip, g in legs[legs.trip.notna() & (legs.action == "ENTER")].groupby("trip", sort=False):
        t0 = g.local_submit_ts.min()
        if pd.notna(t0):
            starts.setdefault(g.pair.iloc[0], []).append(t0)
    far = pd.Timestamp("2262-01-01", tz=NY)
    for trip, g in legs[legs.trip.notna()].groupby("trip", sort=False):
        pair = g.pair.iloc[0]
        L, B = pair.split("/")
        e = _agg_fills(g[(g.action == "ENTER") & (g.status == "filled")])
        if set(e) != {L, B}:
            rows.append(dict(trip=trip, pair=pair, kind="entry failed", complete=False))
            continue
        side = 1 if e[L]["is_buy"] else -1
        qL, qB, Le, Be = e[L]["qty"], e[B]["qty"], e[L]["px"], e[B]["px"]
        exits = g[g.action.isin(["EXIT", "UNWIND"])]
        kind = "unwound" if (exits.action == "UNWIND").any() else "exit"
        x = _agg_fills(exits)
        if not (L in x and B in x):                                                      # no priced exit of its own: the close orders of a flatten after the entry
            t_in = max(e[L]["ts"], e[B]["ts"]) if pd.notna(e[L]["ts"]) and pd.notna(e[B]["ts"]) else pd.NaT
            nxt = min([t for t in starts.get(pair, []) if pd.notna(t_in) and t > t_in], default=far)
            later = closes[(closes.pair.isin([pair, "-"])) & closes.symbol.isin([L, B]) & (closes.local_submit_ts > t_in) & (closes.local_submit_ts < nxt)
                           & ~closes.client_order_id.isin(used_closes)] if pd.notna(t_in) else closes.iloc[0:0]
            x2 = _agg_fills(later) if len(later) else {}
            if L in x2 and B in x2:
                x, kind = x2, str(later.action.iloc[0])
                used_closes |= set(later.client_order_id)
        have = L in x and B in x
        qLx, qBx = (x[L]["qty"], x[B]["qty"]) if have else (np.nan, np.nan)
        Lx, Bx = (x[L]["px"], x[B]["px"]) if have else (np.nan, np.nan)
        pnl = side * (qLx * (Lx - Le) - qBx * (Bx - Be)) if have else np.nan              # long the spread = long the 3x ETF and short the 1x ETF
        rows.append(dict(trip=trip, pair=pair, kind=kind, complete=bool(have), side=side, position="LONG %s / SHORT %s" % ((L, B) if side > 0 else (B, L)), entry_bar=e[L]["time"],
                         exit_bar=(x[L]["time"] if have else None), exit_bars=(len(x[L]["bars"] | x[B]["bars"]) if have else 0), qty_L=qL, qty_B=qB, qty_L_exit=qLx, qty_B_exit=qBx, qty_mismatch=bool(have and (abs(qLx - qL) > 1e-9 or abs(qBx - qB) > 1e-9)),
                         L_entry=Le, B_entry=Be, L_exit=Lx, B_exit=Bx, entry_ts=max(e[L]["ts"], e[B]["ts"]) if pd.notna(e[L]["ts"]) and pd.notna(e[B]["ts"]) else pd.NaT,
                         exit_ts=(max(x[L]["ts"], x[B]["ts"]) if have and pd.notna(x[L]["ts"]) and pd.notna(x[B]["ts"]) else pd.NaT), s_entry=e[L]["s"],
                         hedge_ratio=qB * Be / (qL * Le) if qL * Le else np.nan, pnl_usd=pnl, pnl_bp_L=pnl / (qL * Le) * 1e4 if have and qL * Le else np.nan, notional_L=qL * Le))
    return pd.DataFrame(rows)


# ── the backtest twins ────────────────────────────────────────────────────────────────────────────────────────────────────────
def twin(date, sc: C.SigCfg, c: S.SimCfg, feed: str, history_days: int = 80) -> Optional[dict]:
    """The research backtest of ONE session on the final bars of ``feed`` with the session's own configuration. None when the cached bars do not reach that day."""
    d = pd.Timestamp(date).normalize()
    panel = {D.pair_name(p): D.load_pairdata(p, sc.tf, feed, start=d - pd.Timedelta(days=history_days), end=d, max_missing=1.0) for p in D.PAIRS}
    if any(not len(v.dates) or v.dates[-1] != d for v in panel.values()):
        return None
    sigs = {k: C.build_signals(v, sc) for k, v in panel.items()}
    res = S.simulate(sigs, replace(c, start=str(d.date()), end=str(d.date()), vol_on=False, g_init=1.0))
    return dict(panel=panel, sigs=sigs, res=res, trades=res.trades.reset_index(drop=True), feed=feed)


def match_trips(live: pd.DataFrame, tw: Optional[dict], tf: int, date: str, tol_bars: int = 1) -> pd.DataFrame:
    """Live round trips against the twin's trades of the same pair and side whose entry bar is within ``tol_bars``; each twin trade is used once. Adds live-only and twin-only rows.
    Every matched, complete trip gets its fills priced the backtest's way (``live_price_bp``, ``d_price_bp_all``); the slippage columns and ``d_price_bp`` are filled ONLY for COMPARABLE trips - the same entry bar AND
    the same exit bar as the twin, exited in one go as many shares as entered - because a trip that left a bar earlier or later differs by timing, not by fill quality."""
    out, used = [], set()
    tt = tw["trades"] if tw is not None else pd.DataFrame()
    for r in live.to_dict("records") if len(live) else []:
        best, bd = None, None
        if len(tt) and "side" in r and pd.notna(r.get("side")):
            sd = "long" if r["side"] > 0 else "short"
            t_live = pd.Timestamp(f"{date} {r['entry_bar']}")
            for i, t in tt.iterrows():
                if i in used or t.pair != r["pair"] or t.side != sd:
                    continue
                dist = abs((t.entry_time - t_live).total_seconds()) / 60.0
                if dist <= tol_bars * tf and (bd is None or dist < bd):
                    best, bd = i, dist
        row = dict(r)
        if best is None:
            row["match"] = "live only"
        else:
            used.add(best)
            t = tt.loc[best]
            row.update(match="matched", twin_entry_time=t.entry_time, twin_exit_time=t.exit_time, twin_L_entry=t.L_entry, twin_B_entry=t.B_entry, twin_L_exit=t.L_exit, twin_B_exit=t.B_exit, twin_beta=t.beta,
                       twin_gross_bp=t.gross_bp, twin_net_bp=t.net_bp, twin_reason=t.reason, entry_bar_diff_min=bd)
            exit_bar = r.get("exit_bar")
            same_exit = isinstance(exit_bar, str) and len(exit_bar) == 5 and pd.Timestamp(f"{date} {exit_bar}") == t.exit_time and r.get("exit_bars", 1) == 1       # one exit bar: a trip that left over several bars is timing, not fill quality
            row["comparable"] = bool(r.get("complete")) and bd == 0 and same_exit and not r.get("qty_mismatch", False)
            if r.get("complete"):
                sgn = r["side"]
                row["live_price_bp"] = sgn * ((r["L_exit"] / r["L_entry"] - 1.0) - t.beta * (r["B_exit"] / r["B_entry"] - 1.0)) * 1e4              # the live fills, priced the way the table of the backtest is
                row["d_price_bp_all"] = row["live_price_bp"] - t.gross_bp
                row["twin_gross_usd_same_size"] = t.gross_bp / 1e4 * r["notional_L"]
                row["twin_net_usd_same_size"] = t.net_bp / 1e4 * r["notional_L"]
                if row["comparable"]:
                    bad = lambda a, b_, buy: _sgn(buy) * (a - b_) / b_ * 1e4                        # positive = worse than the twin's price
                    row["d_price_bp"] = row["d_price_bp_all"]
                    row["slip_entry_L_bp"] = bad(r["L_entry"], t.L_entry, sgn > 0)
                    row["slip_entry_B_bp"] = bad(r["B_entry"], t.B_entry, sgn < 0)
                    row["slip_exit_L_bp"] = bad(r["L_exit"], t.L_exit, sgn < 0)
                    row["slip_exit_B_bp"] = bad(r["B_exit"], t.B_exit, sgn > 0)
        out.append(row)
    for i, t in (tt.iterrows() if len(tt) else []):
        if i not in used:
            out.append(dict(trip=None, pair=t.pair, match="twin only", twin_entry_time=t.entry_time, twin_exit_time=t.exit_time, twin_gross_bp=t.gross_bp, twin_net_bp=t.net_bp, twin_reason=t.reason,
                            position=t.position))
    return pd.DataFrame(out)


def decision_compare(dec: pd.DataFrame, twins: dict) -> pd.DataFrame:
    """The live s-score and position after every bar against the twins' (final bars), row by row."""
    if not len(dec) or "j" not in dec:
        return pd.DataFrame()
    rows = []
    for r in dec.itertuples():
        if not isinstance(r.pair, str) or pd.isna(r.j):
            continue
        j = int(r.j)
        row = dict(pair=r.pair, j=j, time=r.time, s_live=getattr(r, "s", np.nan), pos_live=getattr(r, "pos_after", np.nan), wall=getattr(r, "wall", None))
        for feed, tw in twins.items():
            sg = None if tw is None else tw["sigs"].get(r.pair)
            if sg is None:
                continue
            row[f"s_{feed}"] = float(sg.s[-1, j])
            row[f"pos_{feed}"] = int(tw["res"].per_pair[r.pair].dec[-1, j])
        rows.append(row)
    return pd.DataFrame(rows)


def decision_stats(dc: pd.DataFrame) -> dict:
    out = {}
    for feed in ("sip", "iex"):
        if f"s_{feed}" not in dc:
            continue
        x = dc[["s_live", f"s_{feed}"]].astype(float).dropna()
        out[f"s_corr_{feed}"] = float(np.corrcoef(x.s_live, x[f"s_{feed}"])[0, 1]) if len(x) > 3 and x.s_live.std() > 0 and x[f"s_{feed}"].std() > 0 else np.nan
        out[f"s_mad_{feed}"] = float((x.s_live - x[f"s_{feed}"]).abs().mean()) if len(x) else np.nan
        p = dc[["pos_live", f"pos_{feed}"]].astype(float).dropna()
        out[f"pos_agree_{feed}"] = float((p.pos_live == p[f"pos_{feed}"]).mean()) if len(p) else np.nan
    return out


# ── the tape: where the market was when an order arrived ──────────────────────────────────────────────────────────────────
class Tape:
    def quote_at(self, sym: str, t: pd.Timestamp) -> Optional[dict]:
        raise NotImplementedError

    def trade_after(self, sym: str, t: pd.Timestamp) -> Optional[dict]:
        raise NotImplementedError


class AlpacaTape(Tape):
    """SIP quotes and trades from Alpaca's historical API (read-only; the account may query SIP history older than 16 minutes)."""

    def __init__(self, client=None):
        self.cl = client or D.data_client()
        self._q, self._t = {}, {}

    def quote_at(self, sym, t):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest
        key = (sym, t.floor("s"))
        if key not in self._q:
            q = self.cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=(t - pd.Timedelta(seconds=30)).tz_convert("UTC").to_pydatetime(), end=(t + pd.Timedelta(seconds=1)).tz_convert("UTC").to_pydatetime(),
                                                            feed=DataFeed.SIP)).data.get(sym, [])
            self._q[key] = [(pd.Timestamp(x.timestamp).tz_convert(NY), x.bid_price, x.ask_price) for x in q if x.bid_price > 0 and x.ask_price > 0]
        ok = [x for x in self._q[key] if x[0] <= t]
        return dict(t=ok[-1][0], bid=ok[-1][1], ask=ok[-1][2]) if ok else None

    def trade_after(self, sym, t):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockTradesRequest
        key = (sym, t.floor("s"))
        if key not in self._t:
            tr = self.cl.get_stock_trades(StockTradesRequest(symbol_or_symbols=sym, start=t.tz_convert("UTC").to_pydatetime(), end=(t + pd.Timedelta(seconds=15)).tz_convert("UTC").to_pydatetime(), feed=DataFeed.SIP)).data.get(sym, [])
            self._t[key] = [(pd.Timestamp(x.timestamp).tz_convert(NY), x.price) for x in tr]
        ok = [x for x in self._t[key] if x[0] >= t]
        return dict(t=ok[0][0], price=ok[0][1]) if ok else None


def fills_vs_tape(legs: pd.DataFrame, tape: Tape) -> pd.DataFrame:
    """For every filled entry / exit leg: the NBBO at the moment the broker accepted the order, and the fill's slippage against its mid, its touch (ask for a buy, bid for a sell) and the first trade after it.
    Positive = worse than the reference."""
    out = legs.copy()
    for col in ("nbbo_bid", "nbbo_ask", "nbbo_mid", "spread_bp", "slip_vs_mid_bp", "slip_vs_touch_bp", "next_trade_px", "slip_vs_next_trade_bp"):
        out[col] = np.nan
    for i, r in out.iterrows():
        if r.action not in ("ENTER", "EXIT") or r.status != "filled" or pd.isna(r.fill_px):
            continue
        t = r.submitted_ts if pd.notna(r.submitted_ts) else r.local_submit_ts
        if pd.isna(t):
            continue
        q = tape.quote_at(r.symbol, t)
        if q is None:
            continue
        mid = (q["bid"] + q["ask"]) / 2
        sg = _sgn(r.is_buy)
        out.at[i, "nbbo_bid"], out.at[i, "nbbo_ask"], out.at[i, "nbbo_mid"] = q["bid"], q["ask"], mid
        out.at[i, "spread_bp"] = (q["ask"] - q["bid"]) / mid * 1e4
        out.at[i, "slip_vs_mid_bp"] = sg * (r.fill_px - mid) / mid * 1e4
        touch = q["ask"] if r.is_buy else q["bid"]
        out.at[i, "slip_vs_touch_bp"] = sg * (r.fill_px - touch) / touch * 1e4
        nt = tape.trade_after(r.symbol, t)
        if nt is not None:
            out.at[i, "next_trade_px"] = nt["price"]
            out.at[i, "slip_vs_next_trade_bp"] = sg * (r.fill_px - nt["price"]) / nt["price"] * 1e4
    return out


# ── one session ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_day(date, mode: str = "paper", tape: Optional[Tape] = None, feeds=("sip", "iex"), write: bool = True, reports: Optional[str] = None, log=print) -> dict:
    reports = reports or D.REPORTS
    day = load_day(date, mode, reports)
    sc, c = configs(day)
    legs = legs_table(day)
    trips = live_trips(legs) if len(legs) else pd.DataFrame()
    twins = {f: twin(day.date, sc, c, f) for f in feeds}
    for f, tw in twins.items():
        if tw is None:
            log(f"no cached {f.upper()} bars for {day.date}: top them up first (levpairs_compare.py --update-data) - that twin is skipped")
    sip = twins.get("sip")
    matched = match_trips(trips, sip, sc.tf, day.date)
    if len(legs) and len(matched) and "trip" in matched:                                  # the twin's price of every leg
        px = {}
        for r in matched[(matched.match == "matched") & matched.get("comparable", pd.Series(False, index=matched.index)).fillna(False).astype(bool)].itertuples():          # the slippage of a leg is judged only on comparable trips
            L, B = r.pair.split("/")
            px[(r.trip, "ENTER", L)], px[(r.trip, "ENTER", B)] = r.twin_L_entry, r.twin_B_entry
            px[(r.trip, "EXIT", L)], px[(r.trip, "EXIT", B)] = r.twin_L_exit, r.twin_B_exit
        legs["twin_px"] = [px.get((r.trip, r.action, r.symbol), np.nan) for r in legs.itertuples()]
        legs["slip_vs_twin_bp"] = [(_sgn(r.is_buy) * (r.fill_px - r.twin_px) / r.twin_px * 1e4) if pd.notna(r.twin_px) and pd.notna(r.fill_px) else np.nan for r in legs.itertuples()]
        legs["slip_vs_signal_bp"] = [(_sgn(r.is_buy) * (r.fill_px - r.ref) / r.ref * 1e4) if pd.notna(r.ref) and pd.notna(r.fill_px) else np.nan for r in legs.itertuples()]
    if tape is not None and len(legs):
        legs = fills_vs_tape(legs, tape)
    dc = decision_compare(day.decisions, twins)
    st = decision_stats(dc) if len(dc) else {}
    m = matched[matched.match == "matched"] if len(matched) and "match" in matched else pd.DataFrame()
    mc = m[m.complete.fillna(False).astype(bool)] if len(m) and "complete" in m else pd.DataFrame()
    done = trips[trips.complete.fillna(False)] if len(trips) and "complete" in trips else pd.DataFrame()
    traded = legs[legs.action.isin(["ENTER", "EXIT"]) & (legs.status == "filled")] if len(legs) else pd.DataFrame()
    mean = lambda df, col: float(df[col].mean()) if len(df) and col in df and df[col].notna().any() else np.nan
    summary = dict(date=day.date, mode=mode, feed=(day.run.get("starts") or [{}])[0].get("feed", day.run.get("live", {}).get("feed")), live_trips=len(trips), live_complete=len(done),
                   twin_sip_trades=len(sip["trades"]) if sip else np.nan, twin_iex_trades=len(twins["iex"]["trades"]) if twins.get("iex") else np.nan, matched=len(m),
                   live_only=int((matched.match == "live only").sum()) if len(matched) and "match" in matched else 0, twin_only=int((matched.match == "twin only").sum()) if len(matched) and "match" in matched else 0,
                   live_pnl_usd=float(done.pnl_usd.sum()) if len(done) else 0.0, live_pnl_bp_L=mean(done, "pnl_bp_L"), live_price_bp_matched=mean(mc, "live_price_bp"), twin_gross_bp_matched=mean(mc, "twin_gross_bp"),
                   twin_net_bp_matched=mean(mc, "twin_net_bp"), comparable=int(m.comparable.fillna(False).astype(bool).sum()) if len(m) and "comparable" in m else 0, d_price_bp=mean(m, "d_price_bp"),
                   d_price_bp_all=mean(mc, "d_price_bp_all"), twin_net_usd_same_size=float(mc.twin_net_usd_same_size.sum()) if len(mc) and "twin_net_usd_same_size" in mc else np.nan,
                   slip_entry_bp=float(np.nanmean(np.r_[m.slip_entry_L_bp.values, m.slip_entry_B_bp.values])) if len(m) and "slip_entry_L_bp" in m else np.nan,
                   slip_exit_bp=float(np.nanmean(np.r_[m.slip_exit_L_bp.values, m.slip_exit_B_bp.values])) if len(m) and "slip_exit_L_bp" in m else np.nan,
                   slip_vs_signal_bp=mean(traded, "slip_vs_signal_bp"), slip_vs_mid_bp=mean(traded, "slip_vs_mid_bp"), slip_vs_touch_bp=mean(traded, "slip_vs_touch_bp"), slip_vs_next_trade_bp=mean(traded, "slip_vs_next_trade_bp"),
                   spread_bp=mean(traded, "spread_bp"), lat_bar_to_decision_s=mean(traded, "lat_bar_to_decision_s"), lat_decision_to_submit_s=mean(traded, "lat_decision_to_submit_s"),
                   lat_submit_to_fill_s=mean(traded, "lat_submit_to_fill_s"), lat_local_to_broker_ms=mean(traded, "lat_local_to_broker_ms"), **st)
    out = dict(summary=summary, trips=matched, legs=legs, decisions=dc, day=day)
    if write:
        write_day(out, reports)
    return out


def write_day(out: dict, reports: str) -> None:
    """The files of one session. The summary and the month-to-date report are kept PER MODE (paper / dry / live): rehearsal days in DRY RUN never mix into the paper month; non-paper files carry the mode in their name."""
    d, mode = out["summary"]["date"], out["summary"]["mode"]
    tag = "" if mode == "paper" else f"_{mode}"
    folder = os.path.join(reports, "live_compare")
    os.makedirs(folder, exist_ok=True)
    for k in ("trips", "legs", "decisions"):
        if len(out[k]):
            out[k].to_csv(os.path.join(folder, f"{d}_{k}{tag}.csv"), index=False)
    f = os.path.join(folder, "summary.csv")
    old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    if len(old):
        if "mode" not in old:
            old["mode"] = "paper"
        old = old[~((old.date.astype(str) == d) & (old["mode"] == mode))]
    allm = pd.concat([old, pd.DataFrame([out["summary"]])], ignore_index=True).sort_values(["mode", "date"])
    allm.to_csv(f, index=False)
    open(os.path.join(folder, f"REPORT_{d}{tag}.md"), "w", encoding="utf-8").write(day_report(out))
    open(os.path.join(folder, f"MONTH_TO_DATE{tag}.md"), "w", encoding="utf-8").write(month_report(allm[allm["mode"] == mode].reset_index(drop=True), mode))


def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:,.{nd}f}" if isinstance(x, (int, float, np.floating, np.integer)) else str(x))


def day_report(out: dict) -> str:
    s, trips, legs = out["summary"], out["trips"], out["legs"]
    L = [f"# Live {s['mode']} session {s['date']} against the backtest", ""]
    L.append(f"Feed the engine ran on: {s['feed']}. Live round trips {s['live_trips']} ({s['live_complete']} complete), SIP twin {_fmt(s['twin_sip_trades'], 0)}, IEX twin {_fmt(s['twin_iex_trades'], 0)}; "
             f"matched {s['matched']}, live only {s['live_only']}, twin only {s['twin_only']}.")
    L += ["", "## P&L", f"- live realised P&L ${_fmt(s['live_pnl_usd'])}; per trip {_fmt(s['live_pnl_bp_L'])} bp of the 3x leg's notional",
          f"- the SIP twin's same trips: gross {_fmt(s['twin_gross_bp_matched'])} bp, net of modelled costs {_fmt(s['twin_net_bp_matched'])} bp; live fills priced the same way {_fmt(s['live_price_bp_matched'])} bp (difference {_fmt(s['d_price_bp_all'])} bp; "
          f"{_fmt(s['comparable'], 0)} of {s['matched']} matched trips are comparable - same entry AND exit bar - and their difference, the fill quality, is {_fmt(s['d_price_bp'])} bp)",
          f"- the twin's net P&L at the live size ${_fmt(s['twin_net_usd_same_size'])}", "",
          "## Fills (bp, positive = worse than the reference)", f"- vs the signal bar's close {_fmt(s['slip_vs_signal_bp'])}; vs the twin's next-open price (comparable trips): entry {_fmt(s['slip_entry_bp'])}, exit {_fmt(s['slip_exit_bp'])}",
          f"- vs the SIP NBBO mid when the order reached the broker {_fmt(s['slip_vs_mid_bp'])}; vs the touch {_fmt(s['slip_vs_touch_bp'])}; vs the first SIP trade after it {_fmt(s['slip_vs_next_trade_bp'])}; quoted spread {_fmt(s['spread_bp'])}",
          "", "## Latency (seconds)", f"- bar end -> decision {_fmt(s['lat_bar_to_decision_s'])}; decision -> order sent {_fmt(s['lat_decision_to_submit_s'], 3)}; sent -> filled {_fmt(s['lat_submit_to_fill_s'], 3)}; local clock -> broker accept {_fmt(s['lat_local_to_broker_ms'], 0)} ms",
          "", "## Decisions", f"- s-score correlation with the SIP twin {_fmt(s.get('s_corr_sip'))} (mean |diff| {_fmt(s.get('s_mad_sip'))}), with the IEX twin {_fmt(s.get('s_corr_iex'))} ({_fmt(s.get('s_mad_iex'))}); "
          f"position path agrees with the SIP twin on {_fmt(100 * s.get('pos_agree_sip', np.nan), 1)}% of the bars, with the IEX twin on {_fmt(100 * s.get('pos_agree_iex', np.nan), 1)}%"]
    if len(trips):
        cols = [c for c in ("trip", "match", "comparable", "position", "entry_bar", "exit_bar", "kind", "qty_L", "qty_B", "L_entry", "B_entry", "L_exit", "B_exit", "pnl_usd", "pnl_bp_L", "live_price_bp", "twin_gross_bp", "twin_net_bp", "d_price_bp",
                                 "slip_entry_L_bp", "slip_entry_B_bp", "slip_exit_L_bp", "slip_exit_B_bp") if c in trips]
        L += ["", "## Round trips", "```", trips[cols].round(2).to_string(index=False), "```"]
    return "\n".join(L) + "\n"


def month_report(sm: pd.DataFrame, mode: str = "paper") -> str:
    if not len(sm):
        return "# Month to date\n\nno sessions yet\n"
    for col in ("live_trips", "matched", "live_only", "twin_only", "live_pnl_usd", "twin_net_usd_same_size", "live_complete", "live_price_bp_matched", "twin_gross_bp_matched", "twin_net_bp_matched", "d_price_bp",
                "slip_vs_signal_bp", "slip_entry_bp", "slip_exit_bp", "d_price_bp_all", "comparable", "slip_vs_mid_bp", "slip_vs_touch_bp", "s_corr_sip", "s_corr_iex", "pos_agree_sip", "pos_agree_iex", "lat_bar_to_decision_s",
                "lat_decision_to_submit_s", "lat_submit_to_fill_s", "feed"):
        if col not in sm:
            sm[col] = np.nan                                                             # a session without a SIP / IEX twin or without a tape has no such column
    w = lambda col: float(np.average(sm[col].dropna(), weights=sm.loc[sm[col].notna(), "live_complete"].fillna(0).clip(lower=0) + 1e-9)) if sm[col].notna().any() else np.nan
    L = [f"# Live {mode} test, month to date ({sm.date.min()} .. {sm.date.max()}, {len(sm)} sessions)", "",
         f"- round trips: live {int(sm.live_trips.sum())}, matched with the SIP twin {int(sm.matched.sum())}, live only {int(sm.live_only.sum())}, twin only {int(sm.twin_only.sum())}",
         f"- live realised P&L ${_fmt(sm.live_pnl_usd.sum())}; the SIP twin's net P&L at the same size ${_fmt(sm.twin_net_usd_same_size.sum())} (matched trips only)",
         f"- per matched trip: live {_fmt(w('live_price_bp_matched'))} bp vs twin gross {_fmt(w('twin_gross_bp_matched'))} bp (difference {_fmt(w('d_price_bp_all'))} bp, of which fill quality on the comparable trips - same entry and exit bar - {_fmt(w('d_price_bp'))} bp), twin net {_fmt(w('twin_net_bp_matched'))} bp",
         f"- fill slippage (bp, positive = worse): vs signal close {_fmt(w('slip_vs_signal_bp'))}, vs twin entry {_fmt(w('slip_entry_bp'))} / exit {_fmt(w('slip_exit_bp'))} (comparable trips only), vs NBBO mid {_fmt(w('slip_vs_mid_bp'))}, vs touch {_fmt(w('slip_vs_touch_bp'))}",
         f"- decisions: s correlation with the SIP twin {_fmt(sm.s_corr_sip.mean())}, with the IEX twin {_fmt(sm.s_corr_iex.mean())}; positions agree with the SIP twin {_fmt(100 * sm.pos_agree_sip.mean(), 1)}%, IEX twin {_fmt(100 * sm.pos_agree_iex.mean(), 1)}%",
         f"- latency: decision {_fmt(sm.lat_bar_to_decision_s.mean())} s after the bar end, order {_fmt(sm.lat_decision_to_submit_s.mean(), 3)} s later, filled {_fmt(sm.lat_submit_to_fill_s.mean(), 3)} s after that", "",
         "## By session", "```", sm[["date", "feed", "live_trips", "matched", "live_only", "twin_only", "live_pnl_usd", "live_price_bp_matched", "twin_gross_bp_matched", "d_price_bp", "slip_vs_mid_bp", "s_corr_sip", "pos_agree_sip"]].round(2).to_string(index=False), "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--month", action="store_true")
    ap.add_argument("--mode", default="paper", choices=["paper", "live", "dry"])
    ap.add_argument("--no-tape", action="store_true")
    ap.add_argument("--update-data", action="store_true")
    a = ap.parse_args()
    if a.month:
        f = os.path.join(D.REPORTS, "live_compare", "summary.csv")
        allm = pd.read_csv(f)
        if "mode" not in allm:
            allm["mode"] = "paper"
        for md in allm["mode"].unique():
            open(os.path.join(D.REPORTS, "live_compare", f"MONTH_TO_DATE{'' if md == 'paper' else '_' + md}.md"), "w", encoding="utf-8").write(month_report(allm[allm["mode"] == md].reset_index(drop=True), md))
        print("rebuilt MONTH_TO_DATE for", list(allm["mode"].unique()))
        return
    if a.update_data:
        D.update_cache(feeds=("sip", "iex"))
    dates = [a.date] if a.date else sorted(read_jsonl(os.path.join(D.REPORTS, f"levpairs_fills_{a.mode}.jsonl")).get("date", pd.Series(dtype=str)).astype(str).unique()) if a.all else []
    if not dates:
        ap.error("give --date YYYY-MM-DD or --all")
    tape = None if a.no_tape else AlpacaTape()
    for d in dates:
        out = run_day(d, a.mode, tape)
        print(day_report(out))


if __name__ == "__main__":
    main()
