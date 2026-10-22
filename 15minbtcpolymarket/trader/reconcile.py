"""Live vs backtest on the same period, from the trade ledger (Qt-free).

For every recorded session the ledger knows WHEN it was trading and WITH WHICH settings.  ``reconcile`` re-runs the backtest on exactly that span with exactly those settings and sets it against the trades
the session really made, window by window:

  matched         both traded the same market - then the differences are the real market's: the live quote vs the candle close, the real fill price vs the assumed slippage, the size actually filled
  live only       the session traded a window the backtest did not (its quote at the second differed from the candle close, or its equity path made a different sizing decision)
  backtest only   the backtest traded a window the session did not (no fill, not enough book depth, a skipped order ...)

Sessions recorded before the ledger kept settings and heartbeats cannot be compared exactly and are listed as skipped.  ``replay_live`` (trader.replay) is the other half: it proves the CODE paths are identical;
this module measures how closely REAL trading followed them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import WINDOW
from .ledger import Ledger
from .live import LiveConfig, config_from_dict
from .replay import backtest_config_from

COLS = ["session", "status", "window", "market", "side_live", "side_bt", "quoted_live", "quoted_bt", "entry_live", "entry_bt", "slip_c", "contracts_live", "requested_live", "contracts_bt",
        "pnl_live", "pnl_bt", "outcome_live"]


@dataclass
class SessionComparison:
    session: str
    mode: str
    venue: str
    start_ts: int
    end_ts: int
    windows: int
    cfg: LiveConfig
    table: pd.DataFrame
    notes: list = field(default_factory=list)


@dataclass
class Reconciliation:
    sessions: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        parts = [s.table for s in self.sessions if len(s.table)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)

    def summary(self) -> dict:
        t = self.table()
        m = t[t.status == "matched"]
        lv = t[t.status.isin(["matched", "live only"])]
        bt = t[t.status.isin(["matched", "backtest only"])]
        n_all = len(t)
        settled = lv[lv.outcome_live.isin(["WIN", "LOSS"])]
        ms = m[m.outcome_live.isin(["WIN", "LOSS"])]
        req = lv[lv.requested_live > 0]
        out = dict(sessions=len(self.sessions), windows=int(sum(s.windows for s in self.sessions)), live_trades=len(lv), bt_trades=len(bt), matched=len(m),
                   live_only=int((t.status == "live only").sum()), bt_only=int((t.status == "backtest only").sum()),
                   agreement=float(len(m) / n_all) if n_all else float("nan"),
                   quote_diff_c=float(((m.quoted_live - m.quoted_bt).abs() * 100).mean()) if len(m) else float("nan"),
                   slippage_c=float(lv.slip_c.mean()) if len(lv) else float("nan"),
                   assumed_slippage_c=float(np.mean([s.cfg.paper_slippage * 100 for s in self.sessions])) if self.sessions else float("nan"),
                   fill_ratio=float(req.contracts_live.sum() / req.requested_live.sum()) if len(req) and req.requested_live.sum() > 0 else float("nan"),
                   pnl_live=float(settled.pnl_live.sum()) if len(settled) else 0.0, pnl_bt=float(bt.pnl_bt.sum()) if len(bt) else 0.0,
                   matched_pnl_live=float(ms.pnl_live.sum()) if len(ms) else 0.0, matched_pnl_bt=float(ms.pnl_bt.sum()) if len(ms) else 0.0,
                   ev_live_c=float(ms.pnl_live.sum() / ms.contracts_live.sum() * 100) if len(ms) and ms.contracts_live.sum() else float("nan"),
                   ev_bt_c=float(ms.pnl_bt.sum() / ms.contracts_bt.sum() * 100) if len(ms) and ms.contracts_bt.sum() else float("nan"))
        return out

    def text(self) -> str:
        s = self.summary()
        if not self.sessions:
            return "Nothing to compare yet: no recorded session with saved settings." + ("\n" + "\n".join("  - " + x for x in self.skipped) if self.skipped else "")
        f = lambda x, d=1: "n/a" if x != x else f"{x:.{d}f}"          # noqa: E731
        lines = [f"{s['sessions']} session(s), {s['windows']} windows covered: the session traded {s['live_trades']}, the backtest {s['bt_trades']} -> {s['matched']} matched, {s['live_only']} live only, "
                 f"{s['bt_only']} backtest only (agreement {f(s['agreement'] * 100, 0)}%).",
                 f"Matched trades: live quote vs candle close differ by {f(s['quote_diff_c'])}c on average; real fill paid {f(s['slippage_c'])}c above the quote (the backtest assumes {f(s['assumed_slippage_c'])}c); "
                 f"{f(s['fill_ratio'] * 100, 0)}% of the contracts asked for were filled.",
                 f"Edge per contract on matched trades: live {f(s['ev_live_c'])}c vs backtest {f(s['ev_bt_c'])}c.  Total P&L: live ${s['pnl_live']:+,.2f}, backtest ${s['pnl_bt']:+,.2f}."]
        lines += [f"  note: {n}" for ss in self.sessions for n in ss.notes] + [f"  skipped: {x}" for x in self.skipped]
        return "\n".join(lines)


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d %H:%M")


def _session_trades(ledger_state, session: str) -> pd.DataFrame:
    """The session's settled and still-open positions as one frame keyed by market."""
    recs = []
    for r in ledger_state.settled:
        if r.get("session") == session:
            recs.append(dict(market=r["market"], side=r["side"], quoted=r["quoted"], entry=r["entry"], contracts=r["contracts"], requested=r.get("requested", r["contracts"]), pnl=r["pnl"], outcome=r["outcome"],
                             entry_ts=pd.Timestamp(r["entry_time"], tz="UTC").timestamp()))
    for o in ledger_state.open:
        if o.get("session") == session:
            recs.append(dict(market=o["market"], side=o["side"], quoted=o["quoted"], entry=o["entry"], contracts=o["contracts"], requested=o.get("requested") or o["contracts"], pnl=np.nan, outcome="OPEN",
                             entry_ts=float(o["ts"])))
    return pd.DataFrame(recs, columns=["market", "side", "quoted", "entry", "contracts", "requested", "pnl", "outcome", "entry_ts"])


def reconcile(ledger: Ledger, rows: pd.DataFrame, mode: str = "live", session: Optional[str] = None) -> Reconciliation:
    """Compare every recorded ``mode`` session (or just ``session``) with a backtest of the same span and settings on the candles in ``rows``."""
    rec = Reconciliation()
    st = ledger.load(mode=mode, legacy=False)
    data_end = int(rows.open_ts.max()) if len(rows) else 0
    known = {s["session"] for s in ledger.sessions(mode)}
    orphans = {r.get("session") for r in st.settled} | {o.get("session") for o in st.open}
    n_old = len([x for x in orphans if x and x not in known])
    if n_old and not session:
        rec.skipped.append(f"{n_old} older session(s) have trades but no recorded settings/heartbeat (they started before the ledger kept them) - not comparable exactly")
    for s in ledger.sessions(mode):
        if session and s["session"] != session:
            continue
        if not s["cfg"]:
            rec.skipped.append(f"{s['session']}: no settings recorded")
            continue
        cfg = config_from_dict(s["cfg"])
        m0, grace = 60 * min(cfg.params.minutes), cfg.grace_s
        end = s["end_ts"] if s["end_ts"] is not None else s["last_ts"]
        first = int(-(-(s["start_ts"] - m0 - grace) // WINDOW)) * WINDOW                      # the first window whose decision instant the session was already running for
        last = int((end - m0) // WINDOW) * WINDOW                                              # the last window whose decision instant it was still running at
        notes = []
        if last > data_end:
            notes.append(f"{s['session']}: the local candle data ends {_fmt(data_end + WINDOW)} UTC, before the session did - update the data to compare the rest")
            last = data_end
        if last < first:
            rec.skipped.append(f"{s['session']}: no complete window inside the session (or no data yet)")
            continue
        try:
            bcfg = backtest_config_from(cfg, datetime.fromtimestamp(first, timezone.utc).date(), datetime.fromtimestamp(last, timezone.utc).date())
        except ValueError as e:
            rec.skipped.append(f"{s['session']}: {e}")
            continue
        bcfg.start_ts, bcfg.end_ts = first, last + WINDOW
        bt = run_backtest(rows, bcfg).trades
        lv = _session_trades(st, s["session"])
        rec.sessions.append(SessionComparison(s["session"], s["mode"], s["venue"], first, last + WINDOW, (last - first) // WINDOW + 1, cfg, _compare(s["session"], lv, bt), notes))
    return rec


def _compare(session: str, lv: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    a = lv.set_index("market") if len(lv) else pd.DataFrame(columns=lv.columns).set_index("market")
    b = bt.set_index("market") if len(bt) else pd.DataFrame(columns=["side", "quoted", "entry", "contracts", "pnl", "entry_ts"])
    out = []
    for mk in sorted(set(a.index) | set(b.index)):
        x = a.loc[mk] if mk in a.index else None
        y = b.loc[mk] if mk in b.index else None
        when = float((x if x is not None else y)["entry_ts"])
        out.append(dict(session=session, status="matched" if x is not None and y is not None else "live only" if x is not None else "backtest only", window=_fmt(when // WINDOW * WINDOW), market=mk,
                        side_live="" if x is None else x["side"], side_bt="" if y is None else y["side"],
                        quoted_live=np.nan if x is None else float(x["quoted"]), quoted_bt=np.nan if y is None else float(y["quoted"]),
                        entry_live=np.nan if x is None else float(x["entry"]), entry_bt=np.nan if y is None else float(y["entry"]),
                        slip_c=np.nan if x is None else (float(x["entry"]) - float(x["quoted"])) * 100,
                        contracts_live=0 if x is None else int(x["contracts"]), requested_live=0 if x is None else int(x["requested"]), contracts_bt=0 if y is None else int(y["contracts"]),
                        pnl_live=np.nan if x is None else float(x["pnl"]), pnl_bt=np.nan if y is None else float(y["pnl"]), outcome_live="" if x is None else x["outcome"]))
    df = pd.DataFrame(out, columns=COLS)
    return df.sort_values(["window", "market"]).reset_index(drop=True) if len(df) else df

