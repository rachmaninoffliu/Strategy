"""
Live (paper or dry run) vs backtest, from the journal - no broker, no Qt. Run it after the session once the SIP cache is topped up (the next morning, or 16:20 ET + Update data).

For every journaled decision the backtest's decision for the SAME data bar is rebuilt from the SIP history, with the SAME positions the engine held, and compared:
  score_corr      rank correlation of the live scores (live feed: IEX or delayed SIP) and the SIP-history scores
  target_corr     correlation of the live and backtest target weights;  same_side = share of names held on the same side;  diff_gross = sum |w_live - w_bt| / gross
For every fill: slippage against the decision's reference price (the data bar's close) and against the SIP open of the trade bar (the price the backtest fills at).

  python -m resarb.compare <strategy folder> [YYYY-MM-DD] [paper|live]
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import data as D
from . import strategy as S
from .portfolio import approach, decide, hedge_etf, project


def _et(x) -> pd.Timestamp:
    """A broker timestamp (UTC-aware) as naive ET, like the bar index."""
    t = pd.Timestamp(x)
    return t.tz_convert(D.NY).tz_localize(None) if t.tzinfo is not None else t


def load_journal(folder: str, mode: str = "paper") -> pd.DataFrame:
    p = os.path.join(folder, "reports", f"journal_{mode}.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    return pd.DataFrame(rows)


def compare(strat: S.Strategy, mode: str = "paper", date: Optional[str] = None, log: Callable[[str], None] = print) -> dict:
    j = load_journal(strat.folder, mode)
    if not len(j):
        raise FileNotFoundError("no journal yet")
    j["day"] = pd.to_datetime(j.ts).dt.strftime("%Y-%m-%d")
    day = date or j.day.max()
    jd = j[j.day == day]
    dec = jd[jd.kind.isin(["decision", "plan_dry", "plan_no_orders"])]
    fills = jd[jd.kind == "fill"]
    out_d, out_f = [], []
    if len(dec):
        ctx = S.context(strat, start=(pd.Timestamp(day) - pd.Timedelta(days=60)).strftime("%Y-%m-%d"), end=(pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), log=log,
                        cache=False)
        p, res = ctx.p, ctx.res
        sc = strat.live_score(ctx) if hasattr(strat, "live_score") else strat.score(ctx)
        is_etf = np.array([s in D.ETFS for s in p.syms])
        sidx = np.where(~is_etf)[0]
        pc = strat.port_cfg()
        for r in dec.itertuples():
            tdts = pd.Timestamp(r.data_bar)
            td = int(np.searchsorted(p.ts, tdts))
            if td >= len(p.ts) or p.ts[td] != tdts:
                continue
            cap = float(r.capital)
            held = np.zeros(len(p.syms))
            for s, q in (r.held or {}).items():
                if s in p.col and np.isfinite(p.close[td, p.col[s]]):
                    held[p.col[s]] = q * float((r.px or {}).get(s, p.close[td, p.col[s]])) / cap
            w = np.zeros(len(p.syms))
            if not (r.eod or r.stopped):
                trad = np.isfinite(p.open[td, sidx])
                w[sidx] = approach(held[sidx], decide(sc[td, sidx], ctx.member[td, sidx], held[sidx], pc, res.sector_col[sidx], trad), pc.trade_rate, pc.gross, trad)
                if pc.hedge == "project":
                    w = project(w, res, p.day[td])
                if pc.hedge == "etf":                                                         # like the engine: hedge the whole shares that would be held
                    pxr = np.array([float((r.px or {}).get(s, p.close[td, j])) for j, s in enumerate(p.syms)])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        wr = np.where(np.isfinite(pxr) & (pxr > 0), np.trunc(w * cap / pxr) * pxr / cap, w)
                    wr[is_etf] = 0.0
                    w[res.etf_idx] += hedge_etf(wr, res, p.day[td])
            live = pd.Series(r.targets or {}, dtype=float)
            bt = pd.Series(w, index=p.syms)
            bt = bt[bt != 0]
            u = sorted(set(live.index) | set(bt.index))
            a, b = live.reindex(u).fillna(0.0), bt.reindex(u).fillna(0.0)
            stock = [s for s in u if s not in D.ETFS]
            g = max(float(np.abs(b[stock]).sum()), 1e-12)
            ls = pd.Series(r.scores or {}, dtype=float)
            bs = pd.Series(sc[td], index=p.syms)
            common = [s for s in ls.index if s in bs.index and np.isfinite(bs[s])]
            out_d.append(dict(data_bar=tdts, trade_bar=r.trade_bar, kind=r.kind, n_live=int((a[stock] != 0).sum()), n_bt=int((b[stock] != 0).sum()),
                              score_corr=float(ls[common].rank().corr(bs[common].rank())) if len(common) > 5 else np.nan,
                              target_corr=float(np.corrcoef(a[stock], b[stock])[0, 1]) if len(stock) > 2 and a[stock].std() > 0 and b[stock].std() > 0 else np.nan,
                              same_side=float((np.sign(a[stock]) == np.sign(b[stock]))[(a[stock] != 0) | (b[stock] != 0)].mean()) if len(stock) else np.nan,
                              diff_gross=float(np.abs(a[stock] - b[stock]).sum() / g)))
        for r in fills.itertuples():
            s = r.symbol
            tb = pd.Timestamp(r.trade_bar)
            k = int(np.searchsorted(p.ts, tb))
            o = p.open[k, p.col[s]] if s in p.col and k < len(p.ts) and p.ts[k] == tb else np.nan
            sgn = 1 if r.side == "buy" else -1
            px = float(r.avg_px) if r.avg_px is not None else np.nan
            out_f.append(dict(symbol=s, side=r.side, qty=r.qty, filled_qty=r.filled_qty, avg_px=px, ref_px=r.ref_px, bt_open=o,
                              slip_vs_ref_bp=r.slip_vs_ref_bp, slip_vs_bt_open_bp=(px / o - 1) * 1e4 * sgn if np.isfinite(o) and np.isfinite(px) else np.nan,
                              latency_s=(_et(r.filled_at) - tb).total_seconds() if r.filled_at not in (None, "None", "NaT") else np.nan))
    d, f = pd.DataFrame(out_d), pd.DataFrame(out_f)
    summ = {"day": day, "decisions": len(d), "fills": len(f),
            "score_corr": float(d.score_corr.mean()) if len(d) else np.nan, "target_corr": float(d.target_corr.mean()) if len(d) else np.nan,
            "same_side": float(d.same_side.mean()) if len(d) else np.nan, "diff_gross": float(d.diff_gross.mean()) if len(d) else np.nan,
            "slip_vs_ref_bp": float(f.slip_vs_ref_bp.mean()) if len(f) else np.nan, "slip_vs_bt_open_bp": float(f.slip_vs_bt_open_bp.mean()) if len(f) else np.nan,
            "median_latency_s": float(f.latency_s.median()) if len(f) else np.nan}
    rep = os.path.join(strat.folder, "reports")
    os.makedirs(rep, exist_ok=True)
    d.to_csv(os.path.join(rep, f"compare_decisions_{mode}_{day}.csv"), index=False)
    f.to_csv(os.path.join(rep, f"compare_fills_{mode}_{day}.csv"), index=False)
    with open(os.path.join(rep, f"compare_summary_{mode}.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summ, default=str) + "\n")
    return {"decisions": d, "fills": f, "summary": summ}
