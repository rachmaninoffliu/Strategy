"""
Feed-fidelity test. This paper account may NOT query recent SIP data ("subscription does not permit querying recent SIP data"), only IEX, so a live decision can only see
IEX bars while the research (and the model) used the consolidated SIP tape. How much of the edge survives?

For TSLA / SMH / NVDA, 2022-2026 (IEX history starts 2020-07-27), rule actually planned for the paper runner (no new risk before 10:00; decisions 10:00 .. 15:30, the position held at 15:30 is exited at the official close by a market-on-close order):
  SIP>SIP     research baseline (SIP model, SIP features)
  SIP>IEX     SIP-trained model applied to IEX features   (what the runner would do with today's models)
  IEX>IEX     IEX-trained model on IEX features            (retrain on the live feed)
Signals come from the named feed; P&L is always marked on SIP prices (the consolidated market is where the orders would trade). A 5-minute IEX bar with no IEX trade is carried
forward from the previous close with zero volume (the same rule the live runner uses).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_audit as A
import ia_data as D
import ia_study as S

SYMS = ["TSLA", "SMH", "NVDA"]


def align_iex(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IEX bars reindexed to the SIP 78-bar grid; returns (aligned iex bars, per-bar missing flag)."""
    sip, iex = D.load_bars(sym), D.load_bars(sym, "iex")
    iex = iex[iex.index >= iex.index[0]]
    grid = sip.index[sip.index >= iex.index[0].normalize()]
    a = iex.reindex(grid)
    missing = a.close.isna()
    a["close"] = a.close.ffill()
    for c in ("open", "high", "low", "vwap"):
        a[c] = a[c].fillna(a.close)
    a["volume"] = a.volume.fillna(0.0); a["trade_count"] = a.trade_count.fillna(0.0)
    return a, missing


def wf(dd_train: dict, dd_test: dict, kind: str, first_test_year: int = 2022, stride: int = 2, min_train: int = 300) -> np.ndarray:
    """Expanding window by year: fit on every dd_train day of earlier years, predict the dd_test days of the test year."""
    ytr_years, years = dd_train["days"].year.values, dd_test["days"].year.values
    edge = np.full(dd_test["y"].shape, np.nan)
    nf = dd_test["X"].shape[2]
    for Y in range(first_test_year, years.max() + 1):
        tr, te = np.where(ytr_years < Y)[0], np.where(years == Y)[0]
        if len(tr) < min_train or not len(te):
            continue
        Xtr = dd_train["X"][tr][:, ::stride].reshape(-1, nf); ytr = dd_train["y"][tr][:, ::stride].reshape(-1)
        edge[te] = S.fit_predict(Xtr, ytr, dd_test["X"][te].reshape(-1, nf), kind).reshape(len(te), -1)
    return edge


def restrict(dd: dict, days: pd.DatetimeIndex) -> dict:
    ix = dd["days"].get_indexer(days)
    return {k: (v[ix] if isinstance(v, np.ndarray) else days) for k, v in dd.items()}


def run(sym: str) -> dict:
    a, missing = align_iex(sym)
    sip, iex = S.build_days(sym), S.build_days(sym, a)
    days = sip["days"].intersection(iex["days"])
    s, i = restrict(sip, days), restrict(iex, days)
    out = {"days": days, "nxt": s["nxt"], "y": s["y"]}
    for kind in ("ridge", "gbm"):
        out[f"sip_sip_{kind}"] = wf(sip, s, kind)                    # SIP model, full SIP history, SIP features
        out[f"sip_iex_{kind}"] = wf(sip, i, kind)                    # same model, IEX features
        out[f"iex_iex_{kind}"] = wf(iex, i, kind)                    # model trained on IEX history (starts 2020-07-27)
    # coverage of the IEX tape on the regular-hours grid
    mm = missing.groupby(missing.index.year).mean()
    out["missing_by_year"] = mm.round(3).to_dict()
    out["missing_first_hour"] = float(missing[(missing.index.hour == 9) | ((missing.index.hour == 10) & (missing.index.minute < 30))].mean())
    out["vol_ratio"] = float((a.volume.sum()) / D.load_bars(sym).volume.reindex(a.index).sum())
    return {sym: out}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    R = {}
    with ProcessPoolExecutor(3) as ex:
        for r in ex.map(run, SYMS):
            R.update(r)
    SIG = {"TSLA": 2.46, "SMH": 1.18, "NVDA": 2.00}
    series = {m: {} for m in ("sip_sip", "sip_iex", "iex_iex")}
    for sym, o in R.items():
        c = float(sp.spread_bp[sym]) / 2 + S.SLIP_BP; cj = np.full(72, c); m = o["days"].year.values >= 2022
        print(f"\n== {sym}: IEX bars missing by year {o['missing_by_year']} | first hour {o['missing_first_hour']:.3f} | IEX/SIP volume {o['vol_ratio']:.3f}")
        for mode in series:
            edge = (o[f"{mode}_ridge"] + o[f"{mode}_gbm"]) / 2
            e2 = edge.copy(); e2[:, :5] = 0.0
            pos = A.path(e2, c, 3.0)
            net = A.net_j(pos, o["nxt"], cj)
            series[mode][sym] = pd.Series(net, index=o["days"])[m]
            line = f"{mode:8s} Sharpe {A.sh(net[m]):5.2f}  net bp/day {net[m].mean():6.1f}"
            if mode != "sip_sip":
                ref = (o["sip_sip_ridge"] + o["sip_sip_gbm"]) / 2
                ok = m[:, None] & np.isfinite(edge) & np.isfinite(ref)
                line += f" | corr(edge, SIP edge) {np.corrcoef(edge[ok], ref[ok])[0, 1]:.2f}, sign agreement {(np.sign(edge[ok]) == np.sign(ref[ok])).mean():.3f}"
            print(line)
    print("\n== basket TSLA+SMH+NVDA, inverse-vol weights")
    w = pd.Series({s: 1 / SIG[s] for s in SYMS})
    for mode, d in series.items():
        b = (pd.DataFrame(d) * w).sum(axis=1) / w.sum()
        print(f"{mode:8s} Sharpe {A.sh(b.values):.2f} | by year {({int(y): round(A.sh(g.values), 2) for y, g in b.groupby(b.index.year)})}")
