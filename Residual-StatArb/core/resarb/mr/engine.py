"""
No-leverage, trade-by-trade backtest of a mean-reversion model (no Qt).

Each session:
  1. opening auction  - positions held 'overnight' leave at the official open
  2. decision time    - decide() (the same function as live) with the snapshot prices: exits, then entries ranked into the free slots, sized at 1 / slots of the equity
                         (whole shares at the decision-time price; long + short value never above max_gross x equity)
  3. closing auction  - all of those orders fill at the official close; the cost is cost_auction_bp per side
  4. the equity is marked at the official close
Every trade is recorded: entry / exit session and auction, prices, shares, value, direction, P&L, holding, exit reason, the signal and its rank.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D
from .models import ETF_UNIVERSE, MRModel, Pos, Signals, decide, universe_parts

TD = 252


@dataclass
class MRResult:
    trades: pd.DataFrame
    equity: pd.Series                    # official-close equity per session
    returns: pd.Series
    bench: pd.Series                     # SPY buy & hold returns over the same sessions
    exposure: pd.Series                  # gross value held after the close / equity
    stats: dict = field(default_factory=dict)
    model: Optional[MRModel] = None
    npos: Optional[pd.Series] = None     # positions held after the close


def universe_mask(mk, m: MRModel, n_default: int = 100) -> np.ndarray:
    """Which symbols may be ENTERED on each session: the symbols of an ETF list are always eligible; 'topN' follows the point-in-time monthly membership; 'a+b' = either."""
    T, N = mk.C.shape
    out = np.zeros((T, N), bool)
    for part in universe_parts(m.universe):
        if part.startswith("top"):
            n = int(part[3:] or n_default)
            u = D.load_universe()
            u = u[u["rank"] <= n]
            by = u.groupby("month").symbol.apply(set).to_dict()
            months = mk.dates.to_period("M").to_timestamp()
            last_known = max(by) if by else None
            for t, mo in enumerate(months):
                s = by.get(mo) or (by.get(last_known) if last_known is not None and mo > last_known else set())
                for sym in s:
                    j = mk.col.get(sym)
                    if j is not None:
                        out[t, j] = True
        else:
            for sym in _part_symbols(part):
                j = mk.col.get(sym)
                if j is not None:
                    out[:, j] = True
    return out


def _part_symbols(part: str) -> list:
    if part == "etf":
        return list(ETF_UNIVERSE)
    if part.startswith("list:"):
        return [s.strip().upper() for s in part[5:].split(",") if s.strip()]
    if part.startswith("top"):
        u = D.load_universe()
        return ["SPY"] + sorted(set(u[u["rank"] <= int(part[3:] or 100)].symbol))
    raise ValueError(f"unknown universe {part!r} (use etf, topN, list:A,B,C or parts joined with +)")


def model_symbols(m: MRModel) -> list:
    """Every symbol the model can ever trade (and SPY, the calendar anchor of a top-N universe)."""
    out: list = []
    for part in universe_parts(m.universe):
        out += _part_symbols(part)
    return list(dict.fromkeys(out))


def run(m: MRModel, mk, start: str = "2016-09-01", end: Optional[str] = None, capital: float = 100_000.0, member: Optional[np.ndarray] = None,
        log: Callable[[str], None] = lambda s: None, stop: Callable[[], bool] = lambda: False, split: str = "2023-01-01") -> MRResult:
    sig = Signals(mk)
    member = universe_mask(mk, m) if member is None else member
    T, N = mk.C.shape
    t0 = int(np.searchsorted(mk.dates, pd.Timestamp(start)))
    t1 = int(np.searchsorted(mk.dates, pd.Timestamp(end))) if end else T
    O, C = mk.O, mk.C
    bad_open = set(pd.DatetimeIndex(D.BAD_OPENS).normalize())
    cash = float(capital)
    pos: dict = {}
    trades = []
    eq = np.full(T, np.nan)
    expo = np.zeros(T)
    npos = np.zeros(T, int)
    tid = 0
    c_side = m.cost_auction_bp / 1e4
    borrow_day = m.borrow_bp_yr / 1e4 / TD

    def close_trade(p: Pos, t: int, px: float, auction: str, reason: str):
        nonlocal cash
        val = p.shares * px
        fee = abs(val) * c_side
        cash += val - fee
        pnl = p.shares * (px - p.entry_px) - abs(p.shares * p.entry_px) * c_side - fee
        tr = trades[p.trade_id]
        tr.update(exit_date=mk.dates[t], exit_auction=auction, exit_px=px, exit_reason=reason, held_sessions=t - p.entry_t, pnl=pnl,
                  ret=pnl / abs(p.shares * p.entry_px) if p.entry_px else np.nan)

    for t in range(max(t0, 1), t1):
        if stop():
            break
        # 1. opening auction
        for s in [s for s, p in pos.items() if p.exit_open_t is not None and p.exit_open_t <= t]:
            j = mk.col[s]
            px = O[t, j]
            if mk.dates[t] in bad_open or not np.isfinite(px):
                px = mk.sO[t, j] if np.isfinite(mk.sO[t, j]) else np.nan
            if not np.isfinite(px):
                continue                                                                     # no open print: try again tomorrow
            close_trade(pos.pop(s), t, float(px), "open", "overnight hold ends at the open")
        # delisted / no data any more (no close for 3 sessions running - known by today): cash out at the last close
        for s in [s for s in pos if t >= 2 and not np.any(np.isfinite(C[t - 2:t + 1, mk.col[s]]))]:
            j = mk.col[s]
            lastc = C[:t, j][np.isfinite(C[:t, j])]
            close_trade(pos.pop(s), t, float(lastc[-1]) if len(lastc) else pos[s].entry_px, "close", "no more prices (delisted)")
        # 2. decision at the snapshot time
        last = sig.last[t]

        def mark(s, p):                                                                     # decision-time value: last price, else the previous close, else the entry price
            j = mk.col[s]
            for x in (last[j], C[t - 1, j]):
                if np.isfinite(x):
                    return p.shares * x
            return p.shares * p.entry_px
        val_dec = sum(mark(s, p) for s, p in pos.items())
        equity_dec = cash + val_dec
        dec = decide(m, t, sig, member[t], pos, equity_dec)
        # 3. closing auction
        for s, reason in dec.exits:
            j = mk.col[s]
            if np.isfinite(C[t, j]):
                close_trade(pos.pop(s), t, float(C[t, j]), "close", reason)
        for s, sh, sv, rank in dec.entries:                                                # market-on-close orders: the shares decided at the decision price, filled at the close
            j = mk.col[s]                                                                  # (live cannot trim after the fact, so neither does the backtest; the cash reserve
            px = C[t, j]                                                                   #  absorbs a close above the decision price - max_exposure in the stats shows it)
            if not np.isfinite(px) or sh == 0:
                continue
            fee = abs(sh * px) * c_side
            cash -= sh * px + fee
            trades.append(dict(trade_id=tid, symbol=s, direction="LONG" if sh > 0 else "SHORT", shares=abs(sh), entry_date=mk.dates[t], entry_auction="close", entry_px=float(px),
                               entry_value=abs(sh * px), signal=m.entry_signal, signal_value=sv, rank=rank, decision_px=float(last[j]),
                               exit_date=pd.NaT, exit_auction="", exit_px=np.nan, exit_reason="open", held_sessions=np.nan, pnl=np.nan, ret=np.nan))
            pos[s] = Pos(symbol=s, shares=sh, entry_t=t, entry_px=float(px), exit_open_t=t + 1 if m.hold == "overnight" else None, trade_id=tid)
            tid += 1
        # 4. mark at the official close (+ borrow on shorts)
        short_val = sum(-p.shares * C[t, mk.col[s]] for s, p in pos.items() if p.shares < 0 and np.isfinite(C[t, mk.col[s]]))
        cash -= short_val * borrow_day
        v = sum(p.shares * (C[t, mk.col[s]] if np.isfinite(C[t, mk.col[s]]) else p.entry_px) for s, p in pos.items())
        g = sum(abs(p.shares) * (C[t, mk.col[s]] if np.isfinite(C[t, mk.col[s]]) else p.entry_px) for s, p in pos.items())
        eq[t] = cash + v
        expo[t] = g / eq[t] if eq[t] > 0 else 0.0
        npos[t] = len(pos)
        if (t - t0) % 500 == 0:
            log(f"  {mk.dates[t]:%Y-%m-%d}: equity {eq[t]:,.0f}, {len(pos)} positions, {tid} trades")
    rng = slice(max(t0, 1), t1)
    equity = pd.Series(eq[rng], index=mk.dates[rng]).ffill()
    ret = equity.pct_change().fillna(equity.iloc[0] / capital - 1 if len(equity) else 0.0)
    spy = mk.col.get("SPY")
    bench = pd.Series(np.r_[np.nan, C[1:, spy] / C[:-1, spy] - 1][rng], index=mk.dates[rng]) if spy is not None else pd.Series(np.nan, index=mk.dates[rng])
    tr = pd.DataFrame(trades)
    res = MRResult(trades=tr, equity=equity, returns=ret, bench=bench, exposure=pd.Series(expo[rng], index=mk.dates[rng]), model=m)
    res.npos = pd.Series(npos[rng], index=mk.dates[rng])
    res.stats = stats(res, split)
    return res


def _block(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / TD
    sd = r.std()
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(sd * math.sqrt(TD)), "sharpe": float(r.mean() / sd * math.sqrt(TD)) if sd > 0 else float("nan"),
            "max_dd": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1), "days": len(r)}


def stats(res: MRResult, split: str = "2023-01-01") -> dict:
    r, b = res.returns, res.bench
    sp = pd.Timestamp(split)
    s = {"all": _block(r), "is": _block(r[r.index < sp]), "oos": _block(r[r.index >= sp]), "spy_all": _block(b), "spy_is": _block(b[b.index < sp]),
         "spy_oos": _block(b[b.index >= sp])}
    tr = res.trades
    done = tr[tr.exit_date.notna()] if len(tr) else tr
    s["trades"] = int(len(tr))
    if len(done):
        s["win_rate"] = float((done.pnl > 0).mean())
        s["avg_ret"] = float(done.ret.mean())
        s["avg_hold"] = float(done.held_sessions.mean())
        s["profit_factor"] = float(done.pnl[done.pnl > 0].sum() / max(-done.pnl[done.pnl < 0].sum(), 1e-9))
    s["avg_exposure"] = float(res.exposure.mean())
    s["max_exposure"] = float(res.exposure.max()) if len(res.exposure) else 0.0
    ok = r.notna() & b.notna()
    if ok.sum() > 50:
        s["beta_spy"] = float(np.cov(r[ok], b[ok])[0, 1] / b[ok].var())
        s["corr_spy"] = float(np.corrcoef(r[ok], b[ok])[0, 1])
    s["by_year"] = {int(y): float(np.prod(1 + x) - 1) for y, x in r.groupby(r.index.year)}
    s["spy_by_year"] = {int(y): float(np.prod(1 + x.dropna()) - 1) for y, x in b.groupby(b.index.year)}
    return s


def summary(st: dict) -> str:
    a, i, o, sa = st.get("all", {}), st.get("is", {}), st.get("oos", {}), st.get("spy_all", {})
    f = lambda b, k, fmt: fmt.format(b[k]) if k in b and b[k] is not None and np.isfinite(b[k]) else "-"
    return (f"Sharpe {f(a, 'sharpe', '{:.2f}')} (IS {f(i, 'sharpe', '{:.2f}')}, OOS {f(o, 'sharpe', '{:.2f}')}) · CAGR {f(a, 'cagr', '{:.1%}')} vs SPY {f(sa, 'cagr', '{:.1%}')} · "
            f"maxDD {f(a, 'max_dd', '{:.1%}')} · {st.get('trades', 0)} trades · win {st.get('win_rate', float('nan')):.0%} · exposure {st.get('avg_exposure', 0):.0%}")
