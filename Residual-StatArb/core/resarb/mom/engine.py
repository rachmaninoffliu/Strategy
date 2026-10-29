"""
No-leverage, trade-by-trade backtest of the time-shared momentum book (no Qt).

Each session (the live engine can send market DAY orders in place of the auction orders - its "Auction orders" setting; the backtest prices the auctions):
  1. opening auction  - last night's overnight positions are sold at the official open (cost: auction bp per side)
  2. intraday leg      - at every check (10:00, 10:30, ... 30 minutes before the close) the noise-area rule on the intraday symbol: a buy / sell at that bar's close from the
                         intraday feed plus / minus the half spread; the buy takes (1 - cash reserve) of the equity, never more than the cash
  3. decision time     - decide_overnight() (the same function as live) on the snapshot: the k best-ranked names, (1 - cash reserve) / k of the equity each, whole shares at the
                         decision price; the intraday position is valued at its decision-time price
  4. closing auction   - the intraday position (if still held) is sold and the overnight entries are bought at the official close
  5. the equity is marked at the official close
Every trade of both legs is recorded: leg, ticker, shares, entry / exit time, how (auction / market order), prices, value, the ranking score and rank (overnight) or the band
(intraday), the exit reason and the P&L after costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D
from ..mr import engine as MRE
from . import ml
from .features import feature_frame, member_mask, noise_sigma, rankable, spy_trend
from .market import MomMarket
from .models import SECTOR_OF_ETF, MomModel, check_slots, decide_overnight, intraday_shares, noise_action, noise_upper, slot_end

TD = 252


@dataclass
class Pos:
    symbol: str
    shares: int
    entry_t: int
    entry_px: float
    trade_id: int = 0
    exit_open_t: Optional[int] = None


@dataclass
class MomResult:
    trades: pd.DataFrame
    equity: pd.Series                    # official-close equity per session
    returns: pd.Series
    bench: pd.Series                     # SPY buy & hold returns over the same sessions
    exposure: pd.Series                  # overnight value held after the close / equity
    daily: pd.DataFrame                  # per session: equity, P&L of each leg, overnight names, trend, intraday source
    stats: dict = field(default_factory=dict)
    model: Optional[MomModel] = None
    ml_used: dict = field(default_factory=dict)


def model_symbols(m: MomModel) -> list:
    out = MRE.model_symbols(m) + ["SPY"] + ([m.intraday_symbol] if m.intraday_symbol else [])
    return list(dict.fromkeys(out))


def load_market(m: MomModel, start: str = "2015-06-01", log: Callable[[str], None] = lambda s: None) -> MomMarket:
    syms = model_symbols(m)
    log(f"loading {len(syms)} symbols (daily bars + the {m.minute // 100}:{m.minute % 100:02d} snapshots + {m.intraday_symbol or '-'} {m.intraday_feed} bars) ...")
    return MomMarket(syms, minute=m.minute, start=start, intraday_symbol=m.intraday_symbol, intraday_feed=m.intraday_feed)


def training_frame(m: MomModel, mk: MomMarket) -> pd.DataFrame:
    return feature_frame(mk, rankable(mk, member_mask(mk, m)), target=True)[0]


def prepare(m: MomModel, mk: MomMarket, folder: Optional[str] = None, rows=None, log: Callable[[str], None] = lambda s: None) -> dict:
    """The per-session inputs of the decisions: rankable names, SPY trend, today's move, the ML forecast (rule+ml) and the noise widths of the intraday symbol."""
    U = rankable(mk, member_mask(mk, m))
    trend = spy_trend(mk) if m.trend_filter else np.ones(len(mk.dates), bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        d45 = mk.S / mk.prev_close() - 1
    pred, used = None, {}
    if m.uses_ml:
        if folder is None:
            raise RuntimeError("the rule+ml ranking needs the app folder (its models/ directory)")
        log("ML forecast: feature table ...")
        df, _ = feature_frame(mk, U, rows=rows)
        pred, used = ml.predict_matrix(folder, m, df, mk.C.shape)
    sig = noise_sigma(mk, m.intraday_lookback) if m.intraday_symbol and mk.iq else None
    return dict(U=U, trend=trend, d45=d45, pred=pred, ml_used=used, sig=sig)


def sector_list(syms: list) -> list:
    """The sector of each symbol for the sector cap (models.SECTOR_OF_ETF for the sector ETFs; '' = none) - the universe's sector file, the same in the backtest and live."""
    sec = D.sectors()
    return [SECTOR_OF_ETF.get(s) or str(sec.get(s) or "") for s in syms]


def run(m: MomModel, mk: MomMarket, folder: Optional[str] = None, start: str = "2016-09-01", end: Optional[str] = None, capital: float = 100_000.0,
        split: str = "2023-01-01", log: Callable[[str], None] = lambda s: None, stop: Callable[[], bool] = lambda: False, prep: Optional[dict] = None) -> MomResult:
    T, N = mk.C.shape
    syms, col = mk.syms, mk.col
    t0 = int(np.searchsorted(mk.dates, pd.Timestamp(start)))
    t1 = int(np.searchsorted(mk.dates, pd.Timestamp(end))) if end else T
    P = prep or prepare(m, mk, folder, rows=range(max(t0, 1), t1), log=log)
    O, C, S = mk.O, mk.C, mk.S
    bad_open = set(pd.DatetimeIndex(D.BAD_OPENS).normalize())
    c_open, c_close, c_iclose = m.auction_costs()
    at_open, at_close = "opening auction", "closing auction"
    h = m.half_spread_bp / 1e4
    secs = sector_list(syms) if m.sector_cap > 0 else None
    iq = mk.iq if m.intraday_symbol else {}
    jq = col.get(m.intraday_symbol) if iq else None
    cash = float(capital)
    pos: dict = {}
    ip: Optional[dict] = None                                                                  # intraday position
    trades: list = []
    eq = np.full(T, np.nan)
    expo = np.zeros(T)
    on_pnl, id_pnl = np.zeros(T), np.zeros(T)
    n_on = np.zeros(T, int)
    no_ml = 0

    def new_trade(**kw) -> int:
        trades.append(dict(trade_id=len(trades), direction="LONG", **kw))
        return len(trades) - 1

    def finish(tid: int, t: int, ts, how: str, px: float, reason: str, fee_out: float) -> float:
        tr = trades[tid]
        pnl = tr["shares"] * (px - tr["entry_px"]) - tr["fees"] - fee_out
        tr.update(exit_date=mk.dates[t], exit_time=ts, exit_how=how, exit_px=px, exit_reason=reason, fees=tr["fees"] + fee_out, pnl=pnl,
                  ret=pnl / (tr["shares"] * tr["entry_px"]) if tr["entry_px"] else np.nan)
        return pnl

    def close_overnight(s: str, t: int, px: float, how: str, ts, reason: str, c: float) -> None:
        nonlocal cash
        p = pos.pop(s)
        val = p.shares * px
        fee = abs(val) * c
        cash += val - fee
        on_pnl[t] += finish(p.trade_id, t, ts, how, px, reason, fee)

    for t in range(max(t0, 1), t1):
        if stop():
            break
        day = mk.dates[t]
        close_ts = day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * int(mk.nb[t]))
        open_ts = day + pd.Timedelta(hours=9, minutes=30)
        # 1. opening auction
        for s in [s for s, p in pos.items() if p.exit_open_t is not None and p.exit_open_t <= t]:
            j = col[s]
            px = O[t, j]
            if day in bad_open or not np.isfinite(px):
                px = mk.sO[t, j]
            if np.isfinite(px):
                close_overnight(s, t, float(px), at_open, open_ts, "overnight hold ends at the open", c_open)
        for s in [s for s in pos if t >= 2 and not np.any(np.isfinite(C[t - 2:t + 1, col[s]]))]:
            lastc = C[:t, col[s]][np.isfinite(C[:t, col[s]])]
            close_overnight(s, t, float(lastc[-1]) if len(lastc) else pos[s].entry_px, at_close, close_ts, "no more prices (delisted)", c_close)

        def held_value(at_decision: bool) -> float:
            v = 0.0
            for s, p in pos.items():
                j = col[s]
                for x in ((S[t, j], C[t - 1, j]) if at_decision else (C[t - 1, j],)):
                    if np.isfinite(x):
                        v += p.shares * x
                        break
                else:
                    v += p.shares * p.entry_px
            return v
        # 2. intraday leg
        if jq is not None and P["sig"] is not None:
            upper = noise_upper(iq["op"][t], C[t - 1, jq], P["sig"][t], m.intraday_mult)
            for sl in check_slots(int(mk.nb[t]), m.intraday_every):
                px, vw = iq["cl"][t, sl], iq["vwap"][t, sl]
                act = noise_action(ip is not None, px, upper[sl], vw, m.intraday_trail)
                if act == "buy":
                    sh = intraday_shares(m, cash + held_value(False), cash, px)
                    if sh <= 0:
                        continue
                    fill = px * (1 + h)
                    cash -= sh * fill
                    tid = new_trade(leg="INTRADAY", symbol=m.intraday_symbol, shares=sh, entry_date=day, entry_time=slot_end(day, sl), entry_how=f"market ({iq['src'][t]} bar)",
                                    entry_px=fill, entry_value=sh * fill, score="noise band", score_value=float(upper[sl]), rank=np.nan, move=float(px / C[t - 1, jq] - 1),
                                    forecast=np.nan, decision_px=float(px), fees=0.0, exit_date=pd.NaT, exit_time=pd.NaT, exit_how="", exit_px=np.nan, exit_reason="",
                                    pnl=np.nan, ret=np.nan)
                    ip = dict(shares=sh, trade_id=tid, px=float(px))
                elif act == "sell":
                    fill = px * (1 - h)
                    cash += ip["shares"] * fill
                    stop_px = upper[sl] if not np.isfinite(vw) else max(upper[sl], vw)
                    id_pnl[t] += finish(ip["trade_id"], t, slot_end(day, sl), f"market ({iq['src'][t]} bar)", fill,
                                        f"{px:.2f} below {'VWAP' if np.isfinite(vw) and vw >= upper[sl] else 'band'} {stop_px:.2f}", 0.0)
                    ip = None
                else:
                    if ip is not None and np.isfinite(px):
                        ip["px"] = float(px)
        # 3. overnight decision at the snapshot time
        val = held_value(True)
        if ip is not None:
            qp = S[t, jq] if np.isfinite(S[t, jq]) else ip["px"]
            val += ip["shares"] * qp
        cap = cash + val
        pred_row = P["pred"][t] if P["pred"] is not None else None
        if m.uses_ml and (pred_row is None or not np.isfinite(pred_row).any()) and bool(P["trend"][t]) and m.overnight:
            no_ml += 1
        dec = decide_overnight(m, syms, P["U"][t], bool(P["trend"][t]), P["d45"][t], pred_row, S[t], {s: p.shares for s, p in pos.items()}, cap, sectors=secs)
        # 4. closing auction
        if ip is not None:
            px = float(C[t, jq]) if np.isfinite(C[t, jq]) else ip["px"]
            fee = ip["shares"] * px * c_iclose
            cash += ip["shares"] * px - fee
            id_pnl[t] += finish(ip["trade_id"], t, close_ts, at_close, float(px), f"held at the last check: sold in the {at_close}", fee)
            ip = None
        for s, reason in dec.exits:
            if np.isfinite(C[t, col[s]]):
                close_overnight(s, t, float(C[t, col[s]]), at_close, close_ts, reason, c_close)
        for s in dec.keeps:
            pos[s].exit_open_t = t + 1
        for s, sh, sc, rank, mv, fc in dec.entries:
            j = col[s]
            px = C[t, j]
            if not np.isfinite(px):
                continue
            fee = sh * px * c_close
            cash -= sh * px + fee
            tid = new_trade(leg="OVERNIGHT", symbol=s, shares=sh, entry_date=day, entry_time=close_ts, entry_how=at_close, entry_px=float(px), entry_value=sh * px,
                            score=m.ranking, score_value=sc, rank=rank, move=mv, forecast=fc, decision_px=float(S[t, j]), fees=fee, exit_date=pd.NaT, exit_time=pd.NaT,
                            exit_how="", exit_px=np.nan, exit_reason="", pnl=np.nan, ret=np.nan)
            pos[s] = Pos(symbol=s, shares=sh, entry_t=t, entry_px=float(px), trade_id=tid, exit_open_t=t + 1)
        # 5. mark at the official close
        v = sum(p.shares * (C[t, col[s]] if np.isfinite(C[t, col[s]]) else p.entry_px) for s, p in pos.items())
        eq[t] = cash + v
        expo[t] = v / eq[t] if eq[t] > 0 else 0.0
        n_on[t] = len(pos)
        if (t - t0) % 500 == 0:
            log(f"  {day:%Y-%m-%d}: equity {eq[t]:,.0f}, {len(pos)} overnight names, {len(trades)} trades")
    rng = slice(max(t0, 1), t1)
    idx = mk.dates[rng]
    equity = pd.Series(eq[rng], index=idx).ffill()
    ret = equity.pct_change().fillna(equity.iloc[0] / capital - 1 if len(equity) else 0.0)
    spy = col.get("SPY")
    bench = pd.Series(np.r_[np.nan, C[1:, spy] / C[:-1, spy] - 1][rng], index=idx) if spy is not None else pd.Series(np.nan, index=idx)
    prev_eq = equity.shift(1).fillna(capital)
    daily = pd.DataFrame({"equity": equity, "ret": ret, "overnight_pnl": on_pnl[rng], "intraday_pnl": id_pnl[rng], "overnight_names": n_on[rng],
                          "trend": np.asarray(P["trend"])[rng], "intraday_src": iq["src"][rng] if iq else ""}, index=idx)
    daily["overnight_ret"] = daily.overnight_pnl / prev_eq
    daily["intraday_ret"] = daily.intraday_pnl / prev_eq
    tr = pd.DataFrame(trades)
    res = MomResult(trades=tr, equity=equity, returns=ret, bench=bench, exposure=pd.Series(expo[rng], index=idx), daily=daily, model=m, ml_used=P.get("ml_used") or {})
    res.stats = stats(res, split)
    res.stats["sessions_without_forecast"] = no_ml
    return res


def _block(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / TD
    sd = r.std()
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0, "vol": float(sd * math.sqrt(TD)),
            "sharpe": float(r.mean() / sd * math.sqrt(TD)) if sd > 0 else float("nan"), "max_dd": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1), "days": len(r)}


def stats(res: MomResult, split: str = "2023-01-01") -> dict:
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
        s["profit_factor"] = float(done.pnl[done.pnl > 0].sum() / max(-done.pnl[done.pnl < 0].sum(), 1e-9))
        for leg in ("OVERNIGHT", "INTRADAY"):
            x = done[done.leg == leg]
            s[f"{leg.lower()}_trades"] = int(len(x))
            s[f"{leg.lower()}_win"] = float((x.pnl > 0).mean()) if len(x) else float("nan")
            s[f"{leg.lower()}_avg_ret"] = float(x.ret.mean()) if len(x) else float("nan")
            s[f"{leg.lower()}_pnl"] = float(x.pnl.sum())
    for leg in ("overnight", "intraday"):
        x = res.daily[f"{leg}_ret"]
        s[f"{leg}_leg"] = {"all": _block(x), "is": _block(x[x.index < sp]), "oos": _block(x[x.index >= sp])}
    s["invested_nights"] = float((res.daily.overnight_names > 0).mean())
    s["intraday_days"] = float((res.daily.intraday_pnl != 0).mean())
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
            f"maxDD {f(a, 'max_dd', '{:.1%}')} · {st.get('trades', 0)} trades ({st.get('overnight_trades', 0)} overnight, {st.get('intraday_trades', 0)} intraday)")
