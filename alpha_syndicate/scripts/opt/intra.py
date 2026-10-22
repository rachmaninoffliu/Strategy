"""Intraday (1h / 30m ...) replica of the Diversifier. Same state machine as sim._core, with:
  * dip signal (RSI / exit SMA) on the CHART-timeframe proxy bars
  * macro trend = SMA(len) of the proxy's DAILY closes through the PREVIOUS completed day (Pine: ta.sma()[1] + lookahead_on)
  * max-hold in days x barsPerDay bars (barsPerDay = ceil(390 / tfMinutes))
  * risk metrics from compounded DAILY returns (day boundaries by ET date)
"""
import json, math
import numpy as np
import pandas as pd
from numba import njit, prange
from sim import RSI_LENS, RSI_ENTRIES, EXIT_LENS, TREND_LENS, MAX_HOLDS, rsi_wilder, sma, MODE_DIP, MODE_TREND, MODE_TD, TRAIN, TEST, FULL
from sweep import derive, GRID_SHAPE

DATA = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt"


def load_tv(sym, tf):
    v = json.load(open(f"{DATA}\\tv_{sym}_{tf}.json"))
    b = pd.DataFrame(v["bars"], columns=["t", "o", "h", "l", "c", "v"])
    ts = pd.to_datetime(b["t"], unit="s", utc=True).dt.tz_convert("America/New_York")
    b["dt"] = ts.dt.tz_localize(None)
    b["date"] = b["dt"].dt.normalize()
    b["key"] = b["dt"].dt.year * 10000 + b["dt"].dt.month * 100 + b["dt"].dt.day
    return b.set_index("t")


@njit(cache=True)
def _core_intra(op, cl, pc, rsi, sx, st, inwin, lastwin, day, mode, rsi_entry, dip_trend, hold_bars, comm, split_key, tick):
    n = len(op)
    pos = 0; shares = 0.0; cash = 0.0
    entry_bar = -1; pending = 0; in_dip = False
    E = 1.0; entryE = 1.0
    eqD = 1.0; peak = 1.0; mdd = 0.0
    nd = 0; sS = 0.0; sS2 = 0.0; sNeg = 0.0; sB = 0.0; sB2 = 0.0; sSB = 0.0
    trades = 0; wins = 0; gp = 0.0; gl = 0.0
    inpos = 0; nbars = 0
    n1 = 0; a1 = 0.0; a2 = 0.0; n2 = 0; b1 = 0.0; b2 = 0.0
    dayEq = 1.0; dayPx = cl[0]
    prevCl = cl[0]; prevE = 1.0
    for i in range(n):
        if i > 0 and day[i] != day[i - 1]:
            if inwin[i - 1]:
                r = prevE / dayEq - 1.0
                rb = prevCl / dayPx - 1.0
                nd += 1
                sS += r; sS2 += r * r
                if r < 0:
                    sNeg += r * r
                sB += rb; sB2 += rb * rb; sSB += r * rb
                eqD *= 1.0 + r
                if eqD > peak:
                    peak = eqD
                dd = (peak - eqD) / peak
                if dd > mdd:
                    mdd = dd
                if day[i - 1] < split_key:
                    n1 += 1; a1 += r; a2 += r * r
                else:
                    n2 += 1; b1 += r; b2 += r * r
            dayEq = prevE; dayPx = prevCl
        pos_prev = pos
        just_entered = False
        if pending == 1:
            fp = np.round(op[i] / tick) * tick if tick > 0 else op[i]
            entryE = E
            shares = E / (prevCl * (1.0 + comm))
            cash = E - shares * fp * (1.0 + comm)
            pos = 1; entry_bar = i; just_entered = True
        elif pending == 2:
            fp = np.round(op[i] / tick) * tick if tick > 0 else op[i]
            E = cash + shares * fp * (1.0 - comm)
            shares = 0.0; cash = 0.0; pos = 0
            trades += 1
            pnl = E - entryE
            if pnl > 0:
                wins += 1; gp += pnl
            else:
                gl += -pnl
        pending = 0
        if pos == 1:
            E = cash + shares * cl[i]
        held = (i - entry_bar) if pos == 1 else 0
        time_up = (hold_bars > 0) and (held >= hold_bars)
        up = (not np.isnan(st[i])) and (pc[i] > st[i])
        rlow = (not np.isnan(rsi[i])) and (rsi[i] < rsi_entry)
        ddone = pc[i] > sx[i]
        if mode == 1:
            want = up
        elif mode == 0:
            if pos == 1:
                want = not (ddone or time_up)
            else:
                want = rlow and ((not dip_trend) or up)
        else:
            if up:
                want = True
            elif pos == 1:
                want = in_dip and not (ddone or time_up)
            else:
                want = rlow
        if inwin[i] and not lastwin[i]:
            if want and pos == 0:
                pending = 1
                in_dip = (mode == 0) or (mode == 2 and not up)
            elif (not want) and pos == 1:
                pending = 2
            if mode == 2 and up:
                in_dip = False
        elif pos == 1:
            fpc = np.round(cl[i] / tick) * tick if tick > 0 else cl[i]
            E = cash + shares * fpc * (1.0 - comm)
            shares = 0.0; cash = 0.0; pos = 0
            trades += 1
            pnl = E - entryE
            if pnl > 0:
                wins += 1; gp += pnl
            else:
                gl += -pnl
            pending = 0
        if inwin[i]:
            nbars += 1
            if pos_prev == 1:
                inpos += 1
        prevE = E
        prevCl = cl[i]
    # finalize the last (still open) day
    if inwin[n - 1]:
        r = prevE / dayEq - 1.0
        rb = prevCl / dayPx - 1.0
        nd += 1
        sS += r; sS2 += r * r
        if r < 0:
            sNeg += r * r
        sB += rb; sB2 += rb * rb; sSB += r * rb
        eqD *= 1.0 + r
        if eqD > peak:
            peak = eqD
        dd = (peak - eqD) / peak
        if dd > mdd:
            mdd = dd
        if day[n - 1] < split_key:
            n1 += 1; a1 += r; a2 += r * r
        else:
            n2 += 1; b1 += r; b2 += r * r
    out = np.zeros(20)
    out[0] = nd; out[1] = eqD; out[2] = mdd; out[3] = sS; out[4] = sS2; out[5] = sNeg
    out[6] = trades; out[7] = wins; out[8] = gp; out[9] = gl
    out[10] = (inpos / nbars * nd) if nbars > 0 else 0.0
    out[11] = sB; out[12] = sB2; out[13] = sSB; out[14] = n1; out[15] = n2
    out[16] = a1; out[17] = a2; out[18] = b1; out[19] = b2
    return out


@njit(parallel=True, cache=True)
def _sweep_intra(op, cl, pc, rsiB, sxB, stB, inwin, lastwin, day, mode, dip_trend, comm, split_key, tick, rsi_entries, hold_bars_arr):
    nR = rsiB.shape[0]; nE = len(rsi_entries); nX = sxB.shape[0]; nT = stB.shape[0]; nH = len(hold_bars_arr)
    N = nR * nE * nX * nT * nH
    out = np.zeros((N, 20))
    for k in prange(N):
        h = k % nH; r = k // nH
        t = r % nT; r //= nT
        x = r % nX; r //= nX
        e = r % nE; r //= nE
        ri = r
        o = _core_intra(op, cl, pc, rsiB[ri], sxB[x], stB[t], inwin, lastwin, day, mode, rsi_entries[e], dip_trend,
                        hold_bars_arr[h], comm, split_key, tick)
        for j in range(20):
            out[k, j] = o[j]
    return out


class IMarket:
    def __init__(self, traded, proxy, tf="60", bars_per_day=7):
        t = load_tv(traded, tf)
        p = load_tv(proxy, tf)
        self.bpd = bars_per_day
        # proxy aligned on the traded timestamps
        pc = p["c"].reindex(t.index)
        assert pc.notna().mean() > 0.999, f"proxy {proxy} misses {pc.isna().sum()} bars of {traded}"
        pc = pc.ffill().bfill()
        self.t = t
        self.op = t["o"].values.astype(np.float64)
        self.cl = t["c"].values.astype(np.float64)
        self.pc = pc.values.astype(np.float64)
        self.day = t["key"].values.astype(np.int64)
        self.dates = t["date"].values
        # dip signals on the CHART-timeframe proxy series (use the proxy's own full history for warm-up)
        pfull = p["c"].values.astype(np.float64)
        pidx = p.index
        def al(a):
            return pd.Series(a, index=pidx).reindex(t.index).ffill().values.astype(np.float64)
        self.rsi = np.stack([al(rsi_wilder(pfull, int(n))) for n in RSI_LENS])
        self.sx = np.stack([al(sma(pfull, int(n))) for n in EXIT_LENS])
        # macro trend: daily proxy closes (Yahoo split-adjusted == TradingView), SMA through the PREVIOUS day
        dd = pd.read_csv(rf"{DATA}\yf_{proxy}_1D.csv", index_col="date", parse_dates=True)["Close"]
        dvals = dd.values.astype(np.float64)
        dpos = np.searchsorted(dd.index.values, t["date"].values, side="left") - 1     # previous trading day's index
        st = []
        for n in TREND_LENS:
            s = sma(dvals, int(n))
            arr = np.where(dpos >= 0, s[np.clip(dpos, 0, None)], np.nan)
            st.append(arr)
        self.st = np.stack(st)
        self.traded, self.proxy = traded, proxy

    def window(self, win):
        s = pd.Timestamp(win[0]); e = pd.Timestamp(win[1])
        d = pd.DatetimeIndex(self.dates)
        inwin = ((d >= s) & (d <= e)).astype(np.bool_)
        last = np.zeros(len(d), dtype=np.bool_)
        if inwin.any():
            idx = np.where(inwin)[0]
            if d[idx[-1]].normalize() == e.normalize():
                last[idx[-1]] = True      # last bar of the end date (session.islastbar)
        return inwin, last


def run_one(m, win, mode, rsi_len, rsi_entry, exit_len, trend_len, max_hold_days, dip_trend=True, comm=0.001, hold_bars=None):
    inwin, last = m.window(win)
    ri = int(np.where(RSI_LENS == rsi_len)[0][0]); xi = int(np.where(EXIT_LENS == exit_len)[0][0]); ti = int(np.where(TREND_LENS == trend_len)[0][0])
    hb = int(max_hold_days * m.bpd) if hold_bars is None else int(hold_bars)
    o = _core_intra(m.op, m.cl, m.pc, m.rsi[ri], m.sx[xi], m.st[ti], inwin, last, m.day, mode, float(rsi_entry), bool(dip_trend), hb, float(comm), 20160101, 0.01)
    return metrics_from_out(o)


def metrics_from_out(o):
    nd, E, mdd, sS, sS2, sNeg, tr, w, gp, gl, ip, sB, sB2, sSB = o[:14]
    mean = sS / nd; sd = math.sqrt(max(sS2 / nd - mean * mean, 0)); dn = math.sqrt(sNeg / nd)
    vS = nd * sS2 - sS * sS; vB = nd * sB2 - sB * sB
    cagr = E ** (252.0 / nd) - 1
    return dict(n=int(nd), final=E, cagr=cagr, mdd=mdd, sharpe=mean / sd * math.sqrt(252) if sd > 0 else float("nan"),
                sortino=mean / dn * math.sqrt(252) if dn > 0 else float("nan"), calmar=cagr / mdd if mdd > 0 else float("nan"),
                trades=int(tr), win=w / tr if tr else float("nan"), pf=gp / gl if gl > 0 else float("nan"), exposure=ip / nd,
                corr=(nd * sSB - sS * sB) / math.sqrt(vS * vB) if vS > 0 and vB > 0 else float("nan"))


def sweep_intra(m, mode, dip_trend=True, comm=0.001, win=TRAIN, tick=0.01):
    assert pd.Timestamp(win[1]) <= pd.Timestamp(TRAIN[1]) and pd.Timestamp(win[0]) >= pd.Timestamp(TRAIN[0]), "Train only"
    inwin, last = m.window(win)
    hb = (MAX_HOLDS * m.bpd).astype(np.int64)
    out = _sweep_intra(m.op, m.cl, m.pc, m.rsi, m.sx, m.st, inwin, last, m.day, mode, bool(dip_trend), float(comm), 20160101, tick,
                       RSI_ENTRIES.astype(np.float64), hb)
    return derive(out), out
