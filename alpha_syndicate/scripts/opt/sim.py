"""Exact replica of the Diversifier strategy (Dip / Trend / Trend + Dip) for fast sweeps.

Semantics copied from the Pine script:
  * signals on the PROXY close of bar i; orders fill at the NEXT bar's OPEN of the traded asset
  * held = i - (bar index of the entry FILL); timeUp = held >= maxHold
  * inDip bookkeeping for Trend + Dip, dipTrend only used by Dip mode
  * window: entries only if inWin and not lastBarWin; an open trade is liquidated at the CLOSE of the last window bar
  * 100% of equity, commission = `comm` per side on notional, slippage 0
Metrics follow the research script's f_stats / the dashboard: compounded DAILY returns, CAGR = eq^(252/n)-1,
population-stdev Sharpe, downside-deviation Sortino, MDD peak-to-trough of the compounded series.
"""
import numpy as np
import pandas as pd
from numba import njit, prange

DATA = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt"
MODE_DIP, MODE_TREND, MODE_TD = 0, 1, 2
TRAIN = ("2010-02-11", "2021-12-31")
TEST = ("2022-01-01", "2099-12-31")
FULL = ("2010-02-11", "2099-12-31")

RSI_LENS = np.arange(2, 15)             # 13
RSI_ENTRIES = np.arange(10, 41, 2.0)    # 16
EXIT_LENS = np.arange(3, 21)            # 18
TREND_LENS = np.arange(50, 251, 25)     # 9
MAX_HOLDS = np.arange(2, 21, 2)         # 10


def load(sym, src="yf"):
    p = f"{DATA}\\{src}_{sym}_1D.csv"
    df = pd.read_csv(p, index_col="date", parse_dates=True)
    return df[["Open", "High", "Low", "Close"]].astype(float)


@njit(cache=True)
def rsi_wilder(x, n):
    """TradingView ta.rsi: RMA(gain)/RMA(loss) seeded with the SMA of the first n changes."""
    m = len(x)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    up = 0.0
    dn = 0.0
    for i in range(1, n + 1):
        d = x[i] - x[i - 1]
        up += d if d > 0 else 0.0
        dn += -d if d < 0 else 0.0
    up /= n
    dn /= n
    out[n] = 100.0 if dn == 0 else (0.0 if up == 0 else 100.0 - 100.0 / (1.0 + up / dn))
    for i in range(n + 1, m):
        d = x[i] - x[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        up = (up * (n - 1) + g) / n
        dn = (dn * (n - 1) + l) / n
        out[i] = 100.0 if dn == 0 else (0.0 if up == 0 else 100.0 - 100.0 / (1.0 + up / dn))
    return out


@njit(cache=True)
def sma(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    s = 0.0
    for i in range(m):
        s += x[i]
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


class Market:
    """Traded asset + proxy aligned on the traded asset's dates, with all indicator banks precomputed."""

    def __init__(self, traded, proxy, src="yf", traded_src=None):
        t = load(traded, traded_src or src)
        p = load(proxy, src)
        full_p = p["Close"].values
        self.dates = t.index
        self.op = t["Open"].values.copy()
        self.cl = t["Close"].values.copy()
        pc_full = p["Close"]
        self.pc = pc_full.reindex(self.dates).values.copy()
        assert not np.isnan(self.pc).any(), f"proxy {proxy} missing dates for {traded}"
        # indicators on the FULL proxy history (warm-up), then aligned to the traded dates
        idx = p.index
        def al(a):
            return pd.Series(a, index=idx).reindex(self.dates).values.copy()
        self.rsi = np.stack([al(rsi_wilder(full_p, int(n))) for n in RSI_LENS])
        self.sx = np.stack([al(sma(full_p, int(n))) for n in EXIT_LENS])
        self.st = np.stack([al(sma(full_p, int(n))) for n in TREND_LENS])
        self.traded, self.proxy = traded, proxy

    def window(self, win):
        d = self.dates
        s, e = pd.Timestamp(win[0]), pd.Timestamp(win[1])
        inwin = (d >= s) & (d <= e)
        last = np.zeros(len(d), dtype=np.bool_)
        if inwin.any():
            idx = np.where(inwin)[0]
            # last window bar only counts when the end date is actually reached in the data
            if d[idx[-1]] == e:
                last[idx[-1]] = True
        return inwin.astype(np.bool_), last


@njit(cache=True)
def _core(op, cl, pc, rsi, sx, st, inwin, lastwin, mode, rsi_entry, dip_trend, max_hold, comm, split, want_series, tick, qmode):
    n = len(op)
    pos = 0
    shares = 0.0
    cash = 0.0
    entry_bar = -1
    pending = 0
    in_dip = False
    E = 1.0
    prevE = 1.0
    entryE = 1.0
    peak = 1.0
    mdd = 0.0
    nd = 0
    sS = 0.0
    sS2 = 0.0
    sNeg = 0.0
    sB = 0.0
    sB2 = 0.0
    sSB = 0.0
    trades = 0
    wins = 0
    gp = 0.0
    gl = 0.0
    inpos_days = 0
    # half-split sums (for stability)
    n1 = 0
    a1 = 0.0
    a2 = 0.0
    n2 = 0
    b1 = 0.0
    b2 = 0.0
    prevCl = cl[0]
    rets = np.zeros(n) if want_series else np.zeros(1)
    for i in range(n):
        just_entered = False
        just_exited = False
        pos_prev = pos
        # 1) fills at the open (prices rounded to the tick, share-based so rounding is modelled)
        if pending == 1:
            fp = np.round(op[i] / tick) * tick if tick > 0 else op[i]
            entryE = E
            if qmode == 2:
                shares = E / (prevCl * (1.0 + comm))
            elif qmode == 1:
                shares = min(E / prevCl, E / (fp * (1.0 + comm)))
            else:
                shares = E / (fp * (1.0 + comm))
            cash = E - shares * fp * (1.0 + comm)
            pos = 1
            entry_bar = i
            just_entered = True
        elif pending == 2:
            fp = np.round(op[i] / tick) * tick if tick > 0 else op[i]
            E = cash + shares * fp * (1.0 - comm)
            shares = 0.0
            cash = 0.0
            pos = 0
            just_exited = True
            trades += 1
            pnl = E - entryE
            if pnl > 0:
                wins += 1
                gp += pnl
            else:
                gl += -pnl
        pending = 0
        # 2) mark to market to the close
        if pos == 1:
            E = cash + shares * cl[i]
        # 3) decision at the close (signals on the proxy)
        held = (i - entry_bar) if pos == 1 else 0
        time_up = (max_hold > 0) and (held >= max_hold)
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
            # window end / outside window: liquidate AT THE CLOSE
            fpc = np.round(cl[i] / tick) * tick if tick > 0 else cl[i]
            E = cash + shares * fpc * (1.0 - comm)
            shares = 0.0
            cash = 0.0
            pos = 0
            trades += 1
            pnl = E - entryE
            if pnl > 0:
                wins += 1
                gp += pnl
            else:
                gl += -pnl
            pending = 0
        # 4) accounting (window days only)
        if inwin[i]:
            r = E / prevE - 1.0
            rb = cl[i] / prevCl - 1.0
            nd += 1
            sS += r
            sS2 += r * r
            if r < 0:
                sNeg += r * r
            sB += rb
            sB2 += rb * rb
            sSB += r * rb
            if E > peak:
                peak = E
            dd = (peak - E) / peak
            if dd > mdd:
                mdd = dd
            if pos_prev == 1:
                inpos_days += 1
            if i < split:
                n1 += 1
                a1 += r
                a2 += r * r
            else:
                n2 += 1
                b1 += r
                b2 += r * r
            if want_series:
                rets[i] = r
        prevE = E
        prevCl = cl[i]
    # first-bar guard: prevCl initial equals cl[0] so day-0 returns are 0
    out = np.zeros(16)
    out[0] = nd
    out[1] = E
    out[2] = mdd
    out[3] = sS
    out[4] = sS2
    out[5] = sNeg
    out[6] = trades
    out[7] = wins
    out[8] = gp
    out[9] = gl
    out[10] = inpos_days
    out[11] = sB
    out[12] = sB2
    out[13] = sSB
    out[14] = n1
    out[15] = n2
    return out, rets, a1, a2, b1, b2


def metrics(o, a1=None, a2=None, b1=None, b2=None):
    nd, E, mdd, sS, sS2, sNeg, trades, wins, gp, gl, inpos, sB, sB2, sSB = o[:14]
    if nd < 3:
        return dict(n=nd)
    mean = sS / nd
    sd = np.sqrt(max(sS2 / nd - mean * mean, 0.0))
    sharpe = mean / sd * np.sqrt(252) if sd > 0 else np.nan
    dn = np.sqrt(sNeg / nd)
    sortino = mean / dn * np.sqrt(252) if dn > 0 else np.nan
    cagr = E ** (252.0 / nd) - 1.0
    calmar = cagr / mdd if mdd > 0 else np.nan
    vS = nd * sS2 - sS * sS
    vB = nd * sB2 - sB * sB
    corr = (nd * sSB - sS * sB) / np.sqrt(vS * vB) if vS > 0 and vB > 0 else np.nan
    return dict(n=int(nd), final=E, cagr=cagr, mdd=mdd, sharpe=sharpe, sortino=sortino, calmar=calmar,
                trades=int(trades), win=(wins / trades if trades else np.nan),
                pf=(gp / gl if gl > 0 else np.nan), exposure=inpos / nd, corr=corr)


def run_one(m, win, mode, rsi_len, rsi_entry, exit_len, trend_len, max_hold, dip_trend=True, comm=0.001, split=0, series=False, tick=0.01, qmode=2):
    inwin, last = m.window(win)
    ri = int(np.where(RSI_LENS == rsi_len)[0][0])
    xi = int(np.where(EXIT_LENS == exit_len)[0][0])
    ti = int(np.where(TREND_LENS == trend_len)[0][0])
    o, rets, a1, a2, b1, b2 = _core(m.op, m.cl, m.pc, m.rsi[ri], m.sx[xi], m.st[ti], inwin, last, mode,
                                    float(rsi_entry), bool(dip_trend), int(max_hold), float(comm), int(split), series, float(tick), int(qmode))
    mt = metrics(o, a1, a2, b1, b2)
    return (mt, rets) if series else mt
