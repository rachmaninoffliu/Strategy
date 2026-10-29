"""
Triage 2: the mean-reversion legs.
  A. Intraday reversal of attention (Berkman-Koch-Tuttle-Zhang 2012): yesterday's biggest movers are bought at the open (retail attention) and revert during the day.
     Short at the open (MOO) yesterday's top-k winners, cover at the close (MOC); and the mirror for losers.
  B. The full attention round trip: long the day's winners close -> open, then short the same names open -> close (flip in the opening auction).
  C. Connors RSI(2) swing (long-only, close > 200-day mean, enter RSI2 < thr at the close, exit at the close when close > 5-day mean or after H days), K slots, most oversold first.
  D. SPY / QQQ oversold rules (RSI2, IBS) - market-level mean reversion.
Same timing and cost rules as mr_triage.py.
"""
import numpy as np
import pandas as pd

from mr_data import Daily, snapshot_prices, stats, universe

AUC = 1.0
START = "2016-09-01"
d = Daily()
O, H, L, C = d.open, d.high, d.low, d.close
T, N = C.shape
S45 = snapshot_prices(d, 1545)
dates = d.dates
t0 = int(np.searchsorted(dates, pd.Timestamp(START)))
rows = []


def add(r, name, **kw):
    s = pd.Series(r, index=dates)[t0:]
    st = stats(s, name)
    st.update(kw)
    rows.append(st)


def pick(sig, mask, k, lowest=True):
    x = np.where(mask & np.isfinite(sig), sig, np.nan)
    x = x if lowest else -x
    r = pd.DataFrame(x).rank(axis=1, method="first").to_numpy()
    return (r <= k) & np.isfinite(x)


with np.errstate(invalid="ignore", divide="ignore"):
    prevC = np.vstack([np.full((1, N), np.nan), C[:-1]])
    nextO = np.vstack([O[1:], np.full((1, N), np.nan)])
    day = C / prevC - 1
    day45 = S45 / prevC - 1
    r_on = nextO / C - 1
    r_id = C / O - 1


def shift(x, n=1):
    return np.r_[np.zeros(n), x[:-n]]


add(np.r_[np.nan, C[1:, d.j("SPY")] / C[:-1, d.j("SPY")] - 1], "SPY buy & hold")
for n in (100, 500):
    U = universe(d, n)
    M = U & np.isfinite(S45) if n == 100 else U
    sig_today = day45 if n == 100 else day                                                     # chosen before the MOC cut-off (top-500: the close, flagged)
    sig_yday = np.vstack([np.full((1, N), np.nan), day[:-1]])                                   # yesterday's full-day return, known before today's open
    Mprev = np.vstack([np.zeros((1, N), bool), M[:-1]]) & U
    for k in (5, 10, 20):
        w = pick(sig_yday, Mprev, k, lowest=False)
        lo = pick(sig_yday, Mprev, k, lowest=True)
        short_w = (-np.where(w & np.isfinite(r_id), r_id, 0).sum(1) - w.sum(1) * 2 * AUC / 1e4) / k
        long_l = (np.where(lo & np.isfinite(r_id), r_id, 0).sum(1) - lo.sum(1) * 2 * AUC / 1e4) / k
        add(short_w, f"top{n} A: short yesterday's {k} winners open->close", fam="A", n=n, k=k)
        add(long_l, f"top{n} A: long yesterday's {k} losers open->close", fam="A", n=n, k=k)
        add(0.5 * short_w + 0.5 * long_l, f"top{n} A: L/S intraday reversal {k}+{k}", fam="A", n=n, k=k)
        wt = pick(sig_today, M, k, lowest=False)
        on_long = (np.where(wt & np.isfinite(r_on), r_on, 0).sum(1) - wt.sum(1) * 2 * AUC / 1e4) / k        # realised at the next open
        on_long = shift(on_long)
        wt_prev = np.vstack([np.zeros((1, N), bool), wt[:-1]])
        id_short = (-np.where(wt_prev & np.isfinite(r_id), r_id, 0).sum(1) - wt_prev.sum(1) * 2 * AUC / 1e4) / k
        add(on_long, f"top{n} B: long today's {k} winners close->open", fam="B", n=n, k=k)
        add(on_long + id_short, f"top{n} B: round trip (long overnight, short intraday) {k}", fam="B", n=n, k=k)


def rsi(c, n=2):
    dlt = np.diff(c, axis=0, prepend=np.nan)
    up = pd.DataFrame(np.where(dlt > 0, dlt, 0.0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.DataFrame(np.where(dlt < 0, -dlt, 0.0)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 - 100 / (1 + up / dn)


def swing(sig_entry, entry_ok, exit_fn, k, hmax, uni_mask, cost_bp=2 * AUC):
    """Daily slot simulator: exits at the close first (rule or time stop), then entries at the close, most extreme signal first, up to k slots of 1/k equity each."""
    held = {}
    ret = np.zeros(T)
    trades = 0
    for t in range(t0, T - 1):
        if held:
            js = np.array(list(held))
            ret[t] += np.nansum(C[t, js] / C[t - 1, js] - 1) / k if t > t0 else 0.0
        out = [j for j, t_in in held.items() if exit_fn(t, j) or t - t_in >= hmax or not np.isfinite(C[t, j])]
        for j in out:
            held.pop(j)
            ret[t] -= cost_bp / 2 / 1e4 / k
        free = k - len(held)
        if free > 0:
            cand = np.where(entry_ok[t] & uni_mask[t] & np.isfinite(sig_entry[t]))[0]
            cand = [j for j in cand[np.argsort(sig_entry[t, cand])] if j not in held][:free]
            for j in cand:
                held[j] = t
                ret[t] -= cost_bp / 2 / 1e4 / k
                trades += 1
    return ret, trades


with np.errstate(invalid="ignore"):
    sma200 = pd.DataFrame(C).rolling(200, min_periods=150).mean().to_numpy()
    sma5 = pd.DataFrame(C).rolling(5).mean().to_numpy()
    R2 = rsi(C, 2)
    # RSI2 as of 15:45: today's change measured to the 15:45 price (what the MOC decision can see)
    C45 = np.where(np.isfinite(S45), S45, C)
for n in (100, 500):
    U = universe(d, n)
    sigR = R2 if n == 500 else rsi(np.vstack([C[:-1], C45[-1:]]), 2)                           # approximate: RSI on closes (top-500) - flagged
    for thr, k, h in ((5, 10, 10), (10, 10, 10), (10, 20, 10), (20, 20, 10), (5, 5, 10)):
        ok = (sigR < thr) & (C > sma200)
        r, ntr = swing(sigR, ok, lambda t, j: C[t, j] > sma5[t, j], k, h, U)
        add(r, f"top{n} C: RSI2<{thr}, >SMA200, exit >SMA5, {k} slots, max {h}d" + (" [close signal]" if n == 500 else ""), fam="C", n=n, k=k, trades=ntr)
for etf in ("SPY", "QQQ", "IWM"):
    j = d.j(etf)
    c = C[:, j]
    ibs = (c - L[:, j]) / (H[:, j] - L[:, j])
    r1 = np.r_[c[1:] / c[:-1] - 1, np.nan]
    for name, sig in (("RSI2<10", R2[:, j] < 10), ("IBS<0.2", ibs < 0.2), ("RSI2<10 & >SMA200", (R2[:, j] < 10) & (c > sma200[:, j]))):
        pos = np.zeros(T)
        inpos = False
        for t in range(T - 1):
            if not inpos and sig[t]:
                inpos = True
            elif inpos and (c[t] > sma5[t, j] or (name.startswith("IBS") and ibs[t] > 0.8)):
                inpos = False
            pos[t] = 1.0 if inpos else 0.0
        trn = np.abs(np.diff(np.r_[0, pos]))
        r = shift(np.nan_to_num(pos * r1)) - trn * AUC / 1e4
        add(r, f"D: {etf} {name} (exit > SMA5{' / IBS>0.8' if name.startswith('IBS') else ''})", fam="D", exposure=pos[t0:].mean())
df = pd.DataFrame(rows)
df.to_csv("reports/mr_triage2.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
cols = ["name", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all"]
print(df[cols + [c for c in ("trades", "exposure") if c in df]].sort_values("sh_is", ascending=False).round(3).to_string(index=False))
