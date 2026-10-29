"""Valid check: the IBS effect with the information a market-on-close order really has (high / low / price as of 15:45 from the 15-minute bars) vs the textbook version that
decides on the close it trades at."""
import numpy as np
import pandas as pd

from mr_data import D, stats

rows = []
for etf in ("SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"):
    b = D.load_bars(etf)
    day = b.index.normalize()
    g = b.groupby(day)
    daily = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    pre = b[(b.index - b.index.normalize()) <= pd.Timedelta(hours=15, minutes=30)]                 # bars starting <= 15:30 -> data to 15:45
    gp = pre.groupby(pre.index.normalize())
    p45 = pd.DataFrame({"high": gp.high.max(), "low": gp.low.min(), "px": gp.close.last()})
    daily = daily.join(p45, rsuffix="45").loc["2016-09-01":]
    c = daily.close.values
    sma5 = pd.Series(c).rolling(5).mean().values
    r_next = np.r_[c[1:] / c[:-1] - 1, np.nan]
    for label, ibs, cx in (("close (look-ahead)", (daily.close - daily.low) / (daily.high - daily.low), c),
                           ("15:45 (honest)", (daily.px - daily.low45) / (daily.high45 - daily.low45), daily.px.values)):
        ibs = ibs.values
        pos = np.zeros(len(c))
        inpos = False
        for t in range(len(c) - 1):
            if not inpos and ibs[t] < 0.2:
                inpos = True
            elif inpos and (ibs[t] > 0.8 or cx[t] > sma5[t]):
                inpos = False
            pos[t] = float(inpos)
        trn = np.abs(np.diff(np.r_[0, pos]))
        r = pd.Series(np.r_[0, (pos * np.nan_to_num(r_next))[:-1]] - np.r_[0, trn[:-1]] * 1e-4, index=daily.index)
        s = stats(r, f"{etf} IBS<0.2 decided on {label}")
        s["exposure"] = pos.mean()
        rows.append(s)
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df[["name", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "mdd_all", "exposure"]].round(3).to_string(index=False))
