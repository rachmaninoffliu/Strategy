"""
DA: overnight gaps that could be unadjusted distributions. Every member-day whose open is > 8% below the previous close beyond the market move, with the first bar's volume
relative to its 20-day average (an earnings gap trades several times normal volume; a mechanical ex-date usually does not) and whether a fix is already applied.
Review the low-volume ones by hand; real ones go to data.MANUAL_FIXES.
"""
import numpy as np

from common import OUT, log, pd

from resarb import data as D  # noqa: E402

u = D.load_universe()
syms = sorted(set(u[u["rank"] <= 100].symbol))
spy = D.load_bars("SPY", fix=False)
sd = spy.groupby(spy.index.normalize()).agg(o=("open", "first"), c=("close", "last"))
spy_on = sd.o / sd.c.shift() - 1
fixes = pd.read_parquet(D.FIX_FILE) if __import__("os").path.exists(D.FIX_FILE) else pd.DataFrame(columns=["symbol", "ex_date"])
fixed = set(zip(fixes.symbol, pd.to_datetime(fixes.ex_date)))
rows = []
for s in syms:
    x = D.load_bars(s, fix=False)
    if not len(x):
        continue
    day = x.index.normalize()
    g = x.groupby(day).agg(o=("open", "first"), c=("close", "last"))
    v0 = x[x.index.hour * 60 + x.index.minute == 570].volume
    v0.index = v0.index.normalize()
    rel_v = v0 / v0.rolling(20, min_periods=10).mean().shift()
    on = g.o / g.c.shift() - 1
    ex = (1 + on) / (1 + spy_on.reindex(on.index).fillna(0)) - 1
    for d, v in ex[ex < -0.08].items():
        months = u[(u.symbol == s) & (u["rank"] <= 100)].month
        member = ((months <= d) & (months + pd.offsets.MonthEnd(0) >= d)).any()
        rows.append(dict(symbol=s, day=d.date(), gap_vs_spy=v, first_bar_vol_x=float(rel_v.get(d, np.nan)), member=bool(member), fixed=(s, d) in fixed))
df = pd.DataFrame(rows).sort_values("first_bar_vol_x")
df.to_csv(f"{OUT}/r0b_gaps.csv", index=False)
pd.set_option("display.width", 200)
print(df[df.member].round(3).to_string(index=False))
log(f"{len(df)} gaps < -8% vs SPY ({int(df.member.sum())} while a top-100 member); {int((df.member & (df.first_bar_vol_x < 1.5)).sum())} with first-bar volume < 1.5x normal")
