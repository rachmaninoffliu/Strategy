"""DA check: are the residuals really stripped of market / sector moves? Compare raw and residual 15-min returns' correlation with SPY / the sector ETF, intraday vs overnight bars,
and the 15-min vs daily R^2 (Epps effect)."""
import numpy as np

from common import ResidualMeanReversion, S, log, pd

from resarb import data as D  # noqa: E402

st = ResidualMeanReversion({"universe_n": 100, "start": "2016-09-01", "end": "2017-09-01"})
ctx = S.context(st, end="2017-09-01", log=log, cache=False)
p, res = ctx.p, ctx.res
m = p.col["SPY"]
first = p.bar == 0
rows = []
for s in ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "AMZN", "BAC", "WFC", "INTC", "PFE"]:
    if s not in p.col:
        continue
    j = p.col[s]
    sc = int(res.sector_col[j])
    ok = ctx.member[:, j] & np.isfinite(res.r_cc[:, j]) & np.isfinite(res.r_cc[:, m])
    for name, mask in (("intraday bars", ok & ~first), ("first bar (overnight)", ok & first)):
        r, e, mk = res.r_cc[mask, j], res.e_cc[mask, j], res.r_cc[mask, m]
        se = res.r_cc[mask, sc] if sc >= 0 else np.full(mask.sum(), np.nan)
        b = np.polyfit(mk, r, 1)[0]
        rows.append(dict(sym=s, bars=name, n=int(mask.sum()), corr_raw_spy=np.corrcoef(r, mk)[0, 1], corr_res_spy=np.corrcoef(e, mk)[0, 1],
                         corr_res_sector=np.corrcoef(e, se)[0, 1] if np.isfinite(se).all() else np.nan, sd_raw_bp=r.std() * 1e4, sd_res_bp=e.std() * 1e4,
                         beta_15m=b, beta_daily_mean=float(np.nanmean(res.b_m[np.unique(p.day[mask]), j]))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
dr = (p.daily_close()[1:] / p.daily_close()[:-1] - 1)
print("\nmean daily R2 of the regressions (members):", np.nanmean(np.where(ctx.member_d, res.r2, np.nan)).round(3))
