"""
QR: which features predict the forward RESIDUAL return? Rank IC of every feature in resarb/features.py at horizons of 1, 4, 13, 26 and 130 fifteen-minute bars, measured from the bar
the order would fill in (lag 0 = real-time feed; lag 1 = the free plan's delayed SIP), in-sample (2016-09 .. 2022) and out-of-sample (2023 ..). Top-100 universe, ETF factor model.

  python r1_feature_ic.py [lag] [model]
"""
import sys

from common import OUT, SPLIT, START, ResidualMeanReversion, S, log, pd  # noqa: E402

from resarb import features as X  # noqa: E402
from resarb import research as R  # noqa: E402

lag = int(sys.argv[1]) if len(sys.argv) > 1 else 0
model = sys.argv[2] if len(sys.argv) > 2 else "etf"
strat = ResidualMeanReversion({"universe_n": 100, "feed": "iex" if lag == 0 else "sip_delayed", "factor_model": model, "beta_days": 126})
ctx = S.context(strat, log=log)
log(ctx.p.describe())
df = R.feature_ic(ctx, list(X.FEATURES), START, SPLIT, every=2, log=log)
df.to_csv(f"{OUT}/r1_feature_ic_lag{lag}_{model}.csv", index=False)
pd.set_option("display.width", 250)
for h in sorted(df.horizon_bars.unique()):
    d = df[df.horizon_bars == h].sort_values("IS_t")
    print(f"\n=== horizon {h} bars (lag {lag}, {model}) ===")
    print(d[["feature", "IS_ic", "IS_t", "OOS_ic", "OOS_t"]].round(4).to_string(index=False))
