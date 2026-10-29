"""DA: look for unadjusted distributions in the ETF daily bars - an overnight drop far beyond SPY's with no matching move in the rest of the day."""
import sys

import pandas as pd

from mr_run import market_for, MRModel  # noqa: F401

from resarb.mr.market import load_daily  # noqa: E402
from resarb.mr.models import ETF_UNIVERSE  # noqa: E402

d = load_daily(ETF_UNIVERSE)
px = d.pivot_table(index="date", columns="symbol", values="close")
op = d.pivot_table(index="date", columns="symbol", values="open")
on = op / px.shift() - 1
dev = on.sub(on["SPY"], axis=0)
rows = [(c, t, on.loc[t, c], on.loc[t, "SPY"], px.loc[t, c] / op.loc[t, c] - 1) for c in dev for t in dev.index[dev[c] < float(sys.argv[1]) if len(sys.argv) > 1 else dev[c] < -0.08]]
print(pd.DataFrame(rows, columns=["etf", "date", "overnight", "spy_overnight", "rest_of_day"]).sort_values("overnight").to_string(index=False))
