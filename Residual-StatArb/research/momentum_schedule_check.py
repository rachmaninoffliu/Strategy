"""
DRY check of the ResMomentum live schedule on real data (read-only Alpaca calls, never armed): the engine must decide at 15:45 on the last NYSE session of a month and on no other
day, and its targets must equal the backtest's decision for the same bar (from flat). Also prints the upcoming rebalance dates.
"""
import os

import numpy as np

from common import ROOT, ResidualMomentum, S, log, pd

from resarb import data as D  # noqa: E402
from resarb import live as LV  # noqa: E402
from resarb.portfolio import approach, decide, hedge_etf  # noqa: E402

st = ResidualMomentum(None, os.path.join(ROOT, "ResMomentum"))
ends = D.month_end_sessions("2026-09-01", "2027-03-31")
log("month-end sessions (rebalance days): " + ", ".join(f"{d:%a %Y-%m-%d}" for d in ends))
eng = LV.LiveEngine(st, LV.LiveSettings(paper=True, armed=False), log=lambda m, k="info": None)
eng.connect()
eng.positions = lambda: {}                                                                    # preview from flat (the shared account's holdings are not this strategy's)
for day in ("2026-09-29", "2026-09-30"):
    now = pd.Timestamp(f"{day} 15:45:30")
    plan = eng.plan(now)
    b = plan.book
    log(f"{day} 15:45: decision={plan.decision} data bar {plan.data_bar:%H:%M} orders {len(plan.orders)}")
    if plan.decision:
        ctx = S.context(st, start="2025-06-01", end=(pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), log=lambda m: None, cache=False)
        p = ctx.p
        td = int(np.searchsorted(p.ts, plan.data_bar))
        assert p.ts[td] == plan.data_bar
        assert st.decision_bars(p)[td], "the backtest schedule must also decide on this bar"
        sidx = np.where(~np.isin(p.syms, D.ETFS))[0]
        trad = np.isfinite(p.open[td, sidx])
        w = np.zeros(len(p.syms))
        w[sidx] = approach(np.zeros(len(sidx)), decide(st.score(ctx)[td, sidx], ctx.member[td, sidx], np.zeros(len(sidx)), st.port_cfg(), ctx.res.sector_col[sidx], trad),
                           st.port_cfg().trade_rate, st.port_cfg().gross, trad)
        w[ctx.res.etf_idx] += hedge_etf(w, ctx.res, p.day[td])
        live = b.set_index("symbol").target_w
        bt = pd.Series(w, index=p.syms)
        u = sorted(set(live.index) | set(bt[bt != 0].index))
        diff = float(np.abs(live.reindex(u).fillna(0) - bt.reindex(u).fillna(0)).max())
        stocks = b[b.kind == "stock"]
        log(f"   live vs backtest target weights: max |difference| {diff:.2e} over {len(u)} symbols")
        log(f"   book: {int((stocks.tgt_qty > 0).sum())} long / {int((stocks.tgt_qty < 0).sum())} short, stock gross {plan.expo['gross_stocks']:.2f}x, ETF {plan.expo['gross_etf']:.2f}x, "
            f"{len(plan.orders)} orders, notional ${plan.orders.notional.sum():,.0f} on ${plan.capital:,.0f} capital")
        print(stocks.sort_values("target_w")[["symbol", "score", "target_w", "px", "tgt_qty"]].to_string(index=False), flush=True)
