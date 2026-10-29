"""
Integration check of the three live engines against the real Alpaca PAPER account in DRY RUN (read-only: account, clock, positions, assets, live bars - nothing is armed, no order
can be sent). Trains the prediction strategy's live model first. Prints the pre-flight table and the plan for the newest bar.
"""
import os

from common import ROOT, ResidualMeanReversion, ResidualMomentum, ResidualPrediction, S, log, pd

from resarb import live as LV  # noqa: E402

for cls in (ResidualMomentum, ResidualMeanReversion, ResidualPrediction):
    st = cls(None, os.path.join(ROOT, cls.KEY))
    eng = LV.LiveEngine(st, LV.LiveSettings(paper=True, armed=False), log=lambda m, k="info": log(f"   [{k}] {m}"))
    log(f"===== {cls.NAME} (feed {st.v['feed']}, lag {st.lag}) =====")
    if cls is ResidualPrediction:
        end = pd.Timestamp.now()
        ctx = S.context(st, start=(end - pd.DateOffset(years=4)).strftime("%Y-%m-%d"), log=log, cache=False)
        st.fit_latest(ctx, log)
    for c in eng.preflight():
        log(f"   {c.level.upper():4s} {c.name}: {c.msg}")
    assert not eng.live.armed
    plan = eng.plan(force=True)
    b = plan.book
    stocks = b[b.kind == "stock"] if len(b) else b
    log(f"   plan on {plan.data_bar:%Y-%m-%d %H:%M} data: {int((stocks.tgt_qty > 0).sum())} long / {int((stocks.tgt_qty < 0).sum())} short, {len(plan.orders)} orders "
        f"(notional ${plan.orders.notional.sum():,.0f}), capital ${plan.capital:,.0f}, gross {plan.expo['gross_stocks']:.2f}x + ETF {plan.expo['gross_etf']:.2f}x, "
        f"beta after hedge {plan.expo['beta_after_hedge']:+.3f}")
    for n in plan.notes:
        log(f"   note: {n[:200]}")
    print(plan.orders.head(8).to_string(index=False), flush=True)
