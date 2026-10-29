"""
DA / QT: what does the FREE real-time feed (IEX) do to each strategy? 2024-01 .. now, top-100 universe.

For each strategy's chosen settings: the score computed from IEX bars vs from SIP bars (rank correlation per bar), and a backtest whose DECISIONS come from IEX scores while the P&L
is marked at SIP prices (an IEX-only backtest would fill at IEX's own stale prints and flatter the strategy - the LevETF lesson). An all-IEX panel uses IEX for the past days too,
while the live engine uses SIP history + today's IEX bars, so these numbers are a pessimistic bound. Also: the delayed-SIP alternative (lag 1) on the same period.

  python r5_feedgap.py
"""
import json
import os

import numpy as np

from common import OUT, ResidualMeanReversion, ResidualMomentum, ResidualPrediction, S, log, pd, row, BT  # noqa: F401

from resarb import data as D  # noqa: E402

START24 = "2024-01-01"
cfgs = json.load(open(os.path.join(OUT, "chosen.json")))                                    # written after r2-r4: {strategy key: settings}
classes = {"ResMomentum": ResidualMomentum, "ResMeanReversion": ResidualMeanReversion, "ResPrediction": ResidualPrediction}
rows = []
for key, vals in cfgs.items():
    cls = classes[key]
    st = cls({**vals, "start": START24, "split": "2100-01-01", "feed": "iex"})
    ctx_s = S.context(st, start=START24, feed="sip", log=log, cache=False)
    ctx_i = S.context(st, start=START24, feed="iex", log=log, cache=False)
    ci = {s: j for j, s in enumerate(ctx_i.p.syms)}
    rows_i = ctx_i.p.ts.get_indexer(ctx_s.p.ts)
    sc_s = st.score(ctx_s)
    sc_i_raw = st.score(ctx_i)
    sc_i = np.full(sc_s.shape, np.nan)
    for j, s in enumerate(ctx_s.p.syms):
        if s in ci:
            ok = rows_i >= 0
            sc_i[ok, j] = sc_i_raw[rows_i[ok], ci[s]]
    t0 = int(np.searchsorted(ctx_s.p.ts, pd.Timestamp(START24) + pd.Timedelta(days=45)))
    from resarb import features as X
    rr = X.row_corr(X.xs_rank(sc_s[t0:], ctx_s.member[t0:]), X.xs_rank(sc_i[t0:], ctx_s.member[t0:]))
    bt_s = BT.run(ctx_s.p, ctx_s.res, sc_s, ctx_s.member, st.decision_bars(ctx_s.p), st.port_cfg(), st.bt_cfg())
    bt_i = BT.run(ctx_s.p, ctx_s.res, sc_i, ctx_s.member, st.decision_bars(ctx_s.p), st.port_cfg(), st.bt_cfg())
    st1 = cls({**vals, "start": START24, "split": "2100-01-01", "feed": "sip_delayed"})
    bt_d = BT.run(ctx_s.p, ctx_s.res, sc_s, ctx_s.member, st1.decision_bars(ctx_s.p), st1.port_cfg(), st1.bt_cfg())
    for name, bt in (("SIP decisions, lag 0 (paid real-time SIP)", bt_s), ("IEX decisions, SIP marks, lag 0 (free IEX)", bt_i), ("SIP decisions, lag 1 (free delayed SIP)", bt_d)):
        rows.append(row(bt, strategy=key, case=name, score_corr=float(np.nanmedian(rr))))
        log(f"{key} | {name}: Sharpe {rows[-1]['sh_all']:.2f} gross {rows[-1]['gross_sh_all']:.2f} | median score corr IEX vs SIP {np.nanmedian(rr):.2f}")
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/r5_feedgap.csv", index=False)
pd.set_option("display.width", 250)
print(df[["strategy", "case", "score_corr", "sh_all", "gross_sh_all", "cagr_is", "turn"]].round(3).to_string(index=False))
