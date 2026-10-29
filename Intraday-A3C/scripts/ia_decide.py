"""Product decision table: for every candidate, the surrogate agent's net result (config ens_k3: average of ridge and GBM edges, trade when |edge| >= 3 x cost), by year, plus cost and capacity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import ia_data as D, ia_study as S, ia_portfolio as P

if __name__ == "__main__":
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    dc = D.load_daily()
    rows, yearly = [], {}
    for sym in D.ALL:
        dd, E = P.edges_for(sym)
        half = float(sp.spread_bp[sym]) / 2
        cost = half + S.SLIP_BP
        net, gross, lots = S.policy_pnl(E["ens"], dd["nxt"], cost, 3.0)
        yrs = dd["days"].year.values
        def sh(m):
            x = net[m]; x = x[np.isfinite(x)]
            return x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 50 and x.std() > 0 else np.nan
        m_all, m_is, m_oos = yrs >= 2019, (yrs >= 2019) & (yrs <= 2022), yrs >= 2023
        per_year = {int(y): sh(yrs == y) for y in range(2019, 2027)}
        yearly[sym] = per_year
        x = net[m_all]
        adv = (dc[dc.symbol == sym].tail(250).eval("close*volume")).median() / 1e6
        rows.append(dict(symbol=sym, spread_bp=sp.spread_bp[sym], cost_bp=cost, adv_musd=adv, sharpe=sh(m_all), t=x.mean() / x.std(ddof=1) * np.sqrt(len(x)), sharpe_2019_22=sh(m_is), sharpe_2023_26=sh(m_oos),
                         yrs_pos=int(sum(v > 0 for v in per_year.values())), net_bp_day=x.mean(), lots_day=lots[m_all].mean(), pt_net_bp=net[m_all].sum() / lots[m_all].sum(), gross_sharpe=(gross[m_all].mean() / gross[m_all].std() * np.sqrt(252))))
    df = pd.DataFrame(rows).set_index("symbol").sort_values("sharpe", ascending=False)
    df.to_csv(os.path.join(D.REPORTS, "decision_table.csv"))
    pd.DataFrame(yearly).T.to_csv(os.path.join(D.REPORTS, "decision_by_year.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df.round(2).to_string())
    print(pd.DataFrame(yearly).T.loc[df.index[:10]].round(2).to_string())
