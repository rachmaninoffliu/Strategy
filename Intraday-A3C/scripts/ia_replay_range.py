"""
Replay the most recent sessions through the live code path on the IEX feed (what the runner would have seen) and compare with the research version on SIP bars:
the same engine, the same 2026 models (trained on years < 2026, so every day here is out of sample), settled on the consolidated tape and the official close.

  python ia_replay_range.py [n_sessions=15] [n_with_quotes=3]       -> reports/replay_recent.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_data as D
import ia_live as L


def main(n: int = 15, n_quotes: int = 3):
    cfg = L.Cfg(feed="iex")
    costs = L.load_costs(cfg)
    data = L.Data()
    days = [d for d in D.load_daily().query("symbol == 'TSLA'").date.sort_values().tolist()][-n:]
    models = {s: L.ensure_models(s, days[-1].year) for s in cfg.symbols}
    hist = {f: {s: L.History(s, f, top_up=False) for s in cfg.symbols} for f in ("iex", "sip")}
    rows = []
    for i, d in enumerate(days):
        q = i >= len(days) - n_quotes
        a = L.replay_day(d, cfg, models, costs, 100_701.79, data, "api", settle_quotes=q, hist=hist["iex"])
        b = L.replay_day(d, L.Cfg(feed="sip"), models, costs, 100_701.79, None, "parquet", hist=hist["sip"])
        for s in cfg.symbols:
            if not a["res"] or s not in a["res"] or s not in b["res"]:
                continue
            ea, eb = a["store"].read_rows(d), b["store"].read_rows(d)
            ma = ea[(ea.sym == s) & ea.edge.notna()][["j", "edge"]].merge(eb[(eb.sym == s) & eb.edge.notna()][["j", "edge"]], on="j", suffixes=("_iex", "_sip"))
            rows.append(dict(date=d.date(), sym=s, iex_net_bp=a["res"][s]["net_bp"], sip_net_bp=b["res"][s]["net_bp"], iex_net_usd=a["res"][s]["net_usd"], sip_net_usd=b["res"][s]["net_usd"],
                             iex_lots=a["res"][s]["lots"], sip_lots=b["res"][s]["lots"], edge_corr=float(np.corrcoef(ma.edge_iex, ma.edge_sip)[0, 1]),
                             realised_cost_bp=a["res"][s].get("realised_cost_bp"), half_spread_bp=a["res"][s].get("half_spread_bp"), model_cost_bp=costs[s]))
        print(d.date(), {s: (round(a["res"][s]["net_bp"], 1), round(b["res"][s]["net_bp"], 1)) for s in cfg.symbols if a["res"] and s in a["res"] and s in b["res"]}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D.REPORTS, "replay_recent.csv"), index=False)
    g = df.groupby("date")[["iex_net_usd", "sip_net_usd"]].sum()
    print(f"\n{len(g)} sessions | edge corr IEX vs SIP: mean {df.edge_corr.mean():.3f} min {df.edge_corr.min():.3f}")
    print(f"net USD on {100_701.79:,.0f}: IEX-signal {g.iex_net_usd.sum():+,.0f} | SIP-signal {g.sip_net_usd.sum():+,.0f} | daily corr of the two {g.corr().iloc[0, 1]:.2f}")
    print(f"lots traded per product-day: IEX {df.iex_lots.mean():.1f}, SIP {df.sip_lots.mean():.1f}")
    qd = df.dropna(subset=["realised_cost_bp"])
    if len(qd):
        print(f"realised cost vs bar close (last {n_quotes} sessions): {qd.realised_cost_bp.mean():.2f} bp, half quoted spread {qd.half_spread_bp.mean():.2f} bp, model {qd.model_cost_bp.mean():.2f} bp")
        print(qd.groupby("sym")[["realised_cost_bp", "half_spread_bp", "model_cost_bp"]].mean().round(2).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
