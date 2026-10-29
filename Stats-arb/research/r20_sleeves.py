import sys
from rlib import *
if __name__ == "__main__":
    universe, name = sys.argv[1], sys.argv[2]
    panel = panel_for(universe)
    sig = sig_for(universe, name, VARIANTS[name.split("_")[0]], panel, earn=True)
    spy = S.spy_open_to_open(panel)
    def show(label, **kw):
        base = dict(neutral=False, event_filter=False, earn_filter=False, weight_pow=1.0, shock=False, band=0.2, scale_band=0.15, vol_target=0.10, g_init=1.0, g_min=0.25, g_max=2.0, tc_bps=1.5,
                    borrow_bps=50.0, name_cap=0.03, start="2017-06-01", s_close_long=0.5, s_close_short=0.75)
        p = replace(C.PortCfg(), **{**base, **kw})
        r = S.simulate(sig, panel, p)
        ret, d = r.ret, r.daily
        ma, mi, mo = S.metrics(ret, spy), S.metrics(ret[:"2021-12-31"]), S.metrics(ret["2022-01-01":])
        print(f"{label:46s} Sh {ma['sharpe']:5.2f} (IS {mi['sharpe']:5.2f} OOS {mo['sharpe']:5.2f}) CAGR {ma['cagr']:+6.1%} vol {ma['vol']:5.1%} DD {ma['maxdd']:6.1%} beta {ma.get('beta',0):+.2f} gross {d.gross.mean():.2f} turn {d.turnover.mean():.2f} cost/y {d.cost.mean()*252:.1%} nEAP {d.n_eap.mean():.0f}", flush=True)
        return r
    show("EAP only, fixed gross (g=1 -> long1+hedge1)", w_rev=0.0, w_eap=1.0, vol_target=1.0, g_init=1.0, g_min=1.0)
    show("EAP only, vol target 10%", w_rev=0.0, w_eap=1.0)
    show("EAP only, vol target 10% + shock", w_rev=0.0, w_eap=1.0, shock=True)
    show("EAP cheap pool, vol target 10%", w_rev=0.0, w_eap=1.0, eap_pool="cheap")
    show("reversion only (paper rules, vol target 10%)", w_rev=1.0, w_eap=0.0)
    show("70% EAP / 30% reversion", w_rev=0.3, w_eap=0.7)
    show("EAP only, vol target 15%", w_rev=0.0, w_eap=1.0, vol_target=0.15)
