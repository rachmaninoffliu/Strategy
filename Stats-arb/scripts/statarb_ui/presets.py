"""Named parameter sets. 'values' are overrides on FactorCfg / PortCfg defaults; the Settings panel shows every one of them so nothing is hidden behind a preset.

How these were chosen (see STATARB_REPORT.md): the only rule applied to the numbers is 'what survives out-of-sample, costs and a survivorship-free universe'. The paper preset is fixed by the paper; the
others were looked at on 2017-2021 and judged on 2022-2026, and the number of variants tried is recorded in the report (deflated Sharpe)."""
from __future__ import annotations

from dataclasses import fields
from typing import Optional

from statarb_core import FactorCfg, PortCfg

PRESETS: dict = {}
DEFAULT_PRESET = "Earnings premium + OU screen (paper-trade candidate)"

PRESETS[DEFAULT_PRESET] = dict(
    note="Long the stocks that report earnings in the next 2-10 sessions and that the OU model does not call rich (s < 1), beta-hedged with sector SPDRs, 10% volatility target, geopolitical shock "
         "multiplier, point-in-time universe of names trading >= $150M a day (delisted names included), real 4% name cap and 30% sector cap (cash if there are few announcers), gross <= 1.6x. The announcement premium is the only effect in this study that stayed positive after costs, "
         "a survivorship-reduced universe and complete earnings calendars - but only just: net Sharpe about 0.45 (standard error 0.33, in-sample ~0, out-of-sample ~0.8), ~3% a year at 6.5% volatility, "
         "two big years carry it. A risk premium, not an arbitrage. Paper-trade it; do not size it.",
    values=dict(universe="broad", min_adv_m=150.0, w_rev=0.0, w_eap=1.0, eap_lo=2, eap_hi=10, eap_pool="notrich", eap_cap=0.04, eap_sector_cap=0.30, eap_max_names=100, eap_max_move=0.20, eap_hedge=True,
                shock=True, vol_on=True, vol_target=0.10, g_max=1.6, tc_bps=1.5, borrow_bps=50.0))

PRESETS["Earnings premium, unhedged (long book, carries equity beta)"] = dict(
    note="Same stocks, no ETF hedge: higher return and Sharpe in a bull market because it carries the market (beta ~0.3 net of nothing - it is simply long). Not market neutral: it will lose in a bear "
         "market. Shown so the cost of the hedge is visible.",
    values=dict(universe="broad", min_adv_m=150.0, w_rev=0.0, w_eap=1.0, eap_lo=2, eap_hi=10, eap_pool="notrich", eap_cap=0.04, eap_sector_cap=0.30, eap_max_move=0.20, eap_hedge=False, shock=True,
                vol_on=True, vol_target=0.10, g_max=1.0, tc_bps=1.5, borrow_bps=50.0))

PRESETS["Paper replica - Avellaneda & Lee 2008"] = dict(
    note="The paper's rules as published: 15 PCA eigenportfolios, 60-day OU fit, open |s|>1.25, close longs at -0.50 / shorts at +0.75, equal dollars, bang-bang, no filters, no overlays, "
         "5 bp a side. Included so you can see what the 2008 edge is worth today (about nothing after costs).",
    values=dict(universe="core", w_rev=1.0, w_eap=0.0, n_factors=15, est_window=60, volume_time=False, s_open=1.25, s_close_long=0.50, s_close_short=0.75, weight_pow=0.0, neutral=False,
                style_neutral=False, macro_neutral=False, event_filter=False, earn_filter=False, vol_boost=0.0, allow_short=True, hedge="none", shock=False, vol_on=False, g_max=1.0, name_cap=1.0,
                band=0.0, scale_band=0.0, tc_bps=5.0, borrow_bps=0.0))

PRESETS["2026 risk-aware reversion (all overlays) - research"] = dict(
    note="The paper's signal with every 2026 protection on: earnings / news filters, risk-parity sizing, neutral to beta, PCA factors, sectors, momentum / vol / size and oil / rates / USD betas, "
         "10% volatility target, geopolitical shock multiplier, 1.5 bp auction-fill costs. Expect roughly zero or negative return - it is the honest version of the paper.",
    values=dict(universe="broad", min_adv_m=150.0, w_rev=1.0, w_eap=0.0, s_open=1.25, s_close_long=0.50, s_close_short=0.75, weight_pow=1.0, neutral=True, style_neutral=True, macro_neutral=True,
                event_filter=True, earn_filter=True, vol_boost=0.0, allow_short=True, hedge="none", shock=True, vol_on=True, vol_target=0.10, g_max=2.0, name_cap=0.03, band=0.2, tc_bps=1.5,
                borrow_bps=50.0))


def cfgs(name: str, universe: Optional[str] = None, **override) -> tuple:
    """(FactorCfg, PortCfg) for a preset - the same mapping the Settings panel uses, without Qt (used by the report script and the tests)."""
    v = {**PRESETS[name]["values"], **override}
    f = {x.name: getattr(FactorCfg(), x.name) for x in fields(FactorCfg)}
    p = {x.name: getattr(PortCfg(), x.name) for x in fields(PortCfg)}
    for k, val in v.items():
        if k in f:
            f[k] = val
        if k in p:
            p[k] = val
    if "min_adv_m" in v:
        f["min_adv"] = v["min_adv_m"] * 1e6
    if not v.get("vol_on", True):
        p["g_init"] = p["g_min"] = v.get("g_max", p["g_max"])
    if f.get("var_explained") == 0:
        f["var_explained"] = None
    return FactorCfg(**f), PortCfg(**p)
