"""Freeze the selected parameters (BEFORE the Test window is evaluated by any code)."""
import json, hashlib, datetime, os
D = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt"
assert not os.path.exists(f"{D}\\frozen_params.json"), "already frozen"
assert not os.path.exists(f"{D}\\test_run.lock"), "Test already evaluated"
params = dict(mode="Dip", rsiLen=4, rsiEntry=34.0, exitLen=8, trendLen=225, maxHold=20, dipTrend=True)
frozen = dict(
    strategy="Diversifier (proxy QQQ->TQQQ, proxy SPY->SPXL), long-only, 100% equity, 0.1%/side commission, signals at close / fills next open",
    timeframe="D",
    params=params,
    sha256=hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
    frozen_at=datetime.datetime.now().isoformat(timespec="seconds"),
    selected_by="PROTOCOL.md daily Train sweep (highest 3^5-smoothed cross-market score S_smooth=0.730, Dip class, plateau 100%/84% of neighbours >=75% Sharpe)",
    timeframe_decision="1h Train sweep: Dip class far worse than daily (SPXL Sharpe ~0.45 vs 0.78; optimum pinned at grid edges). 1h Trend+Dip top is Trend-250 in disguise (91% exposure, corr 0.84 to B&H) = different strategy, not eligible. 15m/30m not run (15m has no Train coverage).",
    test_end="2026-10-06",
    note="maxHold is a non-binding safety cap: values >=16, OFF and 30 give identical Train results.",
)
json.dump(frozen, open(f"{D}\\frozen_params.json", "w"), indent=1)
print(json.dumps(frozen, indent=1))
