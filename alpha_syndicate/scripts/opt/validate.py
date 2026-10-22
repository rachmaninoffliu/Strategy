"""Validate the numba replica against TradingView results collected earlier (tester + dashboard)."""
import numpy as np
from sim import *

def show(tag, mt, tv):
    print(f"\n== {tag}")
    keys = ["trades", "win", "final", "cagr", "mdd", "sharpe", "sortino", "calmar", "pf", "corr", "exposure"]
    for k in keys:
        if k in tv:
            a = mt[k]
            print(f"  {k:9s} replica {a:12.4f}   TV {tv[k]:12.4f}")

# ---- TQQQ traded (TradingView's own bars), proxy QQQ (yahoo, matches TV closes) ----
mT = Market("TQQQ", "QQQ", src="yf", traded_src="tv")
# A. Dip defaults, FULL window, tester numbers (0.1% commission)
mt = run_one(mT, FULL, MODE_DIP, 2, 20, 5, 200, 10, True)
show("A  Dip defaults TQQQ<-QQQ FULL  (tester: 260 trades, 178 wins, net $1,518,130.99, PF 2.0458)", mt,
     dict(trades=260, win=178/260, final=(100000 + 1518130.9892) / 100000, pf=2.045768))
print("   dashboard Full: CAGR 18.2% MDD 41.9% Sharpe 0.71 Sortino 1.04 Calmar 0.44 win 68.5% PF 2.05 corr 0.47 expo 21.6%")
# D. Train dashboard
mt = run_one(mT, TRAIN, MODE_DIP, 2, 20, 5, 200, 10, True)
show("D1 Dip defaults TQQQ<-QQQ TRAIN (dash: 185 trades, win 65.9%, CAGR 17.2, MDD 41.9, Sharpe 0.67, Sortino 0.97, Calmar 0.41, PF 2.15, corr 0.52, expo 21.8)",
     mt, dict(trades=185, win=0.659, cagr=0.172, mdd=0.419, sharpe=0.67, sortino=0.97, calmar=0.41, pf=2.15, corr=0.52, exposure=0.218))

# ---- signals on TQQQ itself (the user's chart instance variant) ----
mS = Market("TQQQ", "TQQQ", src="yf", traded_src="tv")
mt = run_one(mS, FULL, MODE_TD, 2, 25, 5, 200, 10, True)
show("B  Trend+Dip RSI25 TQQQ<-TQQQ FULL (tester: 148 trades, 83 wins, final $47,469,738.70, PF 2.4301)", mt,
     dict(trades=148, win=83/148, final=(100000 + 36914158.38 + 10455580.32) / 100000, pf=2.430093))
print("   dashboard: CAGR 44.9% MDD 64.3% Sharpe 0.97 Sortino 1.39 Calmar 0.70 win 56.1% PF 2.43 corr 0.86 expo 84.2%")
mt = run_one(mS, FULL, MODE_TREND, 2, 25, 5, 200, 10, True)
show("C  Trend TQQQ<-TQQQ FULL (tester: 45 trades, 18 wins, final $9,998,483.36, PF 5.2948)", mt,
     dict(trades=45, win=18/45, final=(100000 + 7696239.52 + 2202243.84) / 100000, pf=5.294784))

# ---- other assets, TRAIN dashboard numbers ----
mQ = Market("QQQ", "QQQ")
show("D2 QQQ<-QQQ TRAIN (dash: 185, win 65.4%, CAGR 4.4, MDD 16.2, Sharpe 0.47, Sortino 0.67, Calmar 0.27, PF 1.64)", run_one(mQ, TRAIN, MODE_DIP, 2, 20, 5, 200, 10, True),
     dict(trades=185, win=0.654, cagr=0.044, mdd=0.162, sharpe=0.47, sortino=0.67, calmar=0.27, pf=1.64))
mSP = Market("SPY", "SPY")
show("D3 SPY<-SPY TRAIN (dash: 181, win 65.2%, CAGR 1.1, MDD 16.3, Sharpe 0.17, Sortino 0.23, Calmar 0.06, PF 1.16)", run_one(mSP, TRAIN, MODE_DIP, 2, 20, 5, 200, 10, True),
     dict(trades=181, win=0.652, cagr=0.011, mdd=0.163, sharpe=0.17, sortino=0.23, calmar=0.06, pf=1.16))
mX = Market("SPXL", "SPY")
show("D4 SPXL<-SPY TRAIN (dash: 181, win 69.6%, CAGR 6.8, MDD 41.3, Sharpe 0.39, Sortino 0.53, Calmar 0.16, PF 1.40)", run_one(mX, TRAIN, MODE_DIP, 2, 20, 5, 200, 10, True),
     dict(trades=181, win=0.696, cagr=0.068, mdd=0.413, sharpe=0.39, sortino=0.53, calmar=0.16, pf=1.40))
