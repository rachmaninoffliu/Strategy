"""
Residual Momentum - buy the stocks whose RESIDUAL (factor-stripped) return has been strongest, short the weakest, hedge the market and sector with ETFs.

Blitz, Huij & Martens (2011), "Residual momentum", Journal of Empirical Finance 18(3): ranking on residuals instead of total returns removes the momentum strategy's time-varying factor
bets, roughly doubles its risk-adjusted return and avoids most of its crashes. Their signal: residuals of a 36-month Fama-French regression, summed over months t-12 .. t-2 and divided by
their volatility. Here the factor model is SPY + the stock's sector SPDR (or PCA eigenportfolios), re-estimated every morning from daily returns, and the residuals are summed over
``form_days`` days, skipping the last ``skip_days`` (the short-term reversal window, Jegadeesh 1990 / Lehmann 1990).
"""
from __future__ import annotations

import numpy as np

from resarb import features as X
from resarb.strategy import Param, Strategy


class ResidualMomentum(Strategy):
    KEY = "ResMomentum"
    NAME = "Residual Momentum"
    PREFIX = "RESMOM"
    SUMMARY = ("Long the stocks with the strongest residual return over the formation window, short the weakest; the residual is what is left after removing the market (SPY) and sector "
               "(SPDR) moves estimated each morning. Market and sector exposure is hedged with SPY and the sector ETFs.")
    PAPERS = [
        ("Blitz, Huij & Martens (2011), Residual momentum, J. Empirical Finance 18(3) 506-521", "https://repub.eur.nl/pub/22252", "the signal: vol-scaled residual return, 12-1 months"),
        ("Huij & Lansdorp (2017), Residual momentum and reversal strategies revisited", "https://repub.eur.nl/pub/100801", "out-of-sample confirmation"),
        ("Jegadeesh (1990) / Lehmann (1990), short-term reversal", "https://www.nber.org/papers/w2533", "why the last month is skipped"),
        ("Lou, Polk & Skouras (2019), A tug of war: overnight versus intraday expected returns, JFE 134(1)", "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf",
         "momentum is earned overnight: trade near the close"),
        ("Ehsani & Linnainmaa (2022), Factor momentum and the momentum factor, J. Finance 77(3)", "https://www.nber.org/papers/w25551", "why stripping factors matters"),
    ]
    PARAMS = [
        Param("form_days", "Formation window (days)", 252, "int", 5, 504, 21, help="residuals summed over the last N days ..."),
        Param("skip_days", "Skip the most recent (days)", 21, "int", 0, 126, 1, help="... except the most recent M days (short-term reversal)"),
        Param("vol_scale", "Divide by residual volatility", True, "bool", help="Blitz et al.: rank on residual return / residual volatility"),
    ]
    # research choice (in-sample 2016-09..2022 best of 105 variants, research/r2 + r8): 3-month formation, skip 1 month, monthly, top / bottom 10% held until they leave 25%
    DEFAULTS = {"form_days": 63, "skip_days": 21, "rebalance": "monthly", "trade_time": "15:45", "method": "quantile", "q_in": 0.1, "q_out": 0.25, "hedge": "etf", "beta_days": 252,
                "feed": "sip_delayed", "gross": 0.9}                                       # RM: stocks + hedges <= 1.9x overnight at the 95th percentile (research/r9_margin.py)
    PRESETS = {
        "Research choice: 3-1 month, monthly (in-sample best)": {"form_days": 63, "skip_days": 21, "vol_scale": True, "rebalance": "monthly", "trade_time": "15:45", "q_in": 0.1,
                                                                 "q_out": 0.25, "feed": "sip_delayed"},
        "Paper: 12-1 month, monthly (Blitz et al.)": {"form_days": 252, "skip_days": 21, "vol_scale": True, "rebalance": "monthly", "trade_time": "15:45", "q_in": 0.2, "q_out": 0.2},
        "12-1 month, weekly with hysteresis": {"form_days": 252, "skip_days": 21, "vol_scale": True, "rebalance": "weekly", "trade_time": "15:45", "q_in": 0.2, "q_out": 0.35},
        "6-1 month, monthly": {"form_days": 126, "skip_days": 21, "vol_scale": True, "rebalance": "monthly", "trade_time": "15:45", "q_in": 0.2, "q_out": 0.2},
    }
    HOWTO = ("Once a month (or week / day) shortly before the close the engine ranks the members by residual momentum, buys the top fifth, shorts the bottom fifth, and sizes SPY / sector "
             "ETF hedges so the book has no market or sector beta. A name that is held stays until it falls out of the top / bottom 'hold' fraction.")

    def warmup_days(self) -> int:
        # the oldest residual in the formation window needs a full beta window before it: beta_days + form_days sessions (else live betas use shorter windows than the backtest)
        return int(self.v["beta_days"]) + int(self.v["form_days"]) + 40

    def score(self, ctx, log=lambda m: None) -> np.ndarray:
        m = X.res_momentum(ctx.res, int(self.v["form_days"]), int(self.v["skip_days"]), bool(self.v["vol_scale"]))
        return m[ctx.p.day]
