"""
Residual Mean Reversion - bet that a stock's residual (factor-stripped) price moves back to its own equilibrium.

Avellaneda & Lee (2010), "Statistical arbitrage in the U.S. equities market", Quantitative Finance 10(7): the cumulative residual X of each stock is modelled as an Ornstein-Uhlenbeck
process (an AR(1) fit over a rolling window); the s-score s = (X - m) / sigma_eq says how far it is from equilibrium in units of its equilibrium volatility. Open a long at s <= -1.25,
close it at s >= -0.5 (mirror for shorts); trade only residuals that mean-revert fast (Yeo & Papanicolaou: mean-reversion time well inside the window, a good fit). Here the residuals are
15-minute bars, so the same idea runs intraday. The alternative 'xs' mode is the plain cross-sectional residual reversal of Lehmann (1990) / Da, Liu & Schaumburg (2014): short the members
with the highest residual return over the last k bars, buy the lowest.
"""
from __future__ import annotations

import numpy as np

from resarb import features as X
from resarb.strategy import Param, Strategy


class ResidualMeanReversion(Strategy):
    KEY = "ResMeanReversion"
    NAME = "Residual Mean Reversion"
    PREFIX = "RESMR"
    SUMMARY = ("Each stock's cumulative residual (its return after removing the SPY and sector-ETF moves) is fitted with an Ornstein-Uhlenbeck model; when it is stretched far from its "
               "equilibrium (s-score) the engine bets on the snap-back and hedges the market and sector with ETFs.")
    PAPERS = [
        ("Avellaneda & Lee (2010), Statistical arbitrage in the US equities market, Quantitative Finance 10(7) 761-782",
         "https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf", "the model: ETF / PCA residuals, OU s-score, thresholds 1.25 / 0.5 / 0.75"),
        ("Yeo & Papanicolaou, Risk control of mean-reversion time in statistical arbitrage", "https://content.iospress.com/articles/risk-and-decision-analysis/rda132",
         "trade only fast-reverting, well-fitted residuals"),
        ("Da, Liu & Schaumburg (2014), A closer look at the short-term return reversal, Management Science", "https://newyorkfed.org/medialibrary/media/research/staff_reports/sr513.html",
         "reversal lives in the residual (within-industry, non-fundamental) part"),
        ("Lehmann (1990), Fads, martingales, and market efficiency, QJE 105(1)", "https://www.nber.org/papers/w2533", "cross-sectional reversal ('xs' mode)"),
        ("Gatev, Goetzmann & Rouwenhorst (2006), Pairs trading, RFS 19(3)", "https://ideas.repec.org/p/nbr/nberwo/7032.html", "relative-value reversion; profits shrank after 2000"),
        ("Krauss (2017), Statistical arbitrage pairs trading strategies: review and outlook, J. Economic Surveys 31(2)", "https://ideas.repec.org/p/zbw/iwqwdp/092015.html", "survey"),
    ]
    PARAMS = [
        Param("mode", "Signal", "ou", "choice", choices=("ou", "xs"), help="ou = Avellaneda-Lee s-score; xs = cross-sectional residual reversal"),
        Param("ou_window", "OU window (15-min bars)", 130, "int", 26, 780, 13, help="130 bars = 5 sessions"),
        Param("s_open", "Open when |s| >=", 1.25, "float", 0.5, 4.0, 0.05),
        Param("s_close", "Close when |s| <=", 0.5, "float", 0.0, 2.0, 0.05),
        Param("names", "Max positions per side", 30, "int", 5, 100, 5),
        Param("max_tau", "Max mean-reversion time (x window)", 0.5, "float", 0.05, 2.0, 0.05, help="Avellaneda-Lee keep kappa > 2 / window"),
        Param("r2_min", "Min R^2 of the OU fit", 0.0, "float", 0.0, 0.99, 0.05),
        Param("xs_bars", "xs: lookback (bars)", 26, "int", 1, 520, 1),
    ]
    # research choice (in-sample best of 152 variants, research/r3 + r8): reverse the last hour's residual, decided at 15:30, traded at 15:45, held to the next day's 15:45
    DEFAULTS = {"mode": "xs", "xs_bars": 4, "rebalance": "daily", "trade_time": "15:45", "method": "quantile", "hedge": "etf", "q_in": 0.1, "q_out": 0.25, "beta_days": 126,
                "gross": 0.8}                                                              # RM: stocks + hedges <= 1.9x overnight at the 95th percentile (research/r9_margin.py)
    PRESETS = {
        "Research choice: last-hour residual reversal, daily at 15:45 (in-sample best)": {"mode": "xs", "xs_bars": 4, "method": "quantile", "q_in": 0.1, "q_out": 0.25,
                                                                                          "rebalance": "daily", "trade_time": "15:45"},
        "Paper: OU s-score 1.25 / 0.5, 5-day window": {"mode": "ou", "ou_window": 130, "s_open": 1.25, "s_close": 0.5, "method": "threshold", "rebalance": "15min"},
        "OU s-score, 2-day window, hourly": {"mode": "ou", "ou_window": 52, "s_open": 1.5, "s_close": 0.5, "method": "threshold", "rebalance": "1h"},
        "Cross-sectional 1-day reversal, hourly": {"mode": "xs", "xs_bars": 26, "method": "quantile", "q_in": 0.1, "q_out": 0.25, "rebalance": "1h"},
    }
    HOWTO = ("Every bar the engine refits the OU model of every member's cumulative residual over the window, opens a long when the s-score drops below -open (a short above +open), "
             "keeps it until |s| falls under 'close', and hedges each position's market and sector beta with SPY / sector ETFs.")

    def warmup_days(self) -> int:
        return int(self.v["beta_days"]) + int(self.v["ou_window"]) // 26 + 30

    def score(self, ctx, log=lambda m: None) -> np.ndarray:
        if self.v["mode"] == "xs":
            e = ctx.res.e_cc
            return np.where(np.isfinite(e), -X.roll_sum(e, int(self.v["xs_bars"])), np.nan)
        W = int(self.v["ou_window"])
        f = ctx.features(["ou_s", "ou_kappa", "ou_r2"], ou_window=W)
        s, kap, r2 = f["ou_s"].astype(float), f["ou_kappa"].astype(float), f["ou_r2"].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (1.0 / kap <= float(self.v["max_tau"]) * W) & (r2 >= float(self.v["r2_min"]))
        so = float(self.v["s_open"]) - 1e-6
        return np.where(ok, -s, np.clip(-s, -so, so))                                      # a slow / poorly fitted residual can't OPEN a position, but an open one still closes on its s-score
