"""
Ready-made momentum books (the GUI's model picker). All long-only and unlevered (gross <= 1x at every moment): overnight names bought in the closing auction and sold in the
next opening auction, the intraday symbol bought / sold between the open and the close. Research: MOMENTUM_PREDICTION_REPORT.md (3,035 variants; in-sample 2016-09 .. 2022,
out-of-sample 2023 .. 2026-10).

  TRADED  rule + ML forecast, decided at 15:30 (the free 16-minute-delayed SIP gives live exactly the backtest's bars), QQQ intraday on IEX (the free real-time feed);
          the 8 best names, at most 3 per sector (the team's choice, MomentumBook/REPORT.md section 7: the k = 8 return at about k = 20's drawdown)
  SHADOW  the same book ranked by the rule alone - run next to the traded one in the live tab to see what the forecast adds
The models trade in the auctions (MOC / MOO, 1 bp per side). Live can send market DAY orders in their place (the live tab's "Auction orders": 5 s before the close,
3 minutes before the open) - the same model and decisions, only the order type differs.
"""
from __future__ import annotations

from .models import MomModel

_BOOK = dict(universe="top150+etf", k=20, trend_filter=True, intraday_symbol="QQQ", intraday_every=2, intraday_lookback=14, intraday_trail=True, intraday_mult=1.0,
             cash_reserve=0.02, cost_auction_bp=1.0, half_spread_bp=0.5)

TRADED = "Traded: rule + ML forecast, top 8 (max 3 per sector), 15:30, QQQ intraday on IEX"
SHADOW = "Shadow: rule only (today's move), top 8 (max 3 per sector), 15:30, QQQ intraday on IEX"
RENAMED = {"Auction orders: rule + ML forecast, 15:30, MOC / MOO (real-money target)": TRADED,   # presets of earlier versions (saved GUI settings)
           "Traded: rule + ML forecast, 15:30, QQQ intraday on IEX": TRADED, "Shadow: rule only (today's move), 15:30, QQQ intraday on IEX": SHADOW}
_TOP8 = {**_BOOK, "k": 8, "sector_cap": 3}

PRESETS: dict = {
    TRADED: MomModel(
        name="Book rule+ML 1530 IEX top8 cap3", minute=1530, ranking="rule+ml", intraday_feed="iex", **_TOP8,
        note="The traded book. Overnight: the 8 best of the top-150 stocks + 59 ETFs by the rank of today's move plus the rank of the walk-forward ML forecast of the "
             "overnight return (one model per year; the first is 2018), at most 3 of one sector, bought in the closing auction and sold in the next opening auction. "
             "Intraday: QQQ noise-area breakout every 30 minutes on IEX. Backtest from 2018."),
    SHADOW: MomModel(
        name="Book rule 1530 IEX top8 cap3", minute=1530, ranking="rule", intraday_feed="iex", **_TOP8,
        note="The same book without the forecast: the 8 biggest movers of the day, at most 3 of one sector. The live tab can journal its picks next to the traded "
             "model's (shadow)."),
    "Research reference: rule only, 15:45, QQQ intraday on SIP": MomModel(
        name="Book rule 1545 SIP", minute=1545, ranking="rule", intraday_feed="sip", **_BOOK,
        note="The research's a-priori book (Sharpe 1.97 full / 1.57 out-of-sample there). 15:45 and SIP intraday need the paid real-time SIP plan in live."),
    "Overnight leg only: rule + ML forecast, 15:30": MomModel(
        name="Overnight rule+ML 1530", minute=1530, ranking="rule+ml", intraday_feed="iex", **{**_BOOK, "intraday_symbol": ""},
        note="The overnight momentum leg on its own (no intraday position)."),
    "Intraday leg only: QQQ noise area on IEX": MomModel(
        name="Intraday QQQ IEX", minute=1530, overnight=False, ranking="rule", intraday_feed="iex", **_BOOK,
        note="Zarattini-Aziz-Barbon noise-area breakout on QQQ alone, flat every night."),
}


def preset_names() -> list:
    return list(PRESETS)


def get(name: str) -> MomModel:
    return MomModel.from_dict(PRESETS[name].to_dict())
