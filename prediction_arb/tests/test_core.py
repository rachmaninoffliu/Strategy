"""Offline, deterministic tests with hand-computed expectations (run: ``python tests/test_core.py`` or pytest)."""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FeeConfig, ScanSettings  # noqa: E402
from connectors import mapper  # noqa: E402
from core.arb_detector import ArbDetector  # noqa: E402
from core.backtester import Backtester, BacktestConfig, Structure  # noqa: E402
from core.models import ArbType, Candle, Market, Relation, Side, Venue  # noqa: E402
from core.pricing import (LegSpec, annualize, days_to_settle, kalshi_taker_fee, kelly_fraction,  # noqa: E402
                          poly_taker_fee, walk_legs)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
NO_FEES = FeeConfig(kalshi_coef=0.0, poly_mode="zero")


def mk(venue, mid, yes_bid, yes_ask, *, days=30, size=1000.0, title="", event="ev", label="", volume=10_000.0,
       **kw) -> Market:
    d = dict(venue=venue, market_id=mid, event_id=event, title=title or mid, outcome_label=label,
             yes_bid=yes_bid, yes_ask=yes_ask, no_bid=None if yes_ask is None else 1 - yes_ask,
             no_ask=None if yes_bid is None else 1 - yes_bid, yes_ask_size=size, yes_bid_size=size,
             no_ask_size=size, no_bid_size=size, volume=volume, status="open",
             settle_time=NOW + timedelta(days=days), close_time=NOW + timedelta(days=days),
             yes_token="t" + mid, no_token="n" + mid)
    d.update(kw)
    return Market(**d)


K, P = Venue.KALSHI, Venue.POLYMARKET


# ── pricing ───────────────────────────────────────────────────────────────────
def test_kalshi_fee_rounds_up_to_cent():
    assert kalshi_taker_fee(100, 0.5) == 1.75                     # 0.07*100*0.25
    assert kalshi_taker_fee(1, 0.01) == 0.01                      # 0.000693 -> 1 cent
    assert kalshi_taker_fee(0, 0.5) == 0.0


def test_poly_fee_schedule():
    assert abs(poly_taker_fee(100, 0.5, 0.05, 1.0) - 1.25) < 1e-12
    assert abs(poly_taker_fee(100, 0.5, 0.25, 2.0) - 1.5625) < 1e-12


def test_days_floor_and_annualize():
    assert days_to_settle(NOW + timedelta(hours=1), NOW) == 0.25          # floored at 6h
    assert abs(days_to_settle(NOW + timedelta(days=10), NOW) - 10) < 1e-9
    assert abs(annualize(0.01, 30) - ((1.01) ** (365.25 / 30) - 1)) < 1e-9
    assert math.isfinite(annualize(50.0, 0.25))                           # capped exponent, no overflow
    assert annualize(-1.0, 10) == -1.0


def test_kelly():
    assert kelly_fraction(0.10, 0.0) == 1.0
    assert kelly_fraction(0.01, 0.02) == 0.0                              # 1% edge cannot pay for 2% wipe-out risk
    assert abs(kelly_fraction(0.10, 0.02) - 0.78) < 1e-9


def test_walk_legs_matches_hand_calc():
    m1, m2 = mk(K, "a", 0.3, 0.4), mk(P, "b", 0.3, 0.5)
    legs = [LegSpec(m1, Side.YES, [(0.40, 100), (0.45, 100)]), LegSpec(m2, Side.YES, [(0.50, 60), (0.52, 100)])]
    # 30 days, tiny hurdle: consumes 60@0.90, 40@0.92, 60@0.97 then leg 1 runs out of levels -> stops at 160
    r = walk_legs(legs, min_unit_payout=1.0, days=30, hurdle=0.07, fees=NO_FEES)
    assert abs(r.units - 160) < 1e-9
    assert abs(r.total_cost - (60 * 0.90 + 40 * 0.92 + 60 * 0.97)) < 1e-9
    assert abs(r.limit_prices[0] - 0.45) < 1e-12 and abs(r.top_prices[0] - 0.40) < 1e-12
    # 1-year horizon: need >= 7% -> only levels costing <= 0.9346 qualify (0.90 and 0.92), not 0.97
    r2 = walk_legs(legs, min_unit_payout=1.0, days=365.25, hurdle=0.07, fees=NO_FEES)
    assert abs(r2.units - 100) < 1e-9 and abs(r2.total_cost - (60 * 0.90 + 40 * 0.92)) < 1e-9
    assert abs(r2.apy - r2.net_return) < 1e-9                              # D = 365.25 -> APY == R


def test_walk_respects_max_units_and_cost():
    m1, m2 = mk(K, "a", 0.3, 0.4), mk(P, "b", 0.3, 0.5)
    legs = [LegSpec(m1, Side.YES, [(0.40, 1000)]), LegSpec(m2, Side.NO, [(0.50, 1000)])]
    assert abs(walk_legs(legs, min_unit_payout=1, days=30, hurdle=0.07, fees=NO_FEES, max_units=25).units - 25) < 1e-9
    assert abs(walk_legs(legs, min_unit_payout=1, days=30, hurdle=0.07, fees=NO_FEES, max_cost=90).units - 100) < 1e-9


def test_fee_can_kill_a_thin_edge():
    m1, m2 = mk(K, "a", 0.3, 0.48), mk(P, "b", 0.3, 0.5)
    legs = [LegSpec(m1, Side.YES, [(0.48, 100)]), LegSpec(m2, Side.NO, [(0.50, 100)])]      # 2% gross edge
    assert walk_legs(legs, min_unit_payout=1, days=30, hurdle=0.07, fees=NO_FEES) is not None
    assert walk_legs(legs, min_unit_payout=1, days=30, hurdle=0.07, fees=FeeConfig()) is None  # Kalshi fee ~1.7c/unit


# ── detector ──────────────────────────────────────────────────────────────────
def det(**kw) -> ArbDetector:
    return ArbDetector(ScanSettings(min_confidence=0.5, min_volume=0, fees=NO_FEES, **kw))


def test_parity_detected_with_correct_legs_and_bounds():
    k = mk(K, "k1", 0.38, 0.40)
    p = mk(P, "p1", 0.45, 0.47)                                              # NO ask = 0.55
    pair = mapper.MarketPair(k, p, Relation.EQUIVALENT, 0.9)
    c = det().detect_parity([pair], NOW)
    assert len(c) == 1
    o = c[0].opp
    assert [(l.venue, l.side) for l in o.legs] == [(K, Side.YES), (P, Side.NO)]
    assert abs(o.total_cost / o.size - 0.95) < 1e-9
    assert abs(o.net_return - (1 / 0.95 - 1)) < 1e-9
    assert o.guaranteed_payout == o.size and o.max_payout == o.size
    assert o.apy > 0.07 and o.arb_type is ArbType.PARITY


def test_parity_no_arb_when_costs_exceed_payout():
    k = mk(K, "k1", 0.38, 0.52)
    p = mk(P, "p1", 0.50, 0.52)                                              # 0.52 + 0.50 = 1.02
    assert det().detect_parity([mapper.MarketPair(k, p, Relation.EQUIVALENT, 0.9)], NOW) == []


def test_parity_low_yield_capital_trap_filtered_by_hurdle():
    k = mk(K, "k1", 0.38, 0.49, days=700)
    p = mk(P, "p1", 0.50, 0.52, days=700)                                    # NO ask 0.50: 1% edge over ~2y -> APY < 7%
    assert det().detect_parity([mapper.MarketPair(k, p, Relation.EQUIVALENT, 0.9)], NOW) == []
    k2 = mk(K, "k1", 0.38, 0.49, days=20)
    p2 = mk(P, "p1", 0.50, 0.52, days=20)                                    # same 1% over 20 days -> ~20% APY
    assert len(det().detect_parity([mapper.MarketPair(k2, p2, Relation.EQUIVALENT, 0.9)], NOW)) == 1


def test_inverse_pair_buys_same_sides():
    k = mk(K, "k1", 0.38, 0.40)
    p = mk(P, "p1", 0.55, 0.57)                                              # P is the logical negation of K
    c = det().detect_parity([mapper.MarketPair(k, p, Relation.INVERSE, 0.9)], NOW)
    assert len(c) == 1 and [l.side for l in c[0].opp.legs] == [Side.YES, Side.YES]
    assert abs(c[0].opp.total_cost / c[0].opp.size - 0.97) < 1e-9


def test_subset_monotonicity_violation():
    a = mk(K, "a110", 0.62, 0.64, label="110k")           # P(A) is bid 0.62 ...
    b = mk(P, "b100", 0.50, 0.55, label="100k")           # ... but P(B) offered at 0.55 although A implies B
    edge = mapper.SubsetEdge(a, b, 0.9)
    c = det().detect_subset([edge], NOW)
    assert len(c) == 1
    o = c[0].opp
    assert [(l.market_id, l.side) for l in o.legs] == [("b100", Side.YES), ("a110", Side.NO)]
    assert abs(o.total_cost / o.size - (0.55 + 0.38)) < 1e-9
    assert o.unit_min_payout == 1.0 and o.unit_max_payout == 2.0 and o.max_payout == 2 * o.size


def test_subset_no_violation():
    a = mk(K, "a", 0.30, 0.32)
    b = mk(P, "b", 0.50, 0.55)
    assert det().detect_subset([mapper.SubsetEdge(a, b, 0.9)], NOW) == []


def _basket(asks, exhaustive=True, scope="polymarket", spread=0.02):
    ms = [mk(P, f"o{i}", a - spread, a, event="E", label=f"Cand{i}", neg_risk=True) for i, a in enumerate(asks)]
    return mapper.Basket("Event", scope, [[m] for m in ms], exhaustive, "test", NOW + timedelta(days=30))


def test_mece_all_yes_when_sum_below_one():
    d = det()
    c = d.detect_mece([_basket([0.30, 0.30, 0.34])], NOW)
    yes = [x for x in c if x.opp.variant == "buy-all-YES"]
    assert len(yes) == 1
    o = yes[0].opp
    assert abs(o.total_cost / o.size - 0.94) < 1e-9 and o.guaranteed_payout == o.size


def test_mece_non_exhaustive_never_buys_all_yes():
    c = det().detect_mece([_basket([0.30, 0.30, 0.34], exhaustive=False)], NOW)
    assert all(x.opp.variant != "buy-all-YES" for x in c)


def test_mece_buy_all_no_pays_n_minus_1():
    d = det()
    c = d.detect_mece([_basket([0.37, 0.37, 0.37], exhaustive=False, spread=0.01)], NOW)   # YES mids sum 1.095
    no = [x for x in c if x.opp.variant == "buy-all-NO"]
    assert len(no) == 1
    o = no[0].opp
    assert abs(o.total_cost / o.size - (0.64 * 3)) < 1e-9               # NO asks = 1 - bid = 0.64 each
    assert abs(o.guaranteed_payout / o.size - 2.0) < 1e-9               # n-1
    assert abs(o.max_payout / o.size - 3.0) < 1e-9                      # n if none of them occurs


def test_mece_incomplete_basket_rejected_by_mid_sum_gate():
    d = det()
    assert d.detect_mece([_basket([0.10, 0.10, 0.10])], NOW) == []       # mids sum 0.24: outcomes are missing
    assert d.rejected["mece_mid_sum"] >= 1


def test_implausible_edge_is_dropped():
    k = mk(K, "k1", 0.30, 0.31)
    p = mk(P, "p1", 0.20, 0.22)                                          # cost 0.31 + 0.80 = 1.11 -> no arb ...
    p2 = mk(P, "p2", 0.60, 0.62)                                         # cost 0.31 + 0.40 = 0.71 -> +40% edge
    d = det()
    assert d.detect_parity([mapper.MarketPair(k, p2, Relation.EQUIVALENT, 0.9)], NOW) == []
    assert d.rejected["implausible_edge"] == 1


# ── mapper ────────────────────────────────────────────────────────────────────
def test_parse_strike_variants():
    ps = mapper.parse_strike
    assert ps("Will Bitcoin be above $100,000 on Oct 3?")[:3] == ("gt", 100000.0, None)
    assert ps("Bitcoin price on Sep 29, 2026 at 6pm EDT? | $73,100 or above")[:2] == ("gt", 73100.0)
    assert ps("Will Ethereum be below $2,500 on Dec 31, 2026?")[:2] == ("lt", 2500.0)
    assert ps("CPI between 0.2% and 0.3% in August 2026")[:3] == ("between", 0.2, 0.3)
    assert ps("peak RONI at least 2.0 but below 2.25")[0] == "between"           # bounded bin is not 'above'
    assert ps("Will Bitcoin reach $100,000 in August?")[3] == "touch"
    assert ps("What will Gold hit: hit (LOW) $4,100")[:2] == ("lt", 4100.0)
    assert ps("What will Gold hit: hit (HIGH) $4,200")[:2] == ("gt", 4200.0)
    assert ps("Will gold hit $4,000 in September?")[0] == ""                    # ambiguous direction: no structure


def _btc(venue, mid, title, **kw):
    return mk(venue, mid, 0.4, 0.42, title=title, event=mid, **kw)


def test_pair_matching_equivalent_inverse_and_guards():
    k = _btc(K, "k", "Will the price of Bitcoin be above $100,000 on October 3?")
    p_eq = _btc(P, "p1", "Bitcoin above $100,000 on October 3?")
    p_inv = _btc(P, "p2", "Will the price of Bitcoin be below $100,000 on October 3?")
    p_touch = _btc(P, "p3", "Will Bitcoin reach $100,000 on October 3?")
    p_date = _btc(P, "p4", "Will the price of Bitcoin be above $100,000 on October 4?")
    p_strike = _btc(P, "p5", "Will the price of Bitcoin be above $95,000 on October 3?")
    got = {pr.b.market_id: pr for pr in mapper.match_pairs([k], [p_eq], 24, 0)}
    assert got["p1"].relation is Relation.EQUIVALENT and got["p1"].confidence > 0.8
    got = {pr.b.market_id: pr for pr in mapper.match_pairs([k], [p_inv], 24, 0)}
    assert got["p2"].relation is Relation.INVERSE
    for bad in (p_touch, p_date, p_strike):
        assert mapper.match_pairs([_btc(K, "k", k.title)], [bad], 24, 0) == [], bad.title


def test_pair_matching_rejects_different_period():
    k = mk(K, "g1", 0.4, 0.42, title="Will real GDP increase by more than 2.5% in Q4 2026?")
    p = mk(P, "g2", 0.4, 0.42, title="Will US GDP growth in 2026 be greater than 2.5%?")
    assert mapper.match_pairs([k], [p], 24, 0) == []


def test_subset_edges_same_event_only_and_same_template():
    ms = [mk(K, f"b{s}", 0.4, 0.42, event="EV1", title="Bitcoin price on Oct 3?", label=f"${s:,} or above",
             direction="gt", strike=float(s)) for s in (90_000, 100_000, 110_000)]
    other = mk(K, "c", 0.4, 0.42, event="EV2", title="Bitcoin price on Oct 3?", label="$95,000 or above",
               direction="gt", strike=95_000.0)
    edges = mapper.build_subset_edges(ms + [other])
    pairs = {(e.inner.market_id, e.outer.market_id) for e in edges}
    assert pairs == {("b110000", "b100000"), ("b110000", "b90000"), ("b100000", "b90000")}


def test_ladder_covers_line():
    def m(d, lo=None, hi=None):
        return mk(K, f"x{d}{lo}{hi}", 0.4, 0.42, direction=d, strike=lo, strike_hi=hi)
    full = [m("lt", hi=1.0) if False else mk(K, "l", 0.1, 0.2, direction="lt", strike=1.0),
            m("between", 1.0, 2.0), m("between", 2.0, 3.0), mk(K, "g", 0.1, 0.2, direction="gt", strike=3.0)]
    assert mapper._ladder_covers_line(full)
    gap = [full[0], m("between", 1.5, 2.0), full[3]]
    assert not mapper._ladder_covers_line(gap)


# ── backtester (synthetic data, no network) ───────────────────────────────────
D0 = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp())
DAY = 86400


def bt_cfg(**kw) -> BacktestConfig:
    base = dict(start=datetime(2026, 6, 1, tzinfo=timezone.utc), end=datetime(2026, 8, 1, tzinfo=timezone.utc),
                initial_capital=10_000, max_capital_per_trade=1_000, fees=NO_FEES, slippage_bps=0,
                exec_delay_bars=1, poly_half_spread=0.005, assumed_depth_usd=5_000, kalshi_lag_h=0, poly_lag_h=0)
    base.update(kw)
    return BacktestConfig(**base)


def settled_pair(k_result="yes", p_result="yes", settle_day=40, mid="1"):
    settle = datetime.fromtimestamp(D0 + settle_day * DAY, tz=timezone.utc)
    common = dict(status="settled", open_time=datetime.fromtimestamp(D0, tz=timezone.utc), close_time=settle,
                  settle_time=settle, settled_ts=settle, volume=50_000)
    k = mk(K, "K" + mid, None, None, result=k_result, settle_value=1.0 if k_result == "yes" else 0.0, **common)
    p = mk(P, "P" + mid, None, None, result=p_result, settle_value=1.0 if p_result == "yes" else 0.0, **common)
    return k, p


def parity_struct(k, p, sid="s1"):
    return Structure(sid, ArbType.PARITY, "test pair", [(k, Side.YES), (p, Side.NO)], 1.0, 1.0, 0.9, "K YES + P NO")


K_ARB, P_PX = 0.46, 0.53      # K YES ask .46 + P NO ask (1 - (.53 - .005)) = .475  ->  unit cost .935 (mids sum .92)
UNIT = K_ARB + (1 - (P_PX - 0.005))


def series(k_ask_by_day: dict, p_price_by_day: dict, n=45):
    """Kalshi bid = ask - 0.02; Polymarket price-only points; one bar per day at midnight UTC."""
    ks = [Candle(D0 + d * DAY, k_ask_by_day.get(d, 0.60) - 0.02, k_ask_by_day.get(d, 0.60), None, 0.0) for d in range(n)]
    ps = [Candle(D0 + d * DAY, None, None, p_price_by_day.get(d, 0.55), 0.0) for d in range(n)]
    return ks, ps


NOW_BT = datetime(2026, 9, 15, tzinfo=timezone.utc)


def test_backtest_entry_delay_lockup_and_pnl():
    k, p = settled_pair("yes", "yes")
    ks, ps = series({5: K_ARB, 6: K_ARB, 7: K_ARB}, {d: P_PX for d in (5, 6, 7)})   # arb visible on days 5-7
    bt = Backtester(bt_cfg())
    sigs = bt._signals([parity_struct(k, p)], {k.key: ks, p.key: ps}, NOW_BT)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.t_fill == D0 + 6 * DAY                                # signal day 5, fill one bar later
    assert abs(UNIT - 0.935) < 1e-12
    assert abs(s.unit_cost - UNIT) < 1e-9 and abs(s.net_return - (1 / UNIT - 1)) < 1e-9
    res = bt._simulate(sigs, NOW_BT)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.status == "Resolved - Won"
    units = math.floor(1000 / UNIT)
    assert abs(t.size - units) < 1e-9 and abs(t.cost - units * UNIT) < 1e-6
    assert abs(t.realized_payout - units) < 1e-9 and abs(t.pnl - units * (1 - UNIT)) < 1e-6
    assert abs(t.lockup_days - (40 - 6)) < 1e-6
    m = res.metrics
    assert abs(m["Total Net PnL ($)"] - round(t.pnl, 2)) < 0.02
    assert m["Win Rate (%)"] == 100.0 and m["Trades"] == 1
    assert abs(m["Avg Lockup (days)"] - 34.0) < 1e-6
    # capital is locked (equity flat at cost) until release, then jumps by the profit
    eq = res.equity
    assert abs(eq.iloc[10] - 10_000) < 1e-6 and abs(eq.iloc[-1] - (10_000 + t.pnl)) < 1e-6
    assert res.locked.iloc[10] > 0 and abs(res.locked.iloc[-1]) < 1e-6


def test_backtest_resolution_mismatch_is_a_loss_not_assumed_away():
    k, p = settled_pair("no", "yes")                              # venues resolve OPPOSITE ways: YES(K)=0, NO(P)=0
    ks, ps = series({5: K_ARB, 6: K_ARB}, {5: P_PX, 6: P_PX})
    bt = Backtester(bt_cfg())
    res = bt._simulate(bt._signals([parity_struct(k, p)], {k.key: ks, p.key: ps}, NOW_BT), NOW_BT)
    t = res.trades[0]
    assert t.realized_payout == 0.0 and t.status.startswith("Resolved - Loss")
    assert abs(t.pnl + t.cost) < 1e-9
    assert res.metrics["Losing Trades (rule mismatch)"] == 1 and res.metrics["Win Rate (%)"] == 0.0
    assert res.metrics["Max Drawdown (%)"] < 0


def test_backtest_data_gate_rejects_bad_print():
    k, p = settled_pair()
    ks, ps = series({}, {5: 0.10, 6: 0.10})     # a lone Polymarket print of 0.10 -> NO ask .905, K ask .60: sum 1.5 (no arb)
    ks, _ = series({5: 0.30, 6: 0.30}, {})       # K ask .30 + P NO ask .455 = .755 -> would be a 32% "arb"
    ps = [Candle(D0 + d * DAY, None, None, 0.55, 0.0) for d in range(45)]
    ps[5] = Candle(ps[5].ts, None, None, 0.90, 0.0)       # bad print: P mid 0.90 -> NO ask 0.105
    ps[6] = Candle(ps[6].ts, None, None, 0.90, 0.0)
    bt = Backtester(bt_cfg(spike_threshold=0.0))          # spike filter off to isolate the mid-sum gate
    # mids: K YES mid .29 + P NO mid .10 = .39 < 0.9 -> rejected although asks sum (.30 + .105) is 'cheap'
    assert bt._signals([parity_struct(k, p)], {k.key: ks, p.key: ps}, NOW_BT) == []
    assert bt.stats["bars_rejected_by_data_gate"] >= 2


def test_backtest_spike_filter_drops_isolated_print():
    cs = [Candle(D0 + d * DAY, None, None, 0.5, 0.0) for d in range(6)]
    cs[3] = Candle(cs[3].ts, None, None, 0.1, 0.0)
    out = Backtester._despike(cs, 0.12)
    assert len(out) == 5 and all(c.price == 0.5 for c in out)
    trend = [Candle(D0 + d * DAY, None, None, 0.1 * d, 0.0) for d in range(6)]       # a real move is kept
    assert len(Backtester._despike(trend, 0.12)) == 6


def test_backtest_concurrency_cap_enforced():
    pairs = [settled_pair(mid=str(i)) for i in range(3)]
    ks, ps = series({d: K_ARB for d in range(5, 9)}, {d: P_PX for d in range(5, 9)})
    structs, ser = [], {}
    for i, (k, p) in enumerate(pairs):
        structs.append(parity_struct(k, p, f"s{i}"))
        ser[k.key], ser[p.key] = ks, ps
    bt = Backtester(bt_cfg(max_concurrent_capital=1_500, max_capital_per_trade=1_000))
    res = bt._simulate(bt._signals(structs, ser, NOW_BT), NOW_BT)
    # first trade ~$1000 leaves ~$500 headroom: the second is down-sized to fit, the third has < 1 unit of room
    assert len(res.trades) == 2
    assert res.trades[0].cost + res.trades[1].cost <= 1_500 + 1e-6 and res.trades[1].cost < res.trades[0].cost
    assert res.stats["skipped"]["concurrency"] + res.stats["skipped"]["min_size"] >= 1
    assert res.locked.max() <= 1_500 + 1e-6                        # the invariant that matters: C_max is never breached


def test_backtest_kelly_sizes_by_edge_and_basis_risk():
    k, p = settled_pair()
    ks, ps = series({5: 0.49, 6: 0.49}, {5: 0.51, 6: 0.51})         # 1.5% net edge (.49 + .495), cross-venue
    st = parity_struct(k, p)
    bt = Backtester(bt_cfg(sizing="kelly", basis_risk_prob=0.02))
    res = bt._simulate(bt._signals([st], {k.key: ks, p.key: ps}, NOW_BT), NOW_BT)
    assert res.trades == [] and res.stats["skipped"]["kelly_zero"] == 1        # edge < mismatch risk -> no bet
    k2, p2 = settled_pair(mid="2")
    ks2, ps2 = series({5: K_ARB, 6: K_ARB}, {5: P_PX, 6: P_PX})
    bt2 = Backtester(bt_cfg(sizing="kelly", basis_risk_prob=0.02, kelly_mult=0.25, max_capital_per_trade=50_000))
    res2 = bt2._simulate(bt2._signals([parity_struct(k2, p2)], {k2.key: ks2, p2.key: ps2}, NOW_BT), NOW_BT)
    assert len(res2.trades) == 1
    f = kelly_fraction(1 / UNIT - 1, 0.02) * 0.25
    assert abs(res2.trades[0].cost - f * 10_000) < UNIT + 1e-6               # within one unit of f* x equity


def test_metrics_drawdown_sign():
    k, p = settled_pair("no", "yes", mid="3")
    k2, p2 = settled_pair("yes", "yes", settle_day=44, mid="4")
    ks, ps = series({5: K_ARB, 6: K_ARB, 20: K_ARB, 21: K_ARB}, {d: P_PX for d in (5, 6, 20, 21)})
    bt = Backtester(bt_cfg(max_capital_per_trade=2_000))
    sigs = bt._signals([parity_struct(k, p, "a"), parity_struct(k2, p2, "b")],
                       {k.key: ks, p.key: ps, k2.key: ks, p2.key: ps}, NOW_BT)
    res = bt._simulate(sigs, NOW_BT)
    assert res.metrics["Max Drawdown ($)"] > 0 and res.metrics["Max Drawdown (%)"] < 0
    assert (res.drawdown <= 1e-12).all()


# ── long-shot screen ──────────────────────────────────────────────────────────
def _ls(markets, **kw):
    from config import LongShotConfig
    from core.longshots import find_longshots
    cfg = LongShotConfig(**kw)
    return find_longshots(markets, cfg, FeeConfig(), NOW)


def test_longshot_carry_math_matches_hand_calc():
    m = mk(K, "ls1", 0.02, 0.03, days=30, volume_usd=50_000.0, volume_24h_usd=4_000.0, title="Will X happen?")
    (o,) = _ls([m])
    assert abs(o.prob - 0.025) < 1e-12
    assert abs(o.no_ask - 0.98) < 1e-12
    fee = 0.07 * 0.98 * 0.02                                    # continuous Kalshi fee per NO unit
    assert abs(o.fee - fee) < 1e-12
    assert abs(o.carry_gross - 0.02 / 0.98) < 1e-12
    assert abs(o.carry_net - (0.02 - fee) / (0.98 + fee)) < 1e-12
    assert abs(o.carry_apy - ((1 + o.carry_net) ** (365.25 / 30) - 1)) < 1e-9
    assert o.no_depth_usd == 0.98 * 1000.0                       # Kalshi listing carries the size at the best bid


def test_scenario_table_ev():
    from core.longshots import scenario_table
    m = mk(K, "ls1", 0.02, 0.03, days=30, volume_usd=50_000.0)
    (o,) = _ls([m])
    cost = 0.98 + 0.07 * 0.98 * 0.02
    rows = {r["multiple"]: r for r in scenario_table(o)}
    assert abs(rows[1.0]["ev_pct"] - ((1 - 0.025) - cost) / cost) < 1e-12        # slightly negative: fees
    assert rows[0.25]["ev_pct"] > 0 > rows[5.0]["ev_pct"]
    assert all(r["lose_pnl"] == -1.0 for r in rows.values())


def test_longshot_without_a_yes_bid_has_no_carry():
    m = mk(P, "ls2", None, 0.001, volume_usd=1e6, volume_24h_usd=1e4, neg_risk=True)
    (o,) = _ls([m])
    assert o.no_ask is None and o.carry_net is None and o.carry_apy is None
    assert any("NO UNAVAILABLE" in f for f in o.flags) and o.me_event


def test_longshot_filters_flags_and_tags():
    base = dict(volume_usd=60_000.0, volume_24h_usd=5_000.0)
    ok = mk(P, "a", 0.01, 0.02, days=30, title="Will the US confirm that aliens exist by October 31?", **base)
    cat = mk(P, "b", 0.02, 0.03, days=30, title="Will China invade Taiwan by end of 2026?", **base)
    big_p = mk(P, "c", 0.20, 0.22, days=30, **base)                            # not a long-shot
    tiny_vol = mk(P, "d", 0.01, 0.02, days=30, volume_usd=500.0)               # not sizable
    kalshi_oi = mk(K, "e", 0.01, 0.02, days=30, volume_usd=500.0, open_interest=80_000.0)   # sizable via OI
    overdue = mk(P, "f", 0.01, 0.02, days=-40, **base)
    longd = mk(P, "g", 0.01, 0.02, days=800, **base)
    closed = mk(P, "h", 0.01, 0.02, days=30, status="closed", **base)
    got = {o.market_id: o for o in _ls([ok, cat, big_p, tiny_vol, kalshi_oi, overdue, longd, closed])}
    assert set(got) == {"a", "b", "e", "f", "g"}
    assert got["a"].tags == ["absurd"] and got["b"].tags == ["catastrophic"]
    assert any(f.startswith("OVERDUE") for f in got["f"].flags)
    assert any(f.startswith("LONG-DATED") for f in got["g"].flags)
    assert any(f.startswith("CARRY < T-BILL") for f in got["g"].flags)          # 800d at ~1% carry << 5% T-bill
    assert any(f.startswith("INACTIVE") for f in got["e"].flags)


def test_longshot_sort_order_and_cross_venue_annotation():
    a = mk(P, "a", 0.01, 0.02, days=30, volume_usd=60_000.0, volume_24h_usd=1_000.0)
    b = mk(P, "b", 0.01, 0.02, days=30, volume_usd=90_000.0, volume_24h_usd=9_000.0)
    other = mk(K, "ka", 0.02, 0.03, days=30)
    from config import LongShotConfig
    from core.longshots import find_longshots
    out = find_longshots([a, b], LongShotConfig(), FeeConfig(), NOW, {a.key: other})
    assert [o.market_id for o in out] == ["b", "a"]                            # 24h cash volume first
    assert out[1].other_venue == "Kalshi YES 0.020/0.030" and out[0].other_venue == ""


def test_longshot_probability_respects_the_visible_book():
    from core.longshots import _prob
    assert abs(_prob(mk(P, "m", 0.01, 0.03)) - 0.02) < 1e-12
    assert _prob(mk(P, "m", 0.40, None, last_price=0.001)) == 0.40            # bid only: never below the bid
    assert abs(_prob(mk(P, "m", None, 0.05, last_price=0.01)) - 0.01) < 1e-12  # ask only: last trade if below the ask
    assert _prob(mk(P, "m", None, 0.05)) == 0.05


def test_polymarket_second_outcome_flips_last_trade_price():
    """Regression: 'Zeu5' (outcome 1) showed p=0.1% / NO ask 0.001 because lastTradePrice (outcome 0) was not flipped."""
    from connectors.poly_client import markets_from_gamma
    raw = {"conditionId": "0xabc", "question": "Game 2 Winner", "outcomes": '["Maryville", "Zeu5"]',
           "outcomePrices": '["0.001", "0.999"]', "clobTokenIds": '["t0", "t1"]', "bestAsk": 0.001, "bestBid": None,
           "lastTradePrice": 0.001, "volumeNum": 84132.0, "volume24hr": 84122.0, "acceptingOrders": True,
           "enableOrderBook": True, "active": True, "endDate": "2026-10-05T12:00:00Z"}
    m0, m1 = markets_from_gamma(raw, {"id": "e1", "title": "LoL: Maryville vs Zeu5"})
    assert abs(m0.last_price - 0.001) < 1e-12 and abs(m1.last_price - 0.999) < 1e-12
    assert m1.volume_usd == 84132.0 and m1.volume_24h_usd == 84122.0
    from core.longshots import find_longshots
    from config import LongShotConfig
    got = find_longshots([m0, m1], LongShotConfig(), FeeConfig(), NOW)
    assert [o.title.endswith("Maryville") for o in got] == [True]               # only the genuine long-shot remains


def test_kalshi_cash_volume_is_estimated_from_contracts():
    from connectors.kalshi_client import market_from_json
    d = {"ticker": "KX-1", "event_ticker": "KX", "status": "active", "yes_bid_dollars": "0.0200",
         "yes_ask_dollars": "0.0300", "yes_bid_size_fp": "500", "yes_ask_size_fp": "500",
         "last_price_dollars": "0.0200", "volume_fp": "100000", "volume_24h_fp": "5000", "open_interest_fp": "40000",
         "expected_expiration_time": "2026-12-01T00:00:00Z", "close_time": "2026-12-01T00:00:00Z"}
    m = market_from_json(d)
    assert abs(m.volume_usd - 2000.0) < 1e-9 and abs(m.volume_24h_usd - 100.0) < 1e-9 and m.open_interest == 40000.0


# ── Polymarket US (the US-tradable Polymarket) ────────────────────────────────
US_EVENT_SPORTS = {"id": "9", "slug": "nfl-atl-no-2026-10-05", "title": "ATL Falcons vs. NO Saints",
                   "category": "sports", "primaryTag": {"slug": "nfl"}, "endDate": "2026-10-06T04:00:00Z",
                   "teams": [{"name": "Atlanta Falcons"}, {"name": "New Orleans Saints"}]}
US_EVENT_POL = {"id": "7", "slug": "usho-midterms-2026-11-03", "title": "U.S House Midterm Winner",
                "category": "politics", "endDate": "2027-02-02T04:59:00Z"}


def _us_market(slug, long_desc, short_desc, bid, ask, question="Q?", title=None, mtype="futures"):
    q = lambda v: None if v is None else {"value": f"{v:.4f}", "currency": "USD"}      # noqa: E731
    return {"id": "1", "slug": slug, "question": question, "title": title, "description": "rules text",
            "status": "MARKET_STATUS_OPEN", "active": True, "closed": False, "hidden": False,
            "marketType": mtype, "category": "sports", "endDate": "2026-10-06T04:00:00Z", "feeCoefficient": 0.0695,
            "minimumTradeQty": 1, "bestBidQuote": q(bid), "bestAskQuote": q(ask),
            "marketSides": [{"description": long_desc, "long": True, "price": str(ask)},
                            {"description": short_desc, "long": False, "price": str(bid)}]}


def test_polymarket_us_fee_matches_the_published_example():
    from core.pricing import poly_us_fee
    assert poly_us_fee(1000, 0.10, 0.0695) == 6.26             # docs: 0.0695 x 1,000 x 0.10 x 0.90 = $6.255 -> $6.26
    assert poly_us_fee(100, 0.50, 0.0695) == 1.74              # docs: max $1.74 per 100-lot at p = 0.50
    assert poly_us_fee(1, 0.01, 0.0695) == 0.0                 # sub-cent fees round away entirely


def test_polymarket_us_yes_no_market_and_urls():
    from connectors.polyus_client import event_url, markets_from_us
    m = _us_market("paccc-usho-midterms-2026-11-03-dem", "Yes", "No", 0.9000, 0.9010, "U.S House Midterm Winner",
                   "Democratic Party")
    (mk,) = markets_from_us(m, US_EVENT_POL)
    assert mk.us_tradable and mk.venue is Venue.POLYMARKET and mk.market_id == "paccc-usho-midterms-2026-11-03-dem"
    assert (mk.yes_bid, mk.yes_ask) == (0.90, 0.901)
    assert abs(mk.no_ask - 0.10) < 1e-12 and abs(mk.no_bid - 0.099) < 1e-12        # NO ask = 1 - YES bid
    assert mk.outcome_label == "Democratic Party" and mk.fee_rate == 0.0695 and mk.fee_round_cents
    assert mk.url == "https://polymarket.us/event/usho-midterms-2026-11-03"
    assert event_url(US_EVENT_SPORTS) == "https://polymarket.us/sports/nfl/nfl-atl-no-2026-10-05"
    assert not mk.stats_exact and mk.volume_24h_usd < 0                              # listing has no volume


def test_polymarket_us_two_sided_market_gets_a_complement():
    from connectors.polyus_client import markets_from_us
    m = _us_market("aec-nfl-atl-no-2026-10-05", "Falcons", "Saints", 0.8825, 0.8850, "Who will win?", None, "moneyline")
    long_m, short_m = markets_from_us(m, US_EVENT_SPORTS)
    assert long_m.outcome_label == "Falcons" and short_m.outcome_label == "Saints"
    assert short_m.market_id == "aec-nfl-atl-no-2026-10-05#short"
    assert abs(short_m.yes_bid - (1 - 0.885)) < 1e-12 and abs(short_m.yes_ask - (1 - 0.8825)) < 1e-12
    assert long_m.event_title == "Atlanta Falcons vs. New Orleans Saints"            # full names, not 'ATL ... NO ...'


def _us_total(slug, bid, ask):
    m = _us_market(slug, "Over", "Under", bid, ask, "Will the total in ATL Falcons vs. NO Saints be more than X?",
                   None, "totals")
    m["gameStartTime"] = "2026-10-06T00:15:00Z"
    m["endDate"] = "2026-10-20T04:15:00Z"
    return m


def test_polymarket_us_totals_ladders_are_structural_and_never_mix_quantities():
    """Regression: 'game total', 'ATL team total' and 'quarter total' share ONE question text; 40 fake arbs came from it."""
    from connectors.polyus_client import markets_from_us
    specs = [("tsc-nfl-atl-no-2026-10-05-total-60pt5", 0.30, 0.31), ("tsc-nfl-atl-no-2026-10-05-total-50pt5", 0.50, 0.51),
             ("tsc-nfl-atl-no-2026-10-05-tt-atl-33pt5", 0.40, 0.41), ("tsc-nfl-atl-no-2026-10-05-3q-9pt5", 0.60, 0.61)]
    mks = [x for slug, b, a in specs for x in markets_from_us(_us_total(slug, b, a), US_EVENT_SPORTS)]
    by_id = {x.market_id: x for x in mks}
    over, under = by_id["tsc-nfl-atl-no-2026-10-05-total-60pt5"], by_id["tsc-nfl-atl-no-2026-10-05-total-60pt5#short"]
    assert (over.direction, over.strike) == ("gt", 60.5) and (under.direction, under.strike) == ("lt", 60.5)
    assert over.ladder_key == "tsc-nfl-atl-no-2026-10-05-total" and by_id[
        "tsc-nfl-atl-no-2026-10-05-tt-atl-33pt5"].ladder_key == "tsc-nfl-atl-no-2026-10-05-tt-atl"
    edges = {(e.inner.market_id, e.outer.market_id) for e in mapper.build_subset_edges(mks)}
    # only the two GAME-total rungs form a ladder: Over 60.5 implies Over 50.5, and Under 50.5 implies Under 60.5
    assert edges == {("tsc-nfl-atl-no-2026-10-05-total-60pt5", "tsc-nfl-atl-no-2026-10-05-total-50pt5"),
                     ("tsc-nfl-atl-no-2026-10-05-total-50pt5#short", "tsc-nfl-atl-no-2026-10-05-total-60pt5#short")}


def test_polymarket_us_game_markets_settle_at_game_time_and_moneyline_titles_are_short():
    from connectors.polyus_client import markets_from_us
    ml = _us_market("aec-nfl-atl-no-2026-10-05", "Falcons", "Saints", 0.8825, 0.8850,
                    "Who will win in the upcoming football event Atlanta Falcons vs New Orleans Saints scheduled for ...",
                    None, "moneyline")
    ml["gameStartTime"] = "2026-10-06T00:15:00Z"
    ml["endDate"] = "2026-10-20T04:15:00Z"                       # the expiry DEADLINE, two weeks after the game
    a, b = markets_from_us(ml, US_EVENT_SPORTS)
    assert a.title == "Falcons wins" and b.title == "Saints wins"
    assert a.settle_time.isoformat() == "2026-10-06T03:15:00+00:00"          # game start + 3h, not Oct 20
    assert a.date_tokens == ["10-05"] and a.rules.startswith("Who will win")
    fut = _us_market("tec-mlb-nlchamp-2026-09-27-atl", "Yes", "No", 0.17, 0.179, "National League Champion", "Atlanta Braves")
    fut["gameStartTime"] = "2026-09-07T00:00:00Z"
    fut["endDate"] = "2026-11-06T21:20:09Z"
    (f,) = markets_from_us(fut, US_EVENT_SPORTS)
    assert f.settle_time.isoformat() == "2026-11-06T21:20:09+00:00" and f.date_tokens == []   # futures keep endDate


def test_kalshi_moneyline_matches_polymarket_us_moneyline():
    from connectors.polyus_client import markets_from_us
    ml = _us_market("aec-mlb-lad-atl-2026-10-06", "Los Angeles Dodgers", "Atlanta Braves", 0.50, 0.505,
                    "Who will win in the upcoming baseball event Los Angeles Dodgers vs Atlanta Braves scheduled for ...",
                    None, "moneyline")
    ml["gameStartTime"] = "2026-10-06T22:00:00Z"
    ml["endDate"] = "2026-10-21T02:00:00Z"
    ev = {"id": "5", "slug": "mlb-lad-atl-2026-10-06", "title": "LA Dodgers vs. ATL Braves", "category": "sports",
          "primaryTag": {"slug": "mlb"}, "teams": [{"name": "Los Angeles Dodgers"}, {"name": "Atlanta Braves"}]}
    us_long, us_short = markets_from_us(ml, ev)
    ks = [mk(K, "KXMLBGAME-26OCT061800LADATL-LAD", 0.50, 0.51, title="Los Angeles D wins", label="Los Angeles D",
             event_title="Game 3: Los Angeles D vs Atlanta", days=0, settle_time=us_long.settle_time + timedelta(hours=0.5),
             date_tokens=["10-06"]),
          mk(K, "KXMLBGAME-26OCT061800LADATL-ATL", 0.49, 0.50, title="Atlanta wins", label="Atlanta",
             event_title="Game 3: Los Angeles D vs Atlanta", days=0, settle_time=us_long.settle_time + timedelta(hours=0.5),
             date_tokens=["10-06"])]
    got = {(pr.a.market_id[-3:], pr.b.market_id): pr for pr in mapper.match_pairs(ks, [us_long, us_short], 24, 0.0)}
    assert set(got) == {("LAD", "aec-mlb-lad-atl-2026-10-06"), ("ATL", "aec-mlb-lad-atl-2026-10-06#short")}, got.keys()


def _us_game(slug_date, long_t, short_t, league, game_start, bid=0.89, ask=0.90):
    """US moneyline fixture with real-shaped team objects (nickname names, abbreviations like Kalshi's ticker suffix)."""
    m = _us_market(f"aec-{league}-min-mil-{slug_date}", long_t[0], short_t[0], bid, ask, "Who will win?", None, "moneyline")
    m["gameStartTime"] = game_start
    m["endDate"] = "2026-10-20T04:15:00Z"
    m["marketSides"][0]["team"] = {"name": long_t[0], "displayAbbreviation": long_t[1], "abbreviation": long_t[1].lower()}
    m["marketSides"][1]["team"] = {"name": short_t[0], "displayAbbreviation": short_t[1], "abbreviation": short_t[1].lower()}
    ev = {"id": "5", "slug": f"{league}-min-mil-{slug_date}", "title": f"{long_t[1]} {long_t[0]} vs. {short_t[1]} {short_t[0]}",
          "category": "sports", "primaryTag": {"slug": league}, "teams": [{"name": long_t[0]}, {"name": short_t[0]}]}
    return m, ev


def _k_game(ticker_suffix, label, event_title, series="KXNBAGAME", date="10-05", settle_h=3.0, bid=0.93, ask=0.94):
    from connectors.kalshi_client import market_from_json
    d = {"ticker": f"{series}-26OCT05MINMIL-{ticker_suffix}", "event_ticker": f"{series}-26OCT05MINMIL",
         "status": "active", "yes_bid_dollars": f"{bid}", "yes_ask_dollars": f"{ask}", "yes_bid_size_fp": "100",
         "yes_ask_size_fp": "100", "yes_sub_title": label, "title": f"{label} wins",
         "expected_expiration_time": "2026-10-06T03:15:00Z", "close_time": "2026-10-09T00:00:00Z", "volume_fp": "5000"}
    return market_from_json(d, {"event_ticker": d["event_ticker"], "title": event_title, "series_ticker": series})


def test_game_moneylines_match_on_team_abbreviation_even_when_names_differ():
    """Kalshi 'Minnesota' (city) vs Polymarket US 'Timberwolves' (nickname): only the abbreviation + league + date agree."""
    from connectors.polyus_client import markets_from_us
    m, ev = _us_game("2026-10-05", ("Timberwolves", "MIN"), ("Bucks", "MIL"), "nba", "2026-10-06T00:00:00Z")
    us_long, us_short = markets_from_us(m, ev)
    assert us_long.subject_ids == ["min"] and us_short.subject_ids == ["mil"] and us_long.league == "nba"
    k_min = _k_game("MIN", "Minnesota", "Minnesota vs Milwaukee")
    k_mil = _k_game("MIL", "Milwaukee", "Minnesota vs Milwaukee", bid=0.05, ask=0.06)
    assert (k_min.kind, k_min.subject_ids, k_min.league) == ("moneyline", ["min"], "nba")
    got = {(pr.a.market_id[-3:], pr.b.market_id[-6:]): pr.confidence for pr in
           mapper.match_pairs([k_min, k_mil], [us_long, us_short], 24, 0.0)}
    assert set(got) == {("MIN", "-10-05"), ("MIL", "#short")} and all(c >= 0.9 for c in got.values()), got


def test_abbreviation_matching_respects_league_and_date():
    from connectors.polyus_client import markets_from_us
    # same 'MIN' abbreviation but an NFL game (Vikings) on the same date must NOT pair with the Kalshi NBA market
    m, ev = _us_game("2026-10-05", ("Vikings", "MIN"), ("Packers", "GB"), "nfl", "2026-10-06T00:00:00Z")
    nfl_long, _ = markets_from_us(m, ev)
    k_nba = _k_game("MIN", "Minnesota", "Minnesota vs Milwaukee")
    assert mapper.match_pairs([k_nba], [nfl_long], 24, 0.0) == []
    # right league, wrong date
    m2, ev2 = _us_game("2026-10-12", ("Timberwolves", "MIN"), ("Bucks", "MIL"), "nba", "2026-10-13T00:00:00Z")
    later, _ = markets_from_us(m2, ev2)
    assert mapper.match_pairs([k_nba], [later], 24, 0.0) == []


def test_polymarket_us_event_urls_game_vs_futures():
    from connectors.polyus_client import event_url
    game = {"slug": "mlb-nyy-tb-2026-10-05", "category": "sports", "primaryTag": {"slug": "mlb"},
            "teams": [{"name": "a"}, {"name": "b"}]}
    fut = {"slug": "nba-champ-2027-06-18-w", "category": "sports", "primaryTag": {"slug": "nba"},
           "teams": [{"name": str(i)} for i in range(30)]}
    assert event_url(game) == "https://polymarket.us/sports/mlb/mlb-nyy-tb-2026-10-05"
    assert event_url(fut) == "https://polymarket.us/event/nba-champ-2027-06-18-w"


def test_refresh_quote_reprices_against_fresh_books_and_exposes_drift():
    """'Refresh live quote': re-fetch every leg's book NOW, re-price, so a stale scan price is visible before trading."""
    import asyncio
    import core.scanner as sc
    from core.models import OrderBook
    k = mk(K, "KX-1", 0.38, 0.40, volume=9_000.0)
    p = mk(P, "pu-1", 0.45, 0.47, volume=9_000.0)
    det_ = ArbDetector(ScanSettings(min_confidence=0.5, min_volume=0, fees=NO_FEES))
    cand = det_.detect_parity([mapper.MarketPair(k, p, Relation.EQUIVALENT, 0.9)], NOW)[0]
    scanner = sc.LiveScanner(ScanSettings(fees=NO_FEES))
    scanner.last = sc.ScanResult(opportunities=[cand.opp], recipes={cand.opp.key: cand.recipe})
    unit_scan = cand.opp.total_cost / cand.opp.size                                    # 0.40 + 0.55 = 0.95

    class FakeBooks:
        request_count = 0
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get_orderbook(self, ticker):                                         # Kalshi: YES ask now 0.43
            return OrderBook(venue=K, market_id=ticker, yes_asks=[(0.43, 500.0)], no_asks=[(0.60, 500.0)])
        async def get_market_book(self, m):                                            # Polymarket US: NO ask now 0.56
            return OrderBook(venue=P, market_id=m.market_id, yes_asks=[(0.47, 500.0)], no_asks=[(0.56, 500.0)])

    real = (sc.KalshiClient, sc.PolyUSClient, sc.PolyClient)
    sc.KalshiClient = sc.PolyUSClient = sc.PolyClient = FakeBooks
    try:
        fresh = asyncio.run(scanner.refresh_quote(cand.opp, 100.0))
    finally:
        sc.KalshiClient, sc.PolyUSClient, sc.PolyClient = real
    assert fresh is not None and fresh.depth_verified and abs(fresh.size - 100.0) < 1e-9
    unit_live = fresh.total_cost / fresh.size
    assert abs(unit_scan - 0.95) < 1e-9 and abs(unit_live - (0.43 + 0.56)) < 1e-9      # drifted from 0.95 to 0.99
    assert fresh.detected_at > cand.opp.detected_at
    # a vanished book is reported as None, never silently priced
    class Empty(FakeBooks):
        async def get_orderbook(self, ticker): return OrderBook(venue=K, market_id=ticker)
    sc.KalshiClient, sc.PolyUSClient, sc.PolyClient = Empty, Empty, Empty
    try:
        assert asyncio.run(scanner.refresh_quote(cand.opp, 100.0)) is None
    finally:
        sc.KalshiClient, sc.PolyUSClient, sc.PolyClient = real


def test_polymarket_us_closed_or_one_sided_markets():
    from connectors.polyus_client import markets_from_us
    closed = _us_market("x", "Yes", "No", 0.5, 0.6)
    closed["status"] = "MARKET_STATUS_RESOLVED"
    assert markets_from_us(closed, US_EVENT_POL) == []
    ask_only = _us_market("y", "Yes", "No", None, 0.002)
    (mk,) = markets_from_us(ask_only, US_EVENT_POL)
    assert mk.yes_bid is None and mk.no_ask is None and mk.yes_ask == 0.002          # NO cannot be bought
    assert markets_from_us(_us_market("z", "Yes", "No", None, None), US_EVENT_POL) == []


def test_polymarket_us_book_orientation_and_complement():
    from connectors.polyus_client import book_from_us
    raw = {"marketData": {"bids": [{"px": {"value": "0.1650"}, "qty": "2500"}, {"px": {"value": "0.1700"}, "qty": "30"}],
                          "offers": [{"px": {"value": "0.1820"}, "qty": "8"}, {"px": {"value": "0.1790"}, "qty": "166"}],
                          "stats": {"notionalTraded": {"value": "177953.7"}, "openInterest": "149047",
                                    "lastTradePx": {"value": "0.1790"}}}}
    book, st = book_from_us(raw, "s")
    assert book.yes_bids[0] == (0.17, 30.0) and book.yes_asks[0] == (0.179, 166.0)       # sorted best-first
    assert book.no_asks[0] == (0.83, 30.0)                                                 # NO ask = 1 - YES bid
    assert st == {"notional": 177953.7, "shares": None, "oi": 149047.0, "last": 0.179}
    short, _ = book_from_us(raw, "s", complement=True)
    assert short.yes_asks[0] == (0.83, 30.0) and short.no_asks[0] == (0.179, 166.0) and short.market_id == "s#short"


def test_kalshi_and_polymarket_links_are_event_specific():
    from connectors.kalshi_client import market_from_json
    d = {"ticker": "KXBTCD-26OCT0917-T75599.99", "event_ticker": "KXBTCD-26OCT0917", "status": "active",
         "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42", "yes_bid_size_fp": "5", "yes_ask_size_fp": "5",
         "expected_expiration_time": "2026-10-09T21:00:00Z", "close_time": "2026-10-09T21:00:00Z"}
    m = market_from_json(d)
    assert m.url == "https://kalshi.com/events/KXBTCD-26OCT0917" and m.us_tradable
    from connectors.poly_client import markets_from_gamma
    raw = {"conditionId": "0xabc", "question": "Q", "outcomes": '["Yes","No"]', "outcomePrices": '["0.1","0.9"]',
           "clobTokenIds": '["a","b"]', "slug": "will-x", "acceptingOrders": True, "enableOrderBook": True,
           "active": True, "endDate": "2026-12-01T00:00:00Z"}
    (g,) = markets_from_gamma(raw, {"id": "1", "slug": "ev-slug", "title": "T"})
    assert g.url == "https://polymarket.com/event/ev-slug" and g.us_tradable is False


def test_us_only_filter_drops_international_legs():
    import asyncio
    import core.scanner as sc

    k = mk(K, "k1", 0.38, 0.40, volume=5_000.0, event_title="Will X happen by Dec?", label="X", title="Will X happen?")
    p_intl = mk(P, "p1", 0.45, 0.47, volume=5_000.0, event_title="Will X happen by Dec?", label="X",
                title="Will X happen?", us_tradable=False)

    class FakeK:
        request_count = 0
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get_open_events(self, n, cb=None): return [k]

    class FakeG(FakeK):
        warnings: list = []
        async def get_open_events(self, n, cb=None): return [p_intl]

    real = (sc.KalshiClient, sc.PolyClient)
    sc.KalshiClient, sc.PolyClient = FakeK, FakeG
    try:
        base = dict(use_l2_books=False, min_volume=0, min_confidence=0.5, poly_source="global")
        res_all = asyncio.run(sc.LiveScanner(ScanSettings(us_only=False, **base)).scan_once())
        res_us = asyncio.run(sc.LiveScanner(ScanSettings(us_only=True, **base)).scan_once())
    finally:
        sc.KalshiClient, sc.PolyClient = real
    assert any("INTERNATIONAL" in w for w in res_us.warnings)
    assert [o for o in res_us.opportunities if not o.us_tradable] == []
    assert all(not o.us_tradable for o in res_all.opportunities if any(not l.us_tradable for l in o.legs))


# ── regressions from live use ─────────────────────────────────────────────────
def test_exactly_n_is_a_bin_not_a_ladder():
    """Starship: YES('exactly 9') + NO('exactly 6') pays $0 if exactly 6 happens - it is NOT an arbitrage."""
    ps = mapper.parse_strike
    assert ps("Will exactly 9 Starship launches reach space in 2026?")[:2] == ("eq", 9.0)
    assert ps("Will at least 6 Starship launches reach space in 2026?")[:2] == ("gt", 6.0)
    assert ps("Will 7 or more Starship launches reach space?")[:2] == ("gt", 7.0)
    assert ps("Will Bitcoin reach $100,000 in August?")[:2] == ("gt", 100000.0)      # 'reach' + number still counts
    bins = [mk(K, f"b{n}", 0.4, 0.42, event="STARSHIP", title=f"Will exactly {n} Starship launches reach space in 2026?",
               label=f"{n}") for n in (6, 9)]
    assert mapper.build_subset_edges(bins) == []
    # even if the text were mis-parsed, a mutually-exclusive event can never be a nested ladder
    me = [mk(K, f"m{n}", 0.4, 0.42, event="ME", title="How many launches?", label=f"{n}+", direction="gt",
             strike=float(n), mutually_exclusive=True) for n in (6, 9)]
    assert mapper.build_subset_edges(me) == []
    nr = [mk(P, f"n{n}", 0.4, 0.42, event="NR", title="How many launches?", label=f"{n}+", direction="gt",
             strike=float(n), neg_risk=True) for n in (6, 9)]
    assert mapper.build_subset_edges(nr) == []


def _match(k, p):
    return mapper.match_pairs([k], [p], 24, 0.0)


def test_opponent_market_is_never_matched_as_the_same_outcome():
    """'San Lorenzo wins' vs 'Defensa y Justicia win': same event, OPPOSITE sides -> K YES + P NO can pay $0."""
    ev = "San Lorenzo de Almagro vs Defensa y Justicia"
    k = mk(K, "k1", 0.40, 0.42, title="San Lorenzo de Almagro wins", event_title=ev, label="San Lorenzo de Almagro",
           days=2)
    opp = mk(P, "p1", 0.40, 0.42, title="Will CSyD Defensa y Justicia win on 2026-10-03?",
             label="CSyD Defensa y Justicia", event_title=ev, days=2)
    same = mk(P, "p2", 0.40, 0.42, title="Will San Lorenzo de Almagro win on 2026-10-03?",
              label="San Lorenzo de Almagro", event_title=ev, days=2)
    assert _match(k, opp) == []
    assert [pr.b.market_id for pr in _match(k, same)] == ["p2"]          # positive control: right subject still matches


def test_two_way_virtual_market_requires_the_same_team():
    """Kalshi 'Georgia St. wins' vs Polymarket's 'Old Dominion vs. Georgia State - Old Dominion' (outcome = ODU)."""
    k = mk(K, "k1", 0.50, 0.53, title="Georgia St. wins", event_title="Old Dominion vs Georgia St.",
           label="Georgia St.", days=2)
    odu = mk(P, "p1", 0.46, 0.47, title="Old Dominion vs. Georgia State - Old Dominion", label="Old Dominion", days=2)
    assert _match(k, odu) == []


def test_settlement_price_and_intraday_touch_are_different_events():
    """Kalshi 'front-month SETTLE price reach $105' is a subset of Polymarket's intraday 'hit $105'."""
    k = mk(K, "k1", 0.26, 0.268, title="Will the maximum WTI front month settle price reach $105.00 by Oct 30, 2026?",
           days=29)
    p_basis = mk(P, "p1", 0.32, 0.326, title="Will WTI Crude Oil (WTI) hit (HIGH) $105 in October?", days=29)
    p_same = mk(P, "p2", 0.32, 0.326, title="Will the maximum WTI front month settle price reach $105 in October?",
                days=29)
    assert _match(k, p_basis) == []
    assert len(_match(k, p_same)) == 1


def test_same_market_type_on_a_different_game_is_not_a_match():
    """Kalshi 'Croatia vs England: BTTS' vs Polymarket 'Spain vs Czechia: BTTS' (an 11% 'edge' that was two games)."""
    k = mk(K, "k1", 0.57, 0.58, title="Both Teams To Score", event_title="Croatia vs England: BTTS",
           label="Both Teams To Score", days=2)
    other = mk(P, "p1", 0.35, 0.36, title="Spain vs. Czechia: Both Teams to Score",
               event_title="Spain vs. Czechia - More Markets", label="Both Teams to Score", days=2)
    half = mk(P, "p2", 0.35, 0.36, title="Spain vs. England: Both Teams to Score",
              event_title="Spain vs. England - More Markets", label="Both Teams to Score", days=2)
    same = mk(P, "p3", 0.55, 0.56, title="England vs. Croatia: Both Teams to Score",
              event_title="England vs. Croatia - More Markets", label="Both Teams to Score", days=2)
    assert _match(k, other) == []
    assert _match(k, half) == []                      # sharing ONE of two team names is not enough
    assert [pr.b.market_id for pr in _match(k, same)] == ["p3"]          # positive control (home/away order differs)


def test_macro_markets_must_be_about_the_same_economy():
    k = mk(K, "k1", 0.30, 0.32, title="Will real GDP increase by more than 3.5% in Q3 2026?",
           event_title="US real GDP growth in Q3 2026?", label="Above 3.5%", days=28)
    mexico = mk(P, "p1", 0.30, 0.32, title="Will Mexico GDP growth in Q3 2026 be at least 3.5%?",
                event_title="Mexico GDP growth in Q3 2026?", label="3.5%+", days=28)
    euro = mk(P, "p2", 0.30, 0.32, title="Will Eurozone GDP growth in Q3 2026 be at least 3.5%?",
              event_title="Eurozone GDP growth in Q3 2026", label="3.5%+", days=28)
    us = mk(P, "p3", 0.30, 0.32, title="Will US GDP growth in Q3 2026 be greater than 3.5%?",
            event_title="US GDP growth in Q3 2026?", label="≥3.5%", days=28)
    assert _match(k, mexico) == [] and _match(k, euro) == []
    assert [pr.b.market_id for pr in _match(k, us)] == ["p3"]            # positive control


def test_large_parity_edge_carries_a_warning():
    k = mk(K, "k1", 0.38, 0.40)
    p = mk(P, "p1", 0.55, 0.57)                                          # NO ask 0.45 -> cost 0.85 (17.6% edge)
    c = det().detect_parity([mapper.MarketPair(k, p, Relation.EQUIVALENT, 0.9)], NOW)
    assert c and c[0].opp.notes[0].startswith("WARNING")


def test_different_quarters_are_not_one_ladder():
    """Steelers-Browns: '1Q O/U 7.5 Over' and '2Q O/U 10.5 Over' are different quantities, not rungs of a ladder."""
    def tot(seg, line, name, direction="gt"):
        return mk(P, f"{seg}{line}{direction}", 0.4, 0.42, event="NFL", event_title="Steelers vs. Browns",
                  title=f"Steelers vs. Browns: {seg} O/U {line} - {'Over' if direction == 'gt' else 'Under'}",
                  label="Over" if direction == "gt" else "Under", direction=direction, strike=float(line))
    mixed = [tot("1Q", 7.5, "a"), tot("2Q", 10.5, "b")]
    assert mapper.build_subset_edges(mixed) == []
    same = [tot("2Q", 8.5, "c"), tot("2Q", 10.5, "d")]                  # same quarter -> a genuine ladder
    assert {(e.inner.market_id, e.outer.market_id) for e in mapper.build_subset_edges(same)} == {("2Q10.5gt", "2Q8.5gt")}
    assert mapper.stem_key(mixed[0]) != mapper.stem_key(mixed[1])


def test_spread_markets_are_never_auto_paired_across_venues():
    k = mk(K, "k1", 0.43, 0.443, title="Louisiana Tech wins by over 2.5 points", event_title="Army vs Louisiana Tech: Spread",
           label="Louisiana Tech", days=2)
    p = mk(P, "p1", 0.49, 0.50, title="Spread: Army (-2.5) - Louisiana Tech", label="Louisiana Tech", days=2)
    assert _match(k, p) == []


def test_draw_market_is_not_the_team_win_market():
    """K NO('Vasco wins') + P YES('Botafogo-Vasco draw') pays $0 when Vasco wins."""
    k = mk(K, "k1", 0.33, 0.35, title="Vasco da Gama wins", event_title="Botafogo FR vs Vasco da Gama",
           label="Vasco da Gama", days=6)
    draw = mk(P, "p1", 0.28, 0.287, title="Will Botafogo FR vs. CR Vasco da Gama end in a draw?",
              label="Draw (Botafogo FR vs. CR Vasco da Gama)", days=6, event_title="Botafogo FR vs. CR Vasco da Gama")
    win = mk(P, "p2", 0.33, 0.35, title="Will CR Vasco da Gama win on 2026-10-05?", label="CR Vasco da Gama", days=6,
             event_title="Botafogo FR vs. CR Vasco da Gama")
    assert _match(k, draw) == []
    assert [pr.b.market_id for pr in _match(k, win)] == ["p2"]


def test_ladder_rungs_must_resolve_at_the_same_moment():
    a = mk(K, "d1", 0.4, 0.42, event="EV", title="Bitcoin price?", label="$100,000 or above", direction="gt",
           strike=100_000.0, days=10)
    b = mk(K, "d2", 0.4, 0.42, event="EV", title="Bitcoin price?", label="$90,000 or above", direction="gt",
           strike=90_000.0, days=13)                       # three days later: not comparable
    assert mapper.build_subset_edges([a, b]) == []


def test_keyset_pagination_keeps_partial_results_and_warns():
    import asyncio
    from connectors.http_base import ApiError
    from connectors.poly_client import PolyClient
    calls = []

    async def fake_get_json(path, params=None, **kw):
        calls.append((path, dict(params or {})))
        if len(calls) == 1:
            return {"events": [{"id": "1", "volume": 10}, {"id": "2", "volume": 9}], "next_cursor": "abc"}
        raise ApiError(422, path, "offset too large")

    async def go():
        c = PolyClient()
        c.get_json = fake_get_json
        return c, await c._events({"closed": "false"}, 5, None, "open")

    c, evs = asyncio.run(go())
    assert [e["id"] for e in evs] == ["1", "2"]                       # page-1 events are kept
    assert calls[0][0] == "/events/keyset" and calls[1][1]["after_cursor"] == "abc"
    assert c.warnings and "failed" in c.warnings[0]                    # ...and the failure is surfaced, not swallowed


def test_scan_warns_loudly_when_a_venue_returns_nothing():
    import asyncio
    import core.scanner as sc

    class FakeK:
        request_count = 0

        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get_open_events(self, n, cb=None): return [mk(K, "k1", 0.4, 0.42, volume=5_000)]

    class FakePoly(FakeK):
        warnings = ["Polymarket open: page 1 failed (HTTP 422: offset too large)"]

        async def get_open_events(self, n, cb=None, floors=None): return []

    real = (sc.KalshiClient, sc.PolyClient, sc.PolyUSClient)
    sc.KalshiClient, sc.PolyClient, sc.PolyUSClient = FakeK, FakePoly, FakePoly
    try:
        res = asyncio.run(sc.LiveScanner(ScanSettings(use_l2_books=False)).scan_once())          # default source = US
        res_g = asyncio.run(sc.LiveScanner(ScanSettings(use_l2_books=False, poly_source="global")).scan_once())
    finally:
        sc.KalshiClient, sc.PolyClient, sc.PolyUSClient = real
    assert any("Polymarket US returned NO markets" in w for w in res.warnings)
    assert any("offset too large" in w for w in res.warnings)
    assert not any("INTERNATIONAL" in w for w in res.warnings)                  # US source: no non-US banner
    assert any("INTERNATIONAL" in w and "NOT available to US persons" in w for w in res_g.warnings)


if __name__ == "__main__":
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
