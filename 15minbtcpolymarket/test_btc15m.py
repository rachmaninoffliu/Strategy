"""Offline tests for the money-touching pieces of btc15m (fees, settlement P&L, Kelly, decision gating).
Run:  python test_btc15m.py
"""
import numpy as np
import pandas as pd

from strategy import (BucketCalibrator, Calibrator, Config, decide, fee_pc, fee_total, kelly_binary, settle_trade)


def test_fee_rounding():
    assert fee_total(100, 0.50) == 1.75               # 0.07*100*.25 = 1.75 exactly
    assert fee_total(100, 0.90) == 0.63               # 0.07*100*.09 = 0.63 exactly
    assert fee_total(1, 0.50) == 0.02                 # 0.0175 rounds UP to the next cent
    assert fee_total(100, 0.99) == 0.07               # 0.0693 -> 0.07
    assert abs(float(fee_pc(0.5, 100)) - 0.0175) < 1e-12


def test_kelly_formula():
    assert abs(kelly_binary(0.6, 0.5) - 0.2) < 1e-12  # (p-c)/(1-c)
    assert kelly_binary(0.5, 0.5) == 0.0              # zero edge -> zero stake
    assert kelly_binary(0.4, 0.5) == 0.0              # negative edge never bets
    assert kelly_binary(0.99, 0.0) == 0.0 and kelly_binary(0.99, 1.0) == 0.0   # degenerate costs guarded
    # the high-win-rate trap: the same 1c edge is 13% of bankroll at 92.5c but 2% at 50c
    assert abs(kelly_binary(0.935, 0.925) - 0.1333) < 1e-3
    assert abs(kelly_binary(0.51, 0.50) - 0.02) < 1e-12


def test_settle_trade():
    class R:  # minimal row
        pass
    r = R(); r.yes, r.side, r.cost = 1, "YES", 0.90
    assert abs(settle_trade(r, 100) - (100 * 0.10 - 0.63)) < 1e-9          # win: +10.00 - 0.63 fee
    r.yes = 0
    assert abs(settle_trade(r, 100) - (-90.0 - 0.63)) < 1e-9               # loss: -90.00 - 0.63 fee
    r.yes, r.side, r.cost = 0, "NO", 0.15                                  # NO wins when market resolves NO
    assert abs(settle_trade(r, 200) - (200 * 0.85 - fee_total(200, 0.15))) < 1e-9


def _fake_quotes(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    mid = rng.uniform(0.05, 0.95, n)
    bid, ask = np.round(mid - 0.005, 3), np.round(mid + 0.005, 3)
    yes = (rng.random(n) < mid).astype(int)                                # perfectly calibrated market
    return pd.DataFrame(dict(ticker=[f"T{i}" for i in range(n)], minute=rng.integers(1, 14, n), bid_c=bid, ask_c=ask, yes=yes,
                             open_ts=np.arange(n) * 900))


def test_no_trades_on_efficient_market():
    df = _fake_quotes()
    cal = BucketCalibrator.fit(df)
    d = decide(cal, df, Config(legs=("FAV", "FADE"), shrink=1.0, min_edge=0.005))
    # calibrated market + fees + spread => (almost) nothing should clear a 0.5c edge-after-fee hurdle
    assert d.trade.mean() < 0.02, d.trade.mean()


def test_decide_respects_hurdle_and_caps():
    df = _fake_quotes()
    cal = Calibrator.fit(df)
    d = decide(cal, df, Config(min_edge=0.0, kelly_mult=1.0, max_frac=0.05))
    assert (d.frac <= 0.05 + 1e-12).all() and (d.frac >= 0).all()
    assert (d[d.trade].edge >= 0).all()


def test_calibrator_detects_planted_bias():
    # plant a late-window favourite premium: favourites at 0.8-0.9 in minutes 12-13 win 5c less often than quoted
    rng = np.random.default_rng(3)
    n = 40000
    minute = rng.integers(12, 14, n); mid = rng.uniform(0.80, 0.90, n)
    yes = (rng.random(n) < (mid - 0.05)).astype(int)
    df = pd.DataFrame(dict(ticker=[f"T{i}" for i in range(n)], minute=minute, bid_c=mid - .005, ask_c=mid + .005, yes=yes))
    p, se = BucketCalibrator.fit(df).predict(df.minute.values, df.bid_c.values, df.ask_c.values)
    assert abs((p - (df.bid_c + df.ask_c) / 2).mean() + 0.05) < 0.01       # recovers ~-5c
    d = decide(BucketCalibrator.fit(df), df, Config(legs=("FADE",), shrink=1.0, min_edge=0.01))
    assert d.trade.mean() > 0.5 and (d[d.trade].side == "NO").all()       # fades the favourite by buying NO


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all btc15m tests passed")
