"""Run one or more MR models through the trade-level engine and print the summary + sample trades.  python mr_run.py"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import pandas as pd  # noqa: E402

from resarb.mr import engine as E  # noqa: E402
from resarb.mr.market import Market  # noqa: E402
from resarb.mr.models import MRModel  # noqa: E402

_MK: dict = {}


def market_for(m: MRModel) -> Market:
    syms = E.model_symbols(m)
    key = (tuple(syms), m.minute)
    if key not in _MK:
        t = time.time()
        _MK.clear()
        _MK[key] = Market(syms, minute=m.minute, start="2015-06-01")
        print(f"  market: {len(syms)} symbols, {len(_MK[key].dates)} sessions ({time.time() - t:.0f} s)", flush=True)
    return _MK[key]


def run(m: MRModel, **kw):
    mk = market_for(m)
    t = time.time()
    r = E.run(m, mk, **kw)
    print(f"{m.name}: {E.summary(r.stats)} ({time.time() - t:.0f} s)", flush=True)
    return r


if __name__ == "__main__":
    models = [
        MRModel(name="overnight winners top100 (control)", universe="top100", entry_signal="ret1", entry_op=">", entry_value=-9, hold="overnight", slots=10, exit_signal=""),
        MRModel(name="RSI2<10 >SMA200 top100", universe="top100", entry_signal="rsi2", entry_op="<", entry_value=10, trend_n=200, exit_signal="", exit_sma_n=5, max_hold=10, slots=10),
    ]
    for m in models:
        r = run(m)
        pd.set_option("display.width", 250)
        print(r.trades.head(6).to_string(index=False))
