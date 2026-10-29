"""
Headless daily runner of the momentum book (no GUI): the morning step, the intraday checks, the overnight decision, the after-close step.

  python run_live.py                                   DRY RUN: plans and journals every check and decision, sends nothing
  python run_live.py --arm "ARM PAPER"                 sends the orders to the paper account of the project's key file "Alpaca Kay.txt"
                                                       (MB_ALPACA_API_KEY / MB_ALPACA_SECRET_KEY select a separate account)
  python run_live.py --kill "KILL"                     cancel + flatten this book's positions, then exit
Model: <folder>/live_model.json - written by the GUI when its live loop starts (the model you backtested, with its shadow), or File > Save model. The runner prints it first.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading

from . import live as ML
from . import presets as PR
from .models import MomModel


def main(folder: str, argv=None) -> int:
    ap = argparse.ArgumentParser(description="Momentum book: headless live session")
    ap.add_argument("--arm", default="", help=f'type "{ML.ARM_PHRASE}" to send orders')
    ap.add_argument("--kill", default="", help='type "KILL" to cancel + flatten and exit')
    ap.add_argument("--model", default=os.path.join(folder, "live_model.json"))
    ap.add_argument("--shadow", default="file", help="'file' = the shadow saved with the model, 'none', or a preset name")
    ap.add_argument("--capital", type=float, default=0.0, help="fixed capital (0 = account equity)")
    ap.add_argument("--auction-orders", default="file", choices=("file",) + ML.AUCTION_ORDERS,
                    help="file = the choice saved with the model by the GUI (else auction); auction = MOC / MOO orders, as the model; market = market DAY orders in "
                         "their place, 5 s before the close and 3 min before the open")
    ap.add_argument("--until", default="16:10", help="stop at this ET time")
    a = ap.parse_args(argv)
    if not os.path.exists(a.model):
        print(f"no model file {a.model}: start the loop once in the GUI, or File > Save model, then pass --model")
        return 2
    with open(a.model, encoding="utf-8") as fh:
        spec = json.load(fh)
    m = MomModel.from_dict(spec)
    shadow = None
    if a.shadow == "file" and spec.get("_shadow"):
        shadow = MomModel.from_dict(spec["_shadow"])
    elif a.shadow in PR.PRESETS:
        shadow = PR.get(a.shadow)
    armed = a.arm.strip() == ML.ARM_PHRASE
    if a.arm and not armed:
        print(f'--arm needs exactly "{ML.ARM_PHRASE}" - refusing')
        return 2
    log = lambda msg, k="info": print(f"{ML.now_et():%H:%M:%S} [{k}] {msg}", flush=True)
    log(f"model {a.model}: {json.dumps(m.to_dict())}")
    if shadow:
        log(f"shadow (journaled, never traded): {shadow.name}")
    feed = "sip_delayed" if m.minute <= 1530 else "sip"
    orders = a.auction_orders if a.auction_orders != "file" else ML._OLD_ORDERS.get(spec.get("_auction_orders"), spec.get("_auction_orders")) or "auction"
    if orders not in ML.AUCTION_ORDERS:
        print(f"unknown auction orders {orders!r} in {a.model}")
        return 2
    log(f"auction orders: {orders}")
    eng = ML.MomLiveEngine(m, ML.MomLiveSettings(paper=True, armed=armed, capital=a.capital, feed=feed, auction_orders=orders), folder, log=log, shadow=shadow)
    eng.connect()
    if a.kill:
        if a.kill.strip() != "KILL":
            print('--kill needs exactly "KILL"')
            return 2
        for r in eng.kill():
            print(r)
        return 0
    probs = eng.model_problems() + (eng.arm_problems() if armed else [])
    if probs:
        print("cannot run: " + "; ".join(dict.fromkeys(probs)))
        return 3
    stop = threading.Event()
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), lambda *_: stop.set())
    hh, mm = [int(x) for x in a.until.split(":")]

    def watch():
        while not stop.is_set():
            n = ML.now_et()
            if (n.hour, n.minute) >= (hh, mm):
                stop.set()
            stop.wait(20)
    threading.Thread(target=watch, daemon=True).start()
    eng.run(stop)
    return 0
