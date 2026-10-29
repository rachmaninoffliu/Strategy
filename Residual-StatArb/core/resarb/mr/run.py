"""
Headless daily runner of the mean-reversion engine (no GUI): the morning step, the decision at the decision time, the after-close step.

  python run_live.py                                   DRY RUN: plans and journals the decision, sends nothing
  python run_live.py --arm "ARM PAPER"                 sends market-on-close / market-on-open orders to the paper account of the project's key file "Alpaca Key.txt"
                                                       (MR_ALPACA_API_KEY / MR_ALPACA_SECRET_KEY select a separate account)
  python run_live.py --kill "KILL"                     cancel + flatten this model's positions, then exit
Model: <folder>/live_model.json - written by the GUI when its live loop starts (the model you backtested), or File > Save model. The runner prints it before anything else.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading

from . import live as ML
from .models import MRModel


def main(folder: str, argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mean reversion: headless live session")
    ap.add_argument("--arm", default="", help=f'type "{ML.ARM_PHRASE}" to send orders')
    ap.add_argument("--kill", default="", help='type "KILL" to cancel + flatten and exit')
    ap.add_argument("--model", default=os.path.join(folder, "live_model.json"))
    ap.add_argument("--feed", default="", choices=[""] + list(ML.FEEDS), help="default: sip_delayed for a decision at 15:30 or earlier, sip+iex later")
    ap.add_argument("--capital", type=float, default=0.0, help="fixed capital (0 = account equity)")
    ap.add_argument("--until", default="16:10", help="stop at this ET time")
    a = ap.parse_args(argv)
    if not os.path.exists(a.model):
        print(f"no model file {a.model}: start the loop once in the GUI, or File > Save model, then pass --model")
        return 2
    with open(a.model, encoding="utf-8") as fh:
        m = MRModel.from_dict(json.load(fh))
    armed = a.arm.strip() == ML.ARM_PHRASE
    if a.arm and not armed:
        print(f'--arm needs exactly "{ML.ARM_PHRASE}" - refusing')
        return 2
    log = lambda msg, k="info": print(f"{ML.now_et():%H:%M:%S} [{k}] {msg}", flush=True)
    log(f"model {a.model}: {json.dumps(m.to_dict())}")
    feed = a.feed or ML.default_feed(m.minute)
    log(f"today's bars from: {feed}")
    eng = ML.MRLiveEngine(m, ML.MRLiveSettings(paper=True, armed=armed, capital=a.capital, feed=feed), folder, log=log)
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
