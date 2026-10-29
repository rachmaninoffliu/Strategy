"""
Headless session runner for one strategy (no GUI): prepare before the open, one step every 15 minutes, stop after the close.

  python run_live.py                                   DRY RUN: plans and journals every bar, sends nothing
  python run_live.py --arm "ARM PAPER"                 sends orders to the paper account of the project's key file "Alpaca Key.txt" (or of <PREFIX>_ALPACA_API_KEY / _SECRET_KEY)
  python run_live.py --kill "KILL"                     cancel + flatten this strategy's positions, then exit
Settings: the GUI's saved settings file (<folder>/<KEY>_settings.json, File > Save settings) if present, else the strategy's defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading

import pandas as pd

from . import live as LV


def main(strat_cls: type, folder: str, argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{strat_cls.NAME}: headless live session")
    ap.add_argument("--arm", default="", help=f'type "{LV.ARM_PHRASE}" to send orders')
    ap.add_argument("--kill", default="", help='type "KILL" to cancel + flatten and exit')
    ap.add_argument("--capital", type=float, default=0.0)
    ap.add_argument("--settings", default=os.path.join(folder, f"{strat_cls.KEY}_settings.json"))
    ap.add_argument("--until", default="16:05", help="stop at this ET time")
    a = ap.parse_args(argv)
    vals = {}
    if os.path.exists(a.settings):
        vals = json.load(open(a.settings, encoding="utf-8")).get("values", {})
    strat = strat_cls(vals, folder)
    armed = a.arm.strip() == LV.ARM_PHRASE
    if a.arm and not armed:
        print(f'--arm needs exactly "{LV.ARM_PHRASE}" - refusing')
        return 2
    eng = LV.LiveEngine(strat, LV.LiveSettings(paper=True, armed=armed, capital=a.capital), log=lambda m, k="info": print(f"{LV.now_et():%H:%M:%S} [{k}] {m}", flush=True))
    eng.connect()
    if a.kill:
        if a.kill.strip() != "KILL":
            print('--kill needs exactly "KILL"')
            return 2
        for r in eng.kill():
            print(r)
        return 0
    if armed:
        probs = eng.arm_problems()
        if probs:
            print("cannot arm: " + "; ".join(probs))
            return 3
    stop = threading.Event()
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), lambda *_: stop.set())
    hh, mm = [int(x) for x in a.until.split(":")]

    def watch():
        while not stop.is_set():
            n = LV.now_et()
            if (n.hour, n.minute) >= (hh, mm):
                stop.set()
            stop.wait(20)
    threading.Thread(target=watch, daemon=True).start()
    eng.run(stop)
    return 0
