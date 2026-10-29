"""Residual Prediction - headless live session (see resarb/run.py):  python run_live.py [--arm "ARM PAPER"] [--kill "KILL"]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, HERE)

from strategy_respred import ResidualPrediction  # noqa: E402

if __name__ == "__main__":
    from resarb.run import main
    sys.exit(main(ResidualPrediction, HERE))
