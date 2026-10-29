"""Residual Momentum - Backtest + Live trading application (PyQt6). Always starts on Paper + DRY RUN."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, HERE)

from strategy_resmom import ResidualMomentum  # noqa: E402

if __name__ == "__main__":
    from resarb.ui.app import run
    sys.exit(run(ResidualMomentum, HERE))
