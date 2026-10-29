"""Mean Reversion - Backtest + Live trading application (PyQt6). Always starts on Paper + DRY RUN."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

if __name__ == "__main__":
    from resarb.mr.ui.window import run
    sys.exit(run(HERE))
