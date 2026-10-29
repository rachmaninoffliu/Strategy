"""Mean Reversion - headless live session (see resarb/mr/run.py):  python run_live.py [--arm "ARM PAPER"] [--kill "KILL"] [--model live_model.json]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

if __name__ == "__main__":
    from resarb.mr.run import main
    sys.exit(main(HERE))
