"""BTC 15-min Up/Down trader GUI.   Run:  python trader_app.py     (or double-click run_trader.bat)

Live trading tab: connects to your Kalshi account (API Key ID + private key file; Demo or Production), sizes each bet with Kelly
from the strategy's historical win probability. Backtest tab: any date range, every trade, equity curve and full metrics.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from trader.gui import MainWindow, apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
