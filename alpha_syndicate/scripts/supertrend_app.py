"""
Supertrend SOXL/SOXS - Backtest + Live Trading application (PyQt6)
==================================================================

One window, one set of settings:

  LEFT      the strategy inputs, shared by the Backtest and the Live engine (what you test is what trades)
  LIVE      connect to Alpaca, pre-flight go/no-go, dry run / paper / live, orders, positions, session performance
  BACKTEST  any date range, every trade, equity curve vs buy & hold, metrics, robustness (neighbour) check
  MONITOR   what the strategy sees right now: price, Supertrend, re-entry limit, the open position's levels

Run:  python supertrend_app.py       (or the desktop shortcut / launch_supertrend.bat)

The window always starts on Paper + DRY RUN. The trading engine is supertrend_gui.Engine (unchanged), the strategy is supertrend_strategy.
The previous Tkinter window is kept as supertrend_app_tk.py.

RISK: Valid has not signed off. The backtest does not reconcile with the TradingView Pine run, factor 5.0 is a parameter spike and SOXL/SOXS are 3x leveraged.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _install_excepthook() -> None:
    """PyQt6 turns an unhandled exception inside a slot into a silent crash. Show it (and keep the window alive) instead."""
    def hook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        sys.stderr.write(text)
        try:
            import supertrend_gui as GU
            os.makedirs(GU.REPORTS, exist_ok=True)
            with open(os.path.join(GU.REPORTS, "gui_error.log"), "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:                                         # noqa: BLE001
            pass
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Unexpected error", "Something went wrong (the window keeps running):\n\n" + text[-1500:])
        except Exception:                                         # noqa: BLE001
            pass
    sys.excepthook = hook


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from supertrend_ui.main_window import MainWindow
    from supertrend_ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("Supertrend SOXL/SOXS")
    apply_theme(app)
    _install_excepthook()
    win = MainWindow()
    win.show()
    win.raise_()
    win.activateWindow()
    print("  window open - if you cannot see it, look behind other windows or on the taskbar.", flush=True)    # ASCII only: a Windows console is cp1252
    code = app.exec()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)                  # settings are saved by closeEvent; skipping interpreter teardown avoids Qt aborting over a worker thread that is still finishing a network call


def _launch() -> None:
    """Entry point kept for supertrend_gui.py, which hands off to this app when it is run directly."""
    print("=" * 62)
    print("  Supertrend App (Backtest + Live) starting...")
    print("=" * 62, flush=True)
    sys.exit(main())


if __name__ == "__main__":
    try:
        _launch()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except Exception:
            pass
