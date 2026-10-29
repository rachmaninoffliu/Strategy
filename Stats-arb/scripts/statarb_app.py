"""
Statistical Arbitrage (Avellaneda-Lee residual mean reversion + 2026 risk overlays) - Backtest + Live trading application (PyQt6)
================================================================================================================================

  LEFT      the strategy inputs, shared by the Backtest and the Live engine (what you test is what trades)
  LIVE      connect to Alpaca, pre-flight, plan the next session (dry run), submit market-on-open orders only when ARMED and confirmed, reconcile, KILL
  BACKTEST  any window with an in-sample / out-of-sample split, equity vs SPY, trades, exposures, ablation, cost and parameter checks
  SIGNALS   today's s-scores with the reason each candidate is (not) tradable, the macro / geopolitical dashboard, the 2026 risk map

Run:  python statarb_app.py       (or launch_statarb.bat)

The window always starts on Paper + DRY RUN. Look and feel (and the worker plumbing) are shared with the Supertrend app.
RISK: Valid has not signed off for live capital. The raw 2008 signal has largely decayed - read the Ablation and Cost tabs.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _install_excepthook() -> None:
    def hook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        sys.stderr.write(text)
        try:
            import statarb_data as D
            os.makedirs(D.REPORTS, exist_ok=True)
            with open(os.path.join(D.REPORTS, "gui_error.log"), "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:                                                              # noqa: BLE001
            pass
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Unexpected error", "Something went wrong (the window keeps running):\n\n" + text[-1500:])
        except Exception:                                                              # noqa: BLE001
            pass
    sys.excepthook = hook


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from statarb_ui.main_window import MainWindow
    from supertrend_ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("Stat-Arb")
    apply_theme(app)
    _install_excepthook()
    win = MainWindow()
    win.show(); win.raise_(); win.activateWindow()
    print("  window open - if you cannot see it, look behind other windows or on the taskbar.", flush=True)
    code = app.exec()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        print("=" * 62 + "\n  Stat-Arb App (Backtest + Live) starting...\n" + "=" * 62, flush=True)
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except Exception:                                                              # noqa: BLE001
            pass
