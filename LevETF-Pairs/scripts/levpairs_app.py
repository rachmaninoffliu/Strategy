"""
Leveraged-ETF pairs statistical arbitrage (TQQQ/QQQ, SOXL/SMH, SPXL/SPY) - Backtest + Live trading application (PyQt6)
=====================================================================================================================

  LEFT      the strategy inputs, shared by the Backtest and the Live engine (what you test is what trades)
  LIVE      connect to Alpaca, pre-flight, run the intraday session engine (DRY RUN shadow book, or ARMED market orders), replay the last session, reconcile, KILL
  BACKTEST  any window with an in-sample / out-of-sample split, the time-frame scan, equity vs SPY, trades, a one-session spread viewer, ablation, cost and parameter checks
  SIGNALS   each pair's spread and s-score now, the cost of a round trip against what a reversion could capture, the leveraged-ETF risk map

Run:  python levpairs_app.py       (or launch_levpairs.bat)

The window always starts on Paper + DRY RUN.
RISK: Valid has not signed off. With realistic fills the backtest loses money on every time frame - read the Time frames, Ablation and Cost tabs.
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
            import levpairs_data as D
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

    from levpairs_ui.main_window import MainWindow
    from supertrend_ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("Lev-ETF Pairs")
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
        print("=" * 62 + "\n  Lev-ETF Pairs App (Backtest + Live) starting...\n" + "=" * 62, flush=True)
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except Exception:                                                              # noqa: BLE001
            pass
