"""Start the GUI for one strategy:  run(StrategyClass, folder)."""
from __future__ import annotations

import sys


def run(strat_cls: type, folder: str) -> int:
    from PyQt6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .theme import apply_theme
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    w = MainWindow(strat_cls, folder)
    w.show()
    return app.exec()
