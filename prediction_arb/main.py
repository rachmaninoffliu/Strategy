"""Entry point:  python main.py   (requires: pip install pyqt6 aiohttp pydantic matplotlib pandas numpy)"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from database import Database
    from ui.app_window import AppWindow
    from ui.common import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)
    db = Database()
    win = AppWindow(db)
    win.show()
    code = app.exec()
    db.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
