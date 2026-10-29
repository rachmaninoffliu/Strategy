"""Drive the real GUI offscreen on the real cached data (no broker): for each strategy run the Backtest tab, the Research tab's score test and the Signals tab, wait for the workers,
check that results arrived, and save screenshots to research/reports/gui_*.png.

  python gui_smoke.py [strategy key ...]
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RESARB_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(), "s.ini")

from common import OUT, ROOT, ResidualMeanReversion, ResidualMomentum, ResidualPrediction  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from resarb.ui.main_window import MainWindow  # noqa: E402
from resarb.ui.theme import apply_theme  # noqa: E402

CLS = {"ResMomentum": ResidualMomentum, "ResMeanReversion": ResidualMeanReversion, "ResPrediction": ResidualPrediction}


def wait(app, cond, timeout=1800):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
    return cond()


def main(keys):
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    ok = True
    for k in keys:
        cls = CLS[k]
        w = MainWindow(cls, os.path.join(ROOT, k))
        w.resize(1600, 1000)
        w.show()
        w.main_tabs.setCurrentIndex(w.TAB_BT)
        w.backtest.run()
        done = wait(app, lambda: w.backtest.result is not None or "failed" in w.backtest.status.text())
        print(k, "backtest:", w.backtest.status.text()[:160], flush=True)
        ok &= done and w.backtest.result is not None
        w.grab().save(os.path.join(OUT, f"gui_{k}_backtest.png"))
        w.main_tabs.setCurrentIndex(w.TAB_RES)
        w.research.run_score()
        wait(app, lambda: w.research.status.text() in ("done",) or "failed" in w.research.status.text())
        print(k, "research:", w.research.status.text()[:160], flush=True)
        ok &= w.research.t_ic.rowCount() > 0
        w.grab().save(os.path.join(OUT, f"gui_{k}_research.png"))
        w.main_tabs.setCurrentIndex(w.TAB_SIG)
        w.signals.refresh()
        wait(app, lambda: w.signals.status.text().startswith("scores") or "failed" in w.signals.status.text())
        print(k, "signals:", w.signals.status.text()[:160], w.signals.expo.text()[:160], flush=True)
        ok &= w.signals.t.rowCount() > 0
        w.grab().save(os.path.join(OUT, f"gui_{k}_signals.png"))
        w.main_tabs.setCurrentIndex(w.TAB_LIVE)
        w.grab().save(os.path.join(OUT, f"gui_{k}_live.png"))
        w._workers.stop_all(1000)
        w.live.shutdown()
        w.hide()
    print("GUI SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(CLS)))

