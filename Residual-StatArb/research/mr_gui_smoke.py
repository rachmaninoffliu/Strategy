"""Drive the mean-reversion GUI offscreen on the real cached data (no broker): pick a preset, run the Backtest tab, check the trade table (every trade, every column), switch
models and check the live tab's model strip, and save screenshots to research/reports/mr_gui_*.png.

  python mr_gui_smoke.py [preset index]
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")                                    # real glyphs in the offscreen screenshots
os.environ["RESARB_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(), "s.ini")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from resarb.mr import presets as PR  # noqa: E402
from resarb.mr.ui.window import MRWindow  # noqa: E402
from resarb.ui.theme import apply_theme  # noqa: E402

OUT = os.path.join(HERE, "reports")


def wait(app, cond, timeout=1800):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
    return cond()


def main(idx: int) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    w = MRWindow(os.path.join(ROOT, "MeanReversion"))
    w.resize(1600, 1000)
    w.show()
    name = PR.preset_names()[idx]
    w.panel.preset.setCurrentText(name)
    w.main_tabs.setCurrentIndex(0)
    w.backtest.run()
    done = wait(app, lambda: w.backtest.result is not None or "failed" in w.backtest.status.text())
    print("backtest:", w.backtest.status.text()[:200], flush=True)
    ok = done and w.backtest.result is not None
    if ok:
        r = w.backtest.result
        n_view = w.backtest.t_trades.model().rowCount()
        print(f"trades: {len(r.trades):,} in the result, {n_view:,} rows in the table, {w.backtest.t_trades.model().columnCount()} columns", flush=True)
        ok &= n_view == len(r.trades) and len(r.trades) > 0
        hdr = [w.backtest.t_trades.model().headerData(i, __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal) for i in range(w.backtest.t_trades.model().columnCount())]
        print("columns:", hdr, flush=True)
        first = [w.backtest.t_trades.model().index(0, c).data() for c in range(len(hdr))]
        print("first row:", first, flush=True)
        ok &= r.stats["max_exposure"] <= 1.0 + 1e-9
        print(f"max exposure {r.stats['max_exposure']:.3f}x", flush=True)
        s0 = r.trades.symbol.iloc[0]
        w.backtest.filter.setText(s0 + ",")
        app.processEvents()
        n_f = w.backtest.t_trades.model().rowCount()
        print(f"filter '{s0},': {n_f} rows (trades in {s0}: {(r.trades.symbol == s0).sum()})", flush=True)
        ok &= n_f == int((r.trades.symbol == s0).sum())
        w.backtest.filter.setText("")
    w.grab().save(os.path.join(OUT, "mr_gui_backtest.png"))
    w.backtest.tabs.setCurrentIndex(1)
    app.processEvents()
    w.grab().save(os.path.join(OUT, "mr_gui_trades.png"))
    w.main_tabs.setCurrentIndex(1)
    app.processEvents()
    print("live strip:", w.live.model_strip.text()[:220], flush=True)
    ok &= "same model as the last backtest" in w.live.model_strip.text()
    w.grab().save(os.path.join(OUT, "mr_gui_live.png"))
    w.panel.slots.setValue(w.panel.slots.value() + 1)                                         # an edit makes it a different, un-backtested model
    app.processEvents()
    print("after an edit:", w.live.model_strip.text()[-80:], "| preset:", w.panel.preset.currentText(), flush=True)
    ok &= "not backtested" in w.live.model_strip.text()
    w._workers.stop_all(1000)
    w.live.shutdown()
    w.hide()
    print("MR GUI SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
