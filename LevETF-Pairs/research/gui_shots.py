"""Render the app offscreen and save PNGs of the tabs after a backtest, a replay and a signal refresh (visual QA without a display):  python gui_shots.py <output folder>"""
import os
import sys
import tempfile

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["LEVPAIRS_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(), "t.ini")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
out = sys.argv[1] if len(sys.argv) > 1 else HERE
os.makedirs(out, exist_ok=True)

from PyQt6.QtCore import QDate, QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

from levpairs_ui.main_window import MainWindow
from supertrend_ui.theme import apply_theme

app = QApplication([])
apply_theme(app)
w = MainWindow()
w.resize(1560, 980)
w.show()


def wait(cond, limit=300):
    loop = QEventLoop()
    t = QTimer(); t.timeout.connect(lambda: loop.quit() if cond() else None); t.start(250)
    QTimer.singleShot(limit * 1000, loop.quit)
    loop.exec(); t.stop()


def shot(name):
    app.processEvents()
    w.grab().save(os.path.join(out, name + ".png"))


w.backtest.d0.setDate(QDate(2023, 1, 3)); w.backtest.split.setDate(QDate(2025, 1, 1))
w.main_tabs.setCurrentIndex(w.TAB_BT)
w.backtest.run(); wait(lambda: not w.backtest._busy)
shot("1_backtest_equity")
for name, tab in (("pairs", w.backtest.tbl_pair), ("trades", w.backtest.tbl_tr.parentWidget()), ("metrics", w.backtest.tbl_met)):
    w.backtest.tabs.setCurrentWidget(tab); shot("2_backtest_" + name)
w.backtest.show_session(busiest=True); shot("3_backtest_spread_viewer")
w.backtest.run_fill(); wait(lambda: not w.backtest._busy, 600); shot("3b_backtest_fill_models")
w.backtest.run_tf(); wait(lambda: not w.backtest._busy, 600); shot("4_backtest_timeframes")
w.backtest.run_checks(); wait(lambda: not w.backtest._busy, 600); shot("5_backtest_cost_checks")
w.main_tabs.setCurrentIndex(w.TAB_LIVE); w.live.replay_last(); wait(lambda: not w.live._busy, 300); shot("6_live_replay")
w.live.tabs.setCurrentWidget(w.live.t_pairs); shot("7_live_pairs")
w.main_tabs.setCurrentIndex(w.TAB_SIG); w.signals.refresh(); wait(lambda: not w.signals._busy, 300); shot("8_signals")
w.signals.tabs.setCurrentWidget(w.signals.t_cost); shot("9_signals_cost")
w.main_tabs.setCurrentIndex(w.TAB_HELP); shot("10_help")
print("saved to", out)
w.close()
