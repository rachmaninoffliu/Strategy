import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["STATARB_SETTINGS_FILE"] = r"C:\Users\xinyu\AppData\Local\Temp\claude\C--Users-xinyu-PycharmProjects-pythonProject\2e4f679d-1745-4079-91e2-85d1bd1dfe01\scratchpad\qs_smoke.ini"
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication([])
    from supertrend_ui.theme import apply_theme
    apply_theme(app)
    from statarb_ui.main_window import MainWindow

    w = MainWindow()
    w.show()
    print("preset:", w.settings.preset_name(), "| universe:", w.settings.universe(), "| w_eap:", w.settings.port_cfg().w_eap, "| min_adv:", w.settings.factor_cfg().min_adv)

    def wait(cond, timeout, label):
        loop, t0 = QEventLoop(), time.time()
        tm = QTimer(); tm.timeout.connect(lambda: loop.quit() if (cond() or time.time() - t0 > timeout) else None); tm.start(500)
        loop.exec(); tm.stop()
        print(f"{label}: {'ok' if cond() else 'TIMEOUT'} after {time.time() - t0:.0f}s", flush=True)

    w.main_tabs.setCurrentIndex(w.TAB_BT)
    w.backtest.run()
    wait(lambda: w.backtest.out is not None or (w.backtest._busy is None and "Failed" in w.backtest.verdict.text()), 900, "backtest")
    print(w.backtest.verdict.text()[:600])
    print("KPIs:", {k: v.v.text() for k, v in w.backtest.k.items()})
    print("rows: year", w.backtest.tbl_year.rowCount(), "trades", w.backtest.tbl_tr.rowCount(), "metrics", w.backtest.tbl_met.rowCount())

    # signals tab
    w.main_tabs.setCurrentIndex(w.TAB_SIG)
    w.signals.refresh()
    wait(lambda: not w.signals._busy, 600, "signals")
    print("signals status:", w.signals.status.text(), "| banner:", w.signals.banner.text()[:200], "| rows", w.signals.t_sig.rowCount())

    # live tab: preflight + dry-run plan (read-only broker calls: account, clock)
    w.main_tabs.setCurrentIndex(w.TAB_LIVE)
    assert not w.live.armed.isChecked() and w.live.paper()
    w.live.preflight()
    wait(lambda: not w.live._busy, 180, "preflight")
    for r in range(w.live.t_pf.rowCount()):
        print("  PF", [w.live.t_pf.item(r, c).text() for c in range(3)])
    w.live.make_plan(False)
    wait(lambda: not w.live._busy, 900, "plan")
    print("plan:", w.live.plan.decision_date if w.live.plan else None, "| orders", len(w.live.plan.orders) if w.live.plan else None, "| long/short", (w.live.plan.n_long, w.live.plan.n_short) if w.live.plan else None)
    print({k: v.v.text() for k, v in w.live.k.items()})
    print("log tail:\n" + "\n".join(w.live.log_box.toPlainText().splitlines()[-8:]))
    w.close()
    os._exit(0)

if __name__ == '__main__':
    main()
