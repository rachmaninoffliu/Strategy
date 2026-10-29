import os, sys, time, faulthandler
faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["STATARB_SETTINGS_FILE"] = r"C:\Users\xinyu\AppData\Local\Temp\claude\C--Users-xinyu-PycharmProjects-pythonProject\2e4f679d-1745-4079-91e2-85d1bd1dfe01\scratchpad\qs_smoke3.ini"
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")

def main():
    import pandas as pd
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from supertrend_ui.theme import apply_theme
    apply_theme(app)
    from statarb_ui.main_window import MainWindow
    w = MainWindow(); w.show()
    p = w.settings.port_cfg()
    print("preset:", w.settings.preset_name(), "| universe", w.settings.universe(), "| max_names", p.eap_max_names, "| g_max", p.g_max, "| min_adv", w.settings.factor_cfg().min_adv, flush=True)
    print("armed:", w.live.armed.isChecked(), "paper:", w.live.paper(), flush=True)
    def wait(cond, timeout, label):
        loop, t0 = QEventLoop(), time.time()
        tm = QTimer(); tm.timeout.connect(lambda: loop.quit() if (cond() or time.time() - t0 > timeout) else None); tm.start(500)
        loop.exec(); tm.stop()
        print(f"{label}: {'ok' if cond() else 'TIMEOUT'} after {time.time() - t0:.0f}s", flush=True)
    w.live.make_plan(False)
    wait(lambda: not w.live._busy, 900, "plan")
    print("log:\n" + "\n".join(w.live.log_box.toPlainText().splitlines()[-8:]), flush=True)
    pl = w.live.plan
    if pl is None:
        print("NO PLAN"); os._exit(1)
    o = pl.orders
    print("decision", pl.decision_date.date(), "| equity", round(pl.equity), "| deploy", round(pl.deploy), "| g_target", round(pl.g_target, 2), "| shock", round(pl.shock_mult, 2), "| orders", len(o), "| ok", int(o.ok.sum()) if len(o) else 0, flush=True)
    if len(o):
        st = o[o.symbol.isin(["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","SPY"])]
        stocks = o[~o.index.isin(st.index)]
        print("stock orders", len(stocks), "buys $", round(stocks[stocks.side=="buy"].notional.sum()), "| etf orders", len(st), "sells $", round(st[st.side=="sell"].notional.sum()), flush=True)
        print("ticket size $: min %.0f median %.0f max %.0f" % (stocks.notional.min(), stocks.notional.median(), stocks.notional.max()), flush=True)
        print("not ok:", o[~o.ok][["symbol","note"]].head(5).to_dict("records"), flush=True)
        print(o.head(8).to_string(), flush=True)
        print(st.to_string(), flush=True)
    print({k: round(v, 3) for k, v in pl.exposures.items() if not k.startswith("sector")}, flush=True)
    sec = {k.split(":")[1]: round(v, 3) for k, v in pl.exposures.items() if k.startswith("sector")}
    print("sector shares of the stock leg:", sec, flush=True)
    print("notes:", pl.notes, flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
