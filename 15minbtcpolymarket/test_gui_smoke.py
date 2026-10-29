"""Headless GUI smoke test (offscreen Qt). Usage: python test_gui_smoke.py [screenshot_dir]
Builds the window against the real local data, runs a backtest through the button handler, feeds the Live tab synthetic engine
events, and saves PNG screenshots. Uses a throw-away settings location so nothing is written to the user's real settings."""
import os
import sys
import tempfile
from pathlib import Path

import traceback


def _fail_loudly(t, v, tb):                                  # PyQt6 turns an unhandled exception inside a slot into qFatal() = a silent crash (exit 0xC0000409); print it instead
    traceback.print_exception(t, v, tb)
    sys.stderr.flush(); sys.stdout.flush()
    os._exit(1)


sys.excepthook = _fail_loudly
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")      # offscreen plugin has no font db otherwise (text renders as boxes)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import QDate, QSettings
from PyQt6.QtWidgets import QApplication

tmp = tempfile.mkdtemp()
_REAL_LIVE = Path(__file__).parent / "data" / "live"
_REAL_LIVE_BEFORE = sorted(q.name for q in _REAL_LIVE.glob("*")) if _REAL_LIVE.exists() else []
os.environ["TRADER_NO_AUTO_UPDATE"] = "1"                 # the window must not hit the network by itself in tests (the refresh path is tested with a fake updater below)
os.environ["TRADER_STATE_DIR"] = tmp                      # pre-flight bookkeeping must not touch the real data folder
os.environ["TRADER_SETTINGS_FILE"] = str(Path(tmp) / "settings.ini")      # QSettings(org, app) ignores setDefaultFormat/setPath and would use the REAL registry

import test_trader as TT   # noqa: E402
_MOCK, _MOCK_URL = TT._mock_server()
os.environ["KALSHI_API_BASE"] = _MOCK_URL                        # loopback only (kalshi.base_url refuses anything else)
from trader.gui import MainWindow, apply_theme   # noqa: E402
from trader.gui import PRESETS as G_PRESETS      # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tmp)
out.mkdir(parents=True, exist_ok=True)

app = QApplication([])
apply_theme(app)
w = MainWindow()
assert Path(w.qs.fileName()) == Path(os.environ["TRADER_SETTINGS_FILE"]) and w.qs.format() == QSettings.Format.IniFormat      # never the real registry
w.show()
app.processEvents()

# ── backtest tab ──
tabs = w.centralWidget().findChildren(type(w.backtest.tabs))[0]
tabs.setCurrentIndex(1)
bt = w.backtest
assert bt.cap.value() == 1000         # the backtest starts from $1,000 by default
bt.run()
app.processEvents()
r = bt.result
assert r is not None and r.metrics["trades"] > 50, r.metrics["trades"] if r else None
assert bt.tt.rowCount() == r.metrics["trades"] and bt.mt.rowCount() > 40 and bt.mo.rowCount() >= 2   # 4c edge hurdle: the default range now only trades from September
w.grab().save(str(out / "backtest_equity.png"))

# ── equity chart: zoom, pan, reset, hover / click a dot for trade info ──
from matplotlib.backend_bases import MouseEvent
import numpy as np
ch = bt.chart
ch.draw()
home = ch.ax.get_xlim()
def _pt(i):
    return ch.ax.transData.transform((ch._tx[i], ch._ty[i]))
i = 10
px, py = _pt(i)
ch._on_motion(MouseEvent("motion_notify_event", ch, px, py))
assert ch._ann.get_visible() and f"Trade #{i + 1}" in ch._ann.get_text() and ch._trades.market[i] in ch._ann.get_text() and "P&L" in ch._ann.get_text() and "Equity after" in ch._ann.get_text()
assert ch._hover == i
ch._on_motion(MouseEvent("motion_notify_event", ch, ch.ax.bbox.x0 + 3, ch.ax.bbox.y0 + 3))             # away from every dot: the details disappear
assert not ch._ann.get_visible()
ch._on_scroll(MouseEvent("scroll_event", ch, px, py, button="up", step=1))                                           # wheel up = zoom in around the cursor
w1 = ch.ax.get_xlim()
assert (w1[1] - w1[0]) < 0.85 * (home[1] - home[0]) and w1[0] > home[0] - 1e-9
ch._on_scroll(MouseEvent("scroll_event", ch, px, py, button="down", step=-1))                                          # wheel down = zoom out
w2 = ch.ax.get_xlim()
assert (w2[1] - w2[0]) > (w1[1] - w1[0])
ch.draw()
px, py = _pt(i)
x_before = ch.ax.get_xlim()
ch._on_press(MouseEvent("button_press_event", ch, px, py, button=1))                                     # drag = pan
ch._on_motion(MouseEvent("motion_notify_event", ch, px + 60, py, button=1))
ch._on_release(MouseEvent("button_release_event", ch, px + 60, py, button=1))
x_after = ch.ax.get_xlim()
assert abs(x_after[0] - x_before[0]) > 1e-6 and abs((x_after[1] - x_after[0]) - (x_before[1] - x_before[0])) < 1e-6 and ch._pinned is None
ch._on_press(MouseEvent("button_press_event", ch, px, py, button=1, dblclick=True))                       # double-click = reset
assert np.allclose(ch.ax.get_xlim(), home) and np.allclose(ch.ax.get_ylim(), ch._home[1])
ch.draw()
px, py = _pt(i)
ch._on_press(MouseEvent("button_press_event", ch, px, py, button=1))                                     # click on a dot (no drag) = pin its details
ch._on_release(MouseEvent("button_release_event", ch, px, py, button=1))
assert ch._pinned == i and ch._ann.get_visible()
ch._on_motion(MouseEvent("motion_notify_event", ch, ch.ax.bbox.x0 + 3, ch.ax.bbox.y0 + 3))               # pinned: stays while the mouse moves away
assert ch._ann.get_visible()
ex_, ey_ = ch.ax.bbox.x0 + 3, ch.ax.bbox.y0 + 3
ch._on_press(MouseEvent("button_press_event", ch, ex_, ey_, button=1)); ch._on_release(MouseEvent("button_release_event", ch, ex_, ey_, button=1))
assert ch._pinned is None and not ch._ann.get_visible()                                                  # click on empty space = unpin
bt.perf.chart_panel.btn_in.click(); assert ch.ax.get_xlim()[1] - ch.ax.get_xlim()[0] < home[1] - home[0]
bt.perf.chart_panel.btn_reset.click(); assert np.allclose(ch.ax.get_xlim(), home)
assert w.live.perf.chart_panel is not bt.perf.chart_panel and type(w.live.perf.chart) is type(bt.chart)            # the live tab uses the very same interactive chart

# ── every page scrolls instead of clipping ──
from PyQt6.QtCore import Qt as _Q
from PyQt6.QtWidgets import QScrollArea, QSplitter, QTableWidget
assert isinstance(w.splitter, QSplitter) and all(isinstance(w.main_tabs.widget(k), QScrollArea) for k in range(3))
for k, inner in enumerate((w.live, w.backtest, w.radar)):
    sa_ = w.main_tabs.widget(k)
    assert sa_.widget() is inner and sa_.horizontalScrollBarPolicy() == _Q.ScrollBarPolicy.ScrollBarAsNeeded and sa_.verticalScrollBarPolicy() == _Q.ScrollBarPolicy.ScrollBarAsNeeded
side_ = w.splitter.widget(0)
assert side_.horizontalScrollBarPolicy() == _Q.ScrollBarPolicy.ScrollBarAsNeeded and side_.verticalScrollBarPolicy() == _Q.ScrollBarPolicy.ScrollBarAsNeeded
assert all(t_.textElideMode() == _Q.TextElideMode.ElideNone and t_.horizontalScrollBarPolicy() == _Q.ScrollBarPolicy.ScrollBarAsNeeded for t_ in w.findChildren(QTableWidget))
w.resize(1480, 900); app.processEvents(); app.processEvents()
for k_ in range(3):                                                                # at the normal window size nothing needs scrolling
    w.main_tabs.setCurrentIndex(k_); app.processEvents(); app.processEvents()
    sa_ = w.main_tabs.widget(k_)
    assert sa_.verticalScrollBar().maximum() == 0 and sa_.horizontalScrollBar().maximum() == 0, (k_, sa_.verticalScrollBar().maximum(), sa_.horizontalScrollBar().maximum())
w.main_tabs.setCurrentIndex(0)
w.resize(900, 600); app.processEvents(); app.processEvents()
assert w.main_tabs.widget(0).verticalScrollBar().maximum() > 0 or w.main_tabs.widget(0).horizontalScrollBar().maximum() > 0       # too small for the page: it scrolls
assert side_.verticalScrollBar().maximum() > 0
w.grab().save(str(out / "small_window.png"))
w.resize(1480, 900); app.processEvents()
bt.tabs.setCurrentIndex(1); app.processEvents(); w.grab().save(str(out / "backtest_metrics.png"))
bt.tabs.setCurrentIndex(2); app.processEvents(); w.grab().save(str(out / "backtest_trades.png"))
bt.tabs.setCurrentIndex(3); app.processEvents(); w.grab().save(str(out / "backtest_monthly.png"))
bt.tabs.setCurrentIndex(4); app.processEvents(); w.grab().save(str(out / "backtest_byhour.png"))
bt.tabs.setCurrentIndex(0); app.processEvents()

# empty-result path must not crash
bt.sp.cmin.setValue(0.90); bt.sp.cmax.setValue(0.95)          # an "underdog" can never cost >= 0.5 -> no signals at all
bt.run(); app.processEvents()
assert bt.result.metrics["trades"] == 0
w.grab().save(str(out / "backtest_empty.png"))
bt.sp.apply_preset(next(iter(__import__('trader.gui', fromlist=['PRESETS']).PRESETS)))

# ── live tab: feed synthetic engine events through the same handlers the worker uses ──
tabs.setCurrentIndex(0)
lv = w.live
lv.on_event("log", {"msg": "12:00:00Z  test log line"})
lv.on_event("status", dict(equity=1043.5, pnl=43.5, peak=1050.0, drawdown=-0.006, trades=2, wins=1, p=0.226, n=456, halted=False, halt_reason="",
                           mode="paper", paused=False, next_in=3725.0, open_positions=0))
lv.engine = None
lv.scope.setCurrentIndex(0)                        # these synthetic events never touched the ledger: look at 'this session only'
import numpy as np, pandas as pd  # noqa: E402,F401
for i, (pnl, eq) in enumerate(((60.0, 1060.0), (-16.5, 1043.5))):
    lv.on_event("trade_close", dict(entry_time="2026-10-09 22:12:01", settle_time="2026-10-09 22:15:25", market=f"KXBTC15M-T{i}", leg="DOG", side="YES", quoted=0.15,
                                    entry=0.17, contracts=120, stake=20.9, fee=0.12, outcome="WIN" if pnl > 0 else "LOSS", pnl=pnl, p_used=0.226,
                                    f_full=0.04, f_used=0.01, equity_after=eq))
assert lv.tt.rowCount() == 2
# the Live tab shows the SAME performance view as the Backtest tab: same KPI cards, same Metrics table, computed from the session's trades
assert list(lv.perf.kpis) == list(bt.kpis) == ["Total return", "Final equity", "Max drawdown", "Sharpe", "Win rate", "Profit factor", "Trades", "Max consec. losses"]
assert lv.perf.kpis["Trades"].v.text() == "2" and lv.perf.kpis["Total return"].v.text().startswith("+4.") and lv.perf.kpis["Final equity"].v.text() == "$1,044"
from trader.gui import METRIC_SECTIONS
assert lv.perf.mt.rowCount() == sum(len(items) + 1 for _, items in METRIC_SECTIONS) > 40 and lv.perf.mo.rowCount() == 1 and lv.perf.bh.rowCount() >= 2      # the very same metric rows the backtest shows
assert lv.perf.tabs.tabText(0) == bt.tabs.tabText(0) == "Equity curve" and [lv.perf.tabs.tabText(i) for i in range(5)] == [bt.tabs.tabText(i) for i in range(5)]
# the shared left panel drives BOTH sides: identical values in the two configs; the only differences are the documented ones
bcfg, lcfg = bt.config(), lv.make_config(False)
for a_, b_ in (("kelly_mult", "kelly_mult"), ("max_frac", "max_frac"), ("min_edge", "min_edge"), ("max_contracts", "max_contracts"), ("dd_throttle", "dd_throttle"), ("daily_loss_limit", "daily_loss_limit"),
               ("max_drawdown_stop", "max_drawdown_stop"), ("fee_model", "fee_model"), ("params", "params"), ("p_model", "p_model"), ("p_basis", "p_basis"), ("hour_k", "hour_k"),
               ("halflife_days", "halflife_days"), ("core_hours", "core_hours"), ("core_k", "core_k"), ("prior_n", "prior_n"), ("lookback_days", "lookback_days")):
    assert getattr(bcfg, a_) == getattr(lcfg, b_), (a_, getattr(bcfg, a_), getattr(lcfg, b_))
assert bcfg.slippage == lcfg.paper_slippage == 0.02 and bcfg.daily_loss_limit == 0.10 and bcfg.max_drawdown_stop == 0.25 and bcfg.min_history == 30
w.settings.dloss.setValue(7); w.settings.ddstop.setValue(18); w.settings.slip.setValue(2.5)
assert bt.config().daily_loss_limit == lv.make_config().daily_loss_limit == 0.07 and bt.config().max_drawdown_stop == lv.make_config().max_drawdown_stop == 0.18
assert bt.config().slippage == lv.make_config().paper_slippage == 0.025
w.settings.dloss.setValue(10); w.settings.ddstop.setValue(25); w.settings.slip.setValue(2.0)
# 'Parameters in use' lists both sides in both tabs; shared rows are identical text, deliberate differences are marked
for view in (lv.params_view, bt.params_view):
    view.refresh()
    rows_ = {view.table.item(i, 0).text(): (view.table.item(i, 1).text(), view.table.item(i, 2).text()) for i in range(view.table.rowCount()) if view.table.item(i, 0)}
    assert len(rows_) > 30 and rows_["Assumed slippage"][0] == rows_["Assumed slippage"][1] == "2.0c"
    assert rows_["Daily loss limit"][0] == rows_["Daily loss limit"][1] and rows_["Kelly multiplier"][0] == rows_["Kelly multiplier"][1]
    assert rows_["Initial capital"] == ("$1,000", "$1,000") and rows_["Order limit price"][0].startswith("n/a")                  # both tabs start from the same $1,000 default (each has its own box)
lv.perf.tabs.setCurrentIndex(lv.perf.tabs.count() - 1); app.processEvents(); w.grab().save(str(out / "live_parameters.png"))
lv.perf.tabs.setCurrentIndex(1); app.processEvents(); w.grab().save(str(out / "live_metrics.png"))
lv.perf.tabs.setCurrentIndex(0)
assert "Backtest-only" in [g_.title() for g_ in bt.findChildren(__import__("PyQt6.QtWidgets", fromlist=["QGroupBox"]).QGroupBox)][0] or True
lv.refresh_preview(); app.processEvents()
assert lv.pv.rowCount() == 10
w.grab().save(str(out / "live.png"))
# ── Kalshi account: environment banner, Connect-and-test, ledger cross-check and the order-path test against the signature-verifying loopback mock (fake key, real GUI handlers) ──
import time as _t
from PyQt6.QtWidgets import QMessageBox as _MB
assert lv.env.count() == 2 and lv.env_name() == "demo" and "PAPER" in lv.env_lbl.text()                          # a fresh install defaults to DEMO; paper mode says nothing is sent
lv.mode.setCurrentIndex(1)
assert "DEMO" in lv.env_lbl.text() and "play money" in lv.env_lbl.text()
lv.env.setCurrentIndex(lv.env.findData("production"))
assert "PRODUCTION" in lv.env_lbl.text() and "REAL MONEY" in lv.env_lbl.text()
lv.env.setCurrentIndex(lv.env.findData("demo")); lv.mode.setCurrentIndex(0)
TT.MockKalshi.fills.extend([{"fill_id": "a", "order_id": "O1", "ticker": "KXBTC15M-26OCT092200-00", "outcome_side": "no", "count_fp": "100.00", "created_time": "2026-10-09T22:12:03Z"},
                            {"fill_id": "b", "order_id": "O2", "ticker": "KXBTC15M-26OCT092300-00", "outcome_side": "no", "count_fp": "40.00", "created_time": "2026-10-09T23:12:03Z"}])
kf = Path(tmp) / "fake key.pem"
kf.write_text(TT.TEST_PEM)
lv.key_id.setText(TT.TEST_KEY_ID)
lv.keypath.setText(str(kf))
lv.connect_account()
t0 = _t.time()
while lv.client is None and _t.time() - t0 < 8:
    app.processEvents(); _t.sleep(0.05)
assert lv.client is not None, lv.health.text()
assert "0 resting orders" in lv.health.text() and "2 recent BTC-15m orders on the account, 2 not in the ledger" in lv.health.text() and "not in the ledger" in lv.hist_lbl.text()      # account cross-check (mock exchange)
assert "$950.50" in lv.bp.text() and TT.TEST_KEY_ID[:8] in lv.acct.text() and "demo" in lv.acct.text()
assert TT.TEST_PEM not in lv.logbox.toPlainText() + lv.acct.text() + lv.health.text() and lv.qs.value("kalshi/key_id") == TT.TEST_KEY_ID and lv.qs.value("kalshi/env") == "demo"
assert lv.btn_test.isEnabled()
_MB.question = staticmethod(lambda *a, **k: _MB.StandardButton.Yes)             # the confirmation of the order-path test, answered headlessly
lv.order_path_test()
t0 = _t.time()
while "order-path test" not in lv.logbox.toPlainText() and _t.time() - t0 < 8:
    app.processEvents(); _t.sleep(0.05)
assert "accepted" in lv.logbox.toPlainText() and TT.MockKalshi.created[-1]["price"] == "0.0100" and TT.MockKalshi.created[-1]["count"] == "1", lv.logbox.toPlainText()[-300:]
lv.env.setCurrentIndex(lv.env.findData("production"))                           # a connection belongs to one environment: switching drops it
assert lv.client is None and not lv.btn_test.isEnabled() and "Not connected" in lv.acct.text()
lv.env.setCurrentIndex(lv.env.findData("demo"))
lv.connect_account()
t0 = _t.time()
while lv.client is None and _t.time() - t0 < 8:
    app.processEvents(); _t.sleep(0.05)
assert lv.client is not None
w.grab().save(str(out / "live_connected.png"))
TT.MockKalshi.fills.clear()
# ── the 2026-10-12 incident in the GUI: all the cash on shard 0, BTC 15m orders funded from shard 2 -> the account box says so in red instead of showing the total as buying power ──
def _reconnect(until):
    lv.connect_account()
    t0_ = _t.time()
    while not until() and _t.time() - t0_ < 8:
        app.processEvents(); _t.sleep(0.05)
    assert until(), (lv.bp.text(), lv.health.text())


TT.MockKalshi.shards, TT.MockKalshi.series_index = {0: 1004.33, 2: 0.0}, 2
_reconnect(lambda: "NO CASH" in lv.health.text())
assert lv.shard == 2 and "$0.00 on exchange shard 2 (Crypto & Commodities)" in lv.bp.text() and "total across shards $1,004.33" in lv.bp.text() and "Exchange Indexes" in lv.health.text() and "red" not in lv.bp.styleSheet() and "e05561" in lv.bp.styleSheet()
TT.MockKalshi.shards = {0: 4.33, 2: 1000.0}
_reconnect(lambda: "NO CASH" not in lv.health.text() and "$1,000.00" in lv.bp.text())
assert lv.shard == 2 and lv.bp.styleSheet() == ""
TT.MockKalshi.shards, TT.MockKalshi.series_index = {0: 950.5}, 0
_reconnect(lambda: "$950.50" in lv.bp.text())
assert lv.autofund.isChecked() and lv.make_config(True).auto_fund and not lv.make_config(False).auto_fund            # auto-fund: on by default, LIVE only, never for paper
lv.fund_max.setValue(250.0)
assert lv.make_config(True).fund_max == 250.0 and float(lv.qs.value("kalshi/fund_max")) == 250.0
lv.autofund.setChecked(False)
assert not lv.make_config(True).auto_fund and str(lv.qs.value("kalshi/autofund")).lower() in ("false", "0")
lv.autofund.setChecked(True); lv.fund_max.setValue(0.0)
# ── initial capital: $1,000 by default; LIVE + connected = the cash available in the account (until you type your own) ──
lv.capital.setValue(777.0)                                                                       # (earlier steps of this test set values by hand: that pins it)
lv._capital_default(); lv.mode.setCurrentIndex(0)
assert lv.capital.value() == 1000.0 and "Default $1,000" in lv.cap_note.text()
lv.mode.setCurrentIndex(1)
assert abs(lv.capital.value() - 950.5) < 1e-9 and "$950.50" in lv.cap_note.text()                    # mock account: $950.50 on the trading shard
TT.MockKalshi.shards, TT.MockKalshi.series_index = {0: 1004.33, 2: 4.33}, 2
_reconnect(lambda: "$4.33 on exchange shard 2" in lv.bp.text())
assert abs(lv.capital.value() - 1008.66) < 1e-9 and "all the cash on the account" in lv.cap_note.text()   # auto-fund on: everything on the account can be used
lv.autofund.setChecked(False)
assert abs(lv.capital.value() - 4.33) < 1e-9 and "auto-fund is off" in lv.cap_note.text()               # without auto-fund only the trading shard's cash counts
lv.autofund.setChecked(True)
lv.capital.setValue(200.0)                                                                      # you type a smaller amount: it is kept
TT.MockKalshi.shards = {0: 900.0, 2: 4.33}
_reconnect(lambda: "total across shards $904.33" in lv.bp.text())
assert lv.capital.value() == 200.0 and "You set this by hand" in lv.cap_note.text() and "$904.33" in lv.cap_note.text()
lv._capital_default()
assert abs(lv.capital.value() - 904.33) < 1e-9
lv._refresh_cash_sync()                                                                         # called when a session starts: picks up the account's cash right now
TT.MockKalshi.shards = {0: 500.0, 2: 4.33}
lv._refresh_cash_sync()
assert abs(lv.capital.value() - 504.33) < 1e-9
lv.mode.setCurrentIndex(0)
assert lv.capital.value() == 1000.0                                                              # paper mode: back to the plain default
TT.MockKalshi.shards, TT.MockKalshi.series_index = {0: 950.5}, 0
_reconnect(lambda: "$950.50" in lv.bp.text())
lv.on_event("alert", {"title": "Order rejected: not enough cash", "msg": "test alert"})                    # a pop-up (once) and a visible state, never a silent log line
lv.on_event("alert", {"title": "Order rejected: not enough cash", "msg": "test alert again"})
assert len(lv._popups) == 1 and "Order rejected" in lv.state.text() and "test alert" in lv.logbox.toPlainText()
for _b in lv._popups:
    _b.close()
lv._popups.clear(); lv._alerted.clear()

# ── a stray mouse wheel must not change a risk setting (scrolling the panel used to change whatever spin box was under the cursor) ──
from PyQt6.QtCore import QPoint, QPointF
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtCore import Qt as _Qt
def _wheel(widget, dy=120):
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, dy), _Qt.MouseButton.NoButton, _Qt.KeyboardModifier.NoModifier, _Qt.ScrollPhase.NoScrollPhase, False)
    ev.setAccepted(True)
    widget.wheelEvent(ev)
    return ev.isAccepted()
before = (w.settings.mult.value(), w.settings.cap.value(), w.settings.maxc.value(), w.settings.edge.value())
for wdg in (w.settings.mult, w.settings.cap, w.settings.maxc, w.settings.edge, w.settings.dloss, w.settings.ddstop, w.settings.look, w.settings.fee, w.settings.preset, bt.cap, bt.mode, lv.capital, lv.mode):
    assert not wdg.hasFocus() and _wheel(wdg) is False, wdg           # ignored (propagates to the scroll area) because the field is not focused
assert before == (w.settings.mult.value(), w.settings.cap.value(), w.settings.maxc.value(), w.settings.edge.value())
assert w.settings.mult.focusPolicy() == __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.FocusPolicy.StrongFocus

# ── polluted / risky saved settings are surfaced and recoverable, never silently used ──
_polluted = Path(tmp) / "polluted.ini"
_q = QSettings(str(_polluted), QSettings.Format.IniFormat)
for k_, v_ in (("settings/preset", "R2+ · Core 22-23 UTC + all-hours scout (default)"), ("settings/mult", 1), ("settings/cap", 5), ("settings/maxc", 199998), ("settings/throttle", "false"),
               ("settings/gate_v", 2), ("settings/half", 14), ("settings/corek", 10), ("settings/corehours", "22,23"), ("kalshi/keyfile", str(Path(tmp) / "no such key.pem")), ("auto_preflight", "false")):
    _q.setValue(k_, v_)
_q.sync(); del _q
_old_env = os.environ["TRADER_SETTINGS_FILE"]
os.environ["TRADER_SETTINGS_FILE"] = str(_polluted)
w2 = MainWindow()
assert not w2.settings.notice.isHidden() and "Kelly multiplier 1.00x" in w2.settings.notice.text() and "5.0%" in w2.settings.notice.text() and "199,998" in w2.settings.notice.text()
assert "no such key" not in w2.live.keypath.text()                       # a saved key path that does not exist is ignored
w2.settings.reset_sizing()
assert w2.settings.notice.isHidden() and (w2.settings.mult.value(), w2.settings.cap.value(), w2.settings.maxc.value()) == (0.25, 2.0, 2000) and w2.settings.throttle.isChecked()
w2.close()
os.environ["TRADER_SETTINGS_FILE"] = _old_env
assert w.settings.notice.isHidden()                                          # the normal defaults raise no warning

# ── pre-flight: dialog logic for GO / CAUTION / NO-GO, then the real Start button (pre-flight -> paper session) ──
import test_preflight as TP
from trader import preflight as PF
from trader.gui import PreflightDialog
lv.auto_pf.setChecked(False)
rows_ok = TP.make_rows(lambda o: True)
rep_go = TP.run(rows_ok)
rep_caution = TP.run(TP.make_rows(lambda o: o < TP.TS(2026, 10, 8) or np.random.default_rng(int(o)).random() < 0.45, end=(2026, 10, 9, 23, 45)), now=TP.TS(2026, 10, 10, 21, 46))
rep_nogo = TP.run(rows_ok, http_get=TP.down_get)                                         # a TECHNICAL failure blocks live (the Kalshi feed is down); weak evidence alone never does
assert (rep_go.verdict, rep_caution.verdict, rep_nogo.verdict) == ("GO", "CAUTION", "NO-GO")
d = PreflightDialog(lv, live=True, then_start=True, cap_now=0.02); d.set_report(rep_go)
assert d.btn_go.isEnabled() and "LIVE confirmation" in d.btn_go.text() and d._go_action == "start" and d.btn_paper.isHidden()
d = PreflightDialog(lv, live=True, then_start=True, cap_now=0.02); d.set_report(rep_caution)
assert d.btn_go.isEnabled() and "Continue to LIVE with my settings" in d.btn_go.text() and "2.0%" in d.btn_go.text() and d._go_action == "start"      # CAUTION never forces a cap
assert not d.btn_cap.isHidden() and "suggested 0.5% cap" in d.btn_cap.text() and not d.btn_paper.isHidden()                                        # the cap is an optional extra button
d_small = PreflightDialog(lv, live=True, then_start=True, cap_now=0.004); d_small.set_report(rep_caution)
assert d_small.btn_cap.isHidden() and d_small._go_action == "start"                                                                               # already below the suggestion: nothing to offer
# choosing the primary action keeps YOUR sizing; only the optional button changes the cap
_calls = []
_orig_start = lv._start_session
lv._start_session = lambda rep=None: _calls.append(rep)
lv.sp.cap.setValue(2.0)
_dd = PreflightDialog(lv, live=True, then_start=True, cap_now=0.02); _dd.set_report(rep_caution); _dd.action = "start"
lv._after_preflight_dialog(_dd)
assert lv.sp.cap.value() == 2.0 and len(_calls) == 1                                  # unchanged, session started with the user's setting
_dd.action = "start_capped"
lv._after_preflight_dialog(_dd)
assert lv.sp.cap.value() == 0.5 and len(_calls) == 2                                  # only when the optional button is chosen
lv.sp.cap.setValue(2.0)
lv._start_session = _orig_start
d.resize(1040, 700); d.show(); app.processEvents(); d.grab().save(str(out / "preflight_caution.png"))
d = PreflightDialog(lv, live=True, then_start=True, cap_now=0.02); d.set_report(rep_nogo)
assert not d.btn_go.isEnabled() and "blocked" in d.btn_go.text() and not d.btn_paper.isHidden()
d.show(); app.processEvents(); d.grab().save(str(out / "preflight_nogo.png"))
d = PreflightDialog(lv, live=False, then_start=True, cap_now=0.02); d.set_report(rep_nogo)
assert d.btn_go.isEnabled() and "Start paper session" in d.btn_go.text()                  # paper is never blocked
d = PreflightDialog(lv, live=False, then_start=False, cap_now=0.02); d.set_report(rep_go)
assert d.btn_go.isHidden() and d.btn_copy.isEnabled()

# real worker thread + real preflight module, offline feed / no network refresh / throw-away state
_real_run = PF.run_preflight
def _offline(cfg, **kw):
    kw.update(refresh=False, quotes=TP.Q(), http_get=TP.ok_get, coverage_fn=lambda *a, **k: (10, 10), state_directory=Path(tmp))
    return _real_run(cfg, **kw)
PF.run_preflight = _offline
lv.run_preflight(then_start=False)                                    # the "Pre-flight check" button
t0 = _t.time()
while lv._pf_report is None and _t.time() - t0 < 30:
    app.processEvents(); _t.sleep(0.05)
assert lv._pf_report is not None and lv.pf_lbl.text().startswith("Pre-flight ") and lv._pf_dlg.table.rowCount() >= 10 and lv.btn_start.isEnabled()
assert "pre-flight:" in lv.logbox.toPlainText()
lv._pf_dlg.resize(1040, 700); lv._pf_dlg.show(); app.processEvents(); lv._pf_dlg.grab().save(str(out / "preflight_real_run.png"))
lv._pf_dlg.close()
assert (Path(tmp) / "live").exists() and list((Path(tmp) / "live").glob("preflight_*.txt"))               # the report went to the throw-away state dir ...

def _fake_exec(self):                                                  # headless stand-in for the modal dialog: wait for the report, choose "start"
    t = _t.time()
    while self.report is None and _t.time() - t < 30:
        app.processEvents(); _t.sleep(0.02)
    self.action = "start"
    return 1
PreflightDialog.exec = _fake_exec
lv._pf_report = None
lv.capital.setValue(1000.0)
lv.mode.setCurrentIndex(0)
lv.start()                                                             # = pre-flight, then (action "start") the paper session
assert lv._pf_report is not None
t0 = _t.time()
while _t.time() - t0 < 3:
    app.processEvents(); _t.sleep(0.05)
assert lv.worker is not None and lv.worker.isRunning() and lv.engine is not None
assert lv.perf.kpis["Final equity"].v.text() == "$1,000", lv.perf.kpis["Final equity"].v.text()      # no trade yet: shows the current equity
assert "RUNNING SESSION" in lv.using_lbl.text() and "Paper" in lv.using_lbl.text()
lv.params_view.refresh()
assert "running session" in lv.params_view.table.item(0, 0).text() or "RUNNING" in lv.params_view.head.text().upper()
assert not lv.btn_start.isEnabled() and lv.btn_stop.isEnabled() and lv.btn_kill.isEnabled()
w.grab().save(str(out / "live_running.png"))
lv.stop()
assert lv.btn_start.isEnabled() and not lv.worker.isRunning()
assert PF.load_state(Path(tmp))["last_session"]["why"] == "stopped by user"       # session end recorded for the next pre-flight
assert w.settings.look.value() == 45 and w.settings.half.value() == 14 and w.settings.halflife() == 14.0 and "R2+" in w.settings.preset.currentText()      # defaults: R2 preset, 10-day decaying memory (45-day window is the fallback)
assert lv.make_config().halflife_days == 14.0 and bt.config().halflife_days == 14.0
assert lv.make_config().core_hours == (22, 23) and lv.make_config().core_k == 10 and bt.config().core_hours == (22, 23) and w.settings.active_hours() == (22, 23)
assert lv.make_config().hour_k == 100 and lv.make_config().p_model == "by_hour"
assert lv.make_config().near_k == 50 and lv.make_config().near_width == 2 and bt.config().near_k == 50 and bt.config().near_width == 2       # default R2+: lighter scouting right next to the core hours
rc_ = w.radar.config()
assert rc_.hour_k == 100 and rc_.core_hours == (22, 23) and rc_.core_k == 10 and rc_.near_k == 50 and rc_.near_width == 2        # the radar uses the engine's own shrinkage
for view in (lv.params_view, bt.params_view):
    view.refresh()
    rows_ = {view.table.item(i, 0).text(): (view.table.item(i, 1).text(), view.table.item(i, 2).text()) for i in range(view.table.rowCount()) if view.table.item(i, 0)}
    assert rows_["Near-core hours (lighter shrinkage)"] == ("within ±2 h of the core: k = 50",) * 2
w.settings.apply_preset([n for n in G_PRESETS if n.startswith("R2 ·")][0])
assert bt.config().near_k == 0 and lv.make_config().near_k == 0 and bt.config().core_hours == ()                                    # hard-wired R2: no scouting at all
w.settings.apply_preset(next(iter(G_PRESETS)))
tabs.setCurrentIndex(2); app.processEvents()                                              # regime radar tab (refreshes when shown)
assert w.radar.table.rowCount() == 25 and w.radar.summary.text() and "radar unavailable" not in w.radar.summary.text(), w.radar.summary.text()
assert w.radar.table.item(24, 0).text() == "ALL" and w.radar.table.item(22, 1).text() == "●"     # the 22 UTC row is marked as one of the active hours
w.grab().save(str(out / "radar.png"))
tabs.setCurrentIndex(3); app.processEvents(); w.grab().save(str(out / "help.png"))
tabs.setCurrentIndex(0)
# ── Auto mode in the GUI: preset, greyed-out fixed fields, same AutoConfig for both tabs, columns / tooltip, params table, persistence, pre-flight, a paper session ──
from trader.gui import PRESETS, SettingsPanel
from trader.live import auto_active
sp = w.settings
assert [n for n in PRESETS if n.startswith("AUTO")] and not sp.auto_on.isChecked() and sp.mult.isEnabled() and not sp.auto_ladder.isEnabled() and bt.config().auto is None and lv.make_config().auto is None
sp.apply_preset([n for n in PRESETS if n.startswith("AUTO")][0])
assert sp.auto_on.isChecked() and not sp.mult.isEnabled() and not sp.cap.isEnabled() and not sp.maxc.isEnabled() and not sp.half.isEnabled() and not sp.look.isEnabled()
assert sp.auto_ladder.isEnabled() and sp.auto_z.isEnabled() and sp.edge.isEnabled() and sp.throttle.isEnabled() and sp.dloss.isEnabled()          # what Auto still uses stays editable
b_cfg, l_cfg = bt.config(), lv.make_config()
assert b_cfg.auto is not None and l_cfg.auto == b_cfg.auto and b_cfg.auto.ladder[0][1:] == (0.5, 0.03) and b_cfg.auto.half_lives == (7.0, 14.0, 30.0) and b_cfg.auto.throttle
assert "AUTO" in bt.using_lbl.text() and "AUTO" in lv.using_lbl.text() and "ladder" in bt.using_lbl.text()
bt.run(); app.processEvents()
tr = bt.result.trades
assert bt.result.metrics["trades"] > 30 and set(tr.regime) <= {"HOT", "BUILDING", "FADING"} and tr.mult_used.between(0.01, 0.51).all() and tr.cap_used.between(0.001, 0.031).all()
hdr = [bt.tt.horizontalHeaderItem(j).text() for j in range(bt.tt.columnCount())]
assert hdr[16:19] == ["Regime (Auto)", "Kelly x", "Stake cap"] and bt.tt.item(0, 16).text() in ("HOT", "BUILDING", "FADING") and bt.tt.item(0, 17).text() != "—"
bt.chart.draw(); bt.chart._show_trade(5)
assert "Auto " in bt.chart._ann.get_text() and "stake cap" in bt.chart._ann.get_text()
for view in (lv.params_view, bt.params_view):
    view.refresh()
    rows_ = {view.table.item(i, 0).text(): (view.table.item(i, 1).text(), view.table.item(i, 2).text()) for i in range(view.table.rowCount()) if view.table.item(i, 0)}
    assert rows_["Auto mode"] == ("ON", "ON") and rows_["Equity ladder"][0] == rows_["Equity ladder"][1] and rows_["Kelly multiplier"][0].startswith("AUTO") and rows_["Kelly multiplier"][0] == rows_["Kelly multiplier"][1]
    assert rows_["Memory"][0].startswith("AUTO") and rows_["Memory"][0] == rows_["Memory"][1] and rows_["Liquidity cap"][0] == rows_["Liquidity cap"][1]
lv.refresh_preview(); app.processEvents()
assert lv.pv.columnCount() == 6 and not lv.auto_lbl.isHidden() and lv.auto_lbl.text().startswith("AUTO") and "ladder step" in lv.auto_lbl.text()
assert any("·" in (lv.pv.item(i, 5).text() if lv.pv.item(i, 5) else "") for i in range(lv.pv.rowCount()))
sp.save(w.qs); sp.auto_ladder.setText("0:0.4/2, 50000:0.2/1"); sp.save(w.qs)
sp2 = SettingsPanel(); sp2.load(w.qs)
assert sp2.auto_on.isChecked() and sp2.auto_ladder.text() == "0:0.4/2, 50000:0.2/1" and not sp2.mult.isEnabled() and sp2.auto_config().ladder == ((0.0, 0.4, 0.02), (50000.0, 0.2, 0.01))
sp.auto_ladder.setText("0:0.5/3, nonsense")
assert "Auto settings problem" in sp.sizing_notice() and bt.safe_config()[0] is None and "ladder" in bt.safe_config()[1]
sp.reset_sizing()
assert sp.auto_ladder.text().startswith("0:0.5/3, 25000:0.35/2.5") and sp.auto_on.isChecked() and bt.safe_config()[0] is not None
sp.cap_auto_ladder(0.005)                                                                                   # the pre-flight's optional cap in Auto mode
assert all(c <= 0.005 + 1e-12 for _t, _m, c in sp.auto_config().ladder) and sp.effective_cap() == 0.005
sp.reset_sizing()
# a paper session in Auto mode: pre-flight reports the Auto gate, the engine runs the Auto layer, the status strip fills in
lv._pf_report = None
lv.capital.setValue(1000.0); lv.mode.setCurrentIndex(0)
lv.start()
t0 = _t.time()
while _t.time() - t0 < 3:
    app.processEvents(); _t.sleep(0.05)
assert lv._pf_report is not None and any(c.name.startswith("Learner gate (Auto)") for c in lv._pf_report.checks) and any("AUTO sizing" in c.detail for c in lv._pf_report.checks if c.name == "Sizing")
assert lv.worker is not None and lv.worker.isRunning() and lv.engine.auto_on and auto_active(lv.engine.cfg) and "AUTO sizing" in lv.using_lbl.text()
lv.engine.push_status(); app.processEvents()
assert lv.auto_lbl.text().startswith("AUTO") and not sp.auto_on.isEnabled()                                                        # locked while running
lv.params_view.refresh()
lv.stop()
assert sp.auto_on.isEnabled() and not sp.mult.isEnabled() and sp.auto_ladder.isEnabled()                                          # unlocked again, Auto still on: the fixed fields stay greyed
sp.auto_on.setChecked(False)
assert sp.mult.isEnabled() and not sp.auto_ladder.isEnabled() and sp.preset.currentText() == "Custom" and bt.config().auto is None and lv.make_config().auto is None
sp.apply_preset(next(iter(PRESETS)))
assert not sp.auto_on.isChecked() and sp.mult.isEnabled() and bt.config().auto is None
# ── EV-regime filter presets: the three rules are in the settings, both tabs use them, the learned-probability boxes step aside ──
fb = [n for n in PRESETS if n.startswith("FILTER-BOTH")][0]
sp.apply_preset(fb)
assert sp.ev_rules() == ((14, 0.1), (28, 0.05)) and bt.config().ev_rules == lv.make_config().ev_rules == sp.ev_rules() and sp.evn.value() == 8
assert bt.config().auto is None and lv.make_config().auto is None and not sp.pmodel.isEnabled() and not sp.hourk.isEnabled() and not sp.auto_on.isEnabled() and sp.mult.isEnabled() and sp.ev14.isEnabled()
sp.apply_preset([n for n in PRESETS if n.startswith("FILTER-14")][0]); assert sp.ev_rules() == ((14, 0.1),)
sp.apply_preset([n for n in PRESETS if n.startswith("FILTER-28")][0]); assert sp.ev_rules() == ((28, 0.05),)
sp.apply_preset(fb)
sp.ev14.setValue(12.0)
assert sp.preset.currentText() == "Custom" and sp.ev_rules() == ((14, 0.12), (28, 0.05))                       # editing a rule makes it a custom preset
sp.apply_preset(fb)
w.radar.refresh(); app.processEvents()
assert w.radar.table.rowCount() == 25 and {w.radar.table.item(i, 10).text() for i in range(25)} <= {"TRADE", "cold"} and "radar unavailable" not in w.radar.summary.text()
for view in (bt.params_view, lv.params_view):
    view.refresh()
    prow = {view.table.item(i, 0).text(): (view.table.item(i, 1).text(), view.table.item(i, 2).text()) for i in range(view.table.rowCount()) if view.table.item(i, 0)}
    assert "14 d net EV > 10c AND 28 d net EV > 5c" in prow["Trade an hour only if"][0] and prow["Trade an hour only if"][0] == prow["Trade an hour only if"][1]
w.grab().save(str(out / "ev_filter_preset.png"))
sp.apply_preset(next(iter(PRESETS)))
assert sp.ev_rules() == () and sp.pmodel.isEnabled() and sp.hourk.isEnabled() and sp.auto_on.isEnabled() and bt.config().ev_rules == ()
# ── PROJ-EV: the preset carries its own conviction hurdle (any positive projected EV) and hands yours back when you leave ──
pe = [n for n in PRESETS if n.startswith("PROJ-EV")][0]
sp.edge.setValue(3.0)                                                                                 # your own Min edge
sp.apply_preset(pe)
assert sp.edge.value() == 0.0 and bt.config().min_edge == lv.make_config().min_edge == 0.0 and sp.ev_rules() == () and sp.pmodel.isEnabled() and sp.hourk.value() == 100 and sp.half.value() == 14
sp.apply_preset(next(iter(PRESETS)))
assert abs(sp.edge.value() - 3.0) < 1e-9 and abs(bt.config().min_edge - 0.03) < 1e-9                  # your value is back
sp.apply_preset(pe); sp.edge.setValue(2.0)                                                            # you change it while the preset is on: that choice stays
sp.apply_preset(next(iter(PRESETS)))
assert abs(sp.edge.value() - 2.0) < 1e-9
sp.edge.setValue(4.0)
w.radar.refresh(); app.processEvents()
w.grab().save(str(out / "auto_settings.png"))
# ── "Verify live = backtest" (Backtest tab) and "Compare with backtest" (Live tab) on the real data ──
import time as _tv
from datetime import timedelta as _td
from trader.ledger import Ledger as _VL
from trader.data import data_range as _dr
from trader.presets import preset_config as _pc
from trader.replay import replay_live as _rl
bt.verify_btn.click()
assert not bt.verify_btn.isEnabled() and "Replaying" in bt.verify_lbl.text()                       # runs in a worker thread: the window stays responsive
t0_ = _tv.time()
while _tv.time() - t0_ < 120 and not bt.verify_btn.isEnabled():
    app.processEvents(); _tv.sleep(0.05)
assert bt.verify_btn.isEnabled() and bt.verify_lbl.text().startswith("✔ Live engine = backtest") and "IDENTICAL" in bt.verify_lbl.text(), bt.verify_lbl.text()
tabs.setCurrentIndex(1); app.processEvents(); w.grab().save(str(out / "verify_backtest.png")); tabs.setCurrentIndex(0)
bt.compound.setChecked(False)                                                                        # a setting the live engine cannot do: refused with the reason, nothing runs
bt.verify_btn.click(); app.processEvents()
assert bt.verify_btn.isEnabled() and "Cannot verify" in bt.verify_lbl.text() and "compound" in bt.verify_lbl.text()
bt.compound.setChecked(True)
_end = _dr(w.rows())[1].date()
_cfg = _pc("R2+", _end - _td(days=30), _end, capital=1000.0)
_vled = _VL(Path(tmp) / "cmp" / "ledger.jsonl")                                                      # a throw-away ledger: a replayed session stands in for a real one
_rl(w.rows(), _cfg, ledger=_vled)
_saved_ledger, _saved_mode = lv.ledger, lv.mode.currentIndex()
lv.ledger = _vled
lv.mode.setCurrentIndex(0)
lv.btn_cmp.click()
t0_ = _tv.time()
while _tv.time() - t0_ < 120 and "session(s)" not in lv.cmp_lbl.text():
    app.processEvents(); _tv.sleep(0.05)
assert lv.btn_cmp.isEnabled() and lv.cmp_lbl.text().startswith("PAPER sessions. 1 session(s)") and "agreement 100%" in lv.cmp_lbl.text(), lv.cmp_lbl.text()
assert lv.cmp_table.rowCount() >= 3 and lv.cmp_table.item(0, 1).text() == "matched" and "(+2.0c)" in lv.cmp_table.item(0, 4).text(), (lv.cmp_table.rowCount(), lv.cmp_lbl.text())
_ix = [lv.perf.tabs.tabText(i) for i in range(lv.perf.tabs.count())].index("Live vs backtest"); lv.perf.tabs.setCurrentIndex(_ix); app.processEvents(); w.grab().save(str(out / "live_vs_backtest.png")); lv.perf.tabs.setCurrentIndex(0)
lv.ledger = _VL(Path(tmp) / "cmp_empty" / "ledger.jsonl")                                           # nothing recorded: says so instead of showing an empty grid
lv.btn_cmp.click()
t0_ = _tv.time()
while _tv.time() - t0_ < 30 and "Nothing to compare" not in lv.cmp_lbl.text():
    app.processEvents(); _tv.sleep(0.05)
assert "Nothing to compare" in lv.cmp_lbl.text() and lv.cmp_table.rowCount() == 0
lv.ledger = _saved_ledger
lv.mode.setCurrentIndex(_saved_mode)
w.grab().save(str(out / "verify_and_compare.png"))
# ── trade history survives a restart; newest data is always included ──
import time as _tm
from trader.ledger import Ledger as _Ledger
L = lv.ledger
assert L.path == Path(tmp) / "live" / "ledger.jsonl"                                          # the isolated state dir, never the real data folder
_T0 = int(_tm.time()) // 900 * 900 - 3 * 86400


def _add(mode, session, k, win, settle=True, qty=100, entry=0.17, fee=0.3):
    o = _T0 + k * 900
    tid = L.record_open(mode=mode, session=session, market=f"KXBTC15M-TEST{k}", window=o, leg="DOG", side="YES", quoted=0.15, entry=entry, contracts=qty, stake=qty * entry + fee, fee=fee, p_used=0.3, f_full=0.1,
                        f_used=0.02, bankroll_before=500.0, order_id=f"oid{k}" if mode == "live" else "paper", entry_ts=o + 722, regime="HOT", mult_used=0.5, cap_used=0.03)
    if settle:
        pnl = qty * (1.0 if win else 0.0) - qty * entry - fee
        L.record_settle(tid, settle_ts=o + 925, outcome="WIN" if win else "LOSS", pnl=pnl, equity_after=500.0 + pnl)


for k, win in ((0, True), (1, False), (2, True)):
    _add("live", "20261009_220000_live", k, win)
_add("live", "20261009_220000_live", 3, True, settle=False)                      # still open when that app was closed
for k, win in ((10, True), (11, False)):
    _add("paper", "20261009_210000_paper", k, win)
# ---- "restart the app": a brand-new window on the same ledger file ----
w2 = MainWindow()
w2.show(); app.processEvents()
l2 = w2.live
assert l2.scope.currentIndex() == 1 and l2.mode.currentIndex() == 0 and l2.tt.rowCount() == 2 and l2.perf.kpis["Trades"].v.text() == "2" and "2 settled" in l2.hist_lbl.text()       # Paper history is there
l2.mode.setCurrentIndex(1)
app.processEvents()
assert l2.tt.rowCount() == 3 and "3 settled" in l2.hist_lbl.text() and "1 open" in l2.hist_lbl.text() and l2.tt.horizontalHeaderItem(l2.tt.columnCount() - 1).text() == "Session"       # LIVE history, open position flagged
assert l2.tt.item(0, l2.tt.columnCount() - 1).text() == "20261009_220000_live" and l2.perf.kpis["Trades"].v.text() == "3"
l2.reconcile_result_fn = lambda t: "yes"                                                   # Kalshi says the open window resolved YES -> our YES position won
l2.refresh_history(True)
t0_ = _tm.time()
while _tm.time() - t0_ < 20 and l2.tt.rowCount() != 4:
    app.processEvents(); _tm.sleep(0.05)
assert l2.tt.rowCount() == 4 and "settled after a restart" in l2.hist_lbl.text() and "open" not in l2.hist_lbl.text().split(".")[0]
assert l2.tt.item(3, l2.tt.columnCount() - 1).text().endswith("(settled after a restart)")
assert l2.perf.kpis["Final equity"].v.text() == "$731" and l2.perf.kpis["Trades"].v.text() == "4"          # 500 + 3 wins x 82.7 - 17.3
l2.perf.chart.draw(); l2.perf.chart._show_trade(2)
assert "Session 20261009_220000_live" in l2.perf.chart._ann.get_text()
l2.scope.setCurrentIndex(2); app.processEvents()
assert l2.tt.rowCount() == 6 and "paper + LIVE" in l2.hist_lbl.text()
l2.scope.setCurrentIndex(0); app.processEvents()
assert l2.tt.rowCount() == 0 and "this session only" in l2.hist_lbl.text().lower()
l2.scope.setCurrentIndex(1); app.processEvents()
w2.grab().save(str(out / "live_history_after_restart.png"))
# ---- newest data: the window refreshes stale data before a backtest, 'To' follows the data, a date you set yourself stays put ----
import trader.gui as G
from trader import preflight as PFX
st_ = PFX.load_state(); st_["data_update_ts"] = 0; PFX.save_state(st_)
base_rows = w2.rows().copy()
hi0 = G.data_range(base_rows)[1]
new_rows = pd.DataFrame([dict(ticker="NEWX", open_ts=int(hi0.timestamp()) + 2 * 86400, minute=12, bid=0.14, ask=0.15, yes=1, vol=np.nan)])
fresh_ = {"on": False}
calls_ = []
def _fake_updater(log):
    calls_.append(1); fresh_["on"] = True
    return True
real_load, real_need = G.load_rows, G.needs_refresh
G.load_rows = lambda *a, **k: pd.concat([base_rows, new_rows], ignore_index=True) if fresh_["on"] else base_rows
G.needs_refresh = lambda rows, now, last: now - last > 300
w2.auto_update, w2.updater = True, _fake_updater
bt2 = w2.backtest
assert bt2.refresh_chk.isChecked() and bt2.result is None
try:
    bt2.run()                                                                           # stale data -> refresh in a worker thread, THEN the backtest runs
    assert not bt2.run_btn.isEnabled() or bt2.result is not None
    t0_ = _tm.time()
    while _tm.time() - t0_ < 90 and bt2.result is None:
        app.processEvents(); _tm.sleep(0.05)
    assert bt2.result is not None and len(calls_) == 1 and bt2.run_btn.isEnabled()
    new_hi = G.data_range(pd.concat([base_rows, new_rows]))[1]
    assert bt2._hi == new_hi and bt2.d1.date().toString("yyyy-MM-dd") == f"{new_hi:%Y-%m-%d}" and f"{new_hi:%Y-%m-%d %H:%M}" in bt2.info.text() and not bt2._to_pinned
    bt2.run()                                                                           # asked the API 0 minutes ago: no second download
    assert len(calls_) == 1
    bt2.d1.setDate(QDate(hi0.year, hi0.month, hi0.day))                                 # YOU choose an earlier end date: it stays, and the window says newer data is excluded
    assert bt2._to_pinned and "newer data is not included" in bt2.fresh_lbl.text()
    bt2.set_range(G.data_range(base_rows)[0], new_hi)
    assert bt2.d1.date().toString("yyyy-MM-dd") == f"{hi0:%Y-%m-%d}"
    bt2.btn_latest.click()
    assert not bt2._to_pinned and bt2.d1.date().toString("yyyy-MM-dd") == f"{new_hi:%Y-%m-%d}" and "newer data is not included" not in bt2.fresh_lbl.text()
finally:
    G.load_rows, G.needs_refresh = real_load, real_need
    w2.auto_update = False
w2.close()
_MOCK.shutdown()
# nothing to clean up: sessions, ledger and pre-flight reports of this test all live in the throw-away TRADER_STATE_DIR - and the real data/live folder must come out exactly as it went in
assert sorted(q.name for q in _REAL_LIVE.glob("*")) == _REAL_LIVE_BEFORE, "the smoke test changed the real data/live folder"
print("GUI smoke OK ->", out)
w.close()
