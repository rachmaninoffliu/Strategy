"""Headless test of the PyQt6 Supertrend app (offscreen Qt).   Usage:  python test_supertrend_gui.py [screenshot_dir]

Builds the real window against the real cached bars, drives it through the same button handlers a user uses, and replaces EVERYTHING that touches the outside world:
  * Alpaca trading + data clients are fakes (no network, no real account is ever contacted, no order is ever sent),
  * reports / state / log / KILL files and the saved settings go to a throw-away folder.
The trading Engine itself (supertrend_gui.Engine) is the real one, running against those fakes. Screenshots of each page are saved for a visual check."""
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


def _fail_loudly(t, v, tb):                                  # PyQt6 turns an unhandled exception inside a slot into a silent crash; print it instead
    traceback.print_exception(t, v, tb)
    sys.stderr.flush(); sys.stdout.flush()
    os._exit(1)


sys.excepthook = _fail_loudly
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")                 # the offscreen plugin has no font db otherwise (text renders as boxes)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

tmp = tempfile.mkdtemp()
os.environ["SUPERTREND_SETTINGS_FILE"] = str(Path(tmp) / "settings.ini")
os.environ["SUPERTREND_NO_AUTO_UPDATE"] = "1"

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import supertrend_gui as GU                                   # noqa: E402
import supertrend_strategy as S                               # noqa: E402

GU.REPORTS = tmp
GU.STATE_FILE, GU.LOG_FILE, GU.KILL_FILE = (os.path.join(tmp, n) for n in ("gui_state.json", "live_log.jsonl", "KILL"))
GU.POLL_SECONDS = 1                                           # the engine polls every second instead of every minute

from PyQt6.QtCore import QDate, QPoint, QPointF, QSettings, Qt                   # noqa: E402
from PyQt6.QtGui import QWheelEvent                           # noqa: E402
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox, QScrollArea, QSplitter, QTableWidget  # noqa: E402

from supertrend_ui import backtest as B                       # noqa: E402
from supertrend_ui import data as D                           # noqa: E402
from supertrend_ui import metrics as M                        # noqa: E402
from supertrend_ui import params as P                         # noqa: E402
from supertrend_ui import preflight as PF                     # noqa: E402
from supertrend_ui.main_window import MainWindow              # noqa: E402
from supertrend_ui.preflight_dialog import PreflightDialog    # noqa: E402
from supertrend_ui.theme import apply_theme                   # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tmp)
out.mkdir(parents=True, exist_ok=True)
BARS = D.read_cache("SOXL", 15)
LAST_CLOSE = float(BARS.close.iloc[-1])


# ══ fakes ═══════════════════════════════════════════════════════════════════════════
class FakeTrading:
    """Stands in for alpaca's TradingClient. Records every call; keeps its own open-order book so the engine's duplicate-order logic behaves as at a real broker."""
    calls: list = []
    submitted: list = []
    positions: list = []
    shortable = True
    is_open = True
    equity = 100_000.0
    blocked = False

    def __init__(self, key, secret, paper=True):
        self.paper = paper
        self._base_url = "BaseURL.TRADING_PAPER" if paper else "BaseURL.TRADING_LIVE"

    def get_account(self):
        return SimpleNamespace(account_number="PA3812345" if self.paper else "LV9912345", status="AccountStatus.ACTIVE", equity=str(FakeTrading.equity), last_equity="99500", buying_power="400000",
                               shorting_enabled=True, pattern_day_trader=False, daytrade_count=0, trading_blocked=FakeTrading.blocked, account_blocked=False)

    def get_clock(self):
        return SimpleNamespace(is_open=FakeTrading.is_open, next_open="2026-10-21 09:30:00-04:00", next_close="2026-10-20 16:00:00-04:00")

    def get_asset(self, sym):
        return SimpleNamespace(symbol=sym, tradable=True, shortable=FakeTrading.shortable if sym == "SOXL" else True, easy_to_borrow=True)

    def get_all_positions(self):
        return list(FakeTrading.positions)

    def get_orders(self, req=None):
        if "OPEN" in str(getattr(req, "status", "")).upper():
            return [o for o in FakeTrading.submitted if o.open]
        return []

    def submit_order(self, req):
        o = SimpleNamespace(id=f"order-{len(FakeTrading.submitted) + 1}", symbol=req.symbol, side=req.side, qty=req.qty, limit_price=getattr(req, "limit_price", None), status="accepted", open=True, legs=None)
        FakeTrading.submitted.append(o)
        FakeTrading.calls.append(("submit_order", req.symbol, str(req.side), req.qty, getattr(req, "limit_price", None)))
        return o

    def cancel_order_by_id(self, oid):
        for o in FakeTrading.submitted:
            if o.id == oid:
                o.open = False
        FakeTrading.calls.append(("cancel_order", oid))

    def cancel_orders(self):
        for o in FakeTrading.submitted:
            o.open = False
        FakeTrading.calls.append(("cancel_orders",))
        return []

    def get_order_by_id(self, oid):
        o = next(x for x in FakeTrading.submitted if x.id == oid)
        return SimpleNamespace(id=oid, status="canceled" if not o.open else "new")

    def close_all_positions(self, cancel_orders=True):
        FakeTrading.calls.append(("close_all_positions", cancel_orders))
        n = len(FakeTrading.positions)
        FakeTrading.positions = []
        return [object()] * n


class FakeData:
    def __init__(self, key, secret):
        pass

    def get_stock_bars(self, req):
        idx = BARS.index.tz_localize("America/New_York").tz_convert("UTC")
        st = pd.Timestamp(req.start)                                       # alpaca treats a naive start as UTC
        keep = idx >= (st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC"))
        df = BARS[keep].copy()
        df.index = idx[keep]
        df.index.name = "timestamp"
        df["symbol"] = "SOXL"
        return SimpleNamespace(df=df.set_index("symbol", append=True).swaplevel())

    def get_stock_latest_quote(self, req):
        return {s: SimpleNamespace(ask_price=(LAST_CLOSE if s == "SOXL" else 47.8) * 1.0004, bid_price=(LAST_CLOSE if s == "SOXL" else 47.8) * 0.9996) for s in req.symbol_or_symbols}


def fake_position(sym="SOXS", qty=400, avg=47.0, cur=48.0):
    side = SimpleNamespace(value="long")
    return SimpleNamespace(symbol=sym, qty=str(qty), side=side, avg_entry_price=str(avg), market_value=str(qty * cur), unrealized_pl=str(qty * (cur - avg)), unrealized_plpc=str((cur - avg) / avg),
                           current_price=str(cur))


LIVE_KEYS = {"ok": True}


def fake_creds(paper):
    if not paper and not LIVE_KEYS["ok"]:
        raise RuntimeError("No usable credentials for 'alpaca-live' in ~/.claude.json")
    return "KEY", "SECRET"


GU.creds, GU.TradingClient, GU.StockHistoricalDataClient = fake_creds, FakeTrading, FakeData

# message boxes answered headlessly (and recorded)
ASK = {"answer": QMessageBox.StandardButton.Yes, "log": []}
WARN = []
INPUT = {"text": "", "ok": True}
QMessageBox.question = staticmethod(lambda parent, title, text, *a, **k: (ASK["log"].append((title, text)), ASK["answer"])[1])
QMessageBox.warning = staticmethod(lambda parent, title, text, *a, **k: (WARN.append((title, text)), QMessageBox.StandardButton.Ok)[1])
QMessageBox.information = staticmethod(lambda parent, title, text, *a, **k: (WARN.append((title, text)), QMessageBox.StandardButton.Ok)[1])
QInputDialog.getText = staticmethod(lambda parent, title, label, *a, **k: (ASK["log"].append((title, label)), (INPUT["text"], INPUT["ok"]))[1])
QApplication.beep = staticmethod(lambda: None)

app = QApplication([])
apply_theme(app)


def wait(cond, timeout=30.0, what="", diag=None):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); time.sleep(0.03)
    assert cond(), f"timed out waiting for {what or cond}" + (f" | {diag()}" if diag else "")


def shot(widget, name):
    app.processEvents()
    widget.grab().save(str(out / f"{name}.png"))


def json_load(path):
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reset_fake():
    FakeTrading.calls, FakeTrading.submitted, FakeTrading.positions = [], [], []
    FakeTrading.shortable, FakeTrading.is_open, FakeTrading.equity, FakeTrading.blocked = True, True, 100_000.0, False
    LIVE_KEYS["ok"] = True


# ══ 1. pure logic: metrics, data helpers, parameters ════════════════════════════════
vals0 = dict(P.DEFAULTS)
bt0 = P.BacktestSettings(symbol="SOXL", tf=15, start="2016-01-01", end="2026-12-31", capital=100_000.0, prime=True, benchmark=True, strict=False)          # the classic backtest: identical to the strategy file's own run
r0 = B.run_backtest(BARS, vals0, bt0)
m0 = r0.metrics
ref_eq, ref_tr = S.SupertrendStrategy(P.to_config(vals0, initial_capital=100_000.0)).run(BARS)
ref = S.summarize(ref_eq, ref_tr, 26 * 252)
assert m0["trades"] == ref["trades"] == len(r0.trades) and abs(m0["max_drawdown"] * 100 - ref["max_dd_pct"]) < 1e-9, (m0["trades"], ref["trades"])
assert abs(m0["sharpe_bar"] - ref["sharpe"]) < 1e-9 and abs(m0["profit_factor_pct"] - ref["profit_factor"]) < 1e-9 and abs(m0["win_rate"] * 100 - ref["win_rate_pct"]) < 1e-9
assert abs(m0["final_equity"] - ref["final_equity"]) < 1e-3 and abs(m0["expectancy"] * 100 - ref["avg_trade_pct"]) < 1e-9          # identical to supertrend_strategy.summarize on the same bars
assert m0["total_return"] > 100 and 1.0 < m0["sharpe"] < 3.0 and m0["bh_total_return"] > 10 and 0.0 < m0["exposure"] <= 1.0
assert abs(m0["exit_reversal"] + m0["exit_trail"] + m0["exit_target"] - 1.0) < 1e-9 and m0["long_trades"] + m0["short_trades"] == m0["trades"]
mt = M.monthly_table(r0.eq, 100_000.0)
assert list(mt.columns) == list(range(1, 13)) + ["Year"] and abs((1 + mt["Year"]).prod() - r0.eq.iloc[-1] / 100_000.0) / (r0.eq.iloc[-1] / 100_000.0) < 1e-9      # the yearly returns compound to the total
bd = M.breakdown(r0.trades)
assert bd.n.iloc[0] == m0["trades"] and abs(bd.pnl.iloc[0] - r0.trades.pnl_usd.sum()) < 1e-3
# live-accurate (strict) backtest: the strategy's own strict run, and the position it still holds after the last bar
bt0s = P.BacktestSettings(symbol="SOXL", tf=15, start="2016-01-01", end="2026-12-31", capital=100_000.0, prime=True, benchmark=False, strict=True)
r0s = B.run_backtest(BARS, vals0, bt0s)
strat_s = S.SupertrendStrategy(P.to_config(vals0, initial_capital=100_000.0, strict_entries=True))
_, ref_tr_s = strat_s.run(BARS)
assert len(r0s.trades) == len(ref_tr_s) != len(r0.trades) and P.BacktestSettings().strict is True          # strict is the default; it is a different (live-attainable) set of trades
assert not np.array_equal(r0s.trades.entry_time.to_numpy()[:50], r0.trades.entry_time.to_numpy()[:50]) or len(r0s.trades) != len(r0.trades)
ot = r0s.open_trade
assert ot is None or (ot["side"] in ("long", "short") and ot["qty"] > 0 and ot["tracked"] and ot["asof"] == str(BARS.index[-1]) and abs(ot["now"] - float(BARS.close.iloc[-1])) < 1e-9)
if ot is not None:
    sgn = 1 if ot["side"] == "long" else -1
    assert abs(ot["pnl_pct"] - (ot["now"] / ot["price"] - 1) * 100 * sgn) < 1e-9 and abs(ot["pnl_usd"] - ot["qty"] * (ot["now"] - ot["price"]) * sgn) < 1e-6
    assert ot["tp"] > ot["entry_ref"] if ot["side"] == "long" else ot["tp"] < ot["entry_ref"]
    assert (ot["trail"] is None) or ((ot["trail"] >= ot["entry_ref"]) if ot["side"] == "long" else (ot["trail"] <= ot["entry_ref"]))
    assert ot["opened"] <= str(BARS.index[-1]) and (ref_tr_s.empty or pd.Timestamp(ot["opened"]) >= pd.Timestamp(ref_tr_s.entry_time.iloc[-1]))
# the open position is the trade a LONGER run goes on to close: cut the data at many bars and check the trade that is open there is a trade of the full run, with the same entry
full_entries = {(pd.Timestamp(t.entry_time), round(float(t.price), 6)) for t in ref_tr_s.itertuples()}
cuts = list(range(len(BARS) - 4000, len(BARS) - 40, 211))
n_open = 0
CUT = {}                                                                                                 # side -> a cut index at which the strict backtest holds that side ('flat' = none)
for k in cuts + list(range(len(BARS) - 4000, len(BARS) - 40, 37)):
    cut = BARS.iloc[:k]
    sc = S.SupertrendStrategy(P.to_config(vals0, initial_capital=100_000.0, strict_entries=True)); sc.run(cut)
    CUT.setdefault(sc.open_trade["side"] if sc.open_trade is not None else "flat", k)
    if sc.open_trade is not None and k in cuts:
        n_open += 1
        o_ = sc.open_trade
        assert o_["last_time"] == cut.index[-1] and (pd.Timestamp(o_["entry_time"]), round(float(o_["price"]), 6)) in full_entries, (k, o_["entry_time"], o_["price"])
assert n_open >= len(cuts) // 3 and {"long", "short", "flat"} <= set(CUT), (n_open, len(cuts), CUT)                                                 # the strategy is in the market a good part of the time
# the common shape and the comparison verdicts (pure)
v_l = {"side": "long", "symbol": "SOXL", "qty": 100, "opened": "2026-10-20 10:00:00", "price": 100.0, "now": 101.0, "pnl_pct": 1.0, "pnl_usd": 100.0, "entry_ref": 100.0, "tp": 110.0, "target": 104.5,
       "trail": None, "extreme": 102.0, "fixed": True, "tracked": True}
assert GU.compare_open(None, None)["status"] == "match" and GU.compare_open(v_l, dict(v_l))["status"] == "match"
assert GU.compare_open(v_l, {**v_l, "qty": 5000})["status"] == "match"                                      # quantity follows each side's own equity: not compared
assert GU.compare_open(v_l, {**v_l, "tp": 111.0})["status"] == "differs" and "take-profit" in GU.compare_open(v_l, {**v_l, "tp": 111.0})["text"]
assert GU.compare_open(v_l, {**v_l, "tp": 110.2})["status"] == "match"                                       # within 0.5%
assert GU.compare_open(v_l, {**v_l, "trail": 101.0})["status"] == "differs" and "armed in the backtest" in GU.compare_open(v_l, {**v_l, "trail": 101.0})["text"]
assert GU.compare_open({**v_l, "trail": 101.0}, {**v_l, "trail": 101.2})["status"] == "match" and GU.compare_open({**v_l, "trail": 101.0}, {**v_l, "trail": 103.0})["status"] == "differs"
assert GU.compare_open(None, v_l)["status"] == "differs" and GU.compare_open(v_l, None)["status"] == "differs" and GU.compare_open(v_l, {**v_l, "side": "short"})["status"] == "differs"
assert GU.compare_open(None, v_l, forming=True)["status"] == "timing" and "still forming" in GU.compare_open(None, v_l, forming=True)["text"]         # a fill / reversal inside the forming bar reaches the backtest at its close
assert GU.compare_open({**v_l, "tracked": False}, v_l)["status"] == "untracked" and GU.compare_open({**v_l, "tp": 120.0}, v_l, forming=True)["status"] == "timing"
bv = GU.backtest_open_view({"side": "short", "entry_time": pd.Timestamp("2026-10-20 10:00"), "fill_time": pd.Timestamp("2026-10-20 10:15"), "price": 50.0, "qty": 10, "entry_ref": 50.0, "supert_entry": 55.0,
                            "target": 47.0, "tp": 45.0, "trail": None, "active_fixed": True, "extreme": None, "last_time": pd.Timestamp("2026-10-20 11:00"), "last_close": 49.0,
                            "unrealized_pct": 2.0, "unrealized_usd": 10.0})
assert bv["side"] == "short" and bv["symbol"] == "SOXL" and bv["opened"].startswith("2026-10-20 10:15") and bv["fixed"] is True and GU.backtest_open_view(None) is None
# a late start: priming keeps the strategy trading from day one, never before the start date
late = B.run_backtest(BARS, vals0, P.BacktestSettings(symbol="SOXL", tf=15, start="2023-01-02", end="2026-12-31", capital=50_000.0, prime=True, benchmark=False))
cold = B.run_backtest(BARS, vals0, P.BacktestSettings(symbol="SOXL", tf=15, start="2023-01-02", end="2026-12-31", capital=50_000.0, prime=False, benchmark=False))
assert late.trades.entry_time.min() >= late.first_bar and cold.trades.entry_time.min() > late.trades.entry_time.min() and abs(late.eq.iloc[0] - 50_000.0) < 100 and late.warm_bars >= 500          # primed: trading from the first bar of the window (the first equity point is capital less that trade's commission)
try:
    B.run_backtest(BARS, vals0, P.BacktestSettings(symbol="SOXL", tf=15, start="2031-01-01", end="2031-02-01", capital=1000.0))
    raise AssertionError("an empty window must raise")
except ValueError as e:
    assert "no SOXL bars" in str(e)
# live trade rows -> the same schema (broker fills carry only the exit time)
live_rows = [dict(time=pd.Timestamp("2026-10-19 14:00", tz="UTC"), symbol="SOXL", side="LONG", qty=100, entry=100.0, exit=110.0, pnl=1000.0, pct=10.0),
             dict(time=pd.Timestamp("2026-10-20 15:00", tz="UTC"), symbol="SOXS", side="SHORT", qty=50, entry=50.0, exit=45.0, pnl=-250.0, pct=-10.0)]
lt = M.trades_from_live(live_rows, 10_000.0)
assert list(lt.equity_after) == [11_000.0, 10_750.0] and lt.entry_time.isna().all() and lt.exit_time.iloc[0] == pd.Timestamp("2026-10-19 10:00")
lc = M.curve_from_trades(lt, 10_000.0, "2026-10-19", "2026-10-21", 10_900.0)
lm = M.compute_metrics(lc, lt, 10_000.0, fill_days=True)
assert lm["trades"] == 2 and abs(lm["total_return"] - 0.09) < 1e-9 and lm["wins"] == 1 and lm["short_trades"] == 1 and np.isfinite(lm["max_drawdown"])
tiny = M.compute_metrics(M.curve_from_trades(lt.iloc[:0], 1000.0, "2026-10-20", "2026-10-20 00:03", 1000.5), lt.iloc[:0], 1000.0)          # three minutes of data annualised: inf at worst, never a crash
assert tiny["trades"] == 0 and tiny["total_return"] > 0 and not np.isnan(tiny["cagr"])
# data helpers
sat, mon_am, tue_pm = datetime(2026, 10, 17, 12), datetime(2026, 10, 19, 8, 0), datetime(2026, 10, 20, 17, 0)
assert D.expected_last_bar(sat) == datetime(2026, 10, 16, 15, 45) and D.expected_last_bar(mon_am) == datetime(2026, 10, 16, 15, 45) and D.expected_last_bar(tue_pm) == datetime(2026, 10, 20, 15, 45)
assert D.expected_last_bar(datetime(2026, 10, 20, 11, 7)) == datetime(2026, 10, 20, 10, 30)               # 11:07 less the 16-min SIP delay = 10:51 -> the 10:30 bar is the newest complete one
assert D.needs_refresh(datetime(2026, 10, 16, 15, 45), tue_pm, 0.0) and not D.needs_refresh(datetime(2026, 10, 20, 15, 45), tue_pm, 0.0) and not D.needs_refresh(datetime(2026, 10, 16, 15, 45), tue_pm, time.time())
assert D.next_bar_close(datetime(2026, 10, 20, 10, 7, 30)) == datetime(2026, 10, 20, 10, 15) and D.next_bar_close(datetime(2026, 10, 20, 15, 59)) == datetime(2026, 10, 20, 16, 0)
assert D.next_bar_close(datetime(2026, 10, 20, 16, 0)) is None and D.next_open(datetime(2026, 10, 23, 17, 0)) == datetime(2026, 10, 26, 9, 30) and D.fmt_td(125) == "2m 05s" and D.fmt_td(7500) == "2h 05m"
# parameters: percent <-> fraction, greyed-out logic, notices
cfg = P.to_config(vals0)
assert cfg.riskReward == 0.045 and cfg.buyback == 0.015 and cfg.price_perc == 0.10 and cfg.equity_frac == 1.0 and cfg.capacity_usd == 1e18 and cfg.short_mult == 0.5 and cfg.useDateFilter is False
assert P.notices(vals0) == [] and P.used_flags(vals0)["riskfactor"] is False and P.used_flags({**vals0, "strategyMode": "Always Dynamic"})["riskReward"] is False
assert any("ADX" in n for n in P.notices({**vals0, "useDynamic": True})) and any("warm-up" in n for n in P.notices({**vals0, "atrPeriod": 900})) and any("factor 4" in n for n in P.notices({**vals0, "factor": 4.0}))
rows = P.describe_parameters(vals0, bt0, P.LiveSettings(start_date="2026-10-20"), "Alpaca PAPER · DRY RUN")
kind = {r.name: r.kind for r in rows}
assert kind["ATR period"] == kind["Supertrend factor"] == "shared" and kind["ADX dynamic trail"] == "shared" and kind["How a short is held"] == "differs" and kind["Exit execution"] == "differs"
rows2 = {r.name: r for r in P.describe_parameters({**vals0, "useDynamic": True, "strategyMode": "Auto-Switch via Vol"}, bt0, P.LiveSettings(), "x")}
assert rows2["ADX dynamic trail"].kind == "shared" and rows2["Strategy mode"].kind == "shared" and rows2["ATR period"].kind == "shared" and "not computed" not in rows2["ADX dynamic trail"].live      # the Engine now computes the ADX and the volatility regime
assert all(rows2[n].kind == "shared" for n in ("ADX length", "ADX baseline", "Vol ATR length", "Vol baseline length", "High-vol multiple")) and "ADX(14)" in rows2["ADX dynamic trail"].live
rows3 = {r.name: r for r in P.describe_parameters({**vals0, "useDynamic": True, "strategyMode": "Auto-Switch via Vol", "atrPeriod": 900}, bt0, P.LiveSettings(), "x")}
assert rows3["ATR period"].kind == "differs" and rows3["ADX dynamic trail"].kind == "shared" and rows3["Strategy mode"].kind == "shared"      # the 900-bar warm-up limit on the ATR period is a different, still real, gap
rows4 = {r.name: r for r in P.describe_parameters({**vals0, "strategyMode": "Auto-Switch via Vol", "volSmaLength": 4000}, bt0, P.LiveSettings(), "x")}
assert rows4["Strategy mode"].kind == "differs" and "never sees" in rows4["Strategy mode"].note and any("high-vol regime" in n for n in P.notices({**vals0, "strategyMode": "Auto-Switch via Vol", "volSmaLength": 4000}))      # a regime longer than the live warm-up really is run as Always Dynamic
assert not any("Always Dynamic" in n for n in P.notices({**vals0, "strategyMode": "Auto-Switch via Vol"}))
# the monitor's pure signal == the live Engine's own signal() on the same bars
GU.CFG = P.to_config(vals0)
eng_probe = GU.Engine(__import__("queue").Queue(), True, 1000.0, 1.0, "2026-10-20", False)
sig_e, sig_d = eng_probe.signal(BARS.tail(900)), D.compute_signal(BARS.tail(900), GU.CFG)
assert all(sig_e[k] == sig_d[k] for k in ("bar", "close", "st", "want", "limit", "forming", "mon_high", "mon_low", "open", "high", "low")), (sig_e, sig_d)
assert sig_e["adx"] is None and sig_e["high_vol"] is None and sig_d["adx"] is None and sig_d["high_vol"] is None                 # reference settings: nothing extra is computed
GU.CFG = P.to_config({**vals0, "useDynamic": True, "strategyMode": "Auto-Switch via Vol"})
sig_e, sig_d = eng_probe.signal(BARS.tail(900)), D.compute_signal(BARS.tail(900), GU.CFG)
assert np.isfinite(sig_e["adx"]) and isinstance(sig_e["high_vol"], bool) and sig_e["adx"] == sig_d["adx"] and sig_e["high_vol"] == sig_d["high_vol"]
GU.CFG = P.to_config(vals0)
print("1. logic OK")

# ══ 2. the window ═══════════════════════════════════════════════════════════════════
reset_fake()
w = MainWindow()
assert Path(w.qs.fileName()) == Path(os.environ["SUPERTREND_SETTINGS_FILE"]) and w.qs.format() == QSettings.Format.IniFormat          # never the real registry
w.resize(1480, 940); w.show(); app.processEvents()
sp, bt, lv, mon = w.settings, w.backtest, w.live, w.monitor
assert [w.main_tabs.tabText(i) for i in range(4)] == ["Live trading", "Backtest", "Signal monitor", "How it works"]
assert bt._hi == BARS.index[-1] and "70" in sp.data_lbl.text() and "Local data" in sp.data_lbl.text()
assert lv.mode.currentIndex() == 0 and lv.account.currentIndex() == 0 and "DRY RUN" in lv.env_lbl.text() and "PAPER" in lv.env_lbl.text() and not lv.running              # always starts paper + dry run
shot(w, "01_live_initial")

# ── settings panel ──
assert sp.values() == vals0 and sp.preset_name() == P.REFERENCE_PRESET and sp.notice.isHidden()
sp.w["factor"].setValue(4.0)
assert sp.preset_name() == "Neighbour check · factor 4.0" and not sp.notice.isHidden() and "factor 4" in sp.notice.text()
sp.w["factor"].setValue(4.25)
assert sp.preset_name() == P.CUSTOM
sp.apply_preset("Half size · 50% of equity per trade")
assert sp.values()["equity_frac"] == 50.0 and sp.values()["factor"] == 5.0 and sp.notice.isHidden()
sp.w["strategyMode"].setCurrentText("Always Dynamic")
assert not sp.w["riskReward"].isEnabled() and not sp.w["price_perc"].isEnabled() and sp.w["riskfactor"].isEnabled() and not sp.w["volThreshold"].isEnabled() and not sp.notice.isHidden()
sp.w["strategyMode"].setCurrentText("Auto-Switch via Vol")
assert sp.w["volThreshold"].isEnabled() and sp.w["riskReward"].isEnabled() and "Auto-Switch" in sp.notice.text() and "Always Fixed" in sp.notice.text()
sp.w["useDynamic"].setChecked(True)
assert sp.w["adxLength"].isEnabled() and "ADX" in sp.notice.text()
sp.w["atrPeriod"].setValue(900)
assert "warm-up" in sp.notice.text()
sp.btn_reset.click()
assert sp.values() == vals0 and sp.preset_name() == P.REFERENCE_PRESET and sp.notice.isHidden() and not sp.w["adxLength"].isEnabled()
assert isinstance(sp.to_config(), S.Config) and sp.to_config().atrPeriod == 500 and isinstance(sp.to_config().atrPeriod, int)
# a stray mouse wheel must not change a setting (scrolling the panel used to change whatever field was under the cursor)
def _wheel(widget, dy=120):
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, dy), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    ev.setAccepted(True)
    widget.wheelEvent(ev)
    return ev.isAccepted()
before = sp.values()
for wdg in list(sp.w.values()) + [sp.preset, bt.cap, bt.tf, bt.symbol, bt.d0, lv.capital, lv.mode, lv.account, lv.dloss, lv.maxtr, lv.sdate]:
    if hasattr(wdg, "wheelEvent") and not isinstance(wdg, type(sp.w["useDynamic"])):
        assert not wdg.hasFocus() and _wheel(wdg) is False, wdg
assert sp.values() == before
assert w.splitter.widget(0) is w.side and all(isinstance(w.main_tabs.widget(k), QScrollArea) for k in range(3)) and isinstance(w.splitter, QSplitter)
assert all(t.textElideMode() == Qt.TextElideMode.ElideNone for t in w.findChildren(QTableWidget))
print("2. window + settings OK")

# ══ 3. backtest tab ═════════════════════════════════════════════════════════════════
w.main_tabs.setCurrentIndex(1)
bt.refresh_chk.setChecked(False)                                  # run on exactly what is on disk
assert bt.strict.isChecked() and bt.settings().strict is True     # live-accurate entries are the default ...
bt.strict.setChecked(False)                                       # ... but the equalities below are with the classic backtest (strategy file's own run)
assert bt.d0.date().toString("yyyy-MM-dd") == "2016-01-04" and bt.d1.date() == QDate(BARS.index[-1].year, BARS.index[-1].month, BARS.index[-1].day) and not bt._to_pinned
bt.run()
assert not bt.run_btn.isEnabled()                                 # runs in a worker thread: the window stays responsive
wait(lambda: bt.result is not None and bt.run_btn.isEnabled(), 90, "backtest")
res = bt.result
assert res.metrics["trades"] == m0["trades"] == bt.perf.tt.rowCount() and res.metrics["trades"] > 1000 and abs(res.metrics["final_equity"] - m0["final_equity"]) < 1e-3
pv = bt.perf
assert list(pv.kpis) == ["Total return", "Final equity", "CAGR", "Max drawdown", "Sharpe", "Win rate", "Profit factor", "Trades"] and pv.kpis["Trades"].v.text() == f"{m0['trades']:,}"
assert pv.kpis["Total return"].v.text().startswith("+") and "$" in pv.kpis["Final equity"].v.text() and pv.kpis["Max drawdown"].v.text().startswith("-")
assert pv.mt.rowCount() > 45 and pv.mo.rowCount() == 11 and pv.bd.rowCount() >= 5 and f"{m0['trades']:,}" in pv.lbl_n.text()          # was a hard-coded "1,579": the data cache grows, so the trade count does
assert f"{m0['trades']:,} trades" in bt.info.text() and "simulated in" in bt.info.text() and "Data through" in bt.fresh_lbl.text()
assert "Open position at the end of the window" in pv.open_lbl.text() and pv.open_state["primary"] == res.open_trade and "classic (look-ahead)" in bt.using_lbl.text()
shot(w, "02_backtest_equity")
for i, nm in ((1, "metrics"), (2, "trades"), (3, "monthly"), (4, "byside")):
    pv.tabs.setCurrentIndex(i); shot(w, f"03_backtest_{nm}")
pv.tabs.setCurrentIndex(0)
# trades table: sorted numerically, linked to the chart both ways
tt = pv.tt
tt.sortByColumn(9, Qt.SortOrder.DescendingOrder)                                                   # P&L $ descending
best = r0.trades.pnl_usd.max()
assert abs(float(tt.item(0, 9).text().replace(",", "").replace("+", "")) - best) < 0.01
tt.sortByColumn(0, Qt.SortOrder.AscendingOrder)
tt.selectRow(5); app.processEvents()
assert len(pv.chart._marks) >= 3                                                                   # the selected trade is shaded on the curve
pv._row_zoom(5); x0, x1 = pv.chart.ax.get_xlim()
assert pv.chart._pinned == 5 and (x1 - x0) < (pv.chart._home[0][1] - pv.chart._home[0][0]) / 5 and pv.chart._ann.get_visible() and "Trade #6" in pv.chart._ann.get_text()
pv.chart.reset_view(); pv.chart._pinned = None; pv.chart._show_trade(None)

# ── equity chart: hover / pin / zoom / pan / reset (same behaviours as the BTC trader's chart) ──
from matplotlib.backend_bases import MouseEvent                                                    # noqa: E402
ch = pv.chart
ch.draw()
home = ch.ax.get_xlim()
def _pt(i):
    return ch.ax.transData.transform((ch._tx[i], ch._ty[i]))
i = 400
px, py = _pt(i)
ev = MouseEvent("motion_notify_event", ch, px, py)
j = ch._nearest(ev)                                                  # MouseEvent rounds to whole pixels and ~1,600 dots sit < 1 px apart: the chart resolves the nearest dot (a neighbour at worst)
assert j is not None and abs(j - i) <= 3
ch._on_motion(ev)
assert ch._hover == j and ch._ann.get_visible() and f"Trade #{j + 1}" in ch._ann.get_text() and "Equity after" in ch._ann.get_text() and "Return" in ch._ann.get_text()
ch._on_motion(MouseEvent("motion_notify_event", ch, ch.ax.bbox.x0 + 3, ch.ax.bbox.y0 + 3))
assert not ch._ann.get_visible()
ch._on_scroll(MouseEvent("scroll_event", ch, px, py, button="up", step=1))
w1 = ch.ax.get_xlim()
assert (w1[1] - w1[0]) < 0.85 * (home[1] - home[0])
ch._on_scroll(MouseEvent("scroll_event", ch, px, py, button="down", step=-1))
assert (ch.ax.get_xlim()[1] - ch.ax.get_xlim()[0]) > (w1[1] - w1[0])
y_before = ch.ax.get_ylim()
ch._on_scroll(MouseEvent("scroll_event", ch, px, py, button="up", step=1, key="shift"))            # shift+wheel = value axis only
assert ch.ax.get_ylim() != y_before
ch.reset_view(); ch.draw()
px, py = _pt(i)
xb = ch.ax.get_xlim()
ch._on_press(MouseEvent("button_press_event", ch, px, py, button=1))                               # drag = pan (the view must be zoomed in for a pan to be possible)
ch._on_motion(MouseEvent("motion_notify_event", ch, px + 60, py + 20, button=1))
ch._on_release(MouseEvent("button_release_event", ch, px + 60, py + 20, button=1))
xa = ch.ax.get_xlim()
assert abs(xa[0] - xb[0]) > 1e-6 and abs((xa[1] - xa[0]) - (xb[1] - xb[0])) < 1e-6 and ch._pinned is None
ch._on_press(MouseEvent("button_press_event", ch, px, py, button=1, dblclick=True))                # double-click = reset
assert np.allclose(ch.ax.get_xlim(), home) and np.allclose(ch.ax.get_ylim(), ch._home[1])
ch.draw()
px, py = _pt(i)
ev = MouseEvent("button_release_event", ch, px, py, button=1)
j = ch._nearest(ev)
ch._on_press(MouseEvent("button_press_event", ch, px, py, button=1)); ch._on_release(ev)               # click a dot = pin + select its table row
assert ch._pinned == j and ch._ann.get_visible() and tt.currentRow() >= 0 and tt.item(tt.currentRow(), 0).data(Qt.ItemDataRole.UserRole) == j + 1
ex_, ey_ = ch.ax.bbox.x0 + 3, ch.ax.bbox.y0 + 3
ch._on_press(MouseEvent("button_press_event", ch, ex_, ey_, button=1)); ch._on_release(MouseEvent("button_release_event", ch, ex_, ey_, button=1))
assert ch._pinned is None and not ch._ann.get_visible()
got = []
ch.readout.connect(got.append)
ch._on_motion(MouseEvent("motion_notify_event", ch, ch.ax.bbox.x0 + 80, ch.ax.bbox.y0 + 120))
assert got and "equity" in got[-1] and "drawdown" in got[-1] and "buy & hold" in got[-1]
# toolbar: $ / %, log, overlays, zoom buttons
cp = pv.chart_panel
cp.scale.setCurrentIndex(1); app.processEvents(); ch.draw()
assert ch.pct_axis and ch.ax.yaxis.get_major_formatter()(ch._capital * 2) == "100.0%" and ch.ax.get_ylabel() == "Cumulative return"
shot(w, "04_backtest_pct_axis")
cp.scale.setCurrentIndex(0); cp.chk_log.setChecked(False); app.processEvents()
assert ch.ax.get_yscale() == "linear"
cp.chk_log.setChecked(True); cp.chk_bh.setChecked(False); cp.chk_dots.setChecked(False)
assert not ch._bh_art.get_visible() and all(not a.get_visible() for a in ch._dots_art)
cp.chk_bh.setChecked(True); cp.chk_dots.setChecked(True)
cp.btn_in.click(); assert ch.ax.get_xlim()[1] - ch.ax.get_xlim()[0] < home[1] - home[0]
cp.btn_reset.click(); assert np.allclose(ch.ax.get_xlim(), home)

# ── quick ranges, a pinned date, an empty window, a bad range ──
bt.quick_range(365)
assert bt._from_pinned and not bt._to_pinned and abs((bt._hi - pd.Timestamp(bt.d0.date().toString("yyyy-MM-dd")).to_pydatetime()).days - 365) <= 1
bt.run(); wait(lambda: bt.run_btn.isEnabled() and bt.result.settings.start == bt.d0.date().toString("yyyy-MM-dd"), 60, "1Y backtest")
r1y = bt.result
assert abs(r1y.eq.iloc[0] - bt.cap.value()) < 0.001 * bt.cap.value() and r1y.trades.entry_time.min() >= r1y.first_bar and r1y.warm_bars >= 500 and "warm-up" in bt.info.text()          # primed: no dead first month
bt.quick_range(-1)
assert bt.d0.date().toString("yyyy-MM-dd").endswith("-01-01")
bt.quick_range(0)
assert not bt._from_pinned and bt.d0.date().toString("yyyy-MM-dd") == "2016-01-04"
bt.d1.setDate(QDate(2020, 6, 30))
assert bt._to_pinned and "newer data is not included" in bt.fresh_lbl.text()
bt.btn_latest.click()
assert not bt._to_pinned and "newer data is not included" not in bt.fresh_lbl.text()
WARN.clear()
bt.d0.setDate(QDate(2026, 1, 1)); bt.d1.setDate(QDate(2025, 1, 1))
bt.run()
assert WARN and "after" in WARN[-1][1] and bt.run_btn.isEnabled()                                  # a bad range is refused with the reason
bt.quick_range(0); bt.follow_latest()
bt.d0.setDate(QDate(2031, 1, 1)); bt.d1.setDate(QDate(2031, 2, 1)); bt._to_pinned = True
bt.run(); wait(lambda: bt.run_btn.isEnabled(), 30)
assert WARN and "no SOXL bars" in WARN[-1][1] and "no SOXL bars" in bt.info.text()                  # the reason is shown, the tab stays usable
bt.quick_range(0); bt.follow_latest()
# a window with no trades: the empty path must not crash
bt.d0.setDate(QDate(2026, 10, 12)); bt.d1.setDate(QDate(2026, 10, 14)); bt._to_pinned = True
bt.run(); wait(lambda: bt.run_btn.isEnabled() and bt.result.settings.start == "2026-10-12", 30)
shot(w, "05_backtest_short_window")
bt.quick_range(0); bt.follow_latest()
# a what-if on another symbol needs a download: with no network it says so instead of crashing
bt.symbol.setCurrentText("SOXL")
# robustness check: 15 runs in a worker, verdict + coloured grid
bt.run_robustness()
assert not bt.rob_btn.isEnabled()
wait(lambda: bt.robust is not None, 180, "robustness")
rb = bt.robust
assert rb.sharpe.shape == (3, 5) and np.isfinite(rb.sharpe).all() and rb.verdict.split(" ")[0] == "Sharpe" and any(k in rb.verdict for k in ("PLATEAU", "FRAGILE", "SPIKE"))
assert bt.rob_table.rowCount() == 3 and bt.rob_table.columnCount() == 6 and "Sharpe" in bt.rob_table.item(1, 3).text() and bt.rob_table.item(1, 3).font().bold()
assert abs(rb.sharpe[1, 2] - m0["sharpe"]) < 1e-9                                                  # the centre cell is the very run the Backtest shows
bt.perf.tabs.setCurrentIndex(bt.perf.tabs.count() - 2); shot(w, "06_backtest_robustness"); bt.perf.tabs.setCurrentIndex(0)
bt.run(); wait(lambda: bt.run_btn.isEnabled() and bt.result is not None, 60)

# ── live-accurate (strict) entries are the default; the open position; the Verify button ──
bt.quick_range(0); bt.follow_latest()
bt.strict.setChecked(True)
assert "live-accurate" in bt.using_lbl.text()
bt.run(); wait(lambda: bt.run_btn.isEnabled() and bt.result is not None and bt.result.settings.strict, 90, "strict backtest")
rs = bt.result
assert rs.metrics["trades"] == len(r0s.trades) == bt.perf.tt.rowCount() != m0["trades"] and rs.open_trade == r0s.open_trade
txt = bt.perf.open_lbl.text()
if rs.open_trade is None:
    assert bt.perf.open_state["primary"] is None and "FLAT" in txt
else:
    ot_ = rs.open_trade
    assert bt.perf.open_state["primary"] == ot_ and ot_["side"].upper() in txt and f"{ot_['qty']:,} SOXL" in txt and f"{ot_['tp']:,.2f}" in txt and f"{ot_['price']:,.2f}" in txt
shot(w, "03b_backtest_strict_open_position")
# a window that ends while a position is open shows it with its frozen levels (long and short), and a flat one says so
for side_, k_ in (("long", CUT["long"]), ("short", CUT["short"]), ("flat", CUT["flat"])):
    rk = B.run_backtest(BARS.iloc[:k_], vals0, P.BacktestSettings(symbol="SOXL", tf=15, start="2023-01-02", end="2031-01-01", capital=100_000.0, strict=True, benchmark=False))
    bt.show_result(rk)
    t_ = bt.perf.open_lbl.text()
    if side_ == "flat":
        assert rk.open_trade is None and "FLAT" in t_ and bt.perf.open_state["primary"] is None
    else:
        o2 = rk.open_trade
        assert o2["side"] == side_ and ("▼ SHORT" if side_ == "short" else "▲ LONG") in t_ and f"{o2['entry_ref']:,.2f}" in t_ and f"{o2['tp']:,.2f}" in t_ and ("trailing stop" in t_)
        assert f"{o2['pnl_pct']:+.2f}%" in t_ and f"{rk.last_bar:%Y-%m-%d %H:%M}" in t_
shot(w, "03c_backtest_open_flat")
import supertrend_ui.backtest_tab as BTT                                                         # noqa: E402
BOXES = []
QMessageBox.exec = lambda self: (BOXES.append((self.text(), self.detailedText())), 0)[1]          # the result dialog, answered headlessly
SEEN = {}


def _fake_replay(values, start, end, shortable, lookahead=False, log=None, progress=None, cancelled=None):
    SEEN.update(values=values, start=start, end=end, shortable=shortable, lookahead=lookahead)
    progress(5, 10)
    if SEEN.get("fail"):
        raise RuntimeError("the replay process ended without a result (exit code 1)")
    return {"stats": dict(bt_trades=12, live_trades=12 if not lookahead else 11, full_matches=12 if not lookahead else 9), "identical": not lookahead, "text": ("IDENTICAL" if not lookahead else "DIFFERENT") + ": details",
            "start": start, "end": end, "shortable": shortable, "strict": not lookahead}


BTT.run_replay_process = _fake_replay
bt.quick_range(0); bt.follow_latest()
bt.v_months.setValue(2); bt.v_route.setCurrentIndex(0)
bt.verify(); assert not bt.verify_btn.isEnabled() and not bt.run_btn.isEnabled()                    # one job at a time
wait(lambda: bt.verify_btn.isEnabled() and BOXES, 20, "verify")
last_day = pd.Timestamp(BARS.index[-1].date())
assert SEEN["lookahead"] is False and SEEN["shortable"] is True and SEEN["end"] == f"{last_day:%Y-%m-%d}" and SEEN["start"] == f"{last_day - pd.DateOffset(months=2):%Y-%m-%d}" and SEEN["values"] == sp.values()
assert "IDENTICAL" in BOXES[-1][0] and "live-accurate" in BOXES[-1][0] and BOXES[-1][1] == "IDENTICAL: details" and "IDENTICAL" in bt.status.text() and bt.last_verify["identical"]
bt.strict.setChecked(False); bt.v_route.setCurrentIndex(1); BOXES.clear()                          # the classic backtest cannot be matched by a live engine, and the SOXS route is requested
bt.verify(); wait(lambda: bt.verify_btn.isEnabled() and BOXES, 20, "verify classic")
assert SEEN["lookahead"] is True and SEEN["shortable"] is False and "DIFFERENT" in BOXES[-1][0] and "classic" in BOXES[-1][0] and "DIFFERENT" in bt.status.text()
SEEN["fail"] = True; WARN.clear()
bt.verify(); wait(lambda: bt.verify_btn.isEnabled() and WARN, 20, "verify failure")
assert "ended without a result" in WARN[-1][1] and bt.run_btn.isEnabled()                         # a failed child process is reported, the tab stays usable
SEEN["fail"] = False; WARN.clear()
bt.symbol.setCurrentText("TQQQ"); bt.verify()
assert WARN and "SOXL" in WARN[-1][1] and bt.verify_btn.isEnabled()                               # only the instrument the live engine trades
bt.symbol.setCurrentText("SOXL"); bt.strict.setChecked(True)
print("3. backtest tab OK")

# ══ 4. parameters in use ════════════════════════════════════════════════════════════
w.main_tabs.setCurrentIndex(0)
for view in (lv.params_view, bt.params_view):
    view.refresh()
    items = {view.table.item(i, 0).text(): (view.table.item(i, 1).text(), view.table.item(i, 2).text()) for i in range(view.table.rowCount()) if view.table.item(i, 0)}
    assert len(items) > 30 and items["ATR period"] == ("500 bars", "500 bars") and "SOXL" in items["Signal instrument"][0] and items["Max trades per day"][1] == "12"
    assert items["Account"][1].startswith("Alpaca PAPER") and "DRY RUN" in items["Orders"][1]
assert "Yellow rows differ" in lv.params_view.head.text() and "none right now" not in lv.params_view.head.text()          # the reference settings already differ in a few engine behaviours (short route, exits, costs)
sp.w["useDynamic"].setChecked(True); sp.w["strategyMode"].setCurrentText("Auto-Switch via Vol"); lv.params_view.refresh()
adx = [lv.params_view.table.item(i, 2).text() for i in range(lv.params_view.table.rowCount()) if lv.params_view.table.item(i, 0) and lv.params_view.table.item(i, 0).text() == "ADX dynamic trail"][0]
assert "ADX(14)" in adx and "not computed" not in adx                                              # the Engine computes the ADX nowlv.perf.tabs.setCurrentIndex(lv.perf.tabs.count() - 1); shot(w, "07_live_params")
sp.btn_reset.click(); lv.perf.tabs.setCurrentIndex(0)
print("4. parameters in use OK")

# ══ 5. connect + history ════════════════════════════════════════════════════════════
reset_fake()
LIVE_KEYS["ok"] = False
lv.account.setCurrentIndex(1)                                                                      # LIVE account without keys: a clear message, nothing else
assert "LIVE" in lv.env_lbl.text()
lv.connect_account(); wait(lambda: lv.btn_connect.isEnabled(), 10)
assert "No usable credentials" in lv.health.text() and not lv.btn_test.isEnabled() and lv.acct.text() == "Connection failed"
LIVE_KEYS["ok"] = True
lv.account.setCurrentIndex(0)
lv.connect_account()
wait(lambda: lv._acct is not None, 15, "connect")
assert "PA38" in lv.acct.text() and "$100,000.00" in lv.bp.text() and lv.btn_test.isEnabled() and "SOXL: shortable" in lv.health.text() and "0 open position" in lv.health.text()
assert lv.capital.value() == 100_000.0 and "Default = the account's equity" in lv.cap_note.text()
assert lv.kpis["Day P&L"].v.text() == "+$500.00" and "+0.50%" in lv.kpis["Day P&L"].s.text()
lv.capital.setValue(25_000.0)
assert lv._capital_pinned and "by hand" in lv.cap_note.text()
lv.btn_cap.click()
assert not lv._capital_pinned and lv.capital.value() == 100_000.0
wait(lambda: not lv._history_busy and "since" in lv.hist_lbl.text().lower() and "SIMULATED" in lv.hist_lbl.text(), 30, "history", lambda: lv.hist_lbl.text())           # the account has no fills: the strategy is simulated since the start date
assert lv.perf.kpis["Final equity"].v.text().startswith("$")
lv.account.setCurrentIndex(1)                                                                      # a connection belongs to one account: switching drops it
assert lv._acct is None and lv.acct.text() == "Not connected" and not lv.btn_test.isEnabled()
lv.account.setCurrentIndex(0); lv.connect_account(); wait(lambda: lv._acct is not None, 15)
assert "no session running" in lv.perf.open_lbl.text() and "FLAT" in lv.perf.open_lbl.text()                 # idle: the strip still says what the account holds (nothing)
FakeTrading.positions = [fake_position("SOXS", 400, 47.0, 48.0)]
lv._acct = None; lv.connect_account(); wait(lambda: lv._acct is not None and "SOXS" in lv.perf.open_lbl.text(), 15, "idle position", lambda: lv.perf.open_lbl.text())
assert "SHORT" in lv.perf.open_lbl.text() and "400 SOXS" in lv.perf.open_lbl.text() and "did not open" in lv.perf.open_lbl.text() and "+2.13%" in lv.perf.open_lbl.text()     # long SOXS = a SHORT; no frozen levels for a position this app did not open
FakeTrading.positions = []
# order-path test: one tiny limit order that cannot fill, cancelled at once
FakeTrading.calls.clear()
lv.order_path_test()
wait(lambda: "order-path test" in lv.logbox.toPlainText(), 15, "order-path test")
names = [c[0] for c in FakeTrading.calls]
assert names == ["submit_order", "cancel_order"] and FakeTrading.calls[0][2].endswith("BUY") and FakeTrading.calls[0][3] == 1 and FakeTrading.calls[0][4] < LAST_CLOSE * 0.6 and "keys, signing and order format are fine" in lv.logbox.toPlainText()
FakeTrading.submitted.clear(); FakeTrading.calls.clear()
print("5. connect OK")

# ══ 6. pre-flight ═══════════════════════════════════════════════════════════════════
FakeTrading.is_open = False                                                                        # evening: the freshness check expects the cached 15:45 bar
vals = sp.values()
ls = lv.live_settings()
rep = PF.run_preflight(vals, ls)
chk = {c.key: c for c in rep.checks}
assert rep.verdict == "GO" and chk["creds"].status == chk["endpoint"].status == chk["status"].status == chk["warmup"].status == chk["kill"].status == "PASS", [(c.key, c.status, c.detail) for c in rep.checks if c.status != "PASS"]
assert chk["signal"].status == "INFO" and chk["recent"].status in ("INFO", "WARN") and chk["valid"].status == "INFO" and "SOXL" in chk["asset_SOXL"].detail
ls_live = P.LiveSettings(capital=100_000.0, start_date=ls.start_date, paper=False, armed=True)
rep_live = PF.run_preflight(vals, ls_live)
assert rep_live.verdict == "CAUTION" and {c.key: c for c in rep_live.checks}["valid"].status == "WARN" and rep_live.suggested_frac == 50.0 and {c.key: c for c in rep_live.checks}["size_risk"].status == "WARN"
assert "LIVE (real money)" in rep_live.headline() and "GO" in PF.run_preflight({**vals, "equity_frac": 40.0}, P.LiveSettings(capital=100_000.0, start_date=ls.start_date, paper=True, armed=True)).headline()
# failure modes
LIVE_KEYS["ok"] = False
rep_nok = PF.run_preflight(vals, P.LiveSettings(paper=False, armed=True, start_date=ls.start_date))
assert rep_nok.verdict == "NO-GO" and {c.key: c for c in rep_nok.checks}["creds"].status == "FAIL" and "alpaca-live" in {c.key: c for c in rep_nok.checks}["creds"].detail
LIVE_KEYS["ok"] = True
FakeTrading.blocked = True
assert {c.key: c for c in PF.run_preflight(vals, ls).checks}["status"].status == "FAIL"
FakeTrading.blocked = False
FakeTrading.equity = 20_000.0
assert {c.key: c for c in PF.run_preflight(vals, ls).checks}["capital"].status == "WARN" and {c.key: c for c in PF.run_preflight(vals, ls_live).checks}["capital"].status == "FAIL"
FakeTrading.equity = 100_000.0
FakeTrading.positions = [fake_position("SOXS")]
c = {c.key: c for c in PF.run_preflight(vals, ls).checks}["positions"]
assert c.status == "WARN" and "NOT be managed" in c.detail                                         # held, but this app has no saved exit levels for it
with open(GU.STATE_FILE, "w", encoding="utf-8") as fh:
    fh.write('{"pos": -1, "side": "short", "symbol": "SOXS", "qty": 400, "entry_ref": 120.0, "supert_entry": 130.0, "target": 114.6, "active_fixed": true}')
c = {c.key: c for c in PF.run_preflight(vals, ls).checks}["positions"]
assert c.status == "INFO" and "SOXS exits are polled" in c.detail
FakeTrading.positions = []
open(GU.STATE_FILE, "w").write("{}")
assert {c.key: c for c in PF.run_preflight({**vals, "atrPeriod": 5000}, ls).checks}["warmup"].status == "FAIL"
open(GU.KILL_FILE, "w").write("x")
assert PF.run_preflight(vals, ls).verdict == "NO-GO" and {c.key: c for c in PF.run_preflight(vals, ls).checks}["kill"].status == "FAIL"
os.remove(GU.KILL_FILE)
PF.record_session_end("KILLED", "paper")
assert {c.key: c for c in PF.run_preflight(vals, ls).checks}["prev"].status == "WARN"
PF.record_session_end("stopped by user", "paper")
assert {c.key: c for c in PF.run_preflight(vals, ls).checks}["prev"].status == "PASS"
assert {c.key: c for c in PF.run_preflight(vals, P.LiveSettings(start_date="2024-01-01")).checks}["start"].status == "WARN"
saved = PF.save_report(rep)
assert saved and os.path.exists(saved) and "pre-flight" in open(saved, encoding="utf-8").read().lower()
# dialog logic for every mode / verdict
d = PreflightDialog(lv, live=False, armed=False, then_start=True, frac=100.0); d.set_report(rep_nok)
assert d.btn_go.isEnabled() and "dry run" in d.btn_go.text().lower() and d._go_action == "start" and d.btn_dry.isHidden()                  # a dry run is never blocked, even on NO-GO
d = PreflightDialog(lv, live=True, armed=True, then_start=True, frac=100.0); d.set_report(rep_nok)
assert not d.btn_go.isEnabled() and "blocked" in d.btn_go.text() and not d.btn_dry.isHidden()
d = PreflightDialog(lv, live=True, armed=True, then_start=True, frac=100.0); d.set_report(rep_live)
assert d.btn_go.isEnabled() and "Continue to LIVE with my settings (100% of equity per trade)" in d.btn_go.text() and not d.btn_frac.isHidden() and "50%" in d.btn_frac.text() and not d.btn_dry.isHidden()
d.resize(1040, 700); d.show(); shot(d, "08_preflight_caution")
d = PreflightDialog(lv, live=True, armed=True, then_start=True, frac=40.0); d.set_report(rep_live)
assert d.btn_frac.isHidden()                                                                       # already below the suggestion: nothing to offer
d = PreflightDialog(lv, live=False, armed=True, then_start=True, frac=100.0); d.set_report(rep)
assert d.btn_go.isEnabled() and "PAPER session" in d.btn_go.text()
d = PreflightDialog(lv, live=False, armed=False, then_start=False, frac=100.0); d.set_report(rep)
assert d.btn_go.isHidden() and d.btn_copy.isEnabled()
d.resize(1040, 700); d.show(); shot(d, "09_preflight_go")
# the real button, real worker thread, fake broker
lv._pf_report = None
lv.run_preflight(then_start=False)
wait(lambda: lv._pf_report is not None, 60, "pre-flight")
assert lv.pf_lbl.text().startswith("Pre-flight ") and "GO" in lv.pf_lbl.text() and lv._pf_dlg.table.rowCount() >= 15 and lv.btn_start.isEnabled() and "pre-flight:" in lv.logbox.toPlainText()
lv._pf_dlg.close()
assert "Step 3" in lv.hint.text()
print("6. pre-flight OK")

# ══ 7. a DRY-RUN session against the fake broker (the real Engine) ═══════════════════
FakeTrading.is_open = True
lv.capital.setValue(100_000.0)
lv.sdate.setDate(QDate.currentDate())
real_exec = PreflightDialog.exec
def _fake_exec(self):                                                                              # headless stand-in for the modal dialog: wait for the report, choose "start"
    t0 = time.time()
    while self.report is None and time.time() - t0 < 60:
        app.processEvents(); time.sleep(0.02)
    self.action = "start"
    return 1
PreflightDialog.exec = _fake_exec
lv._pf_report = None
lv.start()                                                                                         # = pre-flight, then (action "start") the dry-run session
assert lv.running and lv.engine is not None and lv._pf_report is not None
assert GU.CFG.atrPeriod == 500 and GU.SHORT_EQUITY_MULT == 0.5 and GU.MAX_TRADES_PER_DAY == 12 and GU.DAILY_LOSS_LIMIT_PCT == -8.0 and GU.PREFER_SHORT_SOXL is True      # the tuned settings reach the engine
w._status()
assert not lv.btn_start.isEnabled() and lv.btn_stop.isEnabled() and not sp.w["factor"].isEnabled() and not lv.capital.isEnabled() and not lv.mode.isEnabled() and "SESSION RUNNING" in w.state_lbl.text()
wait(lambda: lv._sig is not None and lv.orders.rowCount() >= 1, 60, "first poll")
assert "Next order:" in lv.plan_lbl.text() and " / 12" in lv.kpis["Trades today"].v.text()
wait(lambda: lv._openpos is not None, 30, "open-position event")
op = lv._openpos
assert op["live"] is None and lv.perf.open_state["primary"] is None and "Account holds" in lv.perf.open_lbl.text() and ("FLAT" in lv.perf.open_lbl.text())      # a dry run sends nothing: the account is flat
assert (op["expected"] is None and op["cmp"]["status"] == "match") or (op["expected"] is not None and lv.perf.open_state["cmp"]["status"] == "dry" and "DRY RUN" in lv.perf.open_lbl.text() and "Backtest holds" in lv.perf.open_lbl.text())
shot(w, "09b_live_open_position_dry")
assert lv.kpis["Trend (last closed bar)"].v.text().startswith(("UP", "DOWN")) and "Supertrend" in lv.sig_lbl.text() and lv.kpis["Position"].v.text() == "FLAT"
assert lv.orders.item(0, 7).text().startswith("DRY RUN") and lv.orders.item(0, 2).text() == "SOXL" and not FakeTrading.submitted and not lv._popups     # a dry run lists the order, sends none, pops nothing up
n_rows = lv.orders.rowCount()
t0 = time.time()
while time.time() - t0 < 3.5:                                                                      # several more polls: the same proposal must not be listed again
    app.processEvents(); time.sleep(0.05)
assert lv.orders.rowCount() == n_rows and "×" in lv.logbox.toPlainText()                           # repeated log lines are folded into one with a count
assert lv.setup.isHidden() and not lv.run_card.isHidden() and lv.btn_pf.isHidden() and "This session" in lv.run_title.text() and "RUNNING SESSION" in lv.using_lbl.text()      # running: a compact summary instead of the locked setup boxes
lv.btn_setup.setChecked(True); assert not lv.setup.isHidden() and not lv.capital.isEnabled()                                     # ... which can be shown (read-only) on request
lv.btn_setup.setChecked(False); assert lv.setup.isHidden()
sa_live = w.main_tabs.widget(0)                                                                    # long KPI text must not widen the page: at the normal window size nothing scrolls sideways
assert sa_live.horizontalScrollBar().maximum() == 0, sa_live.horizontalScrollBar().maximum()
assert lv.perf.kpis["CAGR"].v.text() == "—" and "month" in lv.perf.kpis["CAGR"].s.text()               # a day of results is not annualised
lv.perf.tabs.setCurrentIndex(lv.perf.tabs.count() - 3)                                              # Orders
shot(w, "10_live_dry_run_orders")
lv.perf.tabs.setCurrentIndex(lv.perf.tabs.count() - 2); shot(w, "11_live_dry_run_log"); lv.perf.tabs.setCurrentIndex(0)
assert lv.perf.kpis["Final equity"].v.text().startswith("$") and "SIMULATED" in lv.hist_lbl.text()
shot(w, "12_live_dry_run")
full = lv.logbox.toPlainText()
lv.log_filter.setCurrentIndex(1)                                                                   # warnings and errors only
assert 0 < len(lv.logbox.toPlainText()) < len(full) and "session started" not in lv.logbox.toPlainText()
lv.log_filter.setCurrentIndex(2)
assert "DRY RUN" in lv.logbox.toPlainText()
lv.log_filter.setCurrentIndex(0)
assert lv.logbox.toPlainText() == full
# stop while holding SOXS: the warning says the position becomes unmanaged
lv.on_event({"kind": "positions", "positions": {"SOXS": dict(qty=400.0, side="long", avg=47.0, mv=19200.0, upl=400.0, uplpc=2.1, cur=48.0)}, "baseline": 1e5, "start": "x"})
assert lv.kpis["Position"].v.text() == "SHORT 400 SOXS" and "short SOXL" in lv.kpis["Position"].s.text()
ASK["log"].clear(); ASK["answer"] = QMessageBox.StandardButton.No
lv.stop()
assert lv.running and "UNMANAGED" in ASK["log"][-1][1]                                             # declined: still running
ASK["answer"] = QMessageBox.StandardButton.Yes
lv.stop()
wait(lambda: not lv.running, 20, "stop")
assert lv.btn_start.isEnabled() and not lv.btn_stop.isEnabled() and sp.w["factor"].isEnabled() and lv.capital.isEnabled() and "stopped by user" in lv.state.text()
assert not lv.setup.isHidden() and lv.run_card.isHidden() and not lv.btn_pf.isHidden()                   # idle again: the setup boxes are back
assert PF.load_state()["last_session"]["why"] == "stopped by user"
print("7. dry-run session OK")

# ══ 8. an ARMED paper session: the engine really sends (to the fake) ═════════════════
reset_fake()
lv.mode.setCurrentIndex(1)
assert "PAPER, ARMED" in lv.env_lbl.text()
ASK["log"].clear(); ASK["answer"] = QMessageBox.StandardButton.No
lv._pf_report = None
lv.chk_pf.setChecked(False)
lv.start()                                                                                         # armed + declined -> nothing starts
assert not lv.running and not FakeTrading.submitted and ASK["log"] and "PAPER" in ASK["log"][-1][1]
ASK["answer"] = QMessageBox.StandardButton.Yes
lv.start()
assert lv.running
wait(lambda: FakeTrading.submitted and lv._popups, 40, "armed order")
o = FakeTrading.submitted[0]
assert o.symbol == "SOXL" and o.limit_price is not None and lv.orders.item(0, 7).text().startswith("SENT") and lv.orders.item(0, 1).text() == "PAPER"
assert len(FakeTrading.submitted) == 1 and lv._popups                                              # a real (paper) order: one window, listed once
time.sleep(2.5); app.processEvents()
assert len(FakeTrading.submitted) == 1, "the engine must not re-send the limit order that is already resting"
for b in lv._popups:
    b.close()
lv._popups.clear()
shot(w, "13_live_armed_paper")
# FLATTEN while running: the engine is stopped FIRST (it would re-enter on its next poll), then orders are cancelled and positions closed
FakeTrading.positions = [fake_position("SOXL", 300, 100.0, 101.0)]
FakeTrading.calls.clear()
ASK["log"].clear()
lv.flatten()
wait(lambda: not lv.running and lv.btn_flat.isEnabled(), 40, "flatten")
assert [c[0] for c in FakeTrading.calls][:2] == ["cancel_orders", "close_all_positions"] and "stopped first" in ASK["log"][-1][1]
assert "flattened by user" in lv.state.text() and not FakeTrading.positions and not os.path.exists(GU.KILL_FILE)
assert json_load(GU.STATE_FILE) == {}
# KILL: same + latch
lv.mode.setCurrentIndex(0)
lv.chk_pf.setChecked(False)
lv.start(); wait(lambda: lv.running, 5)
FakeTrading.positions = [fake_position("SOXS")]
FakeTrading.calls.clear()
lv.kill()
wait(lambda: not lv.running and lv.btn_kill.isEnabled(), 40, "kill")
assert os.path.exists(GU.KILL_FILE) and "KILLED" in lv.state.text() and ("close_all_positions", True) in FakeTrading.calls and not FakeTrading.positions
lv._clock_tick()
assert not lv.btn_unlatch.isHidden() and "KILL latch" in lv.hint.text()
WARN.clear()
lv.start()                                                                                         # with the latch set nothing can start
assert not lv.running
assert {c.key: c for c in PF.run_preflight(sp.values(), lv.live_settings()).checks}["kill"].status == "FAIL"
lv.clear_latch()
assert not os.path.exists(GU.KILL_FILE) and lv.btn_unlatch.isHidden()
assert {c.key: c for c in PF.run_preflight(sp.values(), lv.live_settings()).checks}["prev"].status == "WARN"                 # the next pre-flight remembers the KILL
PF.record_session_end("stopped by user", "paper")
print("8. armed paper / flatten / kill OK")

# ══ 9. LIVE account: the typed confirmation ═════════════════════════════════════════
reset_fake()
lv.account.setCurrentIndex(1); lv.mode.setCurrentIndex(1)
assert "REAL MONEY" in lv.env_lbl.text() and "LIVE, ARMED" in lv.env_lbl.text()
INPUT.update(text="yes", ok=True)
lv.start()
assert not lv.running and not FakeTrading.submitted and "TRADE REAL MONEY" in ASK["log"][-1][1]  # wrong phrase: nothing starts
INPUT.update(text="TRADE REAL MONEY", ok=False)
lv.start()
assert not lv.running
INPUT.update(text="TRADE REAL MONEY", ok=True)
lv.start()
w._status()
assert lv.running and "LIVE" in w.state_lbl.text() and "orders SENT" in w.state_lbl.text()
wait(lambda: FakeTrading.submitted, 40, "live order (fake)")
assert lv.orders.item(0, 1).text() == "LIVE" and lv._popups
for b in lv._popups:
    b.close()
lv._popups.clear()
lv._request_stop("stopped by user"); wait(lambda: not lv.running, 20)
lv.account.setCurrentIndex(0); lv.mode.setCurrentIndex(0)
lv.chk_pf.setChecked(True)
print("9. live confirmation OK")

# ══ 10. signal monitor ══════════════════════════════════════════════════════════════
reset_fake()
w.main_tabs.setCurrentIndex(2)
mon.refresh()
wait(lambda: mon.bars is not None, 20, "monitor bars")
assert len(mon.bars) == GU.WARMUP_BARS and mon.summary.text().startswith(("UP", "DOWN")) and mon.table.rowCount() >= 5 and mon.chart._bars is not None and len(mon.chart._bars) == 130
assert mon.table.item(0, 0).text() == "Last closed bar" and "Re-entry limit" in [mon.table.item(i, 0).text() for i in range(mon.table.rowCount())]
shot(w, "14_monitor")
with open(GU.STATE_FILE, "w", encoding="utf-8") as fh:                                              # an open position: its frozen levels appear on the chart and in the table
    fh.write('{"pos": 1, "side": "long", "symbol": "SOXL", "qty": 100, "entry_ref": %.2f, "supert_entry": %.2f, "target": %.2f, "active_fixed": true, "max_high": %.2f, "min_low": null}'
             % (LAST_CLOSE, LAST_CLOSE * 0.9, LAST_CLOSE * 1.045, LAST_CLOSE * 1.06))
mon.redraw()
labels = [mon.table.item(i, 0).text() for i in range(mon.table.rowCount())]
assert "Open position" in labels and "Trailing stop" in labels
assert any(t.get_text().startswith("take-profit") for t in mon.chart.ax.texts) and any(t.get_text().startswith("trailing stop") for t in mon.chart.ax.texts)
lv_ = D.position_levels(json_load(GU.STATE_FILE), P.to_config(vals0))
assert abs(lv_["tp"] - LAST_CLOSE * 1.10) < 1e-6 and lv_["armed"] and abs(lv_["trail"] - LAST_CLOSE) < 1e-6 and lv_["side"] == "long"        # trail factor 0 = breakeven
shot(w, "15_monitor_position")
open(GU.STATE_FILE, "w").write("{}")
mon.show_n.setCurrentIndex(0); app.processEvents()
assert len(mon.chart._bars) == 26
w.main_tabs.setCurrentIndex(3); shot(w, "16_help")
w.main_tabs.setCurrentIndex(0)
print("10. monitor OK")

# ══ 11. menus, shortcuts, persistence, layout, close ═════════════════════════════════
acts = {a.text(): a for a in w.actions()}
assert acts["Run backtest"].shortcut().toString() == "Ctrl+R" and acts["Pre-flight check"].shortcut().toString() == "Ctrl+L" and acts["How it works"].shortcut().toString() == "F1"
acts["Backtest"].trigger(); assert w.main_tabs.currentIndex() == 1
acts["How it works"].trigger(); assert w.main_tabs.currentIndex() == 3
acts["Live trading"].trigger(); assert w.main_tabs.currentIndex() == 0
sp.apply_preset("Cautious · 25% of equity per trade"); sp.w["factor"].setValue(4.5)
bt.cap.setValue(12_345); lv.capital.setValue(777.0); lv.dloss.setValue(5.0); lv.maxtr.setValue(7); lv.short_route.setCurrentIndex(1)
lv.mode.setCurrentIndex(1); lv.account.setCurrentIndex(1)                                          # armed + live must NOT be remembered
w.qs.setValue("window/geometry", w.saveGeometry())
w.close()
w2 = MainWindow()
w2.show(); app.processEvents()
assert w2.settings.values()["equity_frac"] == 25.0 and w2.settings.values()["factor"] == 4.5 and w2.settings.preset_name() == P.CUSTOM
assert w2.backtest.cap.value() == 12_345 and w2.live.capital.value() == 777.0 and w2.live._capital_pinned and w2.live.dloss.value() == 5.0 and w2.live.maxtr.value() == 7 and w2.live.short_route.currentIndex() == 1
assert w2.live.mode.currentIndex() == 0 and w2.live.account.currentIndex() == 0                    # a new launch is always paper + dry run
w2.move(-6000, 100); w2.qs.setValue("window/geometry", w2.saveGeometry()); w2.close()
w3 = MainWindow(); w3.show(); app.processEvents()
scr = QApplication.primaryScreen().availableGeometry()
assert scr.intersected(w3.frameGeometry()).width() > 200                                          # a window saved off-screen is brought back
# close with a running session asks first
w3.live.chk_pf.setChecked(False)
w3.live.start(); wait(lambda: w3.live.running, 5)
ASK["answer"] = QMessageBox.StandardButton.No
w3.close(); assert w3.live.running and w3.isVisible()
ASK["answer"] = QMessageBox.StandardButton.Yes
w3.close(); wait(lambda: not w3.live.running, 15)
assert not w3.isVisible()
print("11. menus, persistence, close OK")

# ══ 12. file dialogs, data refresh, the remaining buttons ═══════════════════════════
from PyQt6.QtWidgets import QFileDialog                                                           # noqa: E402
reset_fake()
w4 = MainWindow(); w4.resize(1480, 940); w4.show(); app.processEvents()
# settings file round trip, and a damaged file is refused rather than half-applied
spath = str(Path(tmp) / "my_settings.json")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (spath, ""))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (spath, ""))
w4.settings.apply_preset("Stress costs · 10 bp/leg, stop-first")
w4.save_settings_file()
saved_vals = json_load(spath)["values"]
assert saved_vals["commission_bps"] == 10.0 and saved_vals["pessimistic"] is True
w4.settings.btn_reset.click()
assert w4.settings.values() == vals0
w4.load_settings_file()
assert w4.settings.preset_name() == "Stress costs · 10 bp/leg, stop-first" and w4.settings.values()["commission_bps"] == 10.0
open(spath, "w").write("this is not json")
WARN.clear(); w4.load_settings_file()
assert WARN and "Could not read" in WARN[-1][1] and w4.settings.values()["commission_bps"] == 10.0
w4.settings.btn_reset.click()
# export the trades to CSV, and the chart to an image
w4.backtest.refresh_chk.setChecked(False)
w4.backtest.strict.setChecked(False)                                                               # the classic backtest, to compare with m0
w4.backtest.run(); wait(lambda: w4.backtest.result is not None and w4.backtest.run_btn.isEnabled(), 60, "backtest")
t0_ = time.time(); w4.backtest.show_result(w4.backtest.result); render_s = time.time() - t0_          # drawing a 1,579-trade result must not freeze the window
assert render_s < 6.0, render_s
csv_path = str(Path(tmp) / "trades_export.csv")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (csv_path, ""))
w4.main_tabs.setCurrentIndex(1); w4.export_trades()
exp = pd.read_csv(csv_path)
assert len(exp) == m0["trades"] and {"entry_time", "exit_time", "side", "return_pct", "pnl_usd", "exit_reason", "equity_after"} <= set(exp.columns) and "entry_alert" not in exp.columns
w4.main_tabs.setCurrentIndex(0)
WARN.clear(); w4.export_trades()
assert WARN and "no trades" in WARN[-1][1].lower()                                                 # the Live tab has no trades yet: it says so instead of writing an empty file
img = str(Path(tmp) / "equity.png")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (img, ""))
w4.backtest.perf.chart.save_image()
assert os.path.getsize(img) > 5000
# data refresh: the top-up runs in a worker (D.load_bars is faked), the window adopts the new bars; a failure leaves the cached bars in use
newer = pd.concat([BARS, BARS.iloc[-1:].set_axis([BARS.index[-1] + pd.Timedelta(minutes=15)])])
calls = []
real_load_bars = D.load_bars
def _fake_load(sym, tf, refresh=True, log=print):
    calls.append((sym, tf, refresh))
    return newer
D.load_bars = _fake_load
w4.update_data()
assert not w4.settings.btn_update.isEnabled()
wait(lambda: w4.settings.btn_update.isEnabled(), 15, "data update")
assert calls == [("SOXL", 15, True)] and w4.backtest._hi == newer.index[-1] and f"{len(newer):,} bars" in w4.settings.data_lbl.text() and "data updated" in w4.live.logbox.toPlainText()
w4.backtest._bars[("SOXL", 15)] = BARS.iloc[:-3000]; w4.backtest._last_refresh_try = 0.0              # a stale cache: the window tops it up by itself (quietly)
w4.ensure_fresh()
wait(lambda: w4.settings.btn_update.isEnabled() and len(calls) == 2, 15, "automatic top-up")
assert w4.backtest._bars[("SOXL", 15)].index[-1] == newer.index[-1]
w4.ensure_fresh(); app.processEvents()
assert len(calls) == 2                                                                             # up to date and tried a moment ago: no second download
def _failing_load(sym, tf, refresh=True, log=print):
    raise RuntimeError("no network")
D.load_bars = _failing_load
w4.update_data(); wait(lambda: w4.settings.btn_update.isEnabled(), 15)
assert "data update problem: RuntimeError: no network" in w4.live.logbox.toPlainText() and w4.backtest._hi == newer.index[-1]
D.load_bars = real_load_bars
# the REAL D.load_bars: when the Alpaca top-up fails it logs why and serves the cached bars (a backtest still runs offline)
real_alpaca = S.load_alpaca_bars
def _no_network(*a, **k):
    raise RuntimeError("no network")
S.load_alpaca_bars = _no_network
captured = []
got_bars = D.load_bars("SOXL", 15, refresh=True, log=captured.append)
assert len(got_bars) == len(BARS) and any("could not refresh from Alpaca" in c and "no network" in c for c in captured) and not D.needs_refresh(BARS.index[-1].to_pydatetime(), datetime(2026, 10, 20, 17, 0), 0.0)
w4.backtest._bars.clear(); w4.backtest._last_refresh_try = 0.0
w4.backtest.refresh_chk.setChecked(True)
orig_needs = D.needs_refresh
D.needs_refresh = lambda *a, **k: True                                                              # force the top-up attempt
before_res = w4.backtest.result
w4.backtest.run()
wait(lambda: w4.backtest.run_btn.isEnabled() and w4.backtest.result is not before_res, 60, "offline backtest")
assert w4.backtest.result.metrics["trades"] == m0["trades"] and w4.backtest._last_refresh_try > 0                  # it tried, failed quietly, and ran on the cache
D.needs_refresh = orig_needs
S.load_alpaca_bars = real_alpaca
# signal monitor hover readout
w4.main_tabs.setCurrentIndex(2); w4.monitor.refresh(); wait(lambda: w4.monitor.bars is not None, 20)
mch = w4.monitor.chart
mch.draw()
k_ = 60
pxm, pym = mch.ax.transData.transform((k_, float(mch._bars.close.iloc[k_])))
mch._on_motion(MouseEvent("motion_notify_event", mch, pxm, pym))
assert "O " in w4.monitor.readout.text() and "Supertrend" in w4.monitor.readout.text() and f"{mch._bars.index[k_]:%H:%M}" in w4.monitor.readout.text()
xl0 = mch.ax.get_xlim()
mch._on_scroll(MouseEvent("scroll_event", mch, pxm, pym, button="up", step=1))
assert (mch.ax.get_xlim()[1] - mch.ax.get_xlim()[0]) < (xl0[1] - xl0[0])
mch._on_press(MouseEvent("button_press_event", mch, pxm, pym, button=1, dblclick=True))
assert np.allclose(mch.ax.get_xlim(), xl0)
# menu actions: About, open the log file
ABOUT = []
QMessageBox.about = staticmethod(lambda parent, title, text: ABOUT.append(text))
{a.text(): a for a in w4.actions()}["About"].trigger()
assert ABOUT and "NOT signed off" in ABOUT[-1] and "DRY RUN" in ABOUT[-1]
opened = []
os.startfile = lambda p: opened.append(p)
w4.live.open_log_file()
assert opened == [GU.LOG_FILE]
w4.close()
print("12. dialogs, data refresh, extras OK")

# ══ 13. the open position: the account vs the strict backtest (real Engine, fake broker) ═════════
import json as _json                                                                              # noqa: E402
import queue as _queue                                                                            # noqa: E402
from dataclasses import replace as _replace                                                       # noqa: E402
reset_fake()
GU.CFG = P.to_config(vals0)
GU.SHORT_EQUITY_MULT = 0.5
OP_START = "2024-01-02"
eng13 = GU.Engine(_queue.Queue(), True, 100_000.0, 1.0, OP_START, False)
EXP = {}                                                                                           # side -> (bars, signal, expected open position) at a cut where the strict backtest holds that side
for k in range(len(BARS) - 3000, len(BARS) - 40, 29):
    bk = BARS.iloc[:k].tail(GU.WARMUP_BARS)
    sg = eng13.signal(bk)
    e_ = eng13.expected_open(bk, sg)
    EXP.setdefault(e_["side"] if e_ else "flat", (bk, sg, e_))
    if len(EXP) == 3:
        break
assert set(EXP) == {"long", "short", "flat"}, set(EXP)


def _write_state(v, sym="SOXL", shift=1.0):
    long_ = v["side"] == "long"
    st = {"pos": 1 if long_ else -1, "side": v["side"], "symbol": sym, "qty": v["qty"], "instr_entry": v["price"], "exit_order_ids": [], "resting_tp": None, "resting_stop": None,
          "entry_ref": v["entry_ref"] * shift, "supert_entry": v["entry_ref"] * 0.9, "target": v["target"] * shift, "active_fixed": v["fixed"],
          "max_high": v["extreme"] if long_ else None, "min_low": None if long_ else v["extreme"], "opened": v["opened"], "fold_from": None, "ext_bar": None}
    with open(GU.STATE_FILE, "w", encoding="utf-8") as fh:
        _json.dump(st, fh)


def _broker(v, sym=None, direct_short=False):
    sym = sym or "SOXL"
    long_ = v["side"] == "long" or sym == "SOXS"                                                   # a SOXL short is a negative SOXL position; the SOXS route is a LONG SOXS position
    q = float(v["qty"]) * (1 if long_ else -1)
    return {sym: dict(qty=q, side="long" if long_ else "short", avg=v["price"], mv=abs(q) * v["now"], upl=v["pnl_usd"], uplpc=v["pnl_pct"], cur=v["now"])}


for side_ in ("long", "short"):
    bk, sg, e_ = EXP[side_]
    ref_s = S.SupertrendStrategy(_replace(GU.CFG, useDateFilter=True, startDate=OP_START, endDate="2099-12-31", initial_capital=100_000.0, equity_frac=1.0, short_mult=0.5, touch_fill=False,
                                          capacity_usd=1e18, strict_entries=True))
    ref_s.run(bk.iloc[:-1] if sg["forming"] else bk)
    assert e_ == GU.backtest_open_view(ref_s.open_trade) and e_["side"] == side_                    # the Engine's expected position IS the strict backtest's open trade
    for sym in ("SOXL",) + (("SOXS",) if side_ == "short" else ()):
        _write_state(e_, sym)
        live = eng13.live_open(sg, _broker(e_, sym))
        assert live["tracked"] and live["side"] == side_ and live["symbol"] == sym and live["qty"] == e_["qty"] and abs(live["tp"] - e_["tp"]) < 1e-6 and abs(live["entry_ref"] - e_["entry_ref"]) < 1e-9
        assert abs(live["target"] - e_["target"]) < 1e-9 and (live["trail"] is None) == (e_["trail"] is None) and (live["trail"] is None or abs(live["trail"] - e_["trail"]) < 1e-6)
        cm = GU.compare_open(live, e_)
        assert cm["status"] == "match", (side_, sym, cm)                                                # same side, same frozen levels -> MATCHES (SOXL levels, whichever instrument is held)
    _write_state(e_, "SOXL", shift=1.02)                                                               # levels frozen 2% off -> DIFFERS and says which
    live = eng13.live_open(sg, _broker(e_))
    cm = GU.compare_open(live, e_)
    assert cm["status"] == "differs" and "entry reference" in cm["text"] and cm["diffs"]
    assert GU.compare_open(live, e_, forming=True)["status"] == "timing"
    open(GU.STATE_FILE, "w").write("{}")                                                               # a position this app did not open: no levels to compare
    live = eng13.live_open(sg, _broker(e_))
    assert live["tracked"] is False and live["entry_ref"] is None and GU.compare_open(live, e_)["status"] == "untracked"
    assert GU.compare_open(None, e_)["status"] == "differs" and "account is flat" in GU.compare_open(None, e_)["text"]
bk, sg, e_ = EXP["flat"]
assert e_ is None and eng13.live_open(sg, {}) is None and GU.compare_open(None, None)["status"] == "match"
live = eng13.live_open(sg, _broker(EXP["long"][2]))
assert GU.compare_open(live, None)["status"] == "differs" and "backtest is flat" in GU.compare_open(live, None)["text"]
# the event the Live tab renders: account, backtest, verdict
bk, sg, e_ = EXP["long"]
_write_state(e_)
eng13.report_open_position(sg, _broker(e_), bk)
ev = eng13.q.get_nowait()
while ev["kind"] != "openpos":
    ev = eng13.q.get_nowait()
assert ev["live"]["side"] == "long" and ev["expected"] == e_ and ev["cmp"]["status"] == "match" and ev["want"] == sg["want"] and abs(ev["limit"] - sg["limit"]) < 1e-9
lv.on_event(ev)
t_ = lv.perf.open_lbl.text()
assert "MATCHES THE BACKTEST" in t_ and "Account holds" in t_ and "Backtest holds" in t_ and f"{e_['tp']:,.2f}" in t_ and lv.perf.open_state["cmp"]["status"] == "match"
shot(w, "17_live_open_position_match")
_write_state(e_, shift=1.02)
eng13.report_open_position(sg, _broker(e_), bk)
ev = eng13.q.get_nowait()
while ev["kind"] != "openpos":
    ev = eng13.q.get_nowait()
lv.on_event(ev)
assert ev["cmp"]["status"] == "differs" and "DIFFERS FROM THE BACKTEST" in lv.perf.open_lbl.text()
eng13.report_open_position(sg, {}, bk)                                                                  # flat account, backtest holds: a flat strip, the resting-order / waiting text, the verdict
ev = eng13.q.get_nowait()
while ev["kind"] != "openpos":
    ev = eng13.q.get_nowait()
lv.on_event(ev)
assert lv.perf.open_state["primary"] is None and "FLAT" in lv.perf.open_lbl.text() and "waiting for a" in lv.perf.open_lbl.text() and ev["cmp"]["status"] in ("differs", "timing")
open(GU.STATE_FILE, "w").write("{}")
# a failure in the comparison never stops the loop: it is logged
eng13.expected_open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
eng13.report_open_position(sg, {}, bk)
logs = []
while not eng13.q.empty():
    logs.append(eng13.q.get_nowait())
assert any(d_["kind"] == "log" and "open-position check failed: boom" in d_["msg"] for d_ in logs)
print("13. open position OK")

# nothing leaked: every file the engine and the app wrote is in the throw-away folder
assert GU.REPORTS == tmp and os.path.exists(GU.LOG_FILE)
print("GUI test OK ->", out)
