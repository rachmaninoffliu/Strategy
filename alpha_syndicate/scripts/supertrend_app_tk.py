"""
Supertrend SOXL/SOXS — Unified Backtest + Live Trading Application
==================================================================

One window, one set of parameters, two tabs:

  PARAMETERS   every Pine input, edited once and shared by both tabs
  BACKTEST     run the tuned parameters over history, see metrics /
               equity curve / trade list
  LIVE         run those SAME parameters against Alpaca (paper or live)

The point of consolidating: what you tune is exactly what trades. The live
engine and the backtest engine are the same code (supertrend_strategy), and
both read the identical Config object built by the Parameters tab.

  python supertrend_app.py

RISK: Valid has not signed off. The backtest does not reconcile with the
TradingView Pine run (my port sits ~3.5x above it at matched sizing), so
expected return is unknown within a wide band. factor=5.0 is a parameter
SPIKE, not a plateau. SOXL/SOXS are 3x leveraged. The trailing stop and
limit take-profit are NOT yet implemented in the live engine.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
from datetime import datetime, timedelta, timezone

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import (FuncFormatter, NullFormatter, NullLocator,
                               Locator, LogLocator, AutoLocator)
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertrend_strategy as S
import supertrend_gui as GU
from supertrend_gui import (Engine, creds, SYMBOL_LONG, SYMBOL_SHORT,
                            SHORT_EQUITY_MULT, LIVE_CONFIRM, KILL_FILE, REPORTS)

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

DATA_DIR = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data"
os.makedirs(DATA_DIR, exist_ok=True)

# (attr, label, kind, default, options)
PARAM_SPEC = [
    ("strategyMode", "Strategy mode", "choice", "Always Fixed",
     ["Always Fixed", "Always Dynamic", "Auto-Switch via Vol"]),
    ("atrPeriod",    "ATR period",        "int",   500,   None),
    ("factor",       "Supertrend factor", "float", 5.0,   None),
    ("riskReward",   "Risk/Reward %",     "pct",   4.5,   None),
    ("buyback",      "Buyback offset %",  "pct",   1.5,   None),
    ("trailFactor",  "Trailing factor",   "float", 0.0,   None),
    ("price_perc",   "Limit TP (frac)",   "float", 0.1,   None),
    ("riskfactor",   "Risk factor",       "float", 1.0,   None),
    ("limitfactor",  "Limit factor",      "float", 1.0,   None),
    ("useDynamic",   "ADX dynamic trail", "bool",  False, None),
    ("adxLength",    "ADX length",        "int",   14,    None),
    ("adxBase",      "ADX baseline",      "float", 25.0,  None),
    ("volLength",    "Vol ATR length",    "int",   20,    None),
    ("volSmaLength", "Vol baseline len",  "int",   100,   None),
    ("volThreshold", "High-vol mult",     "float", 1.3,   None),
    ("equity_frac",  "Equity per trade %", "pct",  100.0, None),
    ("short_mult",   "Short size mult",   "float", 0.5,   None),
    ("commission_bps", "Commission bp/leg", "float", 5.0, None),
    ("touch_fill",   "Touch fills (TV)",  "bool",  False, None),
]



# ============================================================== UX helpers ===
class Tooltip:
    """Hover help. The parameter names come from Pine and are cryptic
    (price_perc, limitfactor, riskfactor) -- without this the panel is
    unreadable to anyone who has not read the source."""

    def __init__(self, widget, text, delay=450):
        self.w, self.text, self.delay = widget, text, delay
        self._id = self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        self._id = self.w.after(self.delay, self._show)

    def _cancel(self):
        if self._id:
            self.w.after_cancel(self._id)
            self._id = None

    def _show(self):
        if self._tip or not self.text:
            return
        try:
            x = self.w.winfo_rootx() + 18
            y = self.w.winfo_rooty() + self.w.winfo_height() + 6
        except Exception:
            return
        self._tip = tk.Toplevel(self.w)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, justify="left",
                 bg="#1f2937", fg="#f9fafb", font=("Segoe UI", 8),
                 relief="solid", borderwidth=1, padx=7, pady=4,
                 wraplength=380).pack()

    def _hide(self, _=None):
        self._cancel()
        if self._tip:
            self._tip.destroy()
            self._tip = None


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container.

    Without this, a window shorter than the content simply clips it with no
    way to reach the hidden widgets -- the Parameters grid and the Live panel
    stack both overflow on a laptop screen.

    Put children into `.body`. The mouse wheel is bound only while the pointer
    is inside, so it never steals scroll from the chart or a Treeview.
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_yview)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body,
                                              anchor="nw")
        self.body.bind("<Configure>", self._on_body)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_yview(self, lo, hi):
        # hide the scrollbar entirely when everything already fits
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.vsb.pack_forget()
        else:
            self.vsb.pack(side="right", fill="y")
        self.vsb.set(lo, hi)

    def _on_body(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, ev):
        self.canvas.itemconfigure(self._win, width=ev.width)

    def _bind_wheel(self, _=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)

    def _unbind_wheel(self, _=None):
        self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, ev):
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return                      # nothing to scroll
        self.canvas.yview_scroll(int(-ev.delta / 120), "units")


PARAM_HELP = {
    "strategyMode": "Always Fixed: take profit at riskReward then trail.\n"
                    "Always Dynamic: target scales with distance to Supertrend.\n"
                    "Auto-Switch: Fixed only while ATR% > baseline x High-vol mult.\n"
                    "Only Always Fixed was profitable in testing.",
    "atrPeriod": "ATR length for Supertrend, in BARS (not days). 500 on 15-min "
                 "is ~19 trading days.\nSits on a plateau: 375-750 all gave "
                 "Sharpe 1.42-1.52.",
    "factor": "Supertrend ATR multiplier. FRAGILE -- Sharpe 1.52 at 5.0 but "
              "~0.85 at 4.0 and ~0.93 at 6.0.\nThis is a spike, not a plateau. "
              "Treat a good result here with suspicion.",
    "riskReward": "Run-up % from entry that ARMS the trailing stop. Until "
                  "price reaches it there is no stop at all.",
    "buyback": "Re-entry offset from the Supertrend line. After an exit, the "
               "strategy waits for a pullback to st x (1 +/- this) before "
               "re-entering the same trend.",
    "trailFactor": "Fraction of the run-up the trail gives back. 0 = breakeven "
                   "stop (stop sits at entry once armed).\nNote: with ADX "
                   "dynamic trail ON, 0 is rewritten to a 0.05 floor.",
    "price_perc": "Limit take-profit as a FRACTION of entry. 0.1 = +10% for a "
                  "long, -10% for a short.",
    "riskfactor": "Scales the measured entry risk |entry - supertrend|. Only "
                  "used in Dynamic mode.",
    "limitfactor": "Scales the dynamic take-profit distance. Only used in "
                   "Dynamic mode.",
    "useDynamic": "Modulate trailFactor by ADX: max(0.05, min(0.95, "
                  "trailFactor + (adxBase - ADX)/100)).\nTurn OFF to use "
                  "trailFactor literally.",
    "adxLength": "ADX lookback in bars. Only used when ADX dynamic trail is on.",
    "adxBase": "ADX neutral point. Above it the trail tightens, below it loosens.",
    "volLength": "ATR length for the volatility regime measure.",
    "volSmaLength": "Baseline window the current ATR% is compared against.",
    "volThreshold": "High-vol trigger: ATR% > baseline x this switches "
                    "Auto-Switch into Fixed mode.",
    "equity_frac": "Percent of running equity per LONG trade. 100% on a 3x ETF "
                   "is ~3x effective exposure; backtested MaxDD was -65%.",
    "short_mult": "Shorts are sized at equity_frac x this. 0.5 reproduces the "
                  "Pine, which hardcodes equity*0.5 for shorts while longs use "
                  "the full fraction.",
    "commission_bps": "Per-leg cost in basis points. Your TradingView run used "
                      "0, which is worth ~10 points of CAGR.",
    "touch_fill": "ON reproduces TradingView: a limit fills when price merely "
                  "TOUCHES it. OFF requires the bar to pierce the level -- "
                  "stricter and more realistic.",
}


# ============================================================ parameters ====
class ParamsFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars: dict[str, tk.Variable] = {}
        self.entries: dict[str, ttk.Entry] = {}

        hint = ttk.Label(self, foreground="#6b7280", font=("Segoe UI", 8),
                         text="  Hover any label for an explanation. Values are "
                              "validated as you leave a field; invalid entries "
                              "turn red and block the run.")
        hint.pack(fill="x", pady=(4, 0))

        scroll = ScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        host = scroll.body

        # group the 19 inputs so the panel reads as sections, not a wall
        groups = [
            ("Signal", ["strategyMode", "atrPeriod", "factor"]),
            ("Exits", ["riskReward", "price_perc", "trailFactor", "buyback",
                       "riskfactor", "limitfactor"]),
            ("ADX dynamic trail", ["useDynamic", "adxLength", "adxBase"]),
            ("Volatility regime", ["volLength", "volSmaLength", "volThreshold"]),
            ("Sizing & costs", ["equity_frac", "short_mult", "commission_bps",
                                "touch_fill"]),
        ]
        spec = {a: (a, l, k, d, o) for a, l, k, d, o in PARAM_SPEC}

        for gname, attrs in groups:
            box = ttk.LabelFrame(host, text=gname)
            box.pack(fill="x", padx=10, pady=5, anchor="n")
            for i, attr in enumerate(attrs):
                if attr not in spec:
                    continue
                _, label, kind, default, opts = spec[attr]
                r, c = divmod(i, 3)
                cell = ttk.Frame(box)
                cell.grid(row=r, column=c, sticky="w", padx=10, pady=5)
                lab = ttk.Label(cell, text=label, width=18)
                lab.pack(side="left")
                Tooltip(lab, PARAM_HELP.get(attr, "").replace("\\n", "\n"))
                if kind == "bool":
                    v = tk.BooleanVar(value=default)
                    w = ttk.Checkbutton(cell, variable=v)
                elif kind == "choice":
                    v = tk.StringVar(value=default)
                    w = ttk.Combobox(cell, textvariable=v, values=opts, width=18,
                                     state="readonly")
                else:
                    v = tk.StringVar(value=str(default))
                    w = ttk.Entry(cell, textvariable=v, width=12)
                    self.entries[attr] = w
                    w.bind("<FocusOut>", lambda e, a=attr: self._validate(a))
                    w.bind("<Return>", lambda e, a=attr: self._validate(a))
                w.pack(side="left")
                Tooltip(w, PARAM_HELP.get(attr, "").replace("\\n", "\n"))
                self.vars[attr] = v

        bar = ttk.Frame(host)
        bar.pack(fill="x", padx=10, pady=12, anchor="w")
        ttk.Button(bar, text="Reset defaults", command=self.reset).pack(side="left")
        ttk.Button(bar, text="Save preset", command=self.save).pack(side="left", padx=6)
        ttk.Button(bar, text="Load preset", command=self.load).pack(side="left")
        self.lbl = ttk.Label(bar, text="", foreground="#166534")
        self.lbl.pack(side="left", padx=14)

    # ------------------------------------------------------- validation ----
    def _kind(self, attr):
        for a, _, k, _, _ in PARAM_SPEC:
            if a == attr:
                return k
        return "float"

    def _validate(self, attr) -> bool:
        """Mark a bad field red immediately instead of failing at run time."""
        w = self.entries.get(attr)
        if w is None:
            return True
        raw = self.vars[attr].get().strip()
        kind = self._kind(attr)
        ok = True
        try:
            val = float(raw)
            if kind == "int" and val != int(val):
                ok = False
            if val < 0:
                ok = False
            if kind == "pct" and val > 1000:
                ok = False
        except ValueError:
            ok = False
        try:
            w.configure(foreground="#111827" if ok else "#dc2626")
        except Exception:
            pass
        return ok

    def validate_all(self) -> list[str]:
        return [a for a in self.entries if not self._validate(a)]

    def reset(self):
        for attr, _, kind, default, _ in PARAM_SPEC:
            self.vars[attr].set(default if kind in ("bool", "choice") else str(default))
        self.lbl.config(text="defaults restored")

    def save(self):
        path = os.path.join(REPORTS, "params_preset.json")
        json.dump({k: v.get() for k, v in self.vars.items()},
                  open(path, "w"), indent=2)
        self.lbl.config(text=f"saved -> {os.path.basename(path)}")

    def load(self):
        path = os.path.join(REPORTS, "params_preset.json")
        if not os.path.exists(path):
            self.lbl.config(text="no preset found"); return
        for k, v in json.load(open(path)).items():
            if k in self.vars:
                self.vars[k].set(v)
        self.lbl.config(text="preset loaded")

    def to_config(self, **extra) -> S.Config:
        kw = {}
        for attr, _, kind, _, _ in PARAM_SPEC:
            raw = self.vars[attr].get()
            if kind == "bool":
                kw[attr] = bool(raw)
            elif kind == "choice":
                kw[attr] = str(raw)
            elif kind == "int":
                kw[attr] = int(float(raw))
            elif kind == "pct":
                kw[attr] = float(raw) / 100.0
            else:
                kw[attr] = float(raw)
        kw.update(extra)
        kw.setdefault("useDateFilter", False)
        kw.setdefault("capacity_usd", 1e18)
        return S.Config(**kw)

    def summary(self) -> str:
        c = self.to_config()
        return (f"{c.strategyMode} | atr{c.atrPeriod} f{c.factor} "
                f"RR{c.riskReward*100:.1f}% bb{c.buyback*100:.1f}% "
                f"trail{c.trailFactor} pp{c.price_perc} dyn={c.useDynamic} | "
                f"{c.equity_frac:.0%}/trade short x{c.short_mult}")


# ============================================================== data ========
def load_bars(symbol, tf_min, start, end, log=print):
    """Bars for exactly the Start/End the user typed in the GUI.

    Delegates to S.load_alpaca_bars, which tops the parquet cache up on both
    ends before returning. Previously this short-circuited on
    `os.path.exists(cache)` and replayed whatever day the file was created.
    """
    p = S.load_alpaca_bars([symbol], tf_min, start, end, rth_only=True,
                           cache_dir=DATA_DIR, dividend_adjusted=False, log=log)
    df = (p[p.symbol == symbol] if "symbol" in p.columns else p)
    df = df.set_index("bar").sort_index()[["open", "high", "low", "close"]].ffill().dropna()
    out = df.loc[(df.index >= start) & (df.index <= end + " 23:59")]
    if len(out):
        log(f"  window: {out.index[0]:%Y-%m-%d %H:%M} -> {out.index[-1]:%Y-%m-%d %H:%M}"
            f"  ({len(out):,} bars)")
    return out


class PercentLocator(Locator):
    """Ticks at round PERCENT values (0, 100%, 1K%, ...) on a dollar axis.

    A log dollar axis ticks on decades, which in percent terms reads
    0% / 900% / 9.9K% / 99.9K% -- correct but unreadable. This places the
    gridlines where a human would put them and recomputes on every zoom.
    """

    def __init__(self, tab):
        self.tab = tab

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        base = self.tab._base or 1.0
        lo, hi = sorted(((vmin / base - 1) * 100.0, (vmax / base - 1) * 100.0))
        top = max(abs(lo), abs(hi))
        if top <= 0:
            return [base]
        pcts = [0.0] if lo <= 0 <= hi else []
        # cap the ladder at ~5 decades below the top, else a 2.4M% range
        # drags in a pile of unreadable 0.1% / -0.0% ticks near the base
        lo_e = max(-2, int(np.floor(np.log10(top))) - 5)
        exps = [e for e in range(lo_e, 11) if 10 ** e <= top * 10]
        mants = (1,) if len(exps) > 5 else (1, 2, 5)
        for e in exps:
            for m in mants:
                for p in (m * 10 ** e, -m * 10 ** e):
                    if lo <= p <= hi:
                        pcts.append(float(p))

        # Drop ticks that would overlap: on a log dollar axis -10% / 0% / +10%
        # are almost the same pixel. Keep 0% first, then the largest moves.
        log = getattr(self.axis.axes, "get_yscale", lambda: "linear")() == "log"
        def pos(p):
            y = base * (1 + p / 100.0)
            if log:
                if y <= 0 or vmin <= 0 or vmax <= 0 or vmax == vmin:
                    return None
                return (np.log10(y) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
            return None if vmax == vmin else (y - vmin) / (vmax - vmin)

        kept = []
        for p in sorted(set(pcts), key=lambda v: (v != 0.0, -abs(v))):
            q = pos(p)
            if q is None:
                kept.append(p)
            elif all(abs(q - pos(k)) >= 0.045 for k in kept if pos(k) is not None):
                kept.append(p)
        return [base * (1 + p / 100.0) for p in sorted(kept)]


# ============================================================ backtest ======
class BacktestTab(ttk.Frame):
    def __init__(self, master, params: ParamsFrame):
        super().__init__(master)
        self.params = params
        self.q: queue.Queue = queue.Queue()

        top = ttk.LabelFrame(self, text="Run")
        top.pack(fill="x", padx=8, pady=6)
        self.sym = tk.StringVar(value="SOXL")
        self.tf = tk.StringVar(value="15")
        self.start = tk.StringVar(value="2016-01-01")
        self.end = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.cap = tk.StringVar(value="100000")
        for lbl, var, w in (("Symbol", self.sym, 8), ("TF (min)", self.tf, 6),
                            ("Start", self.start, 11), ("End", self.end, 11),
                            ("Capital $", self.cap, 12)):
            ttk.Label(top, text=lbl).pack(side="left", padx=(10, 3), pady=8)
            ttk.Entry(top, textvariable=var, width=w).pack(side="left")
        self.btn = ttk.Button(top, text="RUN BACKTEST", command=self.run)
        self.btn.pack(side="left", padx=16)
        self.prog = ttk.Progressbar(top, mode="indeterminate", length=120)
        self.prog.pack(side="left", padx=(0, 8))
        self.status = ttk.Label(top, text="idle")
        self.status.pack(side="left")
        ttk.Label(top, text="Ctrl+R run · Ctrl+E export",
                  foreground="#6b7280", font=("Segoe UI", 8)
                  ).pack(side="right", padx=8)

        self.res = tk.Label(self, text="-", font=("Consolas", 10), anchor="w",
                            justify="left")
        self.res.pack(fill="x", padx=12, pady=(2, 6))

        # PanedWindow: the chart/table split is draggable, so either side can
        # be given the whole width when you are focused on it.
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        # ---- chart: interactive (toolbar pan/zoom + wheel zoom), $ or % y-axis
        chart = ttk.Frame(body); body.add(chart, weight=3)
        self._base = None        # initial capital: the 0% / $0 reference
        self._series = None      # plotted series, kept for the y auto-fit
        self._ymin = 1.0
        self._xlim0 = self._ylim0 = None

        ctl = ttk.Frame(chart); ctl.pack(side="top", fill="x")
        ttk.Label(ctl, text="Y axis:").pack(side="left", padx=(4, 2))
        self.ymode = tk.StringVar(value="$")
        for txt, val in (("$ equity", "$"), ("% return", "%")):
            ttk.Radiobutton(ctl, text=txt, value=val, variable=self.ymode,
                            command=self._apply_axes).pack(side="left", padx=2)
        self.ylog = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="log", variable=self.ylog,
                        command=self._apply_axes).pack(side="left", padx=(12, 2))
        self.yfit = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="auto-fit Y", variable=self.yfit).pack(side="left", padx=2)
        ttk.Button(ctl, text="Reset view", width=11,
                   command=self._reset_view).pack(side="left", padx=(12, 2))
        ttk.Label(ctl, text="wheel = zoom X · ctrl+wheel = zoom Y",
                  foreground="#6b7280", font=("Segoe UI", 8)).pack(side="left", padx=8)

        self.fig = Figure(figsize=(7.4, 4.4), dpi=96)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        self.ax = self.fig.add_subplot(gs[0])
        # underwater plot: the drawdown profile is what the equity curve hides
        self.axd = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.ax.tick_params(labelbottom=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart)
        tb = NavigationToolbar2Tk(self.canvas, chart, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.readout = ttk.Label(chart, text="", font=("Consolas", 9),
                                 foreground="#374151", anchor="w")
        self.readout.pack(side="bottom", fill="x")

        rt = ttk.Frame(body); body.add(rt, weight=2)
        hdr = ttk.Frame(rt); hdr.pack(fill="x")
        self.lbl_n = ttk.Label(hdr, text="trades: -", font=("Segoe UI", 8))
        self.lbl_n.pack(side="left", padx=4)
        ttk.Button(hdr, text="Export CSV", width=11,
                   command=self.export).pack(side="right", padx=4)
        cols = ("entry_time", "exit_time", "side", "qty", "entry", "exit",
                "ret%", "pnl", "why")
        self.tv = ttk.Treeview(rt, columns=cols, show="headings", height=16)
        self._sort_col, self._sort_rev = None, False
        for c, w in zip(cols, (132, 132, 50, 58, 62, 62, 60, 78, 66)):
            self.tv.heading(c, text=c.upper().replace("_", " "),
                            command=lambda cc=c: self._sort_by(cc))
            self.tv.column(c, width=w,
                           anchor="w" if c in ("entry_time", "exit_time", "side", "why")
                           else "e")
        sb = ttk.Scrollbar(rt, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(side="left", fill="both", expand=True)
        # direction colours: long = green, short = red (NOT win/loss)
        self.tv.tag_configure("long", foreground=GU.COL_LONG)
        self.tv.tag_configure("short", foreground=GU.COL_SHORT)
        self.tv.bind("<<TreeviewSelect>>", self._on_pick)
        self._marks = []
        self.after(200, self._drain)

    # ---------------------------------------------------- interaction ----
    def _on_move(self, ev):
        """Live readout under the chart: date, equity, and % from base."""
        if ev.inaxes not in (self.ax, self.axd) or ev.xdata is None:
            self.readout.config(text="")
            return
        try:
            when = mdates.num2date(ev.xdata).replace(tzinfo=None)
        except Exception:
            return
        txt = when.strftime("%Y-%m-%d %H:%M")
        if self._series:
            eq = self._series[0]
            idx = eq.index.searchsorted(when)
            if 0 <= idx < len(eq):
                v = float(eq.iloc[idx])
                pct = (v / (self._base or 1.0) - 1.0) * 100.0
                dd = (v / float(eq.iloc[:idx + 1].max()) - 1.0) * 100.0
                txt += (f"   equity {self._fmt_dollar(v)}"
                        f"   return {pct:+,.1f}%   drawdown {dd:+.1f}%")
        self.readout.config(text="  " + txt)

    def _sort_by(self, col):
        """Click a heading to sort; click again to reverse."""
        rev = not self._sort_rev if self._sort_col == col else False
        self._sort_col, self._sort_rev = col, rev
        idx = self.tv["columns"].index(col)

        def key(iid):
            raw = str(self.tv.item(iid)["values"][idx])
            try:
                return (0, float(raw.replace(",", "").replace("%", "")
                                 .replace("$", "").replace("+", "")))
            except ValueError:
                return (1, raw)

        for pos, iid in enumerate(sorted(self.tv.get_children(""), key=key,
                                         reverse=rev)):
            self.tv.move(iid, "", pos)
        for c in self.tv["columns"]:
            base = c.upper().replace("_", " ")
            self.tv.heading(c, text=base + ("  v" if c == col and not rev
                                            else "  ^" if c == col else ""))

    def _on_pick(self, _=None):
        """Mark the selected trade on the chart -- links table to curve."""
        for m in self._marks:
            try:
                m.remove()
            except Exception:
                pass
        self._marks = []
        sel = self.tv.selection()
        if not sel or self._base is None:
            self.canvas.draw_idle()
            return
        vals = self.tv.item(sel[0])["values"]
        try:
            t0 = pd.Timestamp(str(vals[0]))
            t1 = pd.Timestamp(str(vals[1]))
        except Exception:
            return
        side = str(vals[2]).lower()
        col = GU.COL_LONG if side == "long" else GU.COL_SHORT
        for ax in (self.ax, self.axd):
            self._marks.append(ax.axvspan(t0, t1, color=col, alpha=0.16, zorder=0))
        self._marks.append(self.ax.axvline(t0, color=col, lw=0.9, ls="--", zorder=1))
        self._marks.append(self.ax.axvline(t1, color=col, lw=0.9, ls=":", zorder=1))
        self.canvas.draw_idle()

    # ------------------------------------------------------------ chart ----
    # The curve is always plotted in DOLLARS. Switching to "%" only swaps the
    # tick formatter, so the toggle is instant and never rescales the view --
    # and percent stays log-safe, which (equity/base - 1) would not be once it
    # crosses zero.
    def _fmt_dollar(self, y, _pos=None):
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(y) >= div:
                return f"${y / div:,.2f}{suf}"
        return f"${y:,.0f}"

    def _fmt_pct(self, y, _pos=None):
        p = (y / (self._base or 1.0) - 1.0) * 100.0
        if abs(p) >= 1e6:
            return f"{p / 1e6:,.2f}M%"
        if abs(p) >= 1e3:
            return f"{p / 1e3:,.1f}K%"
        return f"{p:,.1f}%"

    def _apply_axes(self, redraw=True):
        pct = self.ymode.get() == "%"
        log_ok = self.ylog.get() and self._ymin > 0
        self.ax.set_yscale("log" if log_ok else "linear")
        self.ax.yaxis.set_major_formatter(
            FuncFormatter(self._fmt_pct if pct else self._fmt_dollar))
        if pct:
            self.ax.yaxis.set_major_locator(PercentLocator(self))
            self.ax.yaxis.set_minor_locator(NullLocator())
        else:
            self.ax.yaxis.set_major_locator(LogLocator() if log_ok else AutoLocator())
            self.ax.yaxis.set_minor_locator(
                LogLocator(subs="auto") if log_ok else NullLocator())
        if log_ok:                       # else decade minor ticks get labelled too
            self.ax.yaxis.set_minor_formatter(NullFormatter())
        self.ax.set_ylabel("cumulative return" if pct else "equity")
        self.ax.set_title(("Equity — % return" if pct else "Equity — $")
                          + (" (log)" if log_ok else " (linear)"), fontsize=9)
        if redraw:
            self.canvas.draw_idle()

    def _visible(self, s):
        x0, x1 = self.ax.get_xlim()
        lo = mdates.num2date(x0).replace(tzinfo=None)
        hi = mdates.num2date(x1).replace(tzinfo=None)
        return s.loc[(s.index >= lo) & (s.index <= hi)]

    def _fit_y(self):
        """Rescale Y to the data actually in view — the TradingView behaviour."""
        if not self._series:
            return
        lo = hi = None
        for s in self._series:
            w = self._visible(s)
            if len(w):
                lo = w.min() if lo is None else min(lo, w.min())
                hi = w.max() if hi is None else max(hi, w.max())
        if lo is None or hi is None or hi <= lo:
            return
        if self.ax.get_yscale() == "log" and lo > 0:
            pad = (np.log10(hi) - np.log10(lo)) * 0.05 or 0.02
            self.ax.set_ylim(10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad))
        else:
            pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 1.0
            self.ax.set_ylim(lo - pad, hi + pad)

    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax or self._base is None:
            return
        f = 1 / 1.3 if ev.button == "up" else 1.3
        if (ev.key or "") in ("control", "ctrl"):
            y0, y1 = self.ax.get_ylim()
            if self.ax.get_yscale() == "log" and ev.ydata and ev.ydata > 0:
                a, b, c = np.log10(y0), np.log10(y1), np.log10(ev.ydata)
                self.ax.set_ylim(10 ** (c - (c - a) * f), 10 ** (c + (b - c) * f))
            elif ev.ydata is not None:
                c = ev.ydata
                self.ax.set_ylim(c - (c - y0) * f, c + (y1 - c) * f)
        else:
            x0, x1 = self.ax.get_xlim()
            if ev.xdata is None:
                return
            c = ev.xdata
            self.ax.set_xlim(c - (c - x0) * f, c + (x1 - c) * f)
            if self.yfit.get():
                self._fit_y()
        self.canvas.draw_idle()

    def _reset_view(self):
        if self._xlim0 is None:
            return
        self.ax.set_xlim(self._xlim0)
        self.ax.set_ylim(self._ylim0)
        self._apply_axes()

    def run(self):
        bad = self.params.validate_all()
        if bad:
            messagebox.showerror(
                "Invalid parameters",
                "These fields are not valid numbers (shown in red on the "
                "Parameters tab):\n\n  " + "\n  ".join(bad))
            return
        try:
            cfg = self.params.to_config(initial_capital=float(self.cap.get()))
        except Exception as exc:
            messagebox.showerror("Bad parameters", str(exc)); return
        self.btn.config(state="disabled"); self.status.config(text="running…")
        self.prog.start(12)
        threading.Thread(target=self._work, args=(cfg,), daemon=True).start()

    def _work(self, cfg):
        try:
            bars = load_bars(self.sym.get(), int(self.tf.get()),
                             self.start.get(), self.end.get(),
                             log=lambda m: self.q.put(("status", m)))
            self.q.put(("status", f"{len(bars):,} bars — simulating…"))
            eq, tr = S.SupertrendStrategy(cfg).run(bars)
            bpd = round(390 / int(self.tf.get()))
            st = S.summarize(eq, tr, bpd * 252)
            bh = bars["close"] / bars["close"].iloc[0]
            self.q.put(("done", (st, eq, tr, bh, cfg)))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                elif kind == "error":
                    self.prog.stop()
                    self.btn.config(state="normal"); self.status.config(text="failed")
                    messagebox.showerror("Backtest failed", payload[-1500:])
                elif kind == "done":
                    self._render(*payload)
                    self.prog.stop()
                    self.btn.config(state="normal"); self.status.config(text="done")
        except queue.Empty:
            pass
        self.after(200, self._drain)

    def export(self):
        tr = getattr(self, "_trades", None)
        if tr is None or not len(tr):
            messagebox.showinfo("Export", "Run a backtest first."); return
        path = os.path.join(REPORTS,
                            f"backtest_trades_{self.sym.get()}_{self.tf.get()}m.csv")
        tr.to_csv(path, index=False)
        messagebox.showinfo("Export", f"{len(tr):,} trades written to\n{path}")

    def _render(self, st, eq, tr, bh, cfg):
        self.res.config(text=(
            f"CAGR {st['cagr_pct']:.2f}%   Sharpe {st['sharpe']:.2f}   "
            f"MaxDD {st['max_dd_pct']:.2f}%   total {st['total_return_pct']:,.0f}%   "
            f"final ${st['final_equity']:,.0f}\n"
            f"trades {st['trades']:,}   win {st['win_rate_pct']:.1f}%   "
            f"PF {st['profit_factor']:.3f}   avg {st['avg_trade_pct']:.3f}%   |   "
            f"{self.params.summary()}"))
        self.ax.clear(); self.axd.clear()
        self._marks = []            # ax.clear() already removed them
        base = float(eq.iloc[0])
        eq_d = eq.astype(float)
        bh_d = (bh * base).astype(float)      # same starting capital as strategy
        self._base = base
        self._series = (eq_d, bh_d)
        self._ymin = float(min(eq_d.min(), bh_d.min()))
        self.ax.plot(eq_d.index, eq_d.values, lw=1.2, label="strategy")
        self.ax.plot(bh_d.index, bh_d.values, lw=1.2, color="black", alpha=.65,
                     label=f"{self.sym.get()} buy & hold")
        self.ax.grid(alpha=.3, which="both")
        self.ax.legend(loc="upper left", fontsize=8)

        # underwater curve -- the equity line alone hides how deep and how long
        dd = (eq_d / eq_d.cummax() - 1.0) * 100.0
        ddb = (bh_d / bh_d.cummax() - 1.0) * 100.0
        self.axd.fill_between(dd.index, dd.values, 0, color="#dc2626",
                              alpha=.45, lw=0)
        self.axd.plot(ddb.index, ddb.values, lw=0.8, color="black", alpha=.45)
        self.axd.set_ylabel("DD %", fontsize=8)
        self.axd.grid(alpha=.3)
        self.axd.tick_params(labelsize=8)
        self.axd.margins(x=0)

        self._apply_axes(redraw=False)
        self.ax.relim(); self.ax.autoscale()
        self._xlim0, self._ylim0 = self.ax.get_xlim(), self.ax.get_ylim()
        self.fig.tight_layout(); self.canvas.draw()
        for i in self.tv.get_children():
            self.tv.delete(i)
        self._trades = tr
        # EVERY trade in the tested window, with entry AND exit timestamps.
        for _, r in tr.iterrows():
            self.tv.insert("", "end",
                           tags=(GU.strategy_side(side=r.side),),
                           values=(str(r.entry_time)[:19], str(r.exit_time)[:19],
                                   str(r.side).upper(), f"{r.qty:,.0f}",
                                   f"{r.price:.2f}", f"{r.exit_price:.2f}",
                                   f"{r.return_pct:+.2f}",
                                   f"{r.get('pnl_usd', 0):+,.0f}",
                                   str(r.get("exit_reason", ""))[:10]))
        self.lbl_n.config(text=f"trades: {len(tr):,}  "
                               f"({str(tr.entry_time.min())[:10]} -> "
                               f"{str(tr.exit_time.max())[:10]})" if len(tr)
                          else "trades: 0")
        if len(tr):
            self.tv.see(self.tv.get_children()[-1])


# ================================================================ live ======
class LiveTab(ttk.Frame):
    def __init__(self, master, params: ParamsFrame):
        super().__init__(master)
        self.params = params
        self.q: queue.Queue = queue.Queue()
        self.engine: Engine | None = None
        pad = dict(padx=8, pady=4)

        f = ttk.LabelFrame(self, text="Session"); f.pack(fill="x", **pad)
        self.cap = tk.StringVar(value="100000")
        self.sdate = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.mode = tk.StringVar(value="PAPER")
        self.armed = tk.BooleanVar(value=False)
        ttk.Label(f, text="Capital $").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(f, textvariable=self.cap, width=13).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Start date").grid(row=0, column=2, sticky="w", padx=(16, 6))
        ttk.Entry(f, textvariable=self.sdate, width=13).grid(row=0, column=3, sticky="w")
        ttk.Label(f, text="Account").grid(row=0, column=4, sticky="w", padx=(16, 6))
        mb = ttk.Combobox(f, textvariable=self.mode, values=["PAPER", "LIVE"],
                          width=9, state="readonly")
        mb.grid(row=0, column=5, sticky="w"); mb.bind("<<ComboboxSelected>>", self._mode)
        ttk.Checkbutton(f, text="ARM (send real orders)", variable=self.armed) \
            .grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        b = ttk.Frame(f); b.grid(row=1, column=3, columnspan=3, sticky="e", padx=6)
        self.b_start = ttk.Button(b, text="START", command=self.start)
        self.b_stop = ttk.Button(b, text="STOP", command=self.stop, state="disabled")
        self.b_flat = ttk.Button(b, text="FLATTEN", command=self.flatten)
        for w in (self.b_start, self.b_stop, self.b_flat):
            w.pack(side="left", padx=3)

        self.status = tk.Label(self, text="IDLE", bg="#374151", fg="white",
                               font=("Segoe UI", 10, "bold"), anchor="w")
        self.status.pack(fill="x", padx=8)
        self.lbl_par = ttk.Label(self, text="", font=("Consolas", 8),
                                 foreground="#4b5563")
        self.lbl_par.pack(anchor="w", padx=12, pady=(2, 0))

        fe = ttk.LabelFrame(self, text="Session equity"); fe.pack(fill="x", **pad)
        self.lbl_sess = tk.Label(fe, text="-", font=("Consolas", 10), anchor="w",
                                 justify="left"); self.lbl_sess.pack(anchor="w", padx=8, pady=6)
        fn = ttk.LabelFrame(self, text="Next trade"); fn.pack(fill="x", **pad)
        self.lbl_next = tk.Label(fn, text="-", font=("Consolas", 11, "bold"),
                                 fg="#1d4ed8", anchor="w", justify="left")
        self.lbl_next.pack(anchor="w", padx=8, pady=8)
        fp = ttk.LabelFrame(self, text="Open position"); fp.pack(fill="x", **pad)
        self.lbl_pos = tk.Label(fp, text="NO OPEN POSITION", fg="#6b7280",
                                font=("Consolas", 11, "bold"), anchor="w", justify="left")
        self.lbl_pos.pack(anchor="w", padx=8, pady=8)

        fl = ttk.LabelFrame(self, text="Log"); fl.pack(fill="both", expand=True, **pad)
        lc = ttk.Frame(fl); lc.pack(fill="x", padx=6, pady=(4, 0))
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(lc, text="auto-scroll", variable=self.autoscroll
                        ).pack(side="left")
        ttk.Button(lc, text="Clear", width=7,
                   command=lambda: self.log.delete("1.0", "end")).pack(side="left",
                                                                       padx=6)
        ttk.Button(lc, text="Copy all", width=9,
                   command=self._copy_log).pack(side="left")
        ttk.Label(lc, text="full history in reports/live_log.jsonl",
                  foreground="#6b7280", font=("Segoe UI", 8)).pack(side="right")
        lw = ttk.Frame(fl); lw.pack(fill="both", expand=True, padx=6, pady=6)
        lsb = ttk.Scrollbar(lw, orient="vertical")
        self.log = tk.Text(lw, height=10, bg="#0b1020", fg="#d1d5db",
                           font=("Consolas", 9), yscrollcommand=lsb.set)
        lsb.config(command=self.log.yview)
        lsb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        for tag, col in (("err", "#f87171"), ("warn", "#fbbf24"),
                         ("ok", "#34d399"), ("trade", "#60a5fa")):
            self.log.tag_config(tag, foreground=col)
        self.after(200, self._drain)

    def _mode(self, _=None):
        if self.mode.get() != "LIVE":
            return
        top = tk.Toplevel(self); top.title("Confirm LIVE"); top.grab_set()
        tk.Label(top, bg="#7f1d1d", fg="white", padx=14, pady=10, justify="left",
                 text=("LIVE MODE USES REAL MONEY.\n\nValid has not signed off.\n"
                       "SOXL/SOXS are 3x leveraged.\n\n"
                       f"Type   {LIVE_CONFIRM}   to continue.")).pack()
        e = tk.Entry(top, width=32); e.pack(pady=8); e.focus_set()
        def ok():
            if e.get().strip() == LIVE_CONFIRM:
                top.destroy(); self._line("LIVE mode selected", "err")
            else:
                self.mode.set("PAPER"); top.destroy()
        ttk.Button(top, text="Confirm", command=ok).pack(pady=(0, 10))
        top.protocol("WM_DELETE_WINDOW", lambda: (self.mode.set("PAPER"), top.destroy()))

    def start(self):
        try:
            cap = float(self.cap.get())
            datetime.strptime(self.sdate.get(), "%Y-%m-%d")
            cfg = self.params.to_config()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc)); return
        paper = self.mode.get() == "PAPER"
        if self.armed.get() and not messagebox.askyesno(
                "Arm", f"Send REAL orders to the {'PAPER' if paper else 'LIVE'} "
                       f"account?\n\n{self.params.summary()}"):
            return
        # push tuned params into the shared engine config
        import supertrend_gui as GU
        GU.CFG = cfg
        GU.SHORT_EQUITY_MULT = cfg.short_mult
        self.engine = Engine(self.q, paper, cap, cfg.equity_frac,
                             self.sdate.get(), self.armed.get())
        self.engine.start()
        self.b_start.config(state="disabled"); self.b_stop.config(state="normal")
        self.lbl_par.config(text="running with: " + self.params.summary())
        self.status.config(text=f"RUNNING · {self.mode.get()} · "
                                f"{'ARMED' if self.armed.get() else 'DRY RUN'}",
                           bg="#166534" if paper else "#7f1d1d")

    def stop(self):
        if self.engine:
            self.engine.stop()
        self.b_stop.config(state="disabled")
        self.status.config(text="STOPPING…", bg="#374151")

    def flatten(self):
        if not messagebox.askyesno("Flatten", "Close ALL open positions now?"):
            return
        try:
            paper = self.mode.get() == "PAPER"
            k, s = creds(paper)
            TradingClient(k, s, paper=paper).close_all_positions(cancel_orders=True)
            self._line("FLATTEN sent", "trade")
        except Exception as exc:
            self._line(f"flatten failed: {exc}", "err")

    def _line(self, msg, lvl="info"):
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {msg}\n", lvl)
        if self.autoscroll.get():
            self.log.see("end")

    def _copy_log(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self.log.get("1.0", "end"))
            self._line("log copied to clipboard", "ok")
        except Exception:
            pass

    def _drain(self):
        try:
            while True:
                m = self.q.get_nowait(); k = m["kind"]
                if k == "log":
                    self.log.insert("end", f"[{m['ts']}] {m['msg']}\n",
                                    m.get("level", "info"))
                    if self.autoscroll.get():
                        self.log.see("end")
                elif k == "session":
                    self.lbl_sess.config(text=(
                        f"baseline ${m['baseline']:,.2f} from {m['start']}   "
                        f"{m['n_trades']} trades [{m.get('source','')}]\n"
                        f"RUNNING EQUITY ${m['equity']:,.2f}   "
                        f"long ${m['equity']*m['frac']:,.0f} / "
                        f"short ${m['equity']*m['frac']*SHORT_EQUITY_MULT:,.0f}"))
                elif k == "plan":
                    if m["in_place"]:
                        self.lbl_next.config(text=f"holding {m['symbol']}",
                                             fg=GU.COL_FLAT)
                    else:
                        sside = GU.strategy_side(m["symbol"], m["side"])
                        self.lbl_next.config(fg=GU.side_colour(sside), text=(
                            f"{sside.upper()} {m['qty']:,} {m['symbol']} "
                            f"@ ~{m['price']:.2f} = ${m['notional']:,.2f}\n  {m['when']}"))
                elif k == "positions":
                    held = {a: b for a, b in m["positions"].items() if abs(b["qty"]) > 0}
                    if not held:
                        self.lbl_pos.config(text="NO OPEN POSITION", fg=GU.COL_FLAT)
                    else:
                        for sym, p in held.items():
                            sside = GU.strategy_side(sym, p["side"])
                            note = (f"  (long {SYMBOL_SHORT} = short {SYMBOL_LONG})"
                                    if sym == SYMBOL_SHORT else "")
                            self.lbl_pos.config(
                                fg=GU.side_colour(sside),
                                text=(f"{sym} {sside.upper()}{note} "
                                      f"qty {abs(p['qty']):,.0f}  "
                                      f"entry {p['avg']:.2f}  last {p['cur']:.2f}\n"
                                      f"  unrealised ${p['upl']:,.2f} ({p['uplpc']:+.2f}%)"))
                elif k == "trade_popup":
                    self._popup(m)
                elif k == "stopped":
                    self.b_start.config(state="normal")
                    self.b_stop.config(state="disabled")
                    self.status.config(text="STOPPED", bg="#374151")
        except queue.Empty:
            pass
        self.after(200, self._drain)

    def _popup(self, m):
        p = m["payload"]
        title = "DRY RUN — not sent" if m["dry"] else f"ORDER SENT — {m['mode']}"
        top = tk.Toplevel(self); top.title(title); top.attributes("-topmost", True)
        bg = "#92400e" if m["dry"] else ("#166534" if m["mode"] == "PAPER" else "#7f1d1d")
        tk.Label(top, text=title, bg=bg, fg="white", font=("Segoe UI", 11, "bold"),
                 padx=16, pady=8).pack(fill="x")
        tk.Label(top, text=m["why"], padx=16, pady=(8, 2)).pack(anchor="w")
        t = tk.Text(top, height=len(p) + 2, width=50, font=("Consolas", 10),
                    bg="#0b1020", fg="#d1d5db")
        t.insert("end", json.dumps(p, indent=2)); t.config(state="disabled")
        t.pack(padx=14, pady=10)
        ttk.Button(top, text="OK", command=top.destroy).pack(pady=(0, 12))
        self._line(f"{'DRY RUN' if m['dry'] else 'SENT'} {json.dumps(p)}", "trade")


# ================================================================= app ======
class App(tk.Tk):
    GEOM_FILE = os.path.join(REPORTS, "gui_geometry.json")

    def __init__(self):
        super().__init__()
        self.title("Supertrend SOXL/SOXS — Backtest + Live")
        self.minsize(900, 560)
        tk.Label(self, bg="#7f1d1d", fg="white", anchor="w", font=("Segoe UI", 8),
                 text=("  RISK: backtest does not reconcile with the Pine run. "
                       "factor is a parameter spike. SOXL/SOXS are 3x leveraged.")
                 ).pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        self.params = ParamsFrame(nb)
        self.bt = BacktestTab(nb, self.params)

        # the Live panel stack is taller than a laptop screen -> scroll it
        live_host = ScrollFrame(nb)
        self.live = LiveTab(live_host.body, self.params)
        self.live.pack(fill="both", expand=True)

        nb.add(self.params, text="  Parameters  ")
        nb.add(self.bt, text="  Backtest  ")
        nb.add(live_host, text="  Live Trading  ")

        self.sb = ttk.Label(self, anchor="w", relief="sunken",
                            font=("Segoe UI", 8), padding=(6, 2),
                            text="ready")
        self.sb.pack(fill="x", side="bottom")
        nb.bind("<<NotebookTabChanged>>", self._tab_changed)

        # keyboard shortcuts
        self.bind_all("<Control-r>", lambda e: self._run_bt())
        self.bind_all("<F5>", lambda e: self._run_bt())
        self.bind_all("<Control-e>", lambda e: self.bt.export())
        self.bind_all("<Control-s>", lambda e: self.params.save())
        self.bind_all("<Control-w>", lambda e: self._close())
        self.bind_all("<F1>", lambda e: self._help())

        self._restore_geometry()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------ helpers ----
    def _run_bt(self):
        self.nb.select(self.bt)
        self.bt.run()

    def _tab_changed(self, _=None):
        tips = {
            "Parameters": "Hover a label for help · Ctrl+S saves a preset",
            "Backtest": "Wheel = zoom X · Ctrl+wheel = zoom Y · drag the divider "
                        "· click a trade to mark it · Ctrl+R run · Ctrl+E export",
            "Live Trading": "ARM off = dry run (popups only) · PAPER default "
                            "· FLATTEN closes everything",
        }
        try:
            name = self.nb.tab(self.nb.select(), "text").strip()
        except Exception:
            return
        self.sb.config(text=tips.get(name, "ready"))

    def _help(self):
        messagebox.showinfo(
            "Shortcuts & chart controls",
            "Ctrl+R / F5   run backtest\n"
            "Ctrl+E        export trades to CSV\n"
            "Ctrl+S        save parameter preset\n"
            "Ctrl+W        close\n"
            "F1            this help\n\n"
            "CHART\n"
            "  wheel            zoom time axis\n"
            "  Ctrl+wheel       zoom value axis\n"
            "  toolbar          pan, box-zoom, save image, home\n"
            "  Reset view       back to the full range\n"
            "  auto-fit Y       rescale value axis to what is on screen\n"
            "  $ / % toggle     equity in dollars or cumulative return\n\n"
            "TRADES\n"
            "  click a heading  sort (click again to reverse)\n"
            "  click a row      highlight that trade on the chart")

    def _restore_geometry(self):
        try:
            g = json.load(open(self.GEOM_FILE)).get("geometry")
            if g and "x" in g:
                self.geometry(g)
                return
        except Exception:
            pass
        w, h = 1200, 880
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 3, 0)}")

    def _close(self):
        try:
            json.dump({"geometry": self.geometry()},
                      open(self.GEOM_FILE, "w"), indent=2)
        except Exception:
            pass
        try:
            if getattr(self.live, "engine", None):
                self.live.engine.stop()
        except Exception:
            pass
        self.destroy()


def _launch():
    print("=" * 62)
    print("  Supertrend App (Backtest + Live) starting...")
    print("=" * 62, flush=True)
    app = App()
    app.update_idletasks()
    app.deiconify(); app.lift(); app.attributes("-topmost", True)
    app.after(800, lambda: app.attributes("-topmost", False))
    try:
        app.focus_force()
    except Exception:
        pass
    app.update()
    print(f"  window open: {app.winfo_geometry()} viewable={bool(app.winfo_viewable())}")
    print("  If not visible, look behind PyCharm or on the taskbar.", flush=True)
    app.mainloop()
    print("  closed.", flush=True)


if __name__ == "__main__":
    try:
        _launch()
    except Exception:
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except Exception:
            pass
