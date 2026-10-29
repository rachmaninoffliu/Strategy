"""Parameter spec, presets, notices and the 'Parameters in use' comparison (what the Backtest uses vs what the Live engine really does).

Everything here is plain Python (no Qt) so it can be unit-tested headless. The Live engine is supertrend_gui.Engine; where it behaves differently from the
backtest (supertrend_strategy.SupertrendStrategy) that is stated in the rows below instead of being hidden - see ENGINE GAPS in describe_parameters()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import supertrend_gui as GU
import supertrend_strategy as S

# Single switch for the team rule "Valid signs off before a strategy is called ready for live". Flip it only after Valid has actually signed off.
VALID_SIGNED_OFF = False
VALID_NOTE = ("Valid has NOT signed off: the Python backtest does not reconcile with the TradingView Pine run (about 1.8x-3.7x apart), factor 5.0 is a parameter spike "
              "(Sharpe 1.52 vs ~0.85 at 4.0 and ~0.93 at 6.0) and SOXL / SOXS are 3x leveraged.")
SUGGESTED_FRAC = 50.0                  # the optional lower sizing the pre-flight offers for LIVE (percent of equity per trade)
LIVE_BAR_MINUTES = GU.BAR_MINUTES
LIVE_WARMUP_BARS = GU.WARMUP_BARS
FACTOR_TESTED = 5.0                    # the value the Pine run and the research notes were built around


@dataclass(frozen=True)
class P:
    """One strategy input. ``kind``: choice | int | float | pct (shown in %, stored as a fraction in Config) | bool | money (0 = none)."""
    attr: str
    label: str
    kind: str
    default: Any
    group: str
    lo: float = 0.0
    hi: float = 1e9
    step: float = 1.0
    dec: int = 2
    suffix: str = ""
    options: tuple = ()
    help: str = ""


MODES = ("Always Fixed", "Always Dynamic", "Auto-Switch via Vol")

SPEC: tuple = (
    P("strategyMode", "Strategy mode", "choice", "Always Fixed", "Signal", options=MODES,
      help="Always Fixed: take profit at a fixed % (Risk/Reward arms the trail, Limit TP closes the trade).\nAlways Dynamic: the target scales with the distance between entry and the Supertrend line.\n"
           "Auto-Switch: Fixed only while ATR% is above its baseline x High-vol mult, otherwise Dynamic (the live engine decides this on the last closed bar when it opens a position).\nOnly Always Fixed was profitable in testing."),
    P("atrPeriod", "ATR period (bars)", "int", 500, "Signal", 2, 2000, 25, 0,
      help="ATR length for the Supertrend, in BARS (not days). 500 bars on 15-min is about 19 trading days. 375-750 all gave Sharpe 1.42-1.52 (a plateau).\nThe live engine loads 4000 bars of history (about 7 ATR lengths at 570), so keep this below ~570 for the live Supertrend to equal the backtest's."),
    P("factor", "Supertrend factor", "float", 5.0, "Signal", 0.5, 20.0, 0.25, 2,
      help="ATR multiplier of the Supertrend bands. FRAGILE: Sharpe 1.52 at 5.0 but about 0.85 at 4.0 and 0.93 at 6.0 - a spike, not a plateau.\nTreat a good result here with suspicion; the Backtest tab's Robustness check shows the neighbours."),
    P("riskReward", "Run-up that arms the trail", "pct", 4.5, "Exits", 0.1, 100.0, 0.5, 2, " %",
      help="'Risk/Reward %' in the Pine: the run-up from entry that ARMS the trailing stop. Until price reaches it there is no stop at all. (Always Fixed / Auto-Switch in low vol.)"),
    P("price_perc", "Limit take-profit", "pct", 10.0, "Exits", 0.5, 100.0, 0.5, 2, " %",
      help="Limit take-profit as a % of the entry reference: +10% for a long, -10% for a short (Pine 'price_perc' = 0.1). (Fixed mode.)"),
    P("trailFactor", "Trailing factor", "float", 0.0, "Exits", 0.0, 1.0, 0.05, 2,
      help="Fraction of the run-up the trail gives back. 0 = the stop sits at the entry price once armed (breakeven).\nWith 'ADX dynamic trail' ON, 0 is rewritten to a 0.05 floor."),
    P("buyback", "Re-entry offset", "pct", 1.5, "Exits", 0.0, 20.0, 0.25, 2, " %",
      help="After an exit the strategy waits for a pullback to Supertrend x (1 +/- this) before re-entering the same trend (a resting limit order). Omitting it collapses returns."),
    P("riskfactor", "Risk factor", "float", 1.0, "Exits", 0.1, 10.0, 0.1, 2,
      help="Scales the measured entry risk |entry - Supertrend|. Only used in Dynamic mode."),
    P("limitfactor", "Limit factor", "float", 1.0, "Exits", 0.1, 10.0, 0.1, 2,
      help="Scales the dynamic take-profit distance. Only used in Dynamic mode."),
    P("useDynamic", "ADX dynamic trail", "bool", False, "ADX dynamic trail",
      help="Modulate the trailing factor by ADX: max(0.05, min(0.95, trailFactor + (ADX base - ADX)/100)).\nTurn OFF to use the trailing factor literally.\nThe live engine computes the same ADX on its own bars (the last closed bar)."),
    P("adxLength", "ADX length", "int", 14, "ADX dynamic trail", 2, 200, 1, 0, help="ADX lookback in bars. Only used when the ADX dynamic trail is on."),
    P("adxBase", "ADX baseline", "float", 25.0, "ADX dynamic trail", 1.0, 100.0, 1.0, 1, help="ADX neutral point. Above it the trail tightens, below it loosens."),
    P("volLength", "Vol ATR length", "int", 20, "Volatility regime", 2, 500, 1, 0, help="ATR length of the volatility-regime measure (Auto-Switch only)."),
    P("volSmaLength", "Vol baseline length", "int", 100, "Volatility regime", 2, 1000, 5, 0, help="Window the current ATR% is compared with (Auto-Switch only)."),
    P("volThreshold", "High-vol multiple", "float", 1.3, "Volatility regime", 0.5, 5.0, 0.05, 2,
      help="High-vol trigger: ATR% > baseline x this switches Auto-Switch into Fixed mode (Auto-Switch only)."),
    P("equity_frac", "Equity per trade", "pct", 100.0, "Sizing & costs", 1.0, 100.0, 5.0, 0, " %",
      help="Percent of RUNNING equity committed per long trade. 100% on a 3x ETF is about 3x effective exposure: the 10-year backtest drew down about -50% with the reference settings."),
    P("short_mult", "Short size multiple", "float", 0.5, "Sizing & costs", 0.1, 1.0, 0.1, 2,
      help="Shorts are sized at 'Equity per trade' x this. 0.5 reproduces the Pine, which hardcodes equity x 0.5 for shorts while longs use the full fraction."),
    P("commission_bps", "Commission", "float", 5.0, "Sizing & costs", 0.0, 100.0, 1.0, 1, " bp/leg",
      help="Backtest cost per leg in basis points (slippage proxy; Alpaca equities are commission-free). Your TradingView run used 0, which is worth ~10 points of CAGR.\nAlso used by the SIMULATED session history (while the account has no fills), whose running equity sizes the next live order. Real fills never pay it."),
    P("touch_fill", "Touch fills (TradingView)", "bool", False, "Sizing & costs",
      help="ON reproduces TradingView: a limit order fills when price merely TOUCHES it. OFF requires the bar to trade THROUGH the level - stricter and more realistic. (Backtest only.)"),
    P("pessimistic", "Pessimistic intrabar order", "bool", False, "Sizing & costs",
      help="If one bar could hit BOTH the take-profit and the trailing stop, assume the STOP filled first. Conservative; costs some return.\nUsed by the backtest and by the SIMULATED session history (while the account has no fills); real fills are whatever the market gives."),
    P("capacity_usd", "Max $ per trade", "money", 0.0, "Sizing & costs", 0.0, 1e12, 100_000.0, 0, " $", help="Caps the notional of one backtest trade (liquidity realism). 0 = no cap, as in the reference runs. (Backtest only.)"),
)
BY_ATTR = {p.attr: p for p in SPEC}
GROUPS = ("Signal", "Exits", "ADX dynamic trail", "Volatility regime", "Sizing & costs")
DEFAULTS: dict = {p.attr: p.default for p in SPEC}
BACKTEST_ONLY = ("touch_fill", "capacity_usd", "commission_bps")          # never applied to a real fill (commission_bps and pessimistic also feed the simulated session history; their help says so)

REFERENCE_PRESET = "Reference · Always Fixed 500 / 5.0 (default)"
PRESETS: dict = {
    REFERENCE_PRESET: {},
    "Half size · 50% of equity per trade": {"equity_frac": 50.0},
    "Cautious · 25% of equity per trade": {"equity_frac": 25.0},
    "TradingView match · touch fills, no commission": {"touch_fill": True, "commission_bps": 0.0},
    "Stress costs · 10 bp/leg, stop-first": {"commission_bps": 10.0, "pessimistic": True},
    "Neighbour check · factor 4.0": {"factor": 4.0},
    "Neighbour check · factor 6.0": {"factor": 6.0},
}
CUSTOM = "Custom"


def preset_values(name: str) -> Optional[dict]:
    """Full value dict of a preset (defaults + its overrides), or None for 'Custom' / an unknown name."""
    if name not in PRESETS:
        return None
    return {**DEFAULTS, **PRESETS[name]}


def to_config(values: dict, **extra) -> S.Config:
    """UI values -> the strategy's Config (percent fields become fractions; capacity 0 = unlimited)."""
    kw: dict = {}
    for p in SPEC:
        v = values.get(p.attr, p.default)
        if p.kind == "bool":
            kw[p.attr] = bool(v)
        elif p.kind == "choice":
            kw[p.attr] = str(v)
        elif p.kind == "int":
            kw[p.attr] = int(round(float(v)))
        elif p.kind == "pct":
            kw[p.attr] = float(v) / 100.0
        elif p.kind == "money":
            kw[p.attr] = float(v) if float(v) > 0 else 1e18
        else:
            kw[p.attr] = float(v)
    kw.update(extra)
    kw.setdefault("useDateFilter", False)
    return S.Config(**kw)


def used_flags(values: dict) -> dict:
    """Which inputs the chosen mode actually reads (the panel greys out the rest so nobody tunes a dead control)."""
    mode = values.get("strategyMode", "Always Fixed")
    fixed, dyn, auto = mode == "Always Fixed", mode == "Always Dynamic", mode == "Auto-Switch via Vol"
    use = {p.attr: True for p in SPEC}
    use["riskReward"] = use["price_perc"] = fixed or auto
    use["riskfactor"] = use["limitfactor"] = dyn or auto
    use["volLength"] = use["volSmaLength"] = use["volThreshold"] = auto
    adx = bool(values.get("useDynamic"))
    use["adxLength"] = use["adxBase"] = adx
    return use


ATR_CONVERGE = 7        # a Wilder ATR forgets its seed after ~7 lengths: the live window should hold that many ATR periods to give the backtest's full-history value


def atr_fits_live(values: dict) -> bool:
    """Does the live window (LIVE_WARMUP_BARS) hold ~7 ATR periods, so the live Supertrend equals the backtest's full-history one?"""
    return int(values.get("atrPeriod", 500)) * ATR_CONVERGE <= LIVE_WARMUP_BARS


def vol_regime_fits_live(values: dict) -> bool:
    """Does the live engine's warm-up window hold enough bars for the volatility regime (vol ATR length + baseline length, same 50-bar margin as the ATR period)?"""
    return int(values.get("volLength", 20)) + int(values.get("volSmaLength", 100)) <= LIVE_WARMUP_BARS - 50


def notices(values: dict) -> list[str]:
    """Plain-English warnings about settings that move away from the tested reference or that backtest and live treat differently (nothing for the reference settings)."""
    out = []
    mode = values.get("strategyMode")
    if mode != "Always Fixed":
        out.append(f"mode '{mode}': only Always Fixed was profitable in testing")
    if mode == "Auto-Switch via Vol" and not vol_regime_fits_live(values):
        out.append(f"the volatility regime needs {int(values['volLength']) + int(values['volSmaLength'])} bars but the live engine loads {LIVE_WARMUP_BARS}: live would never see a high-vol regime and would run Auto-Switch as Always Dynamic")
    if abs(float(values.get("factor", 5.0)) - FACTOR_TESTED) > 1e-9:
        out.append(f"factor {values.get('factor'):g} is not the tested 5.0 (results move a lot between neighbouring factors)")
    if values.get("useDynamic"):
        out.append("ADX dynamic trail is ON: not part of the tested reference settings (the live engine reads the ADX of the last closed bar)")
    if int(values.get("atrPeriod", 500)) > LIVE_WARMUP_BARS - 50:
        out.append(f"ATR period {int(values['atrPeriod'])} leaves the live engine too few of its {LIVE_WARMUP_BARS} warm-up bars: it would never produce a signal")
    elif not atr_fits_live(values):
        out.append(f"ATR period {int(values['atrPeriod'])} needs about {int(values['atrPeriod']) * ATR_CONVERGE:,} bars of warm-up to converge but the live engine loads {LIVE_WARMUP_BARS}: "
                   "the live Supertrend can differ slightly from the backtest's")
    if float(values.get("commission_bps", 5.0)) < 5.0 - 1e-9:
        out.append(f"commission {values['commission_bps']:g} bp/leg is below the 5 bp reference (optimistic costs)")
    if values.get("touch_fill"):
        out.append("touch fills are ON (TradingView-style): a limit order counts as filled when price only touches it")
    if values.get("pessimistic"):
        out.append("pessimistic intrabar ordering is ON (stricter than the reference: stop before target)")
    if float(values.get("capacity_usd", 0.0)) > 0:
        out.append(f"trade size is capped at ${float(values['capacity_usd']):,.0f} in the backtest only")
    return out


# ── 'Parameters in use' ───────────────────────────────────────────────────────────
@dataclass
class Row:
    group: str
    name: str
    backtest: str
    live: str
    kind: str                       # shared | differs | backtest-only | live-only | not-used
    note: str = ""


@dataclass
class LiveSettings:
    capital: float = 100_000.0
    start_date: str = ""
    paper: bool = True
    armed: bool = False
    prefer_short: bool = True
    daily_loss_pct: float = 8.0
    max_trades: int = 12


@dataclass
class BacktestSettings:
    symbol: str = "SOXL"
    tf: int = 15
    start: str = ""
    end: str = ""
    capital: float = 100_000.0
    prime: bool = True
    benchmark: bool = True
    strict: bool = True        # live-accurate entries: only what a live order can know (see SupertrendStrategy strict_entries)


def _fmt(v: Any, p: P) -> str:
    if p.kind == "bool":
        return "ON" if v else "OFF"
    if p.kind == "money":
        return "no cap" if float(v) <= 0 else f"${float(v):,.0f}"
    if p.kind == "int":
        return f"{int(round(float(v)))}"
    if p.kind == "pct":
        return f"{float(v):g}%"
    if p.kind == "choice":
        return str(v)
    return f"{float(v):g}"


def describe_parameters(values: dict, bt: Optional[BacktestSettings], lv: LiveSettings, mode_label: str, bt_error: str = "") -> list[Row]:
    """Every parameter with the value the Backtest uses next to the value the Live engine uses, and whether they agree.

    ENGINE GAPS (verified against supertrend_gui.Engine, not assumed):
      * ADX dynamic trail and Auto-Switch are no longer gaps: the Engine computes ADX(adxLength) and the volatility regime on the same last CLOSED bar as the entry signal and
        passes them to trail_level() / record_entry(), the formulas of SupertrendStrategy.run (test_supertrend_engine_parity.py replays both against the backtest). What remains is
        timing: the live ADX is the last closed bar's, so mid-bar the trail is one bar behind the backtest's, and a mode is decided when the order is sent, not on the fill bar;
      * the simulated session history (used while the account has no fills) now uses every strategy input of the user's settings, including commission and the pessimistic
        ordering; touch fills stay OFF and the size cap off (it stands in for real fills);
      * the Engine only freezes exit levels (and so the mode) for MARKET entries - reversals and SOXS trigger entries; a SOXL buyback LIMIT entry gets none;
      * the backtest models a SOXL short; live holds SOXS long when SOXL is not shortable (two separately rebalanced 3x funds);
      * live exits: SOXL gets resting broker orders, SOXS exits are polled by this app and sent at market (nothing rests at the broker)."""
    rows: list[Row] = []

    def add(group: str, name: str, backtest: str, live: str, kind: str, note: str = "") -> None:
        rows.append(Row(group, name, backtest, live, kind, note))

    mode = values["strategyMode"]
    auto = mode == "Auto-Switch via Vol"
    use = used_flags(values)

    g = "Instrument and data"
    if bt is not None:
        same_tf = bt.tf == LIVE_BAR_MINUTES
        add(g, "Signal instrument", bt.symbol, GU.SYMBOL_LONG, "shared" if bt.symbol == GU.SYMBOL_LONG else "differs", "The live engine always reads SOXL bars." if bt.symbol != GU.SYMBOL_LONG else "")
        add(g, "Bar size", f"{bt.tf} min", f"{LIVE_BAR_MINUTES} min", "shared" if same_tf else "differs", "" if same_tf else "The live engine is fixed to 15-minute bars.")
    else:
        add(g, "Signal instrument", "—", GU.SYMBOL_LONG, "live-only", bt_error)
    add(g, "Price adjustment", "splits only (like TradingView)", "splits only", "shared")
    add(g, "Entry signal bar", "bar close", "last CLOSED bar", "shared", "A still-forming bar is ignored for entries (it can repaint); exits watch the forming bar so an intrabar touch is caught.")
    add(g, "Indicator warm-up", "bars before the start date" if (bt and bt.prime) else "none (indicators start at the first bar)", f"{LIVE_WARMUP_BARS} bars loaded each poll",
        "backtest-only" if bt else "live-only", f"The live window should hold about {ATR_CONVERGE} ATR lengths ({int(values['atrPeriod']) * ATR_CONVERGE:,} bars for ATR {int(values['atrPeriod'])}).")

    g = "Signal"
    vol_fits = vol_regime_fits_live(values)
    if auto and vol_fits:
        add(g, "Strategy mode", mode, f"{mode} (regime read on the last closed bar)", "shared",
            "Fixed while ATR% is above its baseline x the multiple, else Dynamic - decided from the last closed bar when the position opens.")
    elif auto:
        add(g, "Strategy mode", mode, "Always Dynamic (regime needs more bars than the live engine loads)", "differs",
            f"The volatility regime needs {int(values['volLength']) + int(values['volSmaLength'])} bars; the live engine loads {LIVE_WARMUP_BARS}, so it never sees a high-vol regime.")
    else:
        add(g, "Strategy mode", mode, mode, "shared")
    atr_note = ("Too long for the live warm-up: no live signal." if int(values["atrPeriod"]) > LIVE_WARMUP_BARS - 50 else
                f"The live window ({LIVE_WARMUP_BARS} bars) holds fewer than {ATR_CONVERGE} ATR lengths: its Supertrend can differ slightly from the backtest's full-history one."
                if not atr_fits_live(values) else "")
    add(g, "ATR period", f"{int(values['atrPeriod'])} bars", f"{int(values['atrPeriod'])} bars", "shared" if atr_fits_live(values) else "differs", atr_note)
    add(g, "Supertrend factor", f"{values['factor']:g}", f"{values['factor']:g}", "shared", "Spike value: Sharpe 1.52 at 5.0 vs ~0.9 at 4.0 / 6.0." if abs(values["factor"] - FACTOR_TESTED) < 1e-9 else "")

    if bt is not None:
        add(g, "Entries use only what a live order knows", "yes - live-accurate" if bt.strict else "NO - classic: a flat re-entry may use the entry bar's own close",
            "yes - the last closed bar and the current quote", "shared" if bt.strict else "differs",
            "" if bt.strict else "About 13% of the classic backtest's trades have no live equivalent, and they were unusually good (+2.35% on average). Tick 'Live-accurate entries' in the Backtest tab.")

    g = "Exits"
    for a in ("riskReward", "price_perc", "trailFactor", "buyback", "riskfactor", "limitfactor"):
        p = BY_ATTR[a]
        note = "" if use[a] else f"Not read in '{mode}' mode."
        add(g, p.label, _fmt(values[a], p), _fmt(values[a], p), "shared" if use[a] else "not-used", note)
    dyn_on = bool(values["useDynamic"])
    adx_txt = f"ON: ADX({int(values['adxLength'])}) vs baseline {values['adxBase']:g}"
    add(g, "ADX dynamic trail", adx_txt if dyn_on else "OFF", adx_txt + ", last closed bar" if dyn_on else "OFF", "shared",
        "Same formula. The live engine reads the ADX of the last CLOSED bar, so while a bar is still forming its trail is one bar behind the backtest's (which tests a bar with that bar's own ADX)." if dyn_on else "")
    for a in ("adxLength", "adxBase"):
        p = BY_ATTR[a]
        add(g, p.label, _fmt(values[a], p) if dyn_on else "—", _fmt(values[a], p) if dyn_on else "not used", "shared" if dyn_on else "not-used")
    for a in ("volLength", "volSmaLength", "volThreshold"):
        p = BY_ATTR[a]
        add(g, p.label, _fmt(values[a], p) if auto else "—", _fmt(values[a], p) if auto else "not used", "shared" if auto else "not-used")
    add(g, "Exit execution", "exact level, instant", "SOXL: resting TP (+OCO stop once armed) at the broker; SOXS: polled each minute, closed at market", "differs",
        "Live fills at the next poll / actual market, not at the exact level.")
    add(g, "Same-bar TP and stop", "stop first" if values["pessimistic"] else "target first", "whichever the market fills first", "backtest-only" if values["pessimistic"] else "differs")

    g = "Sizing and costs"
    p = BY_ATTR["equity_frac"]
    add(g, p.label, _fmt(values["equity_frac"], p), _fmt(values["equity_frac"], p), "shared", "Of RUNNING equity (capital compounded by the session's P&L), not of the account balance.")
    p = BY_ATTR["short_mult"]
    add(g, p.label, _fmt(values["short_mult"], p), _fmt(values["short_mult"], p), "shared")
    if bt is not None:
        same_cap = abs(bt.capital - lv.capital) < 0.5
        add(g, "Initial capital", f"${bt.capital:,.0f}", f"${lv.capital:,.0f}", "shared" if same_cap else "differs", "Each tab has its own box." if not same_cap else "")
    else:
        add(g, "Initial capital", "—", f"${lv.capital:,.0f}", "live-only")
    add(g, "Commission / slippage", f"{values['commission_bps']:g} bp per leg", f"real fills (Alpaca: $0 commission); the simulated history uses your {values['commission_bps']:g} bp", "differs",
        "A backtest cost is a slippage proxy; the live result contains the real slippage.")
    add(g, "Touch fills", "ON" if values["touch_fill"] else "OFF", "n/a - real market", "backtest-only")
    add(g, "Max $ per trade", _fmt(values["capacity_usd"], BY_ATTR["capacity_usd"]), "no cap (size = budget / price)", "backtest-only" if values["capacity_usd"] > 0 else "shared")
    short_route = "SOXL short when shortable, else SOXS long" if lv.prefer_short else "always SOXS long"
    add(g, "How a short is held", "short SOXL (P&L = minus the SOXL move)", short_route, "differs",
        "SOXS is a separately rebalanced 3x fund: it does not mirror SOXL exactly, and its exits are polled (no broker-side protection).")

    g = "Live session"
    add(g, "Start date (P&L baseline)", "—", lv.start_date or "—", "live-only", "Each poll re-simulates from here while the account has no fills: keep it recent.")
    add(g, "Simulated history", "—", "while the account has no fills: the backtest engine with your inputs (touch fills OFF, no size cap)", "live-only",
        "Its running equity sizes the next live order. Commission and the pessimistic ordering above are applied to it.")
    add(g, "Account", "—", mode_label, "live-only")
    add(g, "Orders", "—", "SENT to the broker" if lv.armed else "DRY RUN: shown, not sent", "live-only")
    add(g, "Daily loss limit", "not applied", f"-{lv.daily_loss_pct:g}% of the day's start equity pauses new entries", "live-only", "Open positions keep their exits.")
    add(g, "Max trades per day", "not applied", str(lv.max_trades), "live-only")
    add(g, "Trading hours", "regular session bars only", "orders only while the market is open", "shared")
    add(g, "Poll interval", "—", f"{GU.POLL_SECONDS} s", "live-only")

    if bt is not None:
        g = "Backtest range"
        add(g, "From / to", f"{bt.start} → {bt.end}", "—", "backtest-only")
        add(g, "Buy and hold benchmark", "shown" if bt.benchmark else "hidden", "—", "backtest-only")
    return rows


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str
    key: str = ""


@dataclass
class Report:
    created: float = 0.0
    live: bool = False
    armed: bool = False
    capital: float = 0.0
    checks: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    suggested_frac: Optional[float] = None
    account: dict = field(default_factory=dict)
    path: str = ""

    @property
    def verdict(self) -> str:
        sts = {c.status for c in self.checks}
        return "NO-GO" if "FAIL" in sts else "CAUTION" if "WARN" in sts else "GO"

    def counts(self) -> dict:
        d = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0, "SKIP": 0}
        for c in self.checks:
            d[c.status] = d.get(c.status, 0) + 1
        return d

    def headline(self) -> str:
        c = self.counts()
        what = "LIVE (real money)" if self.live and self.armed else "PAPER orders" if self.armed else "a DRY RUN"
        base = {"GO": f"GO — every check passed for {what}.", "CAUTION": f"CAUTION — {c['WARN']} warning(s) for {what}; read them before you continue.",
                "NO-GO": f"NO-GO — {c['FAIL']} check(s) failed for {what}; fix them first."}[self.verdict]
        return base

    def text(self) -> str:
        from datetime import datetime, timezone
        lines = [f"Supertrend pre-flight {datetime.fromtimestamp(self.created, timezone.utc):%Y-%m-%d %H:%M:%S} UTC", self.headline(), ""]
        for c in self.checks:
            lines.append(f"[{c.status:4}] {c.group} · {c.name}: {c.detail}")
        lines += ["", *self.notes]
        return "\n".join(lines)
