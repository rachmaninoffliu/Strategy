"""'How it works' page: the strategy, what every tab does, what is shared and what is not, the safety rules and the known gaps. Everything stated here was checked against the code."""
from __future__ import annotations

from PyQt6.QtWidgets import QTextBrowser

from .theme import BORDER, PANEL

HELP_HTML = """
<h2>What this app is</h2>
<p>A control panel for the <b>Main Supertrend Strategy – SOXL chart / SOXS short alerts</b> (a Python port of your TradingView Pine script). One window, one set of settings, three jobs:
<b>Backtest</b> it over history, <b>watch</b> what it sees right now, and <b>run</b> it against Alpaca (dry run, paper or live) with the very same parameters. The left panel is shared, so
<i>what you test is what trades</i> - except in the few places listed under "Known gaps" below, which the tab <b>Parameters in use</b> shows live.</p>

<h2>The strategy in plain words</h2>
<ol>
<li><b>Trend.</b> A Supertrend (ATR period 500 bars - about 19 sessions of 15-minute bars - times a factor of 5.0) draws a line under the price in an uptrend and above it in a downtrend.
Close above the line = <b>long SOXL</b>; close below = <b>short SOXL</b> (the Pine routes shorts to the inverse fund SOXS).</li>
<li><b>Two ways in.</b> <i>Reversal:</i> holding the opposite side and the trend flips → a market order at the next open. <i>Re-entry ("buyback"):</i> flat but the trend is intact (e.g. after a take-profit)
→ a resting limit at Supertrend × (1 ± 1.5%), so it gets back in on a pullback instead of chasing. Without the re-entry path returns collapse.
<b>The live rule:</b> every poll the engine compares the <i>current</i> SOXL price with that level - at it or better (a long: at or below · a short: at or above) → it enters now,
otherwise it waits and the level is re-priced as the Supertrend moves. For SOXL that is the resting limit itself (a limit already "in the money" fills at once); on the SOXS route the app polls and sends a market order.</li>
<li><b>Exits.</b> A <b>limit take-profit</b> at +10% (long) / −10% (short) from the entry reference. A <b>trailing stop</b> that is <i>armed only after a 4.5% run-up</i>; once armed it sits at
entry + (best price − entry) × trailing factor (0 = breakeven) and never loosens. Otherwise the trade ends when the trend flips.</li>
<li><b>Size.</b> A percentage of <i>running</i> equity per trade (default 100% - on a 3x ETF that is roughly 3x exposure). Shorts are sized at 0.5x, as in the Pine.</li>
</ol>
<p><b>Honest status:</b> Valid has <b>not</b> signed off. The Python backtest does not reconcile with your TradingView run (about 1.8x-3.7x apart depending on sizing), the factor of 5.0 is a spike
(Sharpe about 1.5 there against about 0.9 at 4.0 and 6.0) and SOXL / SOXS are 3x leveraged. Use the <b>Robustness check</b> in the Backtest tab to see how fragile any setting is. All results are in-sample.</p>

<h2>The tabs</h2>
<h3>Live trading</h3>
<ul>
<li><b>Connect and test</b> - read-only: keys, account status, market clock, whether SOXL is shortable, open positions / orders. Sends nothing. Keys are read from <code>~/.claude.json</code>
(<code>alpaca-paper</code> / <code>alpaca-live</code>). The app <b>always starts on Paper + DRY RUN</b>.</li>
<li><b>Orders: DRY RUN / ARMED.</b> A dry run does everything except send: each order the engine <i>would</i> send is listed once in the <b>Orders</b> tab. ARMED sends to the account you picked;
a PAPER session asks for a yes, a LIVE session asks you to type <code>TRADE REAL MONEY</code>.</li>
<li><b>Pre-flight check</b> (also runs when you press Start, and by itself at 09:10 ET on market days if the app is open) - never sends an order. It checks keys and endpoint, account status and blocks, capital vs equity,
buying power, the pattern-day-trader rule, market clock, SOXL / SOXS tradability and the short route, the live data feed (the engine's own fetch), warm-up bars, the signal now, one order's size,
whether the account holds a position this app is not managing, the KILL latch, how the last session ended, your risk limits and settings, a recent backtest, and Valid's sign-off.
<b>GO</b> = all passed · <b>CAUTION</b> = advice, never a gate · <b>NO-GO</b> = armed sessions are blocked (a dry run is never blocked).</li>
<li><b>Starting capital</b> is the session's starting equity; it compounds with the session's P&amp;L and that running figure - not the raw balance - sizes each order. <b>Start date</b> is the P&amp;L baseline.
While the account has no fills since then, the engine <i>simulates</i> the strategy from that date on every poll to know its equity (the tab says "SIMULATED"): keep the start date recent.</li>
<li><b>Short route.</b> A short is held as a direct SOXL short when the broker allows it, else as a long SOXS position. SOXL positions have their take-profit (and, once armed, an OCO stop) resting <i>at the broker</i>.
SOXS exits are <i>polled by this app</i> every minute and sent at market: if the app stops, a SOXS position is unmanaged.</li>
<li><b>Stop</b> ends the engine only; positions stay open. <b>Flatten</b> stops the engine first (it would re-enter on its next poll otherwise), cancels all orders and closes everything.
<b>KILL</b> = Flatten plus a latch file that blocks any restart until you press <b>Clear kill latch</b>.</li>
<li><b>Guard rails in the engine:</b> orders only while the market is open · new entries pause after the daily loss limit (default -8% on the day) · at most 12 orders a day · signals use the last <i>closed</i> bar.</li>
<li>The performance view below is the same one the Backtest uses (KPIs, curve, metrics, trades, monthly, by side); the session's trades come from the broker's fills or the simulation.</li>
<li><b>Open position strip</b> (above the tabs): what the account holds - side, size, entry, now, P&amp;L, take-profit, run-up target and trailing stop (all SOXL levels, also when a short is held as SOXS) - and, under it,
what the <i>live-accurate backtest</i> holds on the very same bars, with a verdict: <b>MATCHES</b> (same side, same frozen levels within 0.5%; size is not compared because each side compounds its own equity) ·
<b>DIFFERS</b> · <b>TIMING</b> (the current 15-minute bar is still forming: a reversal, fill or exit inside it reaches the backtest at its close) · <b>NOT TRACKED</b> (the account holds a position this app did not open, so there are no frozen levels) ·
<b>DRY RUN</b> (nothing is sent, so the account stays flat).</li>
</ul>
<h3>Backtest</h3>
<ul>
<li><b>Live-accurate entries</b> (default ON). The classic backtest lets a flat re-entry use the entry bar's <i>own close</i> (trend, volatility regime, ADX) - information a live order does not have when it is placed.
About 5% of its trades could not have been taken live, and they were unusually good (+3.3% on average). With the box ticked the backtest only uses what a live order could know (the previous bar's close) and the live engine reproduces it trade for trade;
untick it to see the classic numbers. The two give different trade counts and returns.</li>
<li><b>Open position</b> (strip above the tabs): the position the backtest still holds after its last bar, with its frozen take-profit, run-up target and trailing stop - or "FLAT". The Live tab shows the same strip for the account, next to this one.</li>
<li><b>Verify live = backtest:</b> replays the <i>real</i> live engine bar by bar (two polls per bar) against a simulated broker over the last 1-12 months of the window and compares every trade, and the position still open at the end, with this backtest.
It runs in a separate process (about 25 s per month), sends nothing anywhere, and ends "IDENTICAL" or lists the differences. Shorts "as long SOXS" shows the SOXS route, whose exits are polled once a minute and so differ slightly by design.</li>
<li>Pick a window (quick buttons: 1Y / 3Y / 5Y / YTD / All, <b>→ latest</b> follows the newest data), capital and run (<b>Ctrl+R</b>). The strategy runs on cached 15-minute bars (split-adjusted, regular hours, like TradingView);
<b>Refresh data first</b> tops the cache up from Alpaca when it is behind. If that fails the cached bars are used and the log says so.</li>
<li><b>Prime indicators with bars before the start</b> (default on): a late start date no longer wastes the first 500 bars - the bars before it warm the Supertrend and no trade is taken before the date, like the live engine,
which always starts warm. A full-history run from 2016 is identical either way.</li>
<li><b>Robustness check:</b> 15 re-runs around the current factor and ATR period; says <i>plateau</i> (neighbours hold up) or <i>spike</i> (only good at the exact setting).</li>
<li><b>Sharpe appears twice:</b> daily (the standard) and per bar (the basis of the earlier research notes, so you can reconcile). Profit factor likewise ($ basis and the sum-of-% basis).</li>
</ul>
<h3>Signal monitor</h3>
<p>Price bars, the Supertrend line (green = long, red = short), the re-entry limit and - if the app holds a position - its frozen take-profit, run-up target and trailing stop, all in SOXL prices. Read-only.</p>

<h2>Using the charts and tables</h2>
<ul>
<li><b>Equity chart:</b> mouse wheel zooms around the cursor (<b>Shift+wheel</b> = value axis only), drag pans, double-click resets, hover a dot for a trade, click it to pin and select its table row.
The toolbar switches $ / % and log / linear, and hides the buy &amp; hold line or the dots.</li>
<li><b>Trades table:</b> click a heading to sort, click a row to shade that trade on the curve, double-click to zoom to it. <b>Export trades to CSV</b> saves the table.</li>
<li><b>Monthly</b> is a year x month heat-map; <b>By side and exit</b> shows where the money is made (take-profits) and lost (reversals).</li>
<li>Hover any label in the settings panel for an explanation. Inputs the chosen mode does not read are greyed out. Spin boxes ignore the mouse wheel until you click into them, so scrolling the panel never changes a setting.</li>
<li>Shortcuts: <b>Ctrl+R</b> / <b>F5</b> run backtest · <b>Ctrl+E</b> export trades · <b>Ctrl+L</b> pre-flight · <b>Ctrl+1..4</b> tabs · <b>Ctrl+S</b> save settings to a file · <b>Ctrl+O</b> load · <b>F1</b> this page.</li>
</ul>

<h2>Known gaps between Backtest and Live (also shown in "Parameters in use")</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Item</th><th>Backtest</th><th>Live engine</th></tr>
<tr><td>ADX dynamic trail</td><td>ADX(length) of the bar being tested</td><td>the same ADX, taken from the last <i>closed</i> bar: same formula, but while a bar is still forming the live trail is one bar behind</td></tr>
<tr><td>Auto-Switch via Vol</td><td>Fixed while ATR% &gt; baseline x multiple, else Dynamic, decided when the position opens</td><td>the same rule, read on the last closed bar when the order is sent (needs vol ATR length + baseline length bars; the engine loads 4000)</td></tr>
<tr><td>Which entries are tracked</td><td>every entry gets its exit levels and mode</td><td>every entry too: a market entry is frozen when it is sent, a SOXL buyback <i>limit</i> when the poll sees it filled (reference = the open of the bar it filled in, exits count from the next bar)</td></tr>
<tr><td>Look-ahead</td><td>"Live-accurate entries" ON: none (ticked by default) · OFF: the classic backtest uses the entry bar's own close</td><td>only what exists at the time: the last <i>closed</i> bar plus the current quote</td></tr>
<tr><td>A short</td><td>a SOXL short</td><td>a SOXL short, or long SOXS (a separate 3x fund)</td></tr>
<tr><td>Exits</td><td>fill exactly at the level</td><td>SOXL: resting orders at the broker · SOXS: polled each minute, market order</td></tr>
<tr><td>Costs</td><td>basis points per leg (a slippage proxy)</td><td>real fills; the simulated history uses your basis-point setting</td></tr>
<tr><td>Risk limits</td><td>none</td><td>daily loss pause, order cap, market hours</td></tr>
<tr><td>Simulated history</td><td colspan="2">while the account has no fills it runs the backtest engine with your inputs (risk / limit factor, ADX, volatility, commission, pessimistic order); touch fills stay off and the size cap is off</td></tr>
</table>

<h2>Files</h2>
<ul>
<li><code>reports/live_log.jsonl</code> - every engine log line · <code>reports/preflight_*.txt</code> - saved pre-flight reports · <code>reports/gui_state.json</code> - the open position's frozen exit levels ·
<code>reports/KILL</code> - the kill latch · <code>data/SOXL_15m_2016_split.parquet</code> - the bar cache.</li>
<li>The previous Tkinter window is kept as <code>supertrend_app_tk.py</code>; the trading engine itself is <code>supertrend_gui.py</code> and the strategy <code>supertrend_strategy.py</code>
(its <code>strict_entries</code> option is the live-accurate backtest). <code>python -m supertrend_ui.replay --months 3</code> runs the live-vs-backtest replay from a terminal.</li>
</ul>
"""


class HelpTab(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(False)
        self.setHtml(HELP_HTML)
        self.setStyleSheet(f"QTextBrowser {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 12px; font-size: 13px; }}")
