"""How it works."""
from __future__ import annotations

from PyQt6.QtWidgets import QTextBrowser

HTML = """
<h2>Leveraged-ETF pairs - statistical arbitrage on intraday bars</h2>
<p>The same construction as the Stat-Arb app (Avellaneda &amp; Lee 2008), with one factor: each <b>3x ETF</b> is regressed on its <b>1x ETF</b> - <b>TQQQ/QQQ, SOXL/SMH, SPXL/SPY</b>. A daily-reset 3x fund is three times
its index <i>since the previous close</i>, so  <code>X<sub>t</sub> = (L<sub>t</sub>/L<sub>close</sub> - 1) - &beta; (B<sub>t</sub>/B<sub>close</sub> - 1)</code>  (&beta; = 3) is the fund's premium or discount to fair value. X is fitted as an
Ornstein-Uhlenbeck process (AR(1): <code>X<sub>n+1</sub> = a + b X<sub>n</sub> + &zeta;</code>) over a rolling window of sessions, giving the reversion half-life, the equilibrium level m and the equilibrium sd &sigma;<sub>eq</sub>;
the <b>s-score</b> <code>s = (X - m)/&sigma;<sub>eq</sub></code> says how many sd the pair sits from fair value.</p>
<p><b>Rules:</b> long the spread (buy $1 of the 3x ETF, short $&beta; of the 1x ETF) when s &lt; -open, short the spread when s &gt; +open; take profit when s is back inside the close levels; stop if |s| &gt; stop; time stop;
no entry in the first / last minutes; <b>everything is flat before the close</b> - nothing is held overnight. Any bar size (1 / 5 / 15 / 30 minutes) runs the same rules, all times are in minutes.</p>

<h3>The part that matters: how the order fills</h3>
<ul>
<li><b>next_open (default):</b> the signal is computed from the closes of bar t; the order fills at the <i>open of bar t+1</i> (a few seconds later in real life). Costs are the quoted half-spread of each leg (sampled from SIP quotes) plus a slippage allowance,
on <i>both</i> legs - the hedge leg is &beta; = 3 times larger.</li>
<li><b>mid:</b> half way between the signal bar's last trade and the next open (-0.7 Sharpe on the default preset): what a very fast order might get. Not the default, shown so the size of the fill assumption is visible.</li>
<li><b>close (optimistic):</b> the order fills at the last trade of the signal bar. The last prints of two different ETFs are stale or asynchronous by up to seconds, sit at the bid or the ask, and the entries are picked at the extremes of that noise; the spread snaps back in the next seconds and a backtest that fills at the signal bar's close collects the snap-back (8-12 bp per trade, more than the quoted bounce alone can explain) as profit.
On 5-minute bars it turns a Sharpe of about -5 into about +2.9 (1-minute: -8.5 into +5.9). It is in the app so you can see the size of the bias - not something an order can do.</li>
<li>The decisions never depend on the fill, so all three models trade exactly the same round trips; only the price differs. The <b>Fill models</b> button in the Backtest tab shows the three side by side and how far the spread moves between the signal bar's close and the next open (about 10 bp a trip on the default preset in total). P&amp;L is
accounted with fixed share counts from the fill prices, as a real account earns it, so every row of the Trades tab can be reproduced from its own entry and exit prices.</li>
</ul>

<h3>What the research found (LEVPAIRS_REPORT.md)</h3>
<ul>
<li>1,458 rule sets on 1 / 5 / 15 / 30-minute bars, ranked on 2017-2021 only. With next-open fills <b>every one lost money</b>, in and out of sample. The best in-sample Sharpe is about -0.9.</li>
<li>With realistic fills the gross reversion is under 1 bp per trade at 1-15 minutes and about zero at 30 (a Sharpe of about 4 on 1-minute bars if trading were free, 1.6 on 5-minute bars), while a round trip costs about 4.6 bp (TQQQ/QQQ), 5.7 (SPXL/SPY) and 11 (SOXL/SMH). Even with zero slippage (only the half-spread) the 5-minute default loses.</li>
<li>So the honest answer to "which time frame is best" is: none clears the costs for a retail order that crosses the spread. Shorter bars have the most gross edge and the most noise; longer bars have no edge. The Time frames tab shows the table.</li>
<li>What could still work is passive execution (earning the spread instead of paying it), which a bar-based backtest cannot validate. The Live tab's DRY RUN / PAPER mode records real fill costs so that can be measured.</li>
</ul>

<h3>Tabs</h3>
<ul>
<li><b>Live trading</b> - connect to Alpaca (paper or live), pre-flight, start the session engine (DRY RUN keeps a shadow book; ARMED sends market orders for both legs, confirms the fills, unwinds a failed leg), replay the last cached
session through the same decision path, reconcile with the broker, KILL.</li>
<li><b>Backtest</b> - any window with an in-sample / out-of-sample split; equity vs SPY, yearly table, per-pair table, a one-session spread viewer, <b>Time frames</b>, <b>Fill models</b>, <b>Ablation</b> (including the paper's rules, mid and optimistic fills and zero cost),
and <b>cost / threshold neighbourhood</b> checks. The <b>Trades</b> tab has one row per round trip: <i>You hold</i> says which ETF is LONG and which SHORT ("LONG TQQQ / SHORT QQQ" = you bought the 3x ETF and sold the 1x ETF),
the orders at entry per $10,000 of the 3x ETF, entry and exit time, minutes held, the prices of both legs at entry and exit, gross and net bp, net dollars and why it closed. The Live tab says the same: "LONG TQQQ 357 sh / SHORT QQQ 107 sh", "ENTER long spread: BUY ... + SELL ...".</li>
<li><b>Signals &amp; risk</b> - the spread and s-score of each pair now, the cost of a round trip against what a reversion could capture, the leveraged-ETF risk map.</li>
</ul>

<h3>Safety</h3>
<ul>
<li>The window always starts on <b>PAPER + DRY RUN</b>. Sending orders needs the ARM box; a real-money account also needs a typed phrase. Switching account disarms.</li>
<li>The engine only touches the six ETFs and only what its own ledger holds (orders are tagged <code>LP-…</code>). Positions are whole shares, both legs confirmed, a failed leg unwinds the other, a stale feed holds then exits,
a daily-loss halt flattens, a hard flatten runs at 15:52, KILL cancels its orders and flattens its ledger.</li>
<li>This account cannot read real-time SIP data, only IEX. IEX bars of these ETFs are often empty (SPXL has a bar in 30% of the minutes, TQQQ 75%) and a spread of two stale prints is noise: the IEX s-scores correlate only 0.2-0.57 with the consolidated ones and the rules trade 8x as often. So the live engine runs on IEX in DRY RUN only; an ARMED session on IEX is refused unless you accept it knowingly. Real-time SIP is a paid data plan.</li>
<li>A margin account under $25,000 is subject to the pattern-day-trader rule: pre-flight fails a real-money account below it.</li>
<li><b>Valid has not signed off.</b> The backtest does not make money after costs; treat this as a research and fill-measurement tool.</li>
</ul>
"""


class HelpTab(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setHtml(HTML)
