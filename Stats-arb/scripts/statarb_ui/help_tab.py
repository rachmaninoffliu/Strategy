"""How it works."""
from __future__ import annotations

from PyQt6.QtWidgets import QTextBrowser

HTML = """
<h2>Statistical arbitrage - residual mean reversion, built for 2026</h2>
<p>Based on <b>Avellaneda &amp; Lee (2008), "Statistical Arbitrage in the U.S. Equities Market"</b>. Every stock's return is split into a <i>systematic</i> part (15 PCA eigenportfolios of the
252-day correlation matrix, or its sector ETF) and an <i>idiosyncratic residual</i>. On a 60-day window the cumulative residual is fitted as an Ornstein-Uhlenbeck process
(AR(1): <code>X<sub>n+1</sub> = a + b X<sub>n</sub> + ζ</code>), giving the reversion speed κ = −ln(b)·252, the equilibrium level m = a/(1−b) and the equilibrium sd σ<sub>eq</sub>.
The <b>s-score</b> <code>s = (X − m)/σ<sub>eq</sub></code> says how many sd the stock sits from its sector-implied fair value.</p>
<p><b>Rules (paper):</b> open long at s &lt; −1.25, open short at s &gt; +1.25, close longs at s &gt; −0.50, close shorts at s &lt; +0.75. Only 'fast' reverters (κ &gt; 8.4) are used. The book is held market neutral.</p>

<h3>What was added for today's market</h3>
<ul>
<li><b>Idiosyncratic jumps</b> - earnings dates (Yahoo calendar): no new positions within 6 sessions of an announcement, existing ones are closed 2-3 sessions before it; news / volume-spike days and the session after earnings are not faded.</li>
<li><b>Crowding</b> - the book is projected onto the null space of market beta, the top PCA factors, the 11 sectors, 12-1 momentum, volatility and size.</li>
<li><b>Geopolitics / macro</b> - per-stock oil, rates and USD betas are neutralised; a shock score (SPY↓, VIXY↑, |oil|, |TLT|, gold↑ in sigma units) de-grosses the book for a few sessions after a macro shock.</li>
<li><b>Risk targeting</b> - EWMA volatility target on the strategy's own P&amp;L, gross capped at the Reg-T 2x overnight limit, risk-parity sizing (1/residual vol), a no-trade band, per-name cap.</li>
<li><b>Execution</b> - signals use data through the close; orders are <b>market-on-open</b> the next session, P&amp;L is open-to-open, costs scale with liquidity (large cap → micro), short borrow is charged.</li>
</ul>

<h3>Tabs</h3>
<ul>
<li><b>Live trading</b> - connect to Alpaca (paper or live), pre-flight, <i>Plan next session</i> (dry run), <i>Submit</i> (only when ARMED + confirmed), reconcile with the broker, KILL.</li>
<li><b>Backtest</b> - any window with an in-sample / out-of-sample split, equity vs SPY, yearly table, trades, exposures, <b>Ablation</b> (what each ingredient adds), cost and parameter-neighbourhood checks.</li>
<li><b>Signals &amp; risk</b> - today's s-scores with the reason each candidate is or is not tradable; the macro / geopolitical dashboard; the 2026 risk map.</li>
</ul>

<h3>Safety</h3>
<ul>
<li>The window always starts on <b>PAPER + DRY RUN</b>. Sending orders needs the ARM box, a confirmation dialog, and (real money) a typed phrase.</li>
<li>The engine only touches symbols in its own ledger or plan (orders are tagged <code>SA-…</code>); it never liquidates the rest of the account.</li>
<li>Shorts only in easy-to-borrow names; whole shares; minimum ticket; per-name and gross caps; KILL cancels its orders and flattens its ledger.</li>
<li>Alpaca accepts market-on-open orders between 19:00 ET and 09:28 ET. The plan is made after the 16:00 close (data is final ~16:20).</li>
</ul>

<h3>Read this before trusting any number</h3>
<ul>
<li><b>The 2008 signal is dead.</b> Raw Avellaneda-Lee reversion on 2017-2026 data: net Sharpe about 0 (core universe 0.3 at 1.5 bp, negative on the survivorship-free universe). Its only strongly significant footprint is the
overnight close-to-open reversal (IC t-stat ~10), worth ~0.6 bp a day gross - less than any realistic cost. The Ablation tab shows every ingredient switched off in turn.</li>
<li><b>What is left</b> is the <i>earnings-announcement premium</i> sleeve: long the stocks that report in the next 2-10 sessions (OU screen, hedged with sector SPDRs, real 4% name cap and 30% sector cap, cash when few
companies report). With complete earnings calendars and a point-in-time universe of names trading >= $150M a day its net Sharpe is about <b>0.45 (standard error 0.33)</b>, roughly 3% a year at 6.5% volatility; two years (2020, 2024) carry it,
the 10 best days are ~80% of its P&amp;L, it breaks even at ~6 bp of costs and it is <b>not statistically distinguishable from zero</b> after the variants tried. On the hand-picked 'core' list it looks better (0.7+): that is the hindsight in the list.</li>
<li>'core' universe = today's large caps (hindsight-selected, inflated). 'broad' = every U.S. stock that was ever liquid, <i>delisted ones included</i>, filtered point-in-time; earnings calendars exist only for names that still trade, so its
earnings pool is survivor-only too (the report bounds this).</li>
<li>Backtests assume fills at the official opening print with 1.5 bp (more for thinner names). Real fills in thin names will be worse; costs are a parameter - check the cost table.</li>
<li><b>Valid has not signed off for live capital.</b> Paper-trade it first, with the ARM box only on the PAPER account.</li>
</ul>
"""


class HelpTab(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setHtml(HTML)
