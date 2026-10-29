"""How it works: the strategy, the papers behind it, the shared machinery (betas, residuals, timing, feeds, costs) and the safety rules."""
from __future__ import annotations

import html
import os

from PyQt6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

COMMON = """
<h3>Beta extraction and the residual</h3>
<p>Every morning the engine regresses each stock's <b>daily</b> returns over the beta window (ending at the previous close) on the market (SPY) and on its sector SPDR's return in excess
of SPY: <code>r = a + b<sub>m</sub>·r<sub>SPY</sub> + b<sub>s</sub>·(r<sub>sector</sub> − r<sub>SPY</sub>) + e</code> (Avellaneda &amp; Lee's ETF model; the PCA option uses the top
eigenportfolios of the members' correlation matrix instead). Those betas are applied to every 15-minute return of the day: what is left, <code>e</code>, is the <b>residual</b> - the part
of the move the market and the sector do not explain. All three strategies trade only that part, and hedge the rest: for each $1 of stock the book shorts b<sub>m</sub>−b<sub>s</sub> $
of SPY and b<sub>s</sub> $ of the sector ETF.</p>
<h3>Universe</h3>
<p>At each month end the N most liquid U.S. stocks (63-day median dollar volume, price ≥ $10, ≥ 126 sessions of history, one share class) become next month's members. The history is
survivorship-free: names that were later delisted or acquired are in it.</p>
<h3>Timing and data feeds</h3>
<p>Bars are 15 minutes (09:30 … 15:45 bar starts). A decision made from the bars up to <i>t</i> is filled at the OPEN of the next bar - in the backtest and in live. Alpaca's free plan
gives real-time <b>IEX</b> bars (IEX is ~2-3% of U.S. volume: its last print can be seconds old) or <b>SIP</b> (all exchanges) only for data older than 15 minutes. The 'Live data feed'
setting chooses: <i>iex</i> (no delay, noisier bars), <i>sip_delayed</i> (accurate bars, every trade one bar later - the backtest models that delay), or <i>sip_realtime</i> (the paid
plan). History always comes from SIP.</p>
<h3>Costs</h3>
<p>Each traded dollar pays the per-side cost (half the bid-ask spread plus slippage; ~1-2 bp for these large caps, less for SPY / sector ETFs), shorts pay a borrow fee. The Cost
sensitivity table re-prices the same trades at other costs: a strategy whose Sharpe collapses at 3-5 bp is not robust.</p>
<h3>Safety (live)</h3>
<ul>
<li>Starts on PAPER and DRY RUN. ARM needs the typed phrase <code>ARM PAPER</code>. Real money is disabled in code until Valid signs off.</li>
<li>Keys come from the project's key file <code>Alpaca Key.txt</code> (lines <code>Key ID:</code> / <code>Secret Key:</code>) in the project folder. A strategy's own
<code>&lt;PREFIX&gt;_ALPACA_API_KEY</code> / <code>&lt;PREFIX&gt;_ALPACA_SECRET_KEY</code> variables (RESMOM, RESMR, RESPRED) select a separate account instead.
Alpaca allows three paper accounts per login. An account claimed by one strategy is refused to the others: two strategies on one account would net and close each other's
positions.</li>
<li>The engine trades only its universe and the hedge ETFs; it is target-based (each rebalance trades the broker's real positions to the target), sends reducing orders first, never
flips a position in one order, shorts only easy-to-borrow names, caps gross and order count, stops adding risk after the daily-loss limit and flattens at the drawdown stop.</li>
<li>KILL cancels the engine's open orders and closes every position in the strategy's symbols.</li>
</ul>
"""


class HelpTab(QWidget):
    def __init__(self, strat_cls: type, folder: str):
        super().__init__()
        v = QVBoxLayout(self)
        tb = QTextBrowser(); tb.setOpenExternalLinks(True)
        papers = "".join(f"<li><a href='{u}'>{html.escape(c)}</a> - {html.escape(w)}</li>" for c, u, w in strat_cls.PAPERS)
        rep = os.path.join(folder, "REPORT.md")
        tb.setHtml(f"<h2>{html.escape(strat_cls.NAME)}</h2><p>{html.escape(strat_cls.SUMMARY)}</p><p>{html.escape(strat_cls.HOWTO)}</p>"
                   f"<h3>Papers</h3><ul>{papers}</ul>{COMMON}<p>Research results, the team review and the verdict: <code>{html.escape(rep)}</code></p>")
        v.addWidget(tb)
