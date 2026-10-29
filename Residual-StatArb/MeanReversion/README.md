# Mean Reversion (long-only, no leverage, closing-auction execution)

Buys what was pushed down: a close near the day's low (IBS), a 2-day RSI washout, a few days of losses. It sells into the rebound. Every model is decided from the 15-minute bars that ended by the decision time, and trades in the **closing auction** (market-on-close). It is long-only, and the book never holds more than **1x equity**.

* **Run the app**: double-click `launch_meanreversion.bat` (it starts on PAPER + DRY RUN).
  * **Left**: the model picker. Pick a preset or edit any field: universe, decision time, entry rule (+ an optional second condition), trend filter, exit rules, time stop, slots, how candidates are ranked, sizing, cash reserve, costs.
  * **Backtest**: KPIs next to SPY buy & hold, in-sample vs out-of-sample, equity and drawdown vs SPY. The **Trades** tab lists every trade: #, ticker, direction, shares, entry date / auction / price / value, signal / value / rank, decision price, exit date / auction / price / reason, sessions held, P&L $ and return. It can be sorted and filtered by ticker, and exported to CSV. Further tabs: years vs SPY, P&L by ticker, exit reasons, exposure.
  * **Live trading**: trades **exactly the model in the panel**. The strip says whether that model is the one you last backtested, and the panel is locked while the loop runs. Buttons: connect, pre-flight, plan now (dry run), the daily loop, check live vs backtest, KILL.
* **No leverage, and who gets the capital.** A model has *slots*. Each position is (1 − cash reserve) / slots of the equity, in whole shares at the decision price. When more names qualify than there are free slots, *Rank candidates by* decides who gets them:
  * `signal`: the most extreme reading first;
  * `vol_low`: the calmest names first;
  * `vol_high`: the most volatile first.

  The Live tab's *Candidates* table shows the full ranking each day and which names got capital.
* **Backtest = live.** Both call the same `decide()` on the same data and fill in the same auctions. `core/tests/test_mr.py` replays backtests session by session through the live engine with a simulated broker, and requires identical orders, fills and final equity (models: rule / overnight / trend + time stop / hybrid feed).
* **Keys**: `Alpaca Kay.txt` / `Alpaca Key.txt` in the project folder. `MR_ALPACA_API_KEY` / `MR_ALPACA_SECRET_KEY` select a separate paper account. An account that ResMomentum / ResMeanReversion / ResPrediction has claimed is refused (`data/resarb_accounts.json`).
* **Data feed for today's bars** (Live tab):
  * `sip_delayed` (free) is exactly the backtest's data. It allows decisions at 15:30 or earlier: the 16-minute delay plus the 15:50 market-on-close cut-off.
  * `sip+iex` (free) allows 15:45: delayed SIP plus real-time IEX for the last ~30 minutes. `research/mr_hybrid_check.py` measures what that changes.
  * `sip` is Alpaca's paid real-time plan, and is exactly the backtest's data at 15:45.
* **Daily cycle**:
  * 08:45: top up the daily bars; an overnight model sends its market-on-open sells.
  * Decision time: market-on-close orders.
  * 16:05: fills journaled, book rebuilt (entry day / price per position).
* **Headless**: `run_live.bat` trades `live_model.json` (written when the GUI's loop starts, or *File > Save model*). It is DRY by default; the ARMED line is commented out.
* **Files**:
  * `../core/resarb/mr/`: `models.py` (MRModel, signals, `decide()`), `market.py` (daily bars + decision-time snapshots), `engine.py` (backtest), `live.py` (live engine), `presets.py`, `run.py`, `ui/`.
  * `../research/mr_*.py`: the research.
  * `reports/`: journal and state.
  * `REPORT.md`: the research, the team review and the verdict.
