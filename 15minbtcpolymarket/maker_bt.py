"""Naive passive-quoting backtest on the Kalshi trade tape (no maker fee on these series: fee_type == 'quadratic').

At each minute boundary j (candle close quote bid_c / ask_c) we rest, for ONE minute:
  * a BUY-YES order at the bid   (fills when a taker SELLS YES / buys NO at yes_price <= bid  [touch]  or < bid  [through])
  * a SELL-YES order at the ask  (fills when a taker BUYS YES at yes_price >= ask [touch] or > ask [through])
Size = SIZE contracts, filled when cumulative qualifying volume (plus an assumed queue ahead of us) reaches the size.
P&L settles on the real outcome, no fee. 'through' + big queue is the pessimistic convention (we only fill when the
market trades beyond us, i.e. maximum adverse selection); 'touch' + no queue is optimistic.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
SIZE = 100


def load(series: str):
    mp = DATA / "markets.csv" if series == "KXBTC15M" else DATA / "multi" / f"{series}_markets.csv"
    cp = DATA / "candles.csv" if series == "KXBTC15M" else DATA / "multi" / f"{series}_candles.csv"
    m = pd.read_csv(mp)
    m["yes"] = (m.result == "yes").astype(int)
    c = pd.read_csv(cp, on_bad_lines="skip")
    t = pd.read_csv(DATA / "tape" / f"{series}.csv", on_bad_lines="skip")
    c = c[c.ticker.isin(t.ticker.unique())]
    return m.set_index("ticker"), c, t


def simulate(series: str, queue: float = 0.0, through: bool = False, minutes=range(1, 14), size: int = SIZE) -> pd.DataFrame:
    m, c, t = load(series)
    c = c[(c.bid_c > 0) & (c.ask_c < 1) & (c.ask_c >= c.bid_c) & ((c.ask_c - c.bid_c) <= 0.06) & c.minute.isin(list(minutes))]
    tg = {k: g for k, g in t.groupby("ticker")}
    rows = []
    for r in c.itertuples():
        mk = m.loc[r.ticker]
        g = tg.get(r.ticker)
        if g is None:
            continue
        t0 = (mk.open_ts + 60 * r.minute) * 1000
        w = g[(g.ts_ms > t0) & (g.ts_ms <= t0 + 60_000)]
        if w.empty:
            continue
        mid = (r.bid_c + r.ask_c) / 2
        # buy-yes at bid: taker sells yes -> taker_side == 'no'
        s = w[w.taker_side == "no"]
        s = s[s.yes_price < r.bid_c - 1e-9] if through else s[s.yes_price <= r.bid_c + 1e-9]
        if s["count"].sum() >= queue + size:
            rows.append((series, r.ticker, mk.open_ts, r.minute, "BUY", r.bid_c, mid, mk.yes - r.bid_c, mk.yes - mid))
        # sell-yes at ask: taker buys yes -> taker_side == 'yes'
        b = w[w.taker_side == "yes"]
        b = b[b.yes_price > r.ask_c + 1e-9] if through else b[b.yes_price >= r.ask_c - 1e-9]
        if b["count"].sum() >= queue + size:
            rows.append((series, r.ticker, mk.open_ts, r.minute, "SELL", r.ask_c, mid, r.ask_c - mk.yes, mid - mk.yes))
    return pd.DataFrame(rows, columns=["series", "ticker", "open_ts", "minute", "side", "px", "mid", "pnl", "pnl_vs_mid"])


def report(df: pd.DataFrame, label: str, n_markets: int) -> None:
    if df.empty:
        print(f"{label}: no fills")
        return
    per_mkt = df.groupby("ticker").pnl.sum()
    # market-level t-stat on mean per-fill pnl (fills in a market are correlated through the outcome)
    mean = df.pnl.mean() * 100
    mm = df.groupby("ticker").pnl.mean()
    t = mm.mean() / mm.std() * np.sqrt(len(mm)) if len(mm) > 2 else np.nan
    print(f"{label:34s} fills {len(df):6d} ({len(df)/n_markets:5.2f}/mkt) | EV/contract {mean:+6.2f}c | vs mid {df.pnl_vs_mid.mean()*100:+6.2f}c | "
          f"half-spread captured {((df.pnl-df.pnl_vs_mid).mean())*100:+5.2f}c | t(mkt) {t:+5.2f} | $/mkt@{SIZE} {df.pnl.sum()*SIZE/n_markets:+7.2f}")


if __name__ == "__main__":
    series = sys.argv[1]
    m, c, t = load(series)
    nm = t.ticker.nunique()
    print(f"== {series}: {nm} markets with tape, {len(t)} trades")
    for name, q, th in (("optimistic: touch, no queue", 0, False), ("touch, queue 1x size", SIZE, False),
                        ("touch, queue 5x size", 5 * SIZE, False), ("pessimistic: through, no queue", 0, True)):
        d = simulate(series, queue=q, through=th)
        report(d, name, nm)
        if name.startswith("touch, queue 1x"):
            for side, g in d.groupby("side"):
                report(g, f"   side {side}", nm)
            d["zone"] = pd.cut((d.px), [0, .2, .4, .6, .8, 1.0])
            for z, g in d.groupby("zone", observed=True):
                report(g, f"   px zone {z}", nm)
            d["mb"] = pd.cut(d.minute, [0, 4, 8, 11, 13])
            for z, g in d.groupby("mb", observed=True):
                report(g, f"   minutes {z}", nm)
