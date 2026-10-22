"""Cross-asset replication of the late-window / time-of-day calibration pattern (quotes + outcomes only)."""
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

from strategy import fee_pc

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DATA = Path(__file__).parent / "data"


def load_series(series: str, mpath: Path, cpath: Path) -> pd.DataFrame:
    m = pd.read_csv(mpath)
    m["yes"] = (m.result == "yes").astype(int)
    c = pd.read_csv(cpath, on_bad_lines="skip")
    c = c[(c.minute >= 1) & (c.minute <= 14)]
    d = c.merge(m[["ticker", "open_ts", "yes", "volume"]], on="ticker", how="inner", suffixes=("_c", "_m"))
    d["series"] = series
    return d


def load_all() -> pd.DataFrame:
    parts = [load_series("KXBTC15M", DATA / "markets.csv", DATA / "candles.csv")]
    for mp in sorted(glob.glob(str(DATA / "multi" / "*_markets.csv"))):
        s = Path(mp).name.replace("_markets.csv", "")
        cp = Path(mp.replace("_markets", "_candles"))
        if cp.exists():
            parts.append(load_series(s, Path(mp), cp))
    P = pd.concat(parts, ignore_index=True)
    P = P[(P.open_ts >= 1785974400)]
    P = P[(P.bid_c > 0) & (P.ask_c < 1) & (P.ask_c >= P.bid_c) & ((P.ask_c - P.bid_c) <= 0.06)].copy()
    P["mid"] = (P.bid_c + P.ask_c) / 2
    P["lm"] = np.log(P.mid.clip(.01, .99) / (1 - P.mid.clip(.01, .99)))
    dt = pd.to_datetime(P.open_ts, unit="s", utc=True)
    P["hr"] = dt.dt.hour
    P["blk"] = np.where(P.hr.between(20, 23), "20-23", "other")
    P["win"] = P.open_ts
    return P


if __name__ == "__main__":
    P = load_all()
    print("rows", len(P), "| series:", P.series.nunique(), "| markets:", P.ticker.nunique())
    print(P.groupby("series").agg(markets=("ticker", "nunique"), spread_c=("ask_c", lambda s: (s - P.loc[s.index, "bid_c"]).mean() * 100),
                                  mkt_vol_M=("volume_m", lambda s: s.mean() / 1e6)).round(3).T.to_string())

    L = P[(P.minute >= 12) & (P.minute <= 13)]
    print("\n== late-window (min 12-13) calibration slope of logit(yes) on logit(mid); 1 = calibrated ==")
    rows = []
    for ser, s in L.groupby("series"):
        for blk, ss in s.groupby("blk"):
            r = sm.Logit(ss.yes, sm.add_constant(ss[["lm"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ss.win})
            rows.append((ser, blk, len(ss), r.params["lm"], r.bse["lm"]))
    T = pd.DataFrame(rows, columns=["series", "blk", "n", "slope", "se"]).pivot(index="series", columns="blk", values=["slope", "se", "n"])
    print(T.round(3).to_string())
    for blk, ss in L.groupby("blk"):
        r = sm.Logit(ss.yes, sm.add_constant(ss[["lm"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ss.win})
        print(f"POOLED all assets, block {blk}: slope {r.params['lm']:.3f} (se {r.bse['lm']:.3f}, t vs 1 = {(r.params['lm']-1)/r.bse['lm']:.2f}), n={len(ss)}")
    print("\n== pooled late-window slope by UTC hour (clustered by window) ==")
    rows = []
    for h, s in L.groupby("hr"):
        r = sm.Logit(s.yes, sm.add_constant(s[["lm"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": s.win})
        rows.append((h, len(s), r.params["lm"], r.bse["lm"]))
    print(pd.DataFrame(rows, columns=["hr", "n", "slope", "se"]).round(3).T.to_string())
