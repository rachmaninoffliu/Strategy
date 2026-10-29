"""
Fill the 15-minute SIP bars of the top-150 members that have no bar file yet (ranks 101-150 were never downloaded: the residual apps use the top-100).
Same method as data.download_by_day: one multi-symbol request per session, regular hours, only the months a symbol is (or is about to be) a member; monthly chunks are
resumable, then merged into one file per symbol.   python mp_fill_bars.py
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from resarb import data as D  # noqa: E402

N = 150
CDIR = os.path.join(D.DATA_DIR, "chunks_fill_sip")


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def main():
    u = D.load_universe()
    u = u[u["rank"] <= N]
    missing = sorted(s for s in set(u.symbol) if not os.path.exists(D.bars_path(s, "sip")))
    log(f"{len(missing)} members without bars")
    os.makedirs(CDIR, exist_ok=True)
    client = D.data_client()
    lock, stamps = threading.Lock(), []

    def throttle(per_min=160):
        while True:
            with lock:
                now = time.monotonic()
                while stamps and now - stamps[0] > 60:
                    stamps.pop(0)
                if len(stamps) < per_min:
                    stamps.append(now)
                    return
                wait = 60 - (now - stamps[0]) + 0.05
            time.sleep(wait)
    last = pd.Timestamp.now(tz=D.NY).tz_localize(None)
    months = pd.date_range("2016-01-01", last, freq="MS")
    mm = u[u.symbol.isin(missing)]
    t0 = time.time()
    for mo in months:
        lo, hi = mo - pd.DateOffset(months=1), mo + pd.DateOffset(months=13)
        syms = sorted(set(mm[(mm.month >= lo) & (mm.month <= hi)].symbol))
        f = os.path.join(CDIR, f"{mo:%Y-%m}.parquet")
        if not syms or (os.path.exists(f) and mo.to_period("M") != last.to_period("M")):
            continue
        days = [d for d in D.sessions(mo, min(mo + pd.offsets.MonthEnd(0), last.normalize()))]

        def one(d):
            e = min(d + pd.Timedelta(hours=16), pd.Timestamp.now(tz=D.NY).tz_localize(None) - pd.Timedelta(minutes=D.SIP_DELAY_MIN))
            if e <= d + pd.Timedelta(hours=9, minutes=30):
                return {}
            throttle()
            return D.fetch_bars(client, syms, d + pd.Timedelta(hours=9, minutes=30), e, feed="sip")
        frames = []
        with ThreadPoolExecutor(6) as ex:
            for res in ex.map(one, days):
                for s, fr in res.items():
                    if len(fr):
                        frames.append(fr.assign(symbol=s))
        df = pd.concat(frames) if frames else pd.DataFrame(columns=D.BAR_COLS + ["symbol"])
        df.index.name = "ts"
        df.reset_index().to_parquet(f)
        log(f"{mo:%Y-%m}: {len(syms)} symbols, {len(days)} sessions, {len(df):,} bars ({time.time() - t0:.0f} s)")
    files = sorted(os.path.join(CDIR, x) for x in os.listdir(CDIR) if x.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True).drop_duplicates(["symbol", "ts"], keep="last").sort_values(["symbol", "ts"])
    n = 0
    for s, g in df.groupby("symbol"):
        if s in missing:
            g.set_index("ts")[D.BAR_COLS].to_parquet(D.bars_path(s, "sip"))
            n += 1
    log(f"wrote {n} symbol files; still without bars: {sorted(set(missing) - set(df.symbol))}")


if __name__ == "__main__":
    main()
