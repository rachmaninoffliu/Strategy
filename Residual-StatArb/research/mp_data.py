"""
Research data for the momentum and prediction searches (MP): one aligned panel of

  * official daily OHLCV (split + dividend adjusted) of every stock in the survivorship-free Stats-arb cache plus the MR app's 59 ETFs (daily_mr.parquet);
  * the point-in-time liquidity universes: top-50 / 100 / 150 (resarb_universe.parquet - the names that have 15-minute bars, the same membership the MR app uses) and
    top-500 (daily bars only);
  * an intraday cube of the 15-minute SIP bars of the ~400 names that have them: the close of every bar of the session (slot 0 = the bar ending 9:45), the first trade,
    the bar volume, and the number of bars of the session (14 on an early close).

Decision-time prices come from the cube: "k bars before the close" = slot nbars - 1 - k, so a 15:45 decision on a normal day is a 12:45 decision on a 13:00 close.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from resarb import data as D  # noqa: E402
from resarb.mr.models import ETF_UNIVERSE  # noqa: E402

from mr_data import Daily, universe  # noqa: E402

CACHE = os.path.join(HERE, "reports", "mp_cache")
os.makedirs(CACHE, exist_ok=True)
SLOTS = 26
START, SPLIT = "2016-09-01", "2023-01-01"


class Panel:
    def __init__(self):
        d = Daily()
        etf_extra = [e for e in ETF_UNIVERSE if e not in d.col]
        self.dates = d.dates
        self.syms = list(d.syms) + etf_extra
        self.col = {s: j for j, s in enumerate(self.syms)}
        T, N = len(self.dates), len(self.syms)
        n0 = len(d.syms)
        mr = pd.read_parquet(os.path.join(D.DATA_DIR, "daily_mr.parquet"))
        mr = mr[mr.symbol.isin(etf_extra)]
        for k in ("open", "high", "low", "close", "volume"):
            a = np.full((T, N), np.nan)
            a[:, :n0] = getattr(d, k)
            if etf_extra:
                p = mr.pivot(index="date", columns="symbol", values=k).reindex(index=self.dates, columns=etf_extra)
                a[:, n0:] = p.to_numpy(float)
            setattr(self, k[0].upper() if k != "volume" else "V", a)
        self.etf = np.array([s in set(ETF_UNIVERSE) for s in self.syms])
        self._cube = None

    def j(self, s):
        return self.col[s]

    # ---------------------------------------------------------------- universes
    def top(self, n: int) -> np.ndarray:
        """dates x symbols bool, point-in-time member of the top-n (n <= 150 from the app's universe file, else the daily-bar top-n of mr_data)."""
        if n > 150:
            u = universe(_daily_proxy(self), n)
            out = np.zeros((len(self.dates), len(self.syms)), bool)
            out[:, :u.shape[1]] = u
            return out
        f = os.path.join(CACHE, f"top{n}.npy")
        if os.path.exists(f):
            return np.load(f)
        u = D.load_universe()
        u = u[u["rank"] <= n]
        by = u.groupby("month").symbol.apply(set).to_dict()
        months = self.dates.to_period("M").to_timestamp()
        out = np.zeros((len(self.dates), len(self.syms)), bool)
        for t, mo in enumerate(months):
            for s in by.get(mo, ()):
                j = self.col.get(s)
                if j is not None:
                    out[t, j] = True
        np.save(f, out)
        return out

    def etf_mask(self) -> np.ndarray:
        return np.broadcast_to(self.etf, (len(self.dates), len(self.syms))).copy()

    # ---------------------------------------------------------------- intraday cube
    def cube(self):
        """(closes T x N x 26, first trade T x N, volumes T x N x 26, nbars T) as float32; NaN for names without 15-minute bars."""
        if self._cube is not None:
            return self._cube
        f = os.path.join(CACHE, "cube.npz")
        if os.path.exists(f):
            z = np.load(f)
            self._cube = (z["cl"], z["op"], z["vo"], z["nb"])
            return self._cube
        T, N = len(self.dates), len(self.syms)
        cl = np.full((T, N, SLOTS), np.nan, np.float32)
        vo = np.full((T, N, SLOTS), np.nan, np.float32)
        op = np.full((T, N), np.nan, np.float32)
        nb = np.zeros(T, np.int16)
        tix = {d: i for i, d in enumerate(self.dates)}
        for fn in os.listdir(D.bars_dir("sip")):
            s = fn[:-8].replace("_", ".")
            j = self.col.get(s)
            if j is None:
                continue
            b = D.load_bars(s)
            if not len(b):
                continue
            day = b.index.normalize()
            ti = np.array([tix.get(x, -1) for x in day])
            slot = ((b.index - day) / pd.Timedelta(minutes=15)).astype(int) - 38          # 9:30 bar -> 0
            ok = (ti >= 0) & (slot >= 0) & (slot < SLOTS)
            cl[ti[ok], j, slot[ok]] = b.close.to_numpy()[ok]
            vo[ti[ok], j, slot[ok]] = b.volume.to_numpy()[ok]
            first = ok & (slot == 0)
            op[ti[first], j] = b.open.to_numpy()[first]
        # bars per session from the NYSE calendar (14 on a 13:00 close); bars after an early close are after-hours trades and are dropped
        from resarb.mr.market import session_closes
        closes = session_closes(self.dates[0], self.dates[-1] + pd.Timedelta(days=1)).reindex(self.dates)
        mins = ((closes - closes.index) / pd.Timedelta(minutes=1)).fillna(960).to_numpy()
        nb[:] = ((mins - 570) // 15).astype(int)
        for t in np.where(nb < SLOTS)[0]:
            cl[t, :, nb[t]:] = np.nan
            vo[t, :, nb[t]:] = np.nan
        np.savez(f, cl=cl, op=op, vo=vo, nb=nb)
        self._cube = (cl, op, vo, nb)
        return self._cube

    def snap(self, lead_bars: int = 1) -> np.ndarray:
        """T x N: the last price known lead_bars bars before the close (1 = 15:45 on a normal day, 2 = 15:30), forward-filled within the session."""
        cl, _, _, nb = self.cube()
        T = len(self.dates)
        out = np.full(cl.shape[:2], np.nan)
        for t in range(T):
            k = int(nb[t]) - 1 - lead_bars
            if k < 0:
                continue
            seg = cl[t, :, :k + 1]
            last = pd.DataFrame(seg.T).ffill().to_numpy()[-1]
            out[t] = last
        return out

    def at_slot(self, slot: int) -> np.ndarray:
        """T x N: the close of the bar in a fixed open-relative slot (0 = the 9:30-9:45 bar), forward-filled from earlier bars of the session."""
        cl = self.cube()[0]
        seg = cl[:, :, :slot + 1]
        out = seg[:, :, -1].astype(float)
        for k in range(slot - 1, -1, -1):
            m = ~np.isfinite(out)
            out[m] = seg[:, :, k][m]
        return out


def _daily_proxy(p: Panel):
    """The mr_data.Daily object the top-500 universe was cached from (same dates, the stock columns first)."""
    return Daily()


def stats(r: pd.Series, split: str = SPLIT) -> dict:
    r = r.dropna()
    out = {}
    for lab, x in (("all", r), ("is", r[r.index < split]), ("oos", r[r.index >= split])):
        if len(x) < 30:
            continue
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        out[f"cagr_{lab}"] = eq.iloc[-1] ** (1 / yrs) - 1
        out[f"sh_{lab}"] = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
        out[f"mdd_{lab}"] = (eq / eq.cummax() - 1).min()
        out[f"vol_{lab}"] = x.std() * np.sqrt(252)
    return out
