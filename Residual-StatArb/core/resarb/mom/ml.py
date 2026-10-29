"""
The overnight forecast behind the 'rule+ml' ranking (research/mp_features.py, model "hgbraw"): one gradient-boosting model per calendar year.

  target   the overnight return (close -> next official open) as its percentile rank within the session, minus 0.5
  inputs   features.FEATURES (36 raw values known at the decision time)
  fit      sklearn HistGradientBoostingRegressor with fixed hyper-parameters (no tuning), on every (session, name) row of the years before Y; the last session before Y is
           left out (its target is not known on Y's first day)

The model of year Y is fitted on data before Y-01-01 and saved as <folder>/models/<key>_<Y>.pkl. The backtest predicts each year with that year's file and live loads the
current year's file - the same object, so live = backtest. A new year needs its model: 'Train ML models' in the app (or  python -m resarb.mom.ml <folder>) in January.
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .features import FEATURES

try:                                                                                            # sklearn's joblib probes the physical cores with a tool this Windows lacks, prints a traceback
    from joblib.externals.loky.backend import context as _loky                                  # and falls back to the logical count: give it that count up front (same threads, no noise)
    if _loky.physical_cores_cache is None:
        _loky.physical_cores_cache = os.cpu_count() or 1
except Exception:                                                                               # noqa: BLE001
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
FIRST_YEAR = 2018
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, early_stopping=False, random_state=0)
MIN_ROWS = 20_000
_CACHE: dict = {}


def model_key(m) -> str:
    u = m.universe.replace(":", "_").replace(",", "-").replace("+", "_")
    return f"hgb_on_{u}_{m.minute}"


def model_path(folder: str, m, year: int) -> str:
    return os.path.join(folder, "models", f"{model_key(m)}_{int(year)}.pkl")


def load_model(folder: str, m, year: int) -> Optional[dict]:
    """{'model': fitted regressor, 'info': dict} or None. Cached per file and modification time."""
    p = model_path(folder, m, year)
    if not os.path.exists(p):
        return None
    mt = os.path.getmtime(p)
    hit = _CACHE.get(p)
    if hit and hit[0] == mt:
        return hit[1]
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if list(obj["info"].get("features", [])) != FEATURES:
        raise RuntimeError(f"{p} was trained on other features - train it again")
    _CACHE[p] = (mt, obj)
    return obj


def available_years(folder: str, m) -> list:
    d = os.path.join(folder, "models")
    if not os.path.isdir(d):
        return []
    pre = model_key(m) + "_"
    return sorted(int(f[len(pre):-4]) for f in os.listdir(d) if f.startswith(pre) and f.endswith(".pkl") and f[len(pre):-4].isdigit())


def _target(df: pd.DataFrame) -> pd.Series:
    return df.groupby("t").y_on.rank(pct=True) - 0.5


def train_year(df: pd.DataFrame, year: int, params: Optional[dict] = None, min_rows: int = MIN_ROWS) -> Optional[dict]:
    from sklearn import __version__ as skv
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = {**PARAMS, **(params or {})}
    y = _target(df)
    yr = df.date.dt.year.to_numpy()
    before = yr < year
    if not before.any():
        return None
    last_t = df.t[before].max()
    tr = before & y.notna().to_numpy() & (df.t < last_t).to_numpy()
    if tr.sum() < min_rows:
        return None
    X = df[FEATURES].to_numpy(np.float32)
    t0 = time.time()
    mdl = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr])
    info = dict(year=int(year), features=list(FEATURES), params=params, rows=int(tr.sum()), first=str(df.date[tr].min().date()), last=str(df.date[tr].max().date()),
                sklearn=skv, fitted=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0, 1))
    return dict(model=mdl, info=info)


def train(folder: str, m, df: pd.DataFrame, years: Optional[list] = None, retrain: bool = False, log: Callable[[str], None] = print, params: Optional[dict] = None,
          min_rows: int = MIN_ROWS) -> list:
    """Fit and save the models of ``years`` (default: FIRST_YEAR .. this year) that are missing (or all, ``retrain``). ``df`` = features.feature_frame(..., target=True)."""
    years = years or list(range(FIRST_YEAR, pd.Timestamp.now().year + 1))
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)
    done = []
    for Y in years:
        p = model_path(folder, m, Y)
        if os.path.exists(p) and not retrain:
            continue
        obj = train_year(df, Y, params, min_rows)
        if obj is None:
            log(f"  {Y}: not enough rows before {Y} - no model")
            continue
        obj["info"].update(universe=m.universe, minute=m.minute)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, p)
        done.append(p)
        i = obj["info"]
        log(f"  {Y}: fitted on {i['rows']:,} rows {i['first']} .. {i['last']} ({i['seconds']} s)")
    return done


def predict_rows(obj: dict, df: pd.DataFrame) -> np.ndarray:
    return obj["model"].predict(df[FEATURES].to_numpy(np.float32))


def predict_matrix(folder: str, m, df: pd.DataFrame, shape: tuple) -> tuple:
    """T x N forecasts for the rows of ``df`` (each session with its year's model; NaN where that model is missing) and {year: model info or None}."""
    pred = np.full(shape, np.nan)
    used: dict = {}
    if not len(df):
        return pred, used
    yr = df.date.dt.year.to_numpy()
    for Y in sorted(set(yr)):
        obj = load_model(folder, m, Y)
        used[int(Y)] = obj["info"] if obj else None
        if obj is None:
            continue
        sel = yr == Y
        sub = df[sel]
        pred[sub.t.to_numpy(), sub.j.to_numpy()] = predict_rows(obj, sub)
    return pred, used


def main(argv=None) -> int:
    """python -m resarb.mom.ml <app folder> [--retrain]: fit the missing yearly models of every preset that uses the forecast."""
    import argparse
    from . import engine as E
    from . import presets as PR
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args(argv)
    seen = set()
    for name in PR.preset_names():
        m = PR.get(name)
        if not m.uses_ml or model_key(m) in seen:
            continue
        seen.add(model_key(m))
        print(f"{model_key(m)}: building the feature table ...", flush=True)
        mk = E.load_market(m)
        df = E.training_frame(m, mk)
        train(a.folder, m, df, retrain=a.retrain, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
