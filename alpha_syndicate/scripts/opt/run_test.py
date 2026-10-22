"""SINGLE-SHOT Test evaluation (2022-01-01 .. last completed session). Refuses to run unless the frozen params verify
against their sha256, and refuses a second run (writes test_results.json + a lock file)."""
import json, hashlib, os, sys, datetime
import numpy as np
from sim import *

D = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt"
LOCK = f"{D}\\test_run.lock"
frozen = json.load(open(f"{D}\\frozen_params.json"))
chk = hashlib.sha256(json.dumps(frozen["params"], sort_keys=True).encode()).hexdigest()
assert chk == frozen["sha256"], "frozen params do not match their recorded hash - aborting"
DRY = bool(os.environ.get("DRY"))          # DRY=1: exercise the code path on a Train-period window; no lock, no results file
if not DRY:
    if os.path.exists(LOCK):
        sys.exit("Test was already evaluated once (lock file exists). No re-runs / no re-tuning on Test. See test_results.json")
    open(LOCK, "w").write(f"first run {datetime.datetime.now().isoformat()} frozen sha256 {frozen['sha256']}\n")

P = frozen["params"]
MODE = {"Dip": MODE_DIP, "Trend": MODE_TREND, "Trend + Dip": MODE_TD}[P["mode"]]
TESTW = ("2019-01-01", "2021-12-31") if DRY else ("2022-01-01", frozen.get("test_end", "2026-10-06"))
SPEC = {"TQQQ": ("TQQQ", "QQQ", "tv"), "SPXL": ("SPXL", "SPY", None), "QQQ": ("QQQ", "QQQ", None), "SPY": ("SPY", "SPY", None)}
DEFAULT = dict(mode=MODE_DIP, rl=2, re=20, el=5, tl=200, mh=10, dt=True)


def bh(m, win):
    inwin, _ = m.window(win)
    r = np.diff(m.cl, prepend=m.cl[0]) / np.concatenate([[m.cl[0]], m.cl[:-1]]); r = r[inwin]
    eq = np.cumprod(1 + r); pk = np.maximum.accumulate(eq)
    cagr = eq[-1] ** (252 / len(r)) - 1; mdd = float(((pk - eq) / pk).max())
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(252)), cagr=float(cagr), mdd=mdd, calmar=float(cagr / mdd), final=float(eq[-1]))


res = {"frozen_sha256": frozen["sha256"], "test_window": TESTW, "assets": {}}
for a, (tr, px, src) in SPEC.items():
    m = Market(tr, px, src="yf", traded_src=src)
    f_te = run_one(m, TESTW, MODE, P["rsiLen"], P["rsiEntry"], P["exitLen"], P["trendLen"], P["maxHold"], P["dipTrend"])
    f_tr = run_one(m, TRAIN, MODE, P["rsiLen"], P["rsiEntry"], P["exitLen"], P["trendLen"], P["maxHold"], P["dipTrend"])
    d_te = run_one(m, TESTW, DEFAULT["mode"], DEFAULT["rl"], DEFAULT["re"], DEFAULT["el"], DEFAULT["tl"], DEFAULT["mh"], DEFAULT["dt"])
    b_te = bh(m, TESTW)
    res["assets"][a] = dict(frozen_test=f_te, frozen_train=f_tr, default_test=d_te, buyhold_test=b_te)
    print(f"\n{a}")
    for lab, x in (("frozen  Train", f_tr), ("frozen  TEST ", f_te), ("default TEST ", d_te)):
        print(f"  {lab}: Sharpe {x['sharpe']:.2f} Sortino {x['sortino']:.2f} Calmar {x['calmar']:.2f} CAGR {x['cagr']:.1%} MDD {x['mdd']:.1%} trades {x['trades']} win {x['win']:.1%} PF {x['pf']:.2f} expo {x['exposure']:.1%} corr {x['corr']:.2f} final x{x['final']:.2f}")
    print(f"  buy&hold TEST : Sharpe {b_te['sharpe']:.2f} Calmar {b_te['calmar']:.2f} CAGR {b_te['cagr']:.1%} MDD {b_te['mdd']:.1%}")

# pre-registered pass criteria
A = res["assets"]
c1 = all(A[a]["frozen_test"]["sharpe"] >= 0.5 * A[a]["frozen_train"]["sharpe"] for a in ("TQQQ", "SPXL"))
c2 = all(np.sign(A[a]["frozen_test"]["sharpe"]) == np.sign(A[a]["frozen_train"]["sharpe"]) and A[a]["frozen_test"]["sharpe"] > 0 for a in SPEC)
c3 = all(A[a]["frozen_test"]["mdd"] <= 1.5 * A[a]["frozen_train"]["mdd"] for a in ("TQQQ", "SPXL"))
res["criteria"] = dict(test_sharpe_ge_half_train_on_TQQQ_and_SPXL=bool(c1), positive_sharpe_on_all_four=bool(c2), test_mdd_le_1p5x_train=bool(c3), PASS=bool(c1 and c2 and c3))
print("\nPRE-REGISTERED CRITERIA:", res["criteria"])
if DRY:
    print("(DRY RUN on a Train-period window - nothing written)")
else:
    json.dump(res, open(f"{D}\\test_results.json", "w"), indent=1, default=float)
