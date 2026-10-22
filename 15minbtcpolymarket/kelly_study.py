"""Kelly sizing for binary contracts: how it behaves for HIGH win-rate vs LOW win-rate bets when p is estimated with error.

Contract costs c (fee included), pays $1 w.p. p.   f*(p,c) = (p-c)/(1-c)
Log growth per bet at fraction f:  g = p*ln(1 + f*(1-c)/c) + (1-p)*ln(1-f)
"""
import numpy as np

rng = np.random.default_rng(7)


def g(f, p, c):
    f = np.clip(f, 0, 0.999)
    return p * np.log1p(f * (1 - c) / c) + (1 - p) * np.log1p(-f)


def fstar(p, c):
    return np.maximum(0.0, (p - c) / (1 - c))


print("== A. Sensitivity of f* to a 1c error in p  (df*/dp = 1/(1-c)) ==")
print(f"{'cost c':>8} {'edge':>6} {'f* full':>8} {'f* if p is 1c too high':>24} {'1c error moves f* by':>22}")
for c in (0.15, 0.30, 0.50, 0.70, 0.85, 0.925, 0.97):
    e = 0.01
    p = c + e
    print(f"{c:8.3f} {e*100:5.1f}c {fstar(p, c):8.3%} {fstar(p - 0.01, c):24.3%} {0.01/(1-c):22.1%}")

print("\n== B. Realised log-growth/bet (bp) when p_hat has estimation error N(0, s), using f = lam * f*(p_hat) ==")
profiles = {
    "HIGH-WIN  c=.925  true p=.935 (edge 1.0c, win 93.5%)": (0.935, 0.925, 0.007),
    "HIGH-WIN  c=.925  true p=.930 (edge 0.5c)":            (0.930, 0.925, 0.007),
    "LOW-WIN   c=.150  true p=.190 (edge 4.0c, win 19%)":   (0.190, 0.150, 0.020),
    "MID       c=.500  true p=.520 (edge 2.0c)":            (0.520, 0.500, 0.015),
}
lams = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50)
for name, (p, c, s) in profiles.items():
    ph = p + rng.normal(0, s, 200000)                    # what we THINK p is
    print(f"\n{name}   estimation error sd = {s*100:.1f}c   (no-error full-Kelly g = {g(fstar(p, c), p, c)*1e4:.2f} bp)")
    print("   lam      " + "  ".join(f"{l:>7.2f}" for l in lams))
    row = []
    for l in lams:
        f = l * fstar(ph, c)
        row.append(np.mean(g(f, p, c)) * 1e4)
    print("   g (bp)   " + "  ".join(f"{x:7.2f}" for x in row))
    best = lams[int(np.argmax(row))]
    print(f"   best lam = {best}")

print("\n== C. Drawdown risk over 500 sequential bets (one per ~15-min market ~ 5 days), HIGH-WIN profile, true p=.935, c=.925 ==")
n_bets, n_paths = 500, 4000
for label, p_true in (("edge +1.0c", 0.935), ("edge  0.0c (no real edge)", 0.925), ("edge -1.0c", 0.915)):
    print(f"\n  {label}")
    for l in (0.25, 0.5, 1.0):
        f = l * fstar(0.935, 0.925)                          # we THINK p=.935 -> constant fraction
        wins = rng.random((n_paths, n_bets)) < p_true
        r = np.where(wins, f * (1 - 0.925) / 0.925, -f)
        eq = np.cumprod(1 + r, axis=1)
        peak = np.maximum.accumulate(np.c_[np.ones(n_paths), eq], axis=1)[:, 1:]
        mdd = ((eq - peak) / peak).min(axis=1)
        print(f"    lam {l:4.2f} (f={f:5.1%}): median end {np.median(eq[:, -1]):6.3f}x | P(end<1) {np.mean(eq[:, -1] < 1):5.1%} | "
              f"median maxDD {np.median(mdd):6.1%} | P(DD>30%) {np.mean(mdd < -0.30):5.1%} | P(DD>50%) {np.mean(mdd < -0.50):5.1%}")
