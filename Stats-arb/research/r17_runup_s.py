import sys
exec(open("r16_runup_sleeve.py").read().split("def run(")[0])   # reuse panel/earn/hedge setup
sig = pickle.load(open(os.path.join(SP, f"sig_{universe}_pca15.pkl"), "rb")) if os.path.exists(os.path.join(SP, f"sig_{universe}_pca15.pkl")) else None
s_ = sig.s

def run(h_lo, h_hi, mode, tc=1.5e-4, thr=0.5):
    ahead = np.zeros((T, N), bool)
    for h in range(h_lo, h_hi + 1):
        ahead[:-h] |= er[h:]
    rows, wprev = [], np.zeros(N)
    for d in range(t0, T - 2):
        pool = ahead[d] & ok[d] & np.isfinite(s_[d])
        w = np.zeros(N)
        if mode == "all":
            w[pool] = 1.0
        elif mode == "cheap":
            w[pool & (s_[d] < 0)] = 1.0
        elif mode == "ls":
            w[pool & (s_[d] < -thr)] = 1.0; w[pool & (s_[d] > thr)] = -1.0
        elif mode == "tilt":
            w[pool] = np.clip(1.0 - s_[d][pool], 0.0, 3.0)
        if mode == "ls":
            lg, sh_ = w[w > 0].sum(), -w[w < 0].sum()
            if lg < 3 or sh_ < 3:
                w[:] = 0
            else:
                w = np.where(w > 0, 0.5 * w / lg, 0.5 * w / sh_)
        else:
            if w.sum() < 5:
                w[:] = 0
            else:
                w = w / w.sum()
        bd = np.nan_to_num(beta[d], nan=1.0)
        hedge = -np.bincount(sec_e, weights=w * bd, minlength=len(ecols))
        gross = float(w @ ro[d + 1] + hedge @ ero[d + 1])
        turn = np.abs(w - wprev).sum()
        cost = tc * turn + 0.5e-4 * np.abs(hedge).sum() * 0.1
        rows.append((dates[d], gross, cost)); wprev = w
    df = pd.DataFrame(rows, columns=["date", "gross", "cost"]).set_index("date")
    net = df.gross - df.cost
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    IS, OOS = net[:"2021-12-31"], net["2022-01-01":]
    print(f"{mode:6s} [{h_lo},{h_hi}] thr{thr}: gross Sh {sh(df.gross):5.2f} net Sh {sh(net):5.2f} (IS {sh(IS):5.2f} OOS {sh(OOS):5.2f}) ret {net.mean()*252:+.1%} vol {net.std()*np.sqrt(252):.1%} maxDD {((1+net).cumprod()/(1+net).cumprod().cummax()-1).min():.1%}")
    return net

for mode in ("all", "cheap", "tilt", "ls"):
    run(2, 10, mode)
run(2, 10, "ls", thr=0.25)
run(2, 10, "ls", thr=1.0)
