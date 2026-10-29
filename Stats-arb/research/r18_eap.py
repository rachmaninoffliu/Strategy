import sys
from rlib import *

universe = sys.argv[1] if len(sys.argv) > 1 else "core"
signame = sys.argv[2] if len(sys.argv) > 2 else "pca15"
panel = panel_for(universe)
earn = D.load_earnings()
er = D.earnings_reaction_matrix(panel, earn)
O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
T, N = O.shape
dates = panel.dates
secs = sorted(set(panel.sector.values)); sec = np.array([secs.index(panel.sector[c]) for c in panel.stocks])
ro = np.full((T, N), 0.0); r = O[1:] / O[:-1] - 1; r[~np.isfinite(r) | (np.abs(r) > 0.6)] = 0.0; ro[:-1] = r
ecols = list(panel.etf_close.columns)
from statarb_data import SECTOR_ETF
sec_e = np.array([ecols.index(SECTOR_ETF[secs[k]]) for k in sec])
eo = panel.etf_open.to_numpy(float)
ero = np.zeros((T, len(ecols))); ero[:-1] = np.nan_to_num(eo[1:] / eo[:-1] - 1)
beta = S.sector_betas(panel, sec_e, ecols, 60)
sig = pickle.load(open(os.path.join(SP, f"sig_{universe}_{signame}.pkl"), "rb"))
s_ = sig.s; adv = sig.adv; univ = sig.univ
# historical earnings-move size per stock, point-in-time: mean |gap| over the last 8 reaction sessions up to and including t
gap = np.full((T, N), np.nan); gap[1:] = O[1:] / Cc[:-1] - 1; gap[np.abs(gap) > 0.9] = np.nan
emove = np.full((T, N), np.nan)
for j in range(N):
    ev = np.where(er[:, j])[0]
    if len(ev) == 0:
        continue
    g = np.abs(gap[ev, j])
    ma = pd.Series(g).rolling(8, min_periods=3).mean().to_numpy()
    idx = np.searchsorted(ev, np.arange(T), side="right") - 1
    ok_ = idx >= 0
    emove[ok_, j] = ma[idx[ok_]]
t0 = np.searchsorted(dates.values, np.datetime64("2017-06-01"))
cv = S.cost_vec

def run(h_lo=2, h_hi=10, pool="all", wmode="eq", tc_bps=1.5, cap_move=0.20, wcap=0.04, label="", quiet=False):
    ahead = np.zeros((T, N), bool)
    for h in range(h_lo, h_hi + 1):
        ahead[:-h] |= er[h:]
    rows, wprev = [], np.zeros(N)
    for d in range(t0, T - 2):
        sel = ahead[d] & univ[d] & np.isfinite(s_[d])
        if pool == "cheap":
            sel &= s_[d] < 0
        elif pool == "notrich":
            sel &= s_[d] < 1.0
        if cap_move:
            sel &= ~(emove[d] > cap_move)
        w = np.zeros(N)
        if sel.sum() >= 5:
            if wmode == "eq":
                w[sel] = 1.0
            elif wmode == "invmove":
                w[sel] = 1.0 / np.maximum(np.nan_to_num(emove[d][sel], nan=0.05), 0.01)
            w = w / w.sum()
            for _ in range(3):
                w = np.minimum(w, wcap); w = w / w.sum()
        bd = np.nan_to_num(beta[d], nan=1.0)
        hedge = -np.bincount(sec_e, weights=w * bd, minlength=len(ecols))
        gross = float(w @ ro[d + 1] + hedge @ ero[d + 1])
        cost = float(np.abs(w - wprev) @ cv(adv[d], tc_bps)) + 0.5e-4 * 0.1 * np.abs(hedge).sum()
        rows.append((dates[d], gross, cost, sel.sum())); wprev = w
    df = pd.DataFrame(rows, columns=["date", "gross", "cost", "n"]).set_index("date")
    net = df.gross - df.cost
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    IS, OOS = net[:"2021-12-31"], net["2022-01-01":]
    if not quiet:
        print(f"{label:44s} gross Sh {sh(df.gross):5.2f} net Sh {sh(net):5.2f} (IS {sh(IS):5.2f} OOS {sh(OOS):5.2f}) ret {net.mean()*252:+6.1%} vol {net.std()*np.sqrt(252):5.1%} maxDD {((1+net).cumprod()/(1+net).cumprod().cummax()-1).min():6.1%} n~{df.n.mean():.0f}", flush=True)
    return net

if __name__ == "__main__":
    print(f"universe {universe}/{signame}: long announcers (reaction in d+2..d+10) hedged with own-sector SPDR x beta; stock costs tiered from {1.5}bp")
    run(label="all announcers, equal weight, no move cap", cap_move=0)
    run(label="all, equal weight, skip avg earnings move>20%", pool="all")
    run(label="cheap (s<0), equal weight, move cap", pool="cheap")
    run(label="not rich (s<1), equal weight, move cap", pool="notrich")
    run(label="all, inverse earnings move, move cap", wmode="invmove")
    run(label="cheap, inverse earnings move, move cap", pool="cheap", wmode="invmove")
    run(label="all, equal, move cap, window [3,8]", h_lo=3, h_hi=8)
    run(label="all, equal, move cap, window [2,6]", h_hi=6)
    run(label="all, equal, move cap, 3bp costs", tc_bps=3.0)
