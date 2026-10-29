import sys
from rlib import *

universe = sys.argv[1] if len(sys.argv) > 1 else "core"
panel = panel_for(universe)
earn = D.load_earnings()
er = D.earnings_reaction_matrix(panel, earn)
O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
T, N = O.shape
dates = panel.dates
secs = sorted(set(panel.sector.values)); sec = np.array([secs.index(panel.sector[c]) for c in panel.stocks])
ro = np.full((T, N), 0.0); r = O[1:] / O[:-1] - 1; r[~np.isfinite(r) | (np.abs(r) > 0.6)] = 0.0; ro[:-1] = r
# sector hedge: own-sector SPDR open-to-open, beta from 60d rolling regression (known at the decision)
ecols = list(panel.etf_close.columns)
from statarb_data import SECTOR_ETF
sec_e = np.array([ecols.index(SECTOR_ETF[secs[k]]) for k in sec])
eo = panel.etf_open.to_numpy(float)
ero = np.zeros((T, len(ecols))); ero[:-1] = np.nan_to_num(eo[1:] / eo[:-1] - 1)
beta = S.sector_betas(panel, sec_e, ecols, 60)
adv = pd.DataFrame(Cc * panel.volume.to_numpy(float)).rolling(20, min_periods=15).median().to_numpy()
ok = (Cc >= 5) & (adv >= 2e7)
t0 = np.searchsorted(dates.values, np.datetime64("2017-06-01"))

def run(h_lo, h_hi, hold_through_gap=True, tc=1.5e-4, label=""):
    """Decide at the close of d: hold the stocks whose reaction session is d+h_lo..d+h_hi, from open d+1 until the open of the reaction session (+1 if hold_through_gap)."""
    ahead = np.zeros((T, N), bool)
    for h in range(h_lo, h_hi + 1):
        ahead[:-h] |= er[h:]
    rows, wprev = [], np.zeros(N)
    for d in range(t0, T - 2):
        sel = ahead[d] & ok[d]
        w = np.where(sel, 1.0, 0.0)
        if w.sum() < 5:
            w = np.zeros(N)
        else:
            w = w / w.sum()                               # long book gross 1
        bd = np.nan_to_num(beta[d], nan=1.0)
        hedge = -np.bincount(sec_e, weights=w * bd, minlength=len(ecols))
        pnl_long = float(w @ ro[d + 1]); pnl_hedge = float(hedge @ ero[d + 1])
        turn = np.abs(w - wprev).sum() + 0.0
        cost = tc * turn * 2 + 0.5e-4 * np.abs(hedge).sum() * 0.1
        rows.append((dates[d], pnl_long + pnl_hedge, pnl_long, cost))
        wprev = w
    df = pd.DataFrame(rows, columns=["date", "gross", "long", "cost"]).set_index("date")
    net = df.gross - df.cost
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    IS, OOS = net[:"2021-12-31"], net["2022-01-01":]
    print(f"{label:34s} gross Sh {sh(df.gross):5.2f}  net Sh {sh(net):5.2f} (IS {sh(IS):5.2f} OOS {sh(OOS):5.2f})  ann ret {net.mean()*252:+.1%} vol {net.std()*np.sqrt(252):.1%}  turnover {df.cost.mean()/ (2*tc):.2f}/d  positions~{(ahead[t0:]&ok[t0:]).sum(1).mean():.0f}")
    return net

print(f"universe {universe}; long announcers (reaction in d+h_lo..d+h_hi), hedged with own-sector SPDR x beta; costs 1.5bp/side stocks")
for lo, hi in ((2, 7), (2, 4), (2, 3), (3, 8), (2, 10)):
    run(lo, hi, label=f"announcers in [{lo},{hi}] sessions")
