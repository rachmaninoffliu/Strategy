import sys
exec(open("r18_eap.py").read().split("if __name__")[0])
# announcers vs same-sector NON-announcers among the same (surviving) names: removes survivorship from the comparison
def spread(h_lo=2, h_hi=10):
    ahead = np.zeros((T, N), bool)
    for h in range(h_lo, h_hi + 1):
        ahead[:-h] |= er[h:]
    # also exclude names within 1 session of a reaction from the control group
    near = np.zeros((T, N), bool)
    for h in range(-3, 2):
        if h >= 0: near[:T-h] |= er[h:]
        else: near[-h:] |= er[:T+h]
    out = []
    for d in range(t0, T - 2):
        ok_ = univ[d]
        a = ahead[d] & ok_
        c = ok_ & ~ahead[d] & ~near[d]
        tot, n = 0.0, 0
        for k in np.unique(sec):
            m = sec == k
            ak, ck = a & m, c & m
            if ak.sum() >= 1 and ck.sum() >= 5:
                tot += ak.sum() * (ro[d + 1][ak].mean() - ro[d + 1][ck].mean()); n += ak.sum()
        out.append((dates[d], tot / n if n else 0.0))
    return pd.Series(dict(out))
sp_ = spread()
sh = lambda x: x.mean() / x.std() * np.sqrt(252)
print(f"{universe}: announcers minus same-sector non-announcers (survivors only), NO costs: ann {sp_.mean()*252:+.1%} vol {sp_.std()*np.sqrt(252):.1%} Sharpe {sh(sp_):.2f}  IS {sh(sp_[:'2021-12-31']):.2f} OOS {sh(sp_['2022-01-01':]):.2f}")
ev = run(quiet=True, cap_move=0)
print(f"vs sector-ETF-hedged long book, net of costs: Sharpe {sh(ev):.2f}")
