"""Bed aggradation along the canal at the end of the season (reach-mean over a 5 km window, mean of the seeds), per true multiplier."""
import json, os, numpy as np, matplotlib.pyplot as plt
import style; style.setup(); C = style.C; ENT = style.ENT
RES = os.environ.get('RES', '../results/main.json')
R = [r for r in json.load(open(RES)) if 'error' not in r and r['cfg'].get('tag') == 'main']
L, W = 56.6, 5.0
def window(dz):
    ns = len(dz); dx = L / ns; edges = np.arange(ns + 1) * dx; F = np.concatenate([[0.0], np.cumsum(dz) * dx]); xm = (np.arange(ns) + 0.5) * dx
    a = np.clip(xm - 0.5 * W, 0.0, L - W)
    return xm, (np.interp(a + W, edges, F) - np.interp(a, edges, F)) / W
mults = sorted({r['cfg']['mult'] for r in R})
show = ['MPC', 'Fixed', 'Morph-learn']
fig, axs = plt.subplots(1, len(mults), figsize=(6.8, 3.0), constrained_layout=True)
axs = np.atleast_1d(axs)
for ax, m, pl in zip(axs, mults, 'abc'):
    for c in show:
        P = [np.array(r['dz_prof']) for r in R if r['cfg']['mult'] == m and r['cfg']['ctl'] == c]
        if not P: continue
        xm, w = zip(*[window(p) for p in P]); w = 100 * np.mean(w, axis=0)
        ax.plot(xm[0], w, color=ENT[c]['color'], label=ENT[c]['label'], lw=1.5 if c != 'MPC' else 1.2)
    top = ax.get_ylim()[1]
    for xg in (14.2, 28.3, 42.5): ax.axvline(xg, color=style.INK2, lw=0.6, ls=':')
    ax.set_xlim(0, L); ax.set_ylim(0, top)
    ax.plot([10, 22, 35, 44, 50], [0.0] * 5, '^', color=C['gray'], ms=5, clip_on=False, zorder=5)
    ax.set_xlabel('Distance from the pumps (km)')
    if pl == 'a': ax.set_ylabel('Aggradation, 5 km mean (cm)')
    ax.set_title(r'true $\mu$ = %g' % m, fontsize=10); style.panel(ax, pl)
axs[0].legend(loc='upper right', frameon=False, fontsize=9)
fig.savefig('Fig_profiles.png'); print('ok')
