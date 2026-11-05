"""Paired differences with 95% confidence intervals (five seeds) relative to predictive control."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import style; style.setup(); C = style.C; ENT = style.ENT
P = pd.read_csv(os.environ.get('PAIRED', '../results/paired_main.csv'))
ctls = ['Fixed', 'FixedBest', 'Morph-nominal', 'Morph-learn', 'Morph-oracle']
mults = sorted(P.mult.unique())
pos, y = {}, 1.0
for m in mults:
    for c in ctls:
        pos[(m, c)] = y; y += 1
    y += 0.9
ymax = y - 0.9
fig, axs = plt.subplots(1, 3, figsize=(6.8, 4.4), constrained_layout=True, sharey=True)
mets = [('L5', r'$\Delta L_5$ (pp)'), ('short', r'$\Delta$ shortage (pp)'), ('energy', r'$\Delta$ pump energy (GWh)')]
for ax, (k, lab), pl in zip(axs, mets, 'abc'):
    for (m, c), y in pos.items():
        r = P[(P.mult == m) & (P.a == c) & (P.b == 'MPC')]
        if r.empty: continue
        r = r.iloc[0]
        ax.plot([r[k + '_lo'], r[k + '_hi']], [y, y], '-', color=ENT[c]['color'], lw=1.6, solid_capstyle='butt')
        ax.plot(r[k + '_d'], y, ENT[c]['marker'], color=ENT[c]['color'], ms=5, mec='white', mew=0.6, zorder=3)
    ax.axvline(0, color=style.INK2, lw=0.7); ax.set_xlabel(lab); style.panel(ax, pl); ax.grid(axis='y', visible=False)
    for m in mults:
        ax.text(0.015, pos[(m, ctls[0])] - 0.95, r'true $\mu$ = %g' % m, transform=ax.get_yaxis_transform(), fontsize=9, fontweight='bold', ha='left', va='center', color=style.INK)
axs[0].set_yticks(list(pos.values())); axs[0].set_yticklabels([ENT[c]['label'] for (m, c) in pos], fontsize=9)
axs[0].set_ylim(ymax + 0.5, -0.3)
fig.savefig('Fig_paired.png'); print('ok')
