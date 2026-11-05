"""Learning of the capacity multiplier: estimate, separation predicted by Proposition 3, final weights."""
import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import style; style.setup(); C = style.C
RES = os.environ.get('RES', '../results/main.json'); IDF = os.environ.get('IDENT', '../results/ident.csv')
R = [r for r in json.load(open(RES)) if 'error' not in r and r['cfg'].get('tag') == 'main' and r['cfg']['ctl'] == 'Morph-learn' and 'mult_hist' in r]
I = pd.read_csv(IDF) if os.path.exists(IDF) else pd.DataFrame()
mults = sorted({r['cfg']['mult'] for r in R}); col = dict(zip(mults, [C['blue'], C['orange'], C['aqua']]))
mk = dict(zip(mults, ['o', 'D', '^']))
H = np.array([0.35, 0.6, 1.0, 1.7, 3.0])
fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.2), constrained_layout=True)
a, b, c = axs
for m in mults:
    T = [np.array(r['mult_hist']) for r in R if r['cfg']['mult'] == m]
    n = min(len(t) for t in T); M = np.array([t[:n, 1] for t in T]); day = 30 * T[0][:n, 0]
    a.fill_between(day, M.min(0), M.max(0), color=col[m], alpha=0.18, lw=0)
    a.plot(day, M.mean(0), '-' + mk[m], color=col[m], label=r'%g' % m, ms=3.5)
    a.axhline(m, color=col[m], lw=0.8, ls='--')
a.set_yscale('log'); a.set_yticks([0.35, 0.5, 1, 2, 3]); a.set_yticklabels(['0.35', '0.5', '1', '2', '3']); a.minorticks_off()
a.set_xlabel('Time (days)'); a.set_ylabel(r'Estimated multiplier $\hat\mu$'); a.legend(title=r'true $\mu$', loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9, title_fontsize=9, columnspacing=1.0, handlelength=1.6); style.panel(a, 'a')
if len(I):
    for m in mults:
        g = I[I.mult == m].groupby('m').z
        day = 30 * g.median().index.values
        b.fill_between(day, g.min().values, g.max().values, color=col[m], alpha=0.15, lw=0)
        b.plot(day, g.median().values, '-' + mk[m], color=col[m], ms=3.5)
    for z0, lab in ((1.28, '10%'), (3.09, '0.1%')):
        b.axhline(z0, color=style.INK2, lw=0.6, ls=':'); b.text(0.98, z0 * 1.08, lab, fontsize=9, color=style.INK2, ha='right', va='bottom', transform=b.get_yaxis_transform())
    b.set_yscale('symlog', linthresh=1.0, linscale=0.6); b.set_ylim(-150, 30); b.set_yticks([-100, -10, -1, 0, 1, 3, 10, 30]); b.set_yticklabels(['-100', '-10', '-1', '0', '1', '3', '10', '30']); b.minorticks_off(); b.axhline(0, color=style.INK2, lw=0.6)
b.set_xlabel('Time (days)'); b.set_ylabel('Separation $z_m$ of the nearest rival'); style.panel(b, 'b')
# final weights, mean over seeds
for j, m in enumerate(mults):
    Wt = []
    for r in R:
        if r['cfg']['mult'] != m: continue
        l = np.array(r['loglik_hist'][-1]); w = np.exp(l - l.max()); Wt.append(w / w.sum())
    Wt = np.mean(Wt, axis=0); bottom = 0
    for k in range(5):
        cc = [C['gray'], C['yellow'], C['aqua'], C['violet'], C['magenta']][k]
        c.bar(j, Wt[k], bottom=bottom, color=cc, width=0.62, edgecolor='white', linewidth=0.8, label=r'%g' % H[k] if j == 0 else None); bottom += Wt[k]
c.set_xticks(range(len(mults))); c.set_xticklabels([r'%g' % m for m in mults]); c.set_xlabel(r'true $\mu$'); c.set_ylabel('Weight at the end of the season')
c.legend(title=r'hypothesis $\mu_k$', loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=8.5, title_fontsize=8.5, columnspacing=0.7, handlelength=0.9, handletextpad=0.4)
c.grid(False); style.panel(c, 'c')
fig.savefig('Fig_learning.png'); print('ok')
