"""Weekly decisions of the planner: reduction of the level references of the head pools (a) and of pool 3 (b) against time, for the nominal, the learned and the
true-capacity twin; mean over seeds (lines) and range over seeds (bands), one column per true multiplier."""
import json, os, numpy as np, matplotlib.pyplot as plt
import style; style.setup(); C = style.C
RES = os.environ.get('RES', '../results/main.json')
R = [r for r in json.load(open(RES)) if 'error' not in r and r['cfg'].get('tag') == 'main' and r['cfg']['ctl'].startswith('Morph') and 'plan_hist' in r]
mults = sorted({r['cfg']['mult'] for r in R})
ctls = [c for c in ('Morph-nominal', 'Morph-learn', 'Morph-oracle') if any(r['cfg']['ctl'] == c for r in R)]
fig, axs = plt.subplots(2, len(mults), figsize=(6.9, 4.4), sharex=True, sharey=True, constrained_layout=True)
axs = np.atleast_2d(axs)
for j, m in enumerate(mults):
    for i, (idx, lab) in enumerate(((0, 'Head pools, $a$ (m)'), (2, 'Pool 3, $b$ (m)'))):
        ax = axs[i, j]
        for c in ctls:
            P = [np.array(r['plan_hist'])[:, idx] for r in R if r['cfg']['mult'] == m and r['cfg']['ctl'] == c]
            if not P:
                continue
            n = min(len(p) for p in P); X = np.array([p[:n] for p in P]); wk = np.arange(1, n + 1)
            e = style.ENT[c]
            ax.fill_between(wk, X.min(0), X.max(0), color=e['color'], alpha=0.14, lw=0)
            ax.plot(wk, X.mean(0), color=e['color'], marker=e['marker'], ms=3, lw=1.1, label=e['label'])
        ax.set_ylim(-0.03, 0.63)
        if j == 0:
            ax.set_ylabel(lab)
        if i == 0:
            ax.set_title(r'true $\mu$ = %g' % m, fontsize=10)
        if i == 1:
            ax.set_xlabel('Week of the season')
        style.panel(ax, 'abcdef'[i * len(mults) + j])
axs[0, 0].legend(loc='upper left', frameon=False, fontsize=8.5, handlelength=1.6)
fig.savefig('Fig_planner.png'); print('ok')
