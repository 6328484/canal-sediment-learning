"""Open-loop landscape of constant level shifts (design seed 0): capacity loss versus water shortage, one panel per true multiplier."""
import json, os, numpy as np, matplotlib.pyplot as plt
import style; style.setup(); C = style.C
R = json.load(open('../results/landscape.json'))
D = json.load(open('../results/design.json')) if os.path.exists('../results/design.json') else dict(cands=[], fixed=None)
fam = lambda s: ('nominal' if not any(s) else 'pool 1' if s[0] and not any(s[1:]) else 'pools 1-2' if s[0] and s[1] and not (s[2] or s[3])
                 else 'pools 3-4' if not (s[0] or s[1]) else 'all pools' if all(s) else 'pools 1+3')
FAM = {'nominal': ('#0b0b0b', 'o'), 'pool 1': (C['blue'], 'o'), 'pools 1-2': (C['orange'], 'D'), 'pools 3-4': (C['aqua'], '^'),
       'pools 1+3': (C['violet'], 's'), 'all pools': (C['magenta'], 'v')}
mults = sorted({r['cfg']['mult'] for r in R})
fig, axs = plt.subplots(1, len(mults), figsize=(6.5, 2.7), constrained_layout=True, sharey=False)
axs = np.atleast_1d(axs)
for ax, m, pl in zip(axs, mults, 'abc'):
    rs = [r for r in R if r['cfg']['mult'] == m]
    for r in rs:
        s = tuple(r['cfg']['shift']); f = fam(s); col, mk = FAM[f]
        sel = list(s) in [list(c) for c in D['cands']]
        ax.plot(100 * r['short'], 100 * r['L5'], mk, color=col, mfc=col if sel or f == 'nominal' else 'white', mew=1.2, ms=5.5 if sel else 4.5, zorder=3)
        if sel or f == 'nominal' or (D.get('fixed') and list(s) == list(D['fixed'])):
            ax.annotate('(%s)' % ','.join(('%g' % v) for v in s), (100 * r['short'], 100 * r['L5']), xytext=(4, 3), textcoords='offset points', fontsize=6.3, color=style.INK2)
    ax.set_xlabel('Water shortage (% of deliveries)')
    if pl == 'a': ax.set_ylabel('Capacity loss $L_5$ (%)')
    ax.set_title(r'true $\mu$ = %g' % m, fontsize=8.5)
    style.panel(ax, pl)
hs = [plt.Line2D([], [], marker=mk, color=col, ls='', mfc=col, ms=5, label=f) for f, (col, mk) in FAM.items()]
fig.legend(handles=hs, loc='outside lower center', ncol=6, frameon=False, fontsize=7)
fig.savefig('Fig_landscape.png'); print('ok')
