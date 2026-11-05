import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import style; style.setup(); C = style.C
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.1), constrained_layout=True)
# (a) steady relaxation error
r = np.array([0.99, 0.51, 0.25, 0.13])
conv = np.array([1.31e-1, 7.74e-2, 4.22e-2, 2.21e-2]); fit = np.array([1.08e-16, 2.17e-16, 4.33e-16, 1.19e-15])
a.loglog(r, conv, '-o', color=C['orange'], mfc='white', mew=1.2, label='Conventional coefficient')
a.loglog(r, fit, '-s', color=C['blue'], label='Fitted coefficient, Eq. (%d)' % style.eqn('fit'))
a.loglog([0.13, 1.0], [0.13 * 0.115, 0.115], ':', color=style.INK2, lw=0.9)
a.text(0.30, 0.008, 'slope 1', color=style.INK2, fontsize=9)
a.set_xlabel(r'Reach length / adaptation length, $\Delta x/L_a$'); a.set_ylabel('Maximum error of $C(x)$ (relative)')
a.set_ylim(1e-17, 3); a.set_xlim(0.1, 1.3)
a.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1.0])); a.xaxis.set_major_formatter(FixedFormatter(['0.1', '0.2', '0.5', '1'])); a.xaxis.set_minor_locator(NullLocator())
a.axhline(2.2e-16, color=style.INK2, lw=0.6, ls=':'); a.text(0.105, 3.6e-17, 'machine precision', color=style.INK2, fontsize=9)
a.legend(loc='center right', frameon=False, bbox_to_anchor=(1.02, 0.5)); style.panel(a, 'a')
# (b) convergence of outcomes relative to the extrapolated limits
rows = json.load(open('../results/grid_all.json'))
get = lambda N: [x for x in rows if x['cfg']['fit'] is True and x['cfg']['N'] == N][0]
Ns = [29, 57, 114, 228, 456]; dx = 56600.0 / np.array(Ns)
ext = dict(dep=56.6196, L5=0.05996, Ls=0.01841)
ser = {'Deposit mass': ([get(N)['dep_kt'] for N in Ns], ext['dep'], C['blue'], 'o'),
       r'$L_5$': ([get(N)['L_W5000'] for N in Ns], ext['L5'], C['orange'], 'D'),
       r'$L_{sys}$': ([get(N)['Lsys'] for N in Ns], ext['Ls'], C['aqua'], '^')}
for k, (v, e, col, m) in ser.items():
    b.semilogx(dx, np.array(v) / e, '-' + m, color=col, label=k)
b.axhline(1.0, color=style.INK2, lw=0.7)
b.axvline(497, color=style.INK2, lw=0.7, ls=':'); b.text(520, 1.285, 'N = 114\n(used)', color=style.INK2, fontsize=9, va='top')
b.set_xlabel(r'Reach length $\Delta x$ (m)'); b.set_ylabel('Value / extrapolated limit')
b.xaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000])); b.xaxis.set_major_formatter(FixedFormatter(['100', '200', '500', '1000', '2000'])); b.xaxis.set_minor_locator(NullLocator())
b.set_xlim(100, 2400); b.set_ylim(0.95, 1.4); b.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.0, 1.0)); style.panel(b, 'b')
fig.savefig('Fig_verify.png'); print('ok')
