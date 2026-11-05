"""Sensitivity of the learned planner at mu = 0.4 (three seeds): change of L5 and of shortage relative to predictive control."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import style; style.setup(); C = style.C
T = pd.read_csv(os.environ.get('SENS', '../results/sens.csv'))
order = ['default (E1, seeds 1-3)', 'lambda=2e+06', 'lambda=7e+06', 'lambda=6e+07', 'lambda=2e+08', 'survey every 15 d', 'survey every 60 d', 'survey noise 0.01 m', 'survey noise 0.05 m',
         'omega=1.0, gamma=4', 'omega=0.9, gamma=2', 'omega=0.9, gamma=8', 'turbidity x0.5', 'turbidity x2.0']
lab = {'default (E1, seeds 1-3)': 'Default', 'lambda=2e+06': r'$\lambda$ = 2$\times$10$^6$', 'lambda=7e+06': r'$\lambda$ = 7$\times$10$^6$', 'lambda=6e+07': r'$\lambda$ = 6$\times$10$^7$',
       'lambda=2e+08': r'$\lambda$ = 2$\times$10$^8$', 'survey every 15 d': 'Survey 15 d', 'survey every 60 d': 'Survey 60 d', 'survey noise 0.01 m': 'Noise 0.01 m', 'survey noise 0.05 m': 'Noise 0.05 m',
       'omega=1.0, gamma=4': r'$\omega$ = 1, $\gamma$ = 4', 'omega=0.9, gamma=2': r'$\omega$ = 0.9, $\gamma$ = 2', 'omega=0.9, gamma=8': r'$\omega$ = 0.9, $\gamma$ = 8',
       'turbidity x0.5': 'Turbidity $\\times$0.5', 'turbidity x2.0': 'Turbidity $\\times$2'}
T = T.set_index('variant'); keys = [k for k in order if k in T.index]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 3.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
for i, k in enumerate(keys):
    r = T.loc[k]; col = C['orange'] if k.startswith('default') else C['blue']
    a.plot([r.L5_lo, r.L5_hi], [i, i], '-', color=col, lw=1.6, solid_capstyle='butt'); a.plot(r.L5_d, i, 'D' if k.startswith('default') else 'o', color=col, ms=4.5, mec='white', mew=0.6, zorder=3)
a.axvline(0, color=style.INK2, lw=0.7); a.set_yticks(range(len(keys))); a.set_yticklabels([lab[k] for k in keys], fontsize=9); a.invert_yaxis(); a.grid(axis='y', visible=False)
a.set_xlabel(r'Change of $L_5$ relative to predictive control (pp)'); style.panel(a, 'a')
lam = [k for k in keys if k.startswith('lambda') or k.startswith('default')]
xs = [T.loc[k].short_d for k in lam]; ys = [T.loc[k].L5_d for k in lam]
b.plot(xs, ys, '-', color=C['gray'], lw=0.8, zorder=1)
for k, x, y in zip(lam, xs, ys):
    b.plot(x, y, 'D' if k.startswith('default') else 'o', color=C['orange'] if k.startswith('default') else C['blue'], ms=5, mec='white', mew=0.6, zorder=3)
    if k in ('lambda=6e+07', 'lambda=2e+08'):
        continue
    off = (-70, 6) if k.startswith('default') else (4, 4)
    b.annotate(lab[k].replace('Default', r'default 2$\times$10$^7$'), (x, y), xytext=off, textcoords='offset points', fontsize=8.5, color=style.INK2)
b.annotate(r'$\lambda$ = 6$\times$10$^7$, 2$\times$10$^8$', (xs[3], ys[3]), xytext=(-130, -2), textcoords='offset points', fontsize=8.5, color=style.INK2)
b.axhline(0, color=style.INK2, lw=0.7); b.axvline(0, color=style.INK2, lw=0.7); b.set_ylim(-3.7, 0.3)
b.set_xlabel('Change of shortage (pp)'); b.set_ylabel(r'Change of $L_5$ (pp)'); style.panel(b, 'b')
fig.savefig('Fig_sens.png'); print('ok')
