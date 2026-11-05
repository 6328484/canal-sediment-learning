"""Design figure. Top: open-loop landscape of the nine constant shifts (a, a, b, 0) (capacity loss versus water shortage, design seed) for the three
true multipliers. Bottom: bed aggradation at the end of the season for selected shifts (mu = 0.4 and 1) and the mechanism at the head of the canal."""
import json, os, numpy as np, matplotlib.pyplot as plt
import style; style.setup(); C = style.C
R = json.load(open('../results/landscape.json')); MEC = json.load(open('../results/mech.json'))
D = json.load(open('../results/design.json')) if os.path.exists('../results/design.json') else None
GRID = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]
def fam(s):
    a, b = s[0], s[2]
    return 'none' if not (a or b) else 'head (pools 1, 2)' if a and not b else 'pool 3' if b and not a else 'head and pool 3'
FAM = {'none': ('#0b0b0b', 'o'), 'head (pools 1, 2)': (C['orange'], 'D'), 'pool 3': (C['aqua'], '^'), 'head and pool 3': (C['violet'], 's')}
mults = (0.4, 1.0, 2.5)
fig = plt.figure(figsize=(6.9, 6.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
axs = [fig.add_subplot(gs[0, j]) for j in range(3)]
for ax, m, pl in zip(axs, mults, 'abc'):
    pts = [r for r in R if r['cfg']['mult'] == m and tuple(float(v) for v in r['cfg']['shift']) in GRID]
    for r in pts:
        s = tuple(float(v) for v in r['cfg']['shift']); f = fam(s); col, mk = FAM[f]
        ax.plot(100 * r['short'], 100 * r['L5'], mk, color=col, ms=5.5, mew=0.6, mec='white', zorder=3)
        for key, lab, off in (('best', 'B', (-9, 3)), ('fixed', 'F', (4, -11))):
            if D and list(s) == list(D['best'].get('%g' % m, D['fixed']) if key == 'best' else D['fixed']):
                ax.annotate(lab, (100 * r['short'], 100 * r['L5']), xytext=off, textcoords='offset points', fontsize=9, fontweight='bold', color=style.INK)
    ax.set_xlabel('Water shortage (%)')
    if pl == 'a': ax.set_ylabel('Capacity loss $L_5$ (%)')
    ax.set_title(r'true $\mu$ = %g' % m, fontsize=10); style.panel(ax, pl)
hs = [plt.Line2D([], [], marker=mk, color=col, ls='', ms=5.5, label=f) for f, (col, mk) in FAM.items()]
axs[0].legend(handles=hs, loc='upper right', frameon=False, fontsize=8.5, handletextpad=0.2, labelspacing=0.25, borderpad=0.2)
b = [fig.add_subplot(gs[1, j]) for j in range(3)]
def prof(m, s):
    rr = [r for r in R if r['cfg']['mult'] == m and [float(v) for v in r['cfg']['shift']] == list(s)]
    if not rr: return None
    dz = np.array(rr[0]['dz_prof']); return (np.arange(len(dz)) + 0.5) * 56.6 / len(dz), 100 * dz
cols = [style.INK2, C['orange'], C['aqua'], C['violet']]
for ax, m, sh, pl in ((b[0], 0.4, [(0, 0, 0, 0), (0.3, 0.3, 0, 0), (0.3, 0.3, 0.6, 0)], 'd'), (b[1], 1.0, [(0, 0, 0, 0), (0, 0, 0.6, 0), (0.3, 0.3, 0, 0)], 'e')):
    for s, col in zip(sh, cols):
        p = prof(m, s)
        if p: ax.plot(p[0], p[1], color=col, label='(' + ','.join('%g' % v for v in s) + ')')
    ax.set_xlabel('Distance from the pumps (km)'); ax.set_xlim(0, 56.6); style.panel(ax, pl)
    ax.set_title(r'true $\mu$ = %g' % m, fontsize=10)
    for xg in (14.2, 28.3, 42.5): ax.axvline(xg, color=style.INK2, lw=0.5, ls=':')
    ax.legend(title='shift (m)', loc='upper right', frameon=False, fontsize=8.5, title_fontsize=8.5, handlelength=1.4)
b[0].set_ylabel('Aggradation at day 245 (cm)')
for s, col in zip([(0, 0, 0, 0), (0.3, 0.3, 0, 0), (0.6, 0.6, 0, 0)], [style.INK2, C['orange'], C['magenta']]):
    r = MEC['0.4|' + ','.join('%g' % v for v in s)]; x = np.array(r['x_km']); k = x <= 14
    b[2].plot(x[k], 1e3 * np.array(r['ceq'])[k], color=col)
    b[2].plot(x[k], 1e3 * np.array(r['C'])[k], color=col, ls='--', lw=1.1)
b[2].set_xlabel('Distance from the pumps (km)'); b[2].set_ylabel('Concentration (mg L$^{-1}$)'); b[2].set_xlim(0, 14); b[2].set_ylim(0, 24)
b[2].set_title(r'true $\mu$ = 0.4, day 7', fontsize=10); style.panel(b[2], 'f')
b[2].text(0.97, 0.96, 'solid: capacity\ndashed: suspended', transform=b[2].transAxes, fontsize=8.5, ha='right', va='top', color=style.INK2)
fig.savefig('Fig_design.png'); print('ok')
