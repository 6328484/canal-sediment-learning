import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import style; style.setup(); C = style.C
plt.rcParams['axes.grid'] = False
import json, os
NC = len(json.load(open('../results/design.json'))['cands']) if os.path.exists('../results/design.json') else None
fig = plt.figure(figsize=(6.8, 6.0))
# ---------------- (a) canal sketch
ax = fig.add_axes([0.03, 0.60, 0.95, 0.38]); ax.set_xlim(-5, 63); ax.set_ylim(-4.9, 6.3); ax.axis('off')
L = 56.6
x = np.linspace(0, L, 400); bed = -0.0022 * x; wl = bed + 1.4
ax.fill_between(x, bed - 0.7, bed, color='#c9c8c3', lw=0)
ax.fill_between(x, bed, wl, color='#cfe0f6', lw=0)
ax.plot(x, wl, color=C['blue'], lw=1.0)
for x0, x1, hgt in ((0, 12, 0.60), (35, 40, 0.30)):
    xs = np.linspace(x0, x1, 60); prof = hgt * np.exp(-(xs - x0) / ((x1 - x0) / 3.0))
    ax.fill_between(xs, np.interp(xs, x, bed), np.interp(xs, x, bed) + prof, color=C['orange'], lw=0)
ax.text(1.0, 2.75, 'deposit at the head\n(low capacity)', fontsize=9, color=style.INK2, ha='left', va='bottom', linespacing=1.1)
ax.annotate('', xy=(3.0, 0.15), xytext=(3.4, 2.65), arrowprops=dict(arrowstyle='-', color=style.INK2, lw=0.5))
ax.text(31.0, 2.75, 'deposit below the\nlarge offtake', fontsize=9, color=style.INK2, ha='left', va='bottom', linespacing=1.1)
ax.annotate('', xy=(36.8, -0.1), xytext=(36.2, 2.65), arrowprops=dict(arrowstyle='-', color=style.INK2, lw=0.5))
ax.add_patch(Rectangle((-4.4, wl[0] - 1.9), 2.6, 3.1, fc='#52514e', ec='none'))
ax.text(-3.1, wl[0] + 1.35, 'pumps', fontsize=9, ha='center', va='bottom'); ax.text(-3.1, wl[0] - 0.4, '$Q_p$', fontsize=10, ha='center', va='center', color='white')
ax.annotate('', xy=(-0.2, wl[0] - 0.9), xytext=(-1.7, wl[0] - 0.9), arrowprops=dict(arrowstyle='-|>', color=style.INK2, lw=1.0))
for k, xc in enumerate((14.2, 28.3, 42.5)):
    yb = np.interp(xc, x, bed)
    ax.add_patch(Rectangle((xc - 0.35, yb), 0.7, 2.5, fc=style.INK2, ec='none'))
    ax.text(xc + (0.7 if k == 2 else 0.0), yb - 0.95, 'check %d' % (k + 1), fontsize=9, ha='right' if k == 2 else 'center', va='top')
for k, (a_, b_) in enumerate(((0, 14.2), (14.2, 28.3), (28.3, 42.5), (42.5, 56.6))):
    ax.annotate('', xy=(b_ - 0.4, 5.3), xytext=(a_ + 0.4, 5.3), arrowprops=dict(arrowstyle='<->', color=style.INK2, lw=0.7))
    ax.text(0.5 * (a_ + b_), 5.45, 'pool %d' % (k + 1), fontsize=9.5, ha='center', va='bottom')
for xo, sh in ((10, 0.04), (22, 0.05), (35, 0.50), (44, 0.12), (50, 0.24)):
    yb = np.interp(xo, x, wl)
    ax.add_patch(FancyArrowPatch((xo, yb), (xo, yb - 3.3), arrowstyle='-|>', mutation_scale=7, color=C['aqua'], lw=0.9 + 3.0 * sh))
    ax.text(xo + 0.8, yb - 3.55, '%d%%' % round(100 * sh), fontsize=9, color=style.INK2, ha='left', va='center')
ax.add_patch(FancyArrowPatch((L, wl[-1] - 0.5), (L + 4.0, wl[-1] - 0.5), arrowstyle='-|>', mutation_scale=7, color=C['aqua'], lw=1.2))
ax.text(L + 0.6, wl[-1] + 0.2, 'tail', fontsize=9)
for xt in (0, 10, 20, 30, 40, 50):
    ax.plot([xt, xt], [-3.55, -3.7], color=style.INK2, lw=0.6); ax.text(xt, -4.15, str(xt), fontsize=9, ha='center', color=style.INK2)
ax.plot([0, L], [-3.55, -3.55], color=style.INK2, lw=0.6); ax.text(L / 2, -4.55, 'distance from the pumping station (km)', fontsize=9.5, ha='center', va='top', color=style.INK2)
ax.text(-5, 6.3, 'a', fontsize=11, fontweight='bold', va='top')
# ---------------- (b) control architecture
bx = fig.add_axes([0.03, 0.02, 0.95, 0.54]); bx.set_xlim(0, 100); bx.set_ylim(0, 62); bx.axis('off')
bx.text(0, 62, 'b', fontsize=11, fontweight='bold', va='top')
def box(x0, y0, w, h, txt, fc, ec, fs=9):
    bx.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=ec, lw=1.0))
    bx.text(x0 + w / 2, y0 + h / 2, txt, ha='center', va='center', fontsize=fs, linespacing=1.3)
def arr(pts, txt=None, tx=None, ty=None, ha='left', fs=9):
    for p, q in zip(pts[:-2], pts[1:-1]):
        bx.plot([p[0], q[0]], [p[1], q[1]], color=style.INK2, lw=0.9, solid_capstyle='butt')
    bx.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=8, color=style.INK2, lw=0.9, shrinkA=0, shrinkB=0))
    if txt: bx.text(tx, ty, txt, fontsize=fs, ha=ha, va='center', color=style.INK2, linespacing=1.2)
BL, OR = '#e6eefa', '#fdeee6'
box(58, 3, 40, 13, 'Plant (synthetic)\nunsteady flow, sediment transport,\nbed evolution, pump energy', BL, C['blue'])
box(58, 27, 40, 11, 'Predictive controller\nhourly, 24 h horizon', BL, C['blue'])
box(61, 47, 37, 11, 'Planner (after each survey)\n%s level shifts $\\mathbf{s}$,\nrollout of the MAP twin through\na bank of precomputed flows' % ('%d candidate' % NC if NC else 'candidate'), OR, C['orange'])
box(1, 47, 21, 11, 'Bed survey\n14 stations,\nevery 30 days', '#f1f0ed', style.INK2)
box(28, 47, 21, 11, 'Bayesian weights\n$\\hat\\mu$, MAP twin', OR, C['orange'])
box(29, 24, 22, 14, 'Ensemble of $K=5$ twins\n$\\mu_k\\in\\{0.35,\\ldots,3\\}$\nsame flow model', OR, C['orange'])
arr([(70, 16), (70, 27)], 'noisy levels, flows', 71, 21.5)
arr([(88, 27), (88, 16)], 'pump and\ngate\ncommands', 89, 21.5)
arr([(80, 47), (80, 38)], 'level references\n$y^{ref}-\\mathbf{s}$', 81, 42.5)
arr([(22, 52.5), (28, 52.5)])
arr([(49, 52.5), (61, 52.5)], 'MAP twin', 55, 55.0, ha='center')
arr([(40, 38), (40, 47)], 'predicted\naggradation', 41, 42.5)
arr([(58, 8), (12, 8), (12, 47)], 'bed survey\n(with noise)', 13, 29)
arr([(58, 12), (40, 12), (40, 24)], 'actual pump discharge,\nmeasured turbidity', 41, 17.5)
bx.add_patch(Rectangle((1, 0.2), 2.2, 2.2, fc=BL, ec=C['blue'], lw=0.9)); bx.text(4, 1.3, 'operational layer', fontsize=9, va='center', color=style.INK2)
bx.add_patch(Rectangle((22, 0.2), 2.2, 2.2, fc=OR, ec=C['orange'], lw=0.9)); bx.text(25, 1.3, 'morphology layer', fontsize=9, va='center', color=style.INK2)
fig.savefig('Fig_system.png'); print('ok')
