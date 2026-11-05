"""Grid-convergence analysis of the plant (local controller, seed 0, mu=1)."""
import json, sys, numpy as np
from canal import CanalParams
import gvf
par = CanalParams()
Q0 = gvf.capacity(par, np.zeros(10), tol=1e-5)
rows = []
for f in ('../results/grid_diag.json', '../results/grid_diag2.json'):
    try:
        rows += json.load(open(f))
    except Exception:
        pass
def get(fit, N):
    for r in rows:
        if r['cfg']['fit'] == fit and r['cfg']['N'] == N:
            return r
for r in rows:
    if 'Lsys' not in r:
        r['Lsys'] = gvf.capacity_loss(par, np.array(r['dz_prof']), 0.5, q0=Q0)[0]
json.dump(rows, open('../results/grid_all.json', 'w'))
schemes = {False: 'upwind', True: 'fitted', 2: 'fitted+logmean'}
metrics = [('dep_kt', 'deposit (kt)'), ('L_W5000', 'L5'), ('Lsys', 'Lsys'), ('L_W10000', 'L10'), ('L_W2500', 'L2.5'), ('dz_max', 'dz_max (cell, m)'), ('L_peak', 'L_peak(node)')]
for fit in (False, True, 2):
    Ns = [N for N in (29, 57, 114, 228, 456) if get(fit, N)]
    if not Ns:
        continue
    print('=== scheme', schemes[fit])
    print('N      ' + ' '.join('%14s' % m[1] for m in metrics))
    for N in Ns:
        x = get(fit, N)
        print('%-6d ' % N + ' '.join('%14.5f' % x[m[0]] for m in metrics))
    # observed order and GCI from the three finest grids
    if len(Ns) >= 3:
        tri = Ns[-3:]
        for k, lab in metrics[:5]:
            f = [get(fit, N)[k] for N in tri]
            e21, e32 = f[1] - f[0], f[2] - f[1]
            if e21 * e32 > 0 and e32 != 0:
                p = np.log(abs(e21 / e32)) / np.log(2)
                ext = f[2] + (f[2] - f[1]) / (2 ** p - 1)
                gci = 1.25 * abs((f[2] - f[1]) / f[2]) / (2 ** p - 1)
                err57 = (get(fit, 57)[k] - ext) / ext * 100
                print('  %-14s grids %s  p=%.2f  extrap=%.5f  GCI(fine)=%.1f%%  err(N=57)=%+.1f%%' % (lab, tri, p, ext, 100 * gci, err57))
            else:
                print('  %-14s grids %s  non-monotone: %s' % (lab, tri, np.round(f, 5)))
