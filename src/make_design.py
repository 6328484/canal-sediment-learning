"""Design of the planner from the open-loop landscape (design seed 0).

Candidate set (fixed a priori from the two deposition zones found in the landscape): shifts (a, a, b, 0), a = reduction of the level
references of pools 1 and 2 (head of the canal), b = reduction in pool 3 (below the largest offtake), a, b in {0, 0.3, 0.6} m; pool 4 is
left out because it has no deposition zone and a lower level there only costs deliveries.
Benchmarks, chosen from the same set by the seasonal cost J of the planner (same terms, same weights):
  FixedBest  best constant shift for each true multiplier (needs the truth),
  Fixed      constant shift with the smallest mean J over the true multipliers (one shift for all canals).
Writes ../results/design.json and ../results/landscape_J.csv."""
import json, os, sys, numpy as np, pandas as pd
from sim import Scenario, Plant
from canal import CanalParams
from sediment import RHO_S, POROSITY
R = json.load(open('../results/landscape.json'))
LAMS = (2e6, 7e6, 2e7, 6e7, 2e8); LAM0 = 2e7
GRID = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]
sc = Scenario(Plant(seed=0, cp=CanalParams(N=114)).canal, seed=0)
intended = sum(float(np.sum(sc.actual(k * 3600.0))) * 3600.0 for k in range(245 * 24))
price, c_sh, c_sp, c_dep = 0.03, 0.10, 0.02, 3.0
rows = []
for r in R:
    c = r['cfg']; sh = tuple(float(v) for v in c['shift'])
    dep_vol = r['dep_kt'] * 1e6 / (RHO_S * (1 - POROSITY))
    rows.append(dict(mult=c['mult'], shift=sh, grid=sh in GRID, L5=r['L5'], Lsys=r['Lsys'], dep=r['dep_kt'], short=r['short'], energy=r['energy'],
                     J0=price * r['energy'] * 1e6 + c_sh * r['short'] * intended + c_sp * r['spill'] * intended + c_dep * dep_vol))
D = pd.DataFrame(rows)
for lam in LAMS: D['J_%.0e' % lam] = D.J0 + lam * D.L5
D.to_csv('../results/landscape_J.csv', index=False)
G = D[D.grid]
mults = sorted(G.mult.unique())
ok = [m for m in mults if len(G[G.mult == m]) == len(GRID)]
print('landscape: %d runs, grid runs per multiplier:' % len(D), {m: len(G[G.mult == m]) for m in mults}, '| intended volume %.4e m3' % intended)
best = {}
for m in ok:
    g = G[G.mult == m]
    best[m] = {lam: g.loc[g['J_%.0e' % lam].idxmin(), 'shift'] for lam in LAMS}
    print('  mu=%g best constant shift by weight: ' % m + '  '.join('%.0e->%s' % (lam, ','.join('%g' % v for v in best[m][lam])) for lam in LAMS))
col = 'J_%.0e' % LAM0
piv = G[G.mult.isin(ok)].pivot(index='shift', columns='mult', values=col)
mean = piv.mean(axis=1); fixed = mean.idxmin()
print('\nseasonal cost J at the default weight (M USD), rows = constant shift, columns = true multiplier:')
print((piv / 1e6).round(3).assign(mean=(mean / 1e6).round(3)).to_string())
zero = (0, 0, 0, 0)
print('\nFixed benchmark (smallest mean J):', fixed, '| mean J lower than no shift by %.0f k$' % ((mean[zero] - mean[fixed]) / 1e3))
bestm = {('%g' % m): list(best[m][LAM0]) for m in ok}
print('FixedBest per true multiplier:', bestm)
Pd = {(tuple(float(v) for v in idx), m): float(piv[m].iloc[j]) for j, idx in enumerate(piv.index) for m in ok}
vi = sum(Pd[(tuple(float(v) for v in bestm['%g' % m]), m)] for m in ok) / len(ok)
print('mean J: no shift %.3f | Fixed %.3f | best per multiplier %.3f (M USD); value of knowing the multiplier = %.0f k$ (Fixed - best), no shift - best = %.0f k$' % (
    piv[ok].mean(axis=1)[zero] / 1e6, mean[fixed] / 1e6, vi / 1e6, (mean[fixed] - vi) / 1e3, (mean[zero] - vi) / 1e3))
json.dump(dict(cands=[list(c) for c in GRID], fixed=list(fixed), best=bestm, lam=LAM0, lams=list(LAMS)), open('../results/design.json', 'w'))
