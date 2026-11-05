"""Open-loop landscape of constant level shifts: seasonal cost J (same terms as the planner) for each true multiplier."""
import json, numpy as np, pandas as pd
from sim import Scenario, Plant
from canal import CanalParams
from sediment import SedimentParams, RHO_S, POROSITY
R = json.load(open('../results/landscape.json'))
sc = Scenario(Plant(seed=0, cp=CanalParams(N=114)).canal, seed=0)
intended = sum(float(np.sum(sc.actual(k * 3600.0))) * 3600.0 for k in range(245 * 24))
price, c_sh, c_sp, c_dep, lam = 0.03, 0.10, 0.02, 3.0, 2e7
rows = []
for r in R:
    c = r['cfg']; dz = np.array(r['dz_prof'])
    dep_vol = r['dep_kt'] * 1e6 / (RHO_S * (1 - POROSITY))
    rows.append(dict(mult=c['mult'], shift=tuple(c['shift']), L5=100 * r['L5'], Lsys=100 * r['Lsys'], dep_kt=r['dep_kt'], short=100 * r['short'], spill=100 * r['spill'],
                     E=r['energy'], hmin=r['hmin'],
                     J_E=price * r['energy'] * 1e6, J_sh=c_sh * r['short'] * intended, J_sp=c_sp * r['spill'] * intended, J_dep=c_dep * dep_vol, J_L=lam * r['L5']))
D = pd.DataFrame(rows)
D['J'] = D[['J_E', 'J_sh', 'J_sp', 'J_dep', 'J_L']].sum(axis=1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
print('intended volume %.3e m3' % intended)
for m, g in D.groupby('mult'):
    g = g.copy(); z = g[g['shift'] == (0, 0, 0, 0)]
    if len(z): g['dJ_k$'] = (g.J - z.J.iloc[0]) / 1e3
    print('\n=== true multiplier', m)
    print(g.sort_values('J')[['shift', 'L5', 'Lsys', 'dep_kt', 'short', 'E', 'hmin', 'dJ_k$']].round(3).to_string(index=False))
D.to_csv('../results/landscape.csv', index=False)
