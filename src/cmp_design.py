"""Compare the closed-loop planner runs of the design check (seed 0) with the open-loop landscape of constant shifts (same seed, same cost J)."""
import json, sys, numpy as np, pandas as pd
from sim import Scenario, Plant
from canal import CanalParams
from sediment import RHO_S, POROSITY
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 2e7
FILE = sys.argv[2] if len(sys.argv) > 2 else '../results/design_check.json'
sc = Scenario(Plant(seed=0, cp=CanalParams(N=114)).canal, seed=0)
intended = sum(float(np.sum(sc.actual(k * 3600.0))) * 3600.0 for k in range(245 * 24))
price, c_sh, c_sp, c_dep = 0.03, 0.10, 0.02, 3.0


def J(r):
    dep_vol = r['dep_kt'] * 1e6 / (RHO_S * (1 - POROSITY))
    return price * r['energy'] * 1e6 + c_sh * r['short'] * intended + c_sp * r['spill'] * intended + c_dep * dep_vol + LAM * r['L5']


def row(r, name):
    return dict(name=name, L5=100 * r['L5'], Lsys=100 * r['Lsys'], dep=r['dep_kt'], short=100 * r['short'], spill=100 * r['spill'], E=r['energy'], J=J(r) / 1e6)


L = json.load(open('../results/landscape.json'))
D = json.load(open(FILE))
pd.set_option('display.width', 200)
for m in (0.4, 1.0, 2.5):
    rows = []
    for r in L:
        if r['cfg']['mult'] == m:
            rows.append(row(r, 'const ' + ','.join('%g' % v for v in r['cfg']['shift'][:3])))
    for r in D:
        if r['cfg']['mult'] == m:
            rows.append(row(r, r['cfg']['ctl']))
    if not rows:
        continue
    T = pd.DataFrame(rows).sort_values('J')
    print('\n=== true multiplier %g  (lambda %.0e, J in M USD)' % (m, LAM))
    print(T.round(3).to_string(index=False))
