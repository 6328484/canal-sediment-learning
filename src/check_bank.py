"""Does the bank-based rollout from the start of the season rank the candidate shifts like the coupled plant (open-loop landscape, design seed 0)?
Uses the true multiplier (oracle twin) and the climatological turbidity and demand."""
import json, sys, copy, time, numpy as np, pandas as pd
from sim import Plant, Scenario
from canal import CanalParams
from pump import PumpParams
from sediment import SedimentParams, RHO_S, POROSITY
from forecast import Bank, Climatology
GRID = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 2e7
price, c_sh, c_sp, c_dep = 0.03, 0.10, 0.02, 3.0
cp, pp = CanalParams(N=114), PumpParams()
bank = Bank.get(cp, pp, GRID, dt=3600.0)
sc = Scenario(Plant(seed=0, cp=cp).canal, seed=0)
intended = sum(float(np.sum(sc.actual(k * 3600.0))) * 3600.0 for k in range(245 * 24))
clim = Climatology(Plant(seed=0, cp=cp).canal, 60, 245)
L = json.load(open('../results/landscape.json'))
pd.set_option('display.width', 220)
for mu in (0.4, 1.0, 2.5):
    sp = SedimentParams(mult=mu); sp.exp_fit = 1; sp.win = 5000.0
    pl = Plant(seed=500, sp=sp, pp=pp, cp=copy.deepcopy(cp)); pl.canal.par.dt = 3600.0
    c_bed = (1 - sp.trap) * sp.f_bed * clim.seas[np.arange(bank.hours) // 24]
    rows = []
    t0 = time.time()
    for s in GRID:
        J, d = bank.evaluate(bank.index(s), pl, 0, c_bed, price, c_sh, c_sp, c_dep, LAM)
        r = [x for x in L if x['cfg']['mult'] == mu and [float(v) for v in x['cfg']['shift']] == list(s)]
        row = dict(shift=','.join('%g' % v for v in s[:3]), J_pred=J / 1e6, L5_pred=100 * d['L5'], dep_pred=d['dep_vol'] * RHO_S * (1 - POROSITY) / 1e6, short_pred=100 * d['short'] / intended, E_pred=d['E'] / 1e6)
        if r:
            r = r[0]
            Jp = price * r['energy'] * 1e6 + c_sh * r['short'] * intended + c_sp * r['spill'] * intended + c_dep * r['dep_kt'] * 1e6 / (RHO_S * (1 - POROSITY)) + LAM * r['L5']
            row.update(J_plant=Jp / 1e6, L5_plant=100 * r['L5'], dep_plant=r['dep_kt'], short_plant=100 * r['short'], E_plant=r['energy'])
        rows.append(row)
    T = pd.DataFrame(rows)
    print('\n=== true multiplier %g (%.0f s)  J in M USD (lambda %.0e)' % (mu, time.time() - t0, LAM))
    print(T.round(3).to_string(index=False))
    if 'J_plant' in T:
        print(' rank correlation of J (pred vs plant): %.2f | best predicted %s, best in plant %s' % (T.J_pred.rank().corr(T.J_plant.rank()), T['shift'][T.J_pred.idxmin()], T['shift'][T.J_plant.idxmin()]))
