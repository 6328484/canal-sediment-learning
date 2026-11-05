"""Fixed pool-level shifts under the predictive controller: landscape of siltation metrics (design of the planner)."""
import json, os, sys, time, numpy as np
from multiprocessing import Pool
from sim import *
from sediment import SedimentParams
from canal import CanalParams
from mpc import MPC
import gvf
F = '../results/landscape.json'
FIT = int(os.environ.get('FIT', '1'))
NGRID = int(os.environ.get('NGRID', '114'))
Q0 = gvf.capacity(CanalParams(), np.zeros(10), tol=1e-5)


from morph import FixedShift


def one(cfg):
    sp = SedimentParams(mult=cfg['mult']); sp.exp_fit = FIT; sp.win = 5000.0
    cp = CanalParams(**{'N': NGRID, **cfg.get('cp', {})})
    plant = Plant(seed=cfg['seed'], sp=sp, cp=cp); scen = Scenario(plant.canal, seed=cfg['seed'])
    t0 = time.time()
    rec, tot = run(plant, scen, FixedShift(cfg['shift']))
    s = plant.sed; he = np.array([r['h_end'] for r in rec])
    Lsys = gvf.capacity_loss(plant.canal.par, s.dz, s.sp.kappa_n, q0=Q0, dxf=100.0)[0]
    return dict(cfg=cfg, L5=float(s.capacity_loss()[0]), L2500=float(s.capacity_loss(window=2500.0)[0]),
                L10k=float(s.capacity_loss(window=10000.0)[0]), Lsys=float(Lsys), dz_max=float(s.dz.max()),
                dep_kt=float(s.mass_dep / 1e6), short=tot['short'] / tot['intended'], spill=tot['spill'] / tot['intended'],
                energy=float(plant.pump.energy_kwh / 1e6), hmin=float(he.min()), secs=time.time() - t0,
                dz_prof=[float(v) for v in s.dz])


def key(c): return json.dumps(c, sort_keys=True)


if __name__ == "__main__":
    res = json.load(open(F)) if os.path.exists(F) else []
    done = {key(r['cfg']) for r in res}
    # depth reduction (m) of the level references of pools 1-4; the deposit forms in pool 1 (low capacity) and below the large offtake (pool 3)
    grid = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]                       # planner candidates: head pools 1-2 x pool 3
    extra = [(0.3, 0, 0, 0), (0.6, 0, 0, 0), (0.9, 0, 0, 0), (0.6, 0, 0.6, 0), (0, 0, 0.9, 0), (0, 0, 0.6, 0.6), (0.3, 0.3, 0.3, 0.3), (0.6, 0.6, 0.6, 0.6), (0.9, 0, 0.6, 0)]   # exploratory
    shifts = grid + extra if os.environ.get('STAGE', 'all') == 'all' else (grid if os.environ.get('STAGE') == 'grid' else extra)
    J = [dict(shift=list(s), seed=0, mult=m, fit=FIT) for s in shifts for m in (0.4, 1.0, 2.5)]
    J = [j for j in J if key(j) not in done]
    print('to run', len(J), flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(one, J):
            res.append(r); json.dump(res, open(F, 'w'))
            print(r['cfg']['mult'], r['cfg']['shift'], 'L5=%.4f Lsys=%.4f dep=%.1f short=%.4f E=%.1f hmin=%.2f' % (r['L5'], r['Lsys'], r['dep_kt'], r['short'], r['energy'], r['hmin']), '%ds' % r['secs'], flush=True)
