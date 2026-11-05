"""Mechanism of the level-shift lever: steady-ish profiles (last day of a short run) of depth, velocity, equilibrium and actual concentration."""
import json, sys, time, numpy as np
from sim import *
from sediment import SedimentParams
from canal import CanalParams, trapezoid
from morph import FixedShift

def run_one(shift, mult, days=7, N=114):
    sp = SedimentParams(mult=mult); sp.exp_fit = 1; sp.win = 5000.0
    plant = Plant(seed=0, sp=sp, cp=CanalParams(N=N)); scen = Scenario(plant.canal, seed=0)
    ctl = FixedShift(shift)
    acc = {k: [] for k in ('ceq', 'C', 'v', 'h', 'Q', 'Ceq_over_mult')}
    orig = ctl.post_step
    def post(pl, sc, info, cmd, out):
        orig(pl, sc, info, cmd, out)
        k = int(round(info['t'] / 3600.0))
        if k >= (days - 1) * 24:
            s, c = pl.sed, pl.canal
            h = 0.5 * (c.h[s.p] + c.h[s.q]); Q = 0.5 * (c.Q[s.p] + c.Q[s.q])
            A, T, P = trapezoid(h, c.par.b, c.par.m)
            acc['ceq'].append(s.capacity().copy()); acc['C'].append(s.C.copy()); acc['v'].append(Q / A); acc['h'].append(h); acc['Q'].append(Q)
    ctl.post_step = post
    run(plant, scen, ctl, days=days)
    out = {k: np.mean(v, axis=0).tolist() for k, v in acc.items() if v}
    out['dz'] = plant.sed.dz.tolist(); out['days'] = days; out['x_km'] = ((np.arange(N) + 0.5) * 56.6 / N).tolist()
    return out

if __name__ == '__main__':
    res = {}
    for mult in (0.4, 1.0, 2.5):
        for shift in ((0, 0, 0, 0), (0.3, 0.3, 0, 0), (0.6, 0.6, 0, 0), (0, 0, 0.6, 0.6)):
            t0 = time.time()
            res['%g|%s' % (mult, ','.join('%g' % v for v in shift))] = run_one(shift, mult)
            print(mult, shift, '%.0fs' % (time.time() - t0), flush=True)
    json.dump(res, open('../results/mech.json', 'w'))
