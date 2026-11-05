import numpy as np, time, json, itertools, sys
from multiprocessing import Pool
from sim import *
from controllers import *
from mpc import MPC

def one(args):
    shift, seed, turb = args
    plant = Plant(seed=seed); scen = Scenario(plant.canal, seed=seed, turb_scale=turb)
    ctl = MPC()
    orig_reset = ctl.reset
    def reset(plant, scen):
        orig_reset(plant, scen); ctl.setpoint_shift = np.array(shift, float)
    ctl.reset = reset
    rec, tot = run(plant, scen, ctl)
    he = np.array([r['h_end'] for r in rec])
    return dict(shift=list(shift), seed=seed, turb=turb, dz_max=float(plant.sed.dz.max()), dz_mean=float(plant.sed.dz.mean()),
                dep_kt=float(plant.sed.mass_dep/1e6), cap_loss=float(plant.sed.capacity_loss()[0]),
                spill=tot['spill']/tot['intended'], short=tot['short']/tot['intended'], abs_err=tot['deliv_abs_err']/tot['intended'],
                energy=float(plant.pump.energy_kwh/1e6), pumped=float(plant.pump.water_m3/1e6), hmin=float(he.min()),
                dz_prof=[float(x) for x in plant.sed.dz])
if __name__=="__main__":
    cands = [(0,0,a,b) for a in (0,0.3,0.6) for b in (0,0.3,0.6)] + [(0.3,0.3,0.3,0.3),(0.6,0.6,0.6,0.6)]
    jobs = [(s,0,1.0) for s in cands]
    t0=time.time()
    with Pool(2) as p: out = p.map(one, jobs)
    json.dump(out, open('../results/grid_setpoints.json','w'))
    print("done", time.time()-t0)
