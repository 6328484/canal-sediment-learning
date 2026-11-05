"""Verification: steady suspended-load relaxation against the analytic solution (uniform flow, constant C_eq)."""
import numpy as np, copy, json
from canal import PumpedCanal, CanalParams
from sediment import Sediment, SedimentParams

def run(N, fit, steps=400, dt=1800.0):
    par = CanalParams(N=N)
    par.seep = 0.0
    c = PumpedCanal(par)
    c.h[:] = 5.0; c.Q[:] = 75.0
    sp = SedimentParams(mult=1.0); sp.exp_fit = fit
    s = Sediment(c, sp)
    s._apply_bed = lambda ddz: None            # frozen bed and hydraulics
    Ceq = s.capacity()[0]
    cin = 0.0                                   # clean inflow: concentration relaxes up from 0 towards C_eq
    for _ in range(steps):
        s.step(dt, cin)
    x = (np.arange(N) + 1) * c.dx               # outlet of each segment
    La = 75.0 / (par.b * s.ws)
    exact = Ceq + (cin - Ceq) * np.exp(-x / La)
    # exact solution of the cell outlet for the discrete problem differs only through the scheme
    err = np.max(np.abs(s.C - exact)) / Ceq
    # deposit/erosion: bed is non-erodible (dz = 0) so erosion is limited -> use deposit test with c_in > C_eq instead
    return err, La, Ceq

def run_dep(N, fit, steps=400, dt=1800.0):
    par = CanalParams(N=N); par.seep = 0.0
    c = PumpedCanal(par); c.h[:] = 5.0; c.Q[:] = 75.0
    sp = SedimentParams(mult=1.0); sp.exp_fit = fit
    s = Sediment(c, sp); s._apply_bed = lambda ddz: None
    Ceq = s.capacity()[0]; cin = 0.40            # inflow well above capacity: deposition decays exponentially
    for _ in range(steps):
        s.step(dt, cin)
    x = (np.arange(N) + 1) * c.dx
    La = 75.0 / (par.b * s.ws)
    exact = Ceq + (cin - Ceq) * np.exp(-x / La)
    err = np.max(np.abs(s.C - exact)) / (cin - Ceq)
    # integrated deposit over the canal vs exact: Q (cin - C(L)) * time
    dep_num = s.mass_dep; dep_ex_rate = 75.0 * (cin - exact[-1])
    return err, Ceq, La

print('Ceq(kg/m3)=%.5f' % run_dep(57, 0)[1], ' adaptation length La=%.0f m' % run_dep(57, 0)[2])
for fit in (0, 1, 2):
    print('scheme', {0: 'upwind', 1: 'fitted', 2: 'fitted+logmean'}[fit])
    for N in (29, 57, 114, 228):
        e, _, La = run_dep(N, fit)
        print('  N=%3d dx/La=%.2f  max relative error of C(x) = %.2e' % (N, 56600 / N / La, e))
