"""Code verification at the production configuration (N reaches, fitted exchange): Jacobian, water balance, sediment balance."""
import sys, numpy as np
from canal import trapezoid, CanalParams
from sim import *
from sediment import SedimentParams
from controllers import FFP
N = int(sys.argv[1]) if len(sys.argv) > 1 else 114
sp = SedimentParams(mult=1.0); sp.exp_fit = 1; sp.win = 5000.0
plant = Plant(seed=1, sp=sp, cp=CanalParams(N=N)); scen = Scenario(plant.canal, seed=1)
ctl = FFP(); rec, tot = run(plant, scen, ctl, days=20)
c = plant.canal; sed = plant.sed
a_g = c.gate_a.copy(); a_t = c.tail_a; sched = scen.scheduled(20 * 86400.0 - 1)[:-1]; Qin = float(c.Q[0])
h0, Q0 = c.h.copy(), c.Q.copy(); rng = np.random.default_rng(1)
hn = h0 * (1 + 0.01 * rng.standard_normal(c.M)); Qn = Q0 * (1 + 0.01 * rng.standard_normal(c.M))
ql = c._lateral(c.offtake_flows(sched), h0)
res, J = c._eval(hn, Qn, h0, Q0, 1800.0, Qin, a_g, a_t, ql)
Jn = np.zeros_like(J)
for k in range(2 * c.M):
    hp, Qp = hn.copy(), Qn.copy()
    if k % 2 == 0: hp[k // 2] += 1e-6
    else: Qp[k // 2] += 1e-6
    r2, _ = c._eval(hp, Qp, h0, Q0, 1800.0, Qin, a_g, a_t, ql); Jn[:, k] = (r2 - res) / 1e-6
print("N=%d nodes M=%d" % (N, c.M))
print("Jacobian max abs err / max|J|: %.2e" % (np.max(np.abs(J - Jn)) / np.max(np.abs(J))))
V0 = c.volume(); inflow = out = 0.0
for _ in range(96):
    c.step(Qin + 5.0, a_g, a_t, sched)
    inflow += c.Q[0] * c.par.dt; out += (np.sum(c.last_off_q) + c.last_seep_flow + c.tail_flow()) * c.par.dt
dV = c.volume() - V0
print("water balance rel err (|dV-(in-out)|/in): %.2e" % (abs(dV - (inflow - out)) / inflow))
s = sed; hm = 0.5 * (c.h[s.p] + c.h[s.q]); A, T, P = trapezoid(hm, c.par.b, c.par.m); V = A * c.dx
resid = s.mass_in - (s.mass_dep + np.sum(V * s.C) + s.mass_out)
print("sediment budget residual / mass_in over 20 d + 2 d: %.2e   (mass_in %.1f t)" % (resid / s.mass_in, s.mass_in / 1e3))
hs = np.linspace(1, 8, 4000); A_, T_, P_ = trapezoid(hs, 17.0, 1.7); Qn_ = A_ * (A_ / P_) ** (2 / 3) * np.sqrt(6e-5) / 0.03
print("normal depth 75 m3/s: %.3f m; velocity %.3f m/s" % (np.interp(75.0, Qn_, hs), 75.0 / np.interp(75.0, Qn_, A_)))
