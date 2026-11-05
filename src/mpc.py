"""
Economic MPC for the pumped canal (fast layer, 1 h steps, 24 h horizon).

Prediction model: integrator per pool (storage area A_j identified from steady backwater profiles of the
(nominal or twin) geometry):
    y_j[k+1] = y_j[k] + dt/A_j * ( Qin_j - Qout_j - d_j )
    inputs  u = [Qp, Qg0, Qg1, Qg2, Qtail],  Qin_0 = Qp, Qin_j = Qg_{j-1}, Qout_j = Qg_j, Qout_3 = Qtail
Disturbances: scheduled withdrawals (known 24 h ahead, stepwise at 06:00) + seepage + an estimated bias that
absorbs unscheduled withdrawals and model error (offset-free).
Cost ($-like units): energy of the pump cascade + level-tracking + tail delivery deviation (spill / shortage)
+ move penalties.  Solved with L-BFGS-B and an analytic (adjoint) gradient.
"""
import numpy as np
from scipy.optimize import minimize
from controllers import Base
from model_id import pool_areas

RHO_G = 1000.0 * 9.81


class MPC(Base):
    name = "MPC"

    def __init__(self, N=24, w_y=5000.0, w_band=2.0e5, band=0.12, w_tail=30.0, w_du=4.0, w_energy=1.0,
                 L_obs=0.25, name=None, adaptive_areas=False, level_offsets=None):
        self.N = N
        self.w_y, self.w_band, self.band, self.w_tail, self.w_du, self.w_energy = w_y, w_band, band, w_tail, w_du, w_energy
        self.L_obs = L_obs
        self.adaptive_areas = adaptive_areas
        if name:
            self.name = name
        self.level_offsets = level_offsets

    # ------------------------------------------------------------------ setup
    def reset(self, plant, scen):
        self.common_setup(plant, scen)
        c = plant.canal
        self.E = np.array([[1, -1, 0, 0, 0],
                           [0, 1, -1, 0, 0],
                           [0, 0, 1, -1, 0],
                           [0, 0, 0, 1, -1]], float)
        self.A = self.A0.copy()
        self.dhat = np.zeros(self.nP)
        self.U_prev = None
        self.y_pred = None
        self.last_net = None
        self.freeze = True
        self.price = plant.pp.price
        self.pp = plant.pp
        self.Qmax = plant.pp.Q_max
        self.setpoint_shift = np.zeros(self.nP)            # set by the morphology planner (m, depth reduction)

    def after_warmup(self, plant, scen):
        self.freeze = False
        self.dhat[:] = 0.0

    def yref_now(self):
        return self.y_ref - self.setpoint_shift

    # ------------------------------------------------------------------ disturbance forecast
    def disturbance(self, t, scen):
        D = np.zeros((self.N, self.nP))
        for k in range(self.N):
            s = scen.scheduled(t + k * self.dtc)
            D[k] = self.demand_by_pool(s) + self.seep_pool
        return D

    def tail_sched(self, t, scen):
        return np.array([scen.scheduled(t + k * self.dtc)[-1] for k in range(self.N)])

    # ------------------------------------------------------------------ energy model (smooth, convex in Qp)
    def energy_cost(self, Qp, Yhead):
        pp = self.pp
        Hs = pp.H_static + (Yhead - pp.Y_head_ref)
        eta0 = pp.eta_bep0
        # use best-efficiency (the wear-reduced efficiency is the same for all candidates within a step)
        P = RHO_G * Qp * (Hs + pp.kf * Qp ** 2) / (1000.0 * eta0)
        dP = RHO_G * (Hs + 3 * pp.kf * Qp ** 2) / (1000.0 * eta0)
        return self.price * P * self.dtc / 3600.0, self.price * dP * self.dtc / 3600.0

    # ------------------------------------------------------------------ solve
    def solve(self, y0, D, tail_s, Yhead, Uinit, dhat, Qp_prev, gate_prev, tail_prev):
        N, nP = self.N, self.nP
        dt = self.dtc
        k_dt = dt / self.A
        E = self.E
        yref = self.yref_now()
        u_prev0 = np.concatenate([[Qp_prev], gate_prev, [tail_prev]])
        Dt = D + dhat[None, :]

        def fun(uvec):
            U = uvec.reshape(N, 5)
            net = U @ E.T - Dt
            y = y0[None, :] + k_dt[None, :] * np.cumsum(net, axis=0)
            e = y - yref[None, :]
            J = self.w_y * np.sum(e ** 2)
            gy = 2 * self.w_y * e
            hi = e - self.band
            lo = -e - self.band
            vi = np.maximum(hi, 0.0)
            vl = np.maximum(lo, 0.0)
            J += self.w_band * (np.sum(vi ** 2) + np.sum(vl ** 2))
            gy += 2 * self.w_band * (vi - vl)
            # tail delivery deviation
            et = U[:, 4] - tail_s
            J += self.w_tail * np.sum(et ** 2)
            gU = np.zeros_like(U)
            gU[:, 4] += 2 * self.w_tail * et
            # energy
            ce, dce = self.energy_cost(U[:, 0], Yhead)
            J += self.w_energy * np.sum(ce)
            gU[:, 0] += self.w_energy * dce
            # moves
            dU = np.diff(np.vstack([u_prev0[None, :], U]), axis=0)
            J += self.w_du * np.sum(dU ** 2)
            gdu = 2 * self.w_du * dU
            gU += gdu
            gU[:-1] -= gdu[1:]
            # adjoint of the level model
            gnet = k_dt[None, :] * np.cumsum(gy[::-1], axis=0)[::-1]
            gU += gnet @ E
            return J, gU.ravel()

        bounds = [(0.0, self.Qmax)] + [(0.0, 140.0)] * 3 + [(0.0, 60.0)]
        bounds = bounds * N
        res = minimize(fun, Uinit.ravel(), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
        return res.x.reshape(N, 5), res

    # ------------------------------------------------------------------ interface
    def act(self, info):
        t = info["t"]
        scen = info["scen"]
        nP = self.nP
        y = info["y"]
        # disturbance observer (offset-free): compare measured level with the one predicted last step
        if self.y_pred is not None and not self.freeze:
            e = y - self.y_pred
            self.dhat = np.clip(self.dhat - self.L_obs * (self.A / self.dtc) * e, -8.0, 8.0)
        D = self.disturbance(t, scen)
        tail_s = self.tail_sched(t, scen)
        Qp_prev, gate_prev, tail_prev = info["Qp_prev"], info["gate_q"], info["tail"]
        if self.U_prev is None:
            Uinit = np.tile(np.concatenate([[Qp_prev], gate_prev, [tail_prev]]), (self.N, 1))
        else:
            Uinit = np.vstack([self.U_prev[1:], self.U_prev[-1:]])
        U, res = self.solve(y, D, tail_s, info["Yhead"], Uinit, self.dhat, Qp_prev, gate_prev, tail_prev)
        self.U_prev = U
        u0 = U[0].copy()
        # first-move limit (pump start/stop practice)
        u0[0] = float(np.clip(u0[0], Qp_prev - 15.0, Qp_prev + 15.0))
        net = self.E @ u0
        self.y_pred = y + (self.dtc / self.A) * (net - D[0] - self.dhat)
        return dict(Qp=u0[0], gate_Q=u0[1:4], tail_Q=u0[4])
