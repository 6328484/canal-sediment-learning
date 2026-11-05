"""
Controllers:  FFP (feed-forward / proportional water-balance local control + slow operator correction)
              MPC family is in mpc.py
Interface:    reset(plant, scen); observe(plant, scen, t_override) -> info; act(info) -> cmd;
              after_warmup(plant, scen); post_step(plant, scen, info, cmd, out)
"""
import numpy as np
from model_id import pool_areas


class Base:
    name = "base"

    def common_setup(self, plant, scen):
        c = plant.canal
        self.nP = c.n_struct + 1
        Qnom = np.interp(c.x, [0, c.par.L], [70.0, 5.0])
        self.A0 = pool_areas(c, Qnom, c.zb0, c.n_rough)       # clean-bed identification
        self.pool_of_off = c.off_pool.copy()
        self.y_ref = plant.y_ref_base.copy()
        self.seep_total = 6.0
        L = np.array([c.x[c.pool_end_node[j]] - c.x[c.pool_start_node[j]] for j in range(self.nP)])
        self.seep_pool = self.seep_total * L / L.sum()
        self.dtc = 3600.0
        s0 = scen.sched[0]
        d0 = np.zeros(self.nP)
        for k, pl in enumerate(self.pool_of_off):
            d0[pl] += s0[k]
        d0 += self.seep_pool
        self.Qp_prev = float(s0.sum() + self.seep_total)
        self.gate_q_prev = self.Qp_prev - np.cumsum(d0)[:-1]
        self.tail_prev = s0[-1]

    def reset(self, plant, scen):
        self.common_setup(plant, scen)

    def after_warmup(self, plant, scen):
        pass

    def observe(self, plant, scen, t_override=None):
        c = plant.canal
        t = c.t if t_override is None else t_override
        y = c.pool_end_levels() + 0.02 * plant.rng.standard_normal(self.nP)
        sched = scen.scheduled(t)
        return dict(t=t, y=y, sched=sched, Qp_prev=self.Qp_prev, gate_q=c.gate_flows(), tail=c.tail_flow(),
                    Yhead=c.zb[0] + c.h[0], turb=scen.turbidity(t) * np.exp(0.1 * plant.rng.standard_normal()),
                    plant=plant, scen=scen)

    def demand_by_pool(self, sched):
        d = np.zeros(self.nP)
        for k, pl in enumerate(self.pool_of_off):
            d[pl] += sched[k]
        return d

    def post_step(self, plant, scen, info, cmd, out):
        self.Qp_prev = cmd["Qp"]


class FFP(Base):
    """Water-balance feed-forward + proportional level feedback at every gate (upstream-level control),
    pump flow = scheduled demand + estimated seepage + slow operator correction from tail excess."""
    name = "FFP"

    def __init__(self, tau_c=6 * 3600.0, T_I=12 * 3600.0, margin=0.0):
        self.tau_c = tau_c
        self.T_I = T_I
        self.margin = margin

    def reset(self, plant, scen):
        super().reset(plant, scen)
        self.corr = 0.0
        self.freeze = True

    def after_warmup(self, plant, scen):
        self.corr = 0.0
        self.freeze = False

    def act(self, info):
        sched = info["sched"]
        d = self.demand_by_pool(sched) + self.seep_pool
        y, A = info["y"], self.A0
        yref = self.y_ref
        Qin = np.concatenate([[info["Qp_prev"]], info["gate_q"]])      # inflow to each pool (previous actual)
        Qout = Qin - d + A * (y - yref) / self.tau_c
        Qout = np.maximum(Qout, 0.0)
        gate_Q = Qout[:-1]
        tail_Q = Qout[-1]
        # operator slow correction of the inflow from the tail excess
        if not self.freeze:
            self.corr += -(self.dtc / self.T_I) * (info["tail"] - sched[-1])
            self.corr = float(np.clip(self.corr, -10.0, 10.0))
        Qp = (sched.sum() + self.seep_total) * (1.0 + self.margin) + self.corr
        Qp = float(np.clip(Qp, self.Qp_prev - 15.0, self.Qp_prev + 15.0))
        return dict(Qp=Qp, gate_Q=gate_Q, tail_Q=tail_Q)
