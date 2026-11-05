"""
Scenario generation and the closed-loop simulation harness (plant = PumpedCanal + Sediment + PumpStation).

Time bases: plant step 30 min, control step 1 h, schedule changes (stepwise, 'discrete delivery') at 06:00.
"""
import numpy as np
from canal import PumpedCanal, CanalParams, trapezoid
from sediment import Sediment, SedimentParams
from pump import PumpStation, PumpParams

# Monthly pumped discharge from the Amudarya, 2004-2019 mean, digitised by eye from Fig. 2 of
# Ikramova et al. (2019), World Science 8(48): values for months 1..12 (m3/s).  Integral ~ 1.31 km3/yr,
# consistent with ADB PCR abstraction 1,283-1,332 Mm3.  (+-3 m3/s reading uncertainty)
MONTHLY_PUMPING = np.array([6, 12, 38, 50, 52, 57, 77, 72, 53, 43, 30, 10], float)
MID_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], float)


def profile_q(doy):
    x = np.concatenate([MID_DOY - 365, MID_DOY, MID_DOY + 365])
    y = np.tile(MONTHLY_PUMPING, 3)
    return np.interp(doy, x, y)


class Scenario:
    """Demand (scheduled and actual) and turbidity for one irrigation season."""

    def __init__(self, canal: PumpedCanal, seed=0, doy_start=60, days=245, turb_scale=1.0,
                 unsched_rate=0.15, unsched_amp=0.2):
        rng = np.random.default_rng(seed)
        self.days = days
        self.doy0 = doy_start
        self.share = np.concatenate([canal.off_share, [canal.tail_share]])   # 5 offtakes + tail
        nshare = len(self.share)
        ar = 0.0
        sched = np.zeros((days, nshare))
        actual = np.zeros((days, nshare, 1))
        self.D_total = np.zeros(days)
        for d in range(days):
            base = max(profile_q(doy_start + d + 0.25) - 6.0, 5.0)
            ar = 0.7 * ar + 0.04 * rng.standard_normal()
            D = base * (1 + ar)
            self.D_total[d] = D
            noise = 1 + 0.04 * rng.standard_normal(nshare)
            sched[d] = D * self.share * noise
        self.sched = sched                       # known to controllers (m3/s per offtake, per day)
        # unscheduled changes: per day and offtake, probability, multiplicative factor, start hour
        self.unsched = np.ones((days, nshare))
        self.unsched_hour = np.zeros((days, nshare), int)
        for d in range(days):
            for k in range(nshare):
                if rng.random() < unsched_rate:
                    self.unsched[d, k] = 1 + unsched_amp * (2 * rng.random() - 1)
                    self.unsched_hour[d, k] = int(rng.integers(0, 24))
        # turbidity (kg/m3): seasonal flood bump + AR(1) lognormal daily fluctuation
        doy = doy_start + np.arange(days)
        seas = 0.45 + 1.4 * np.exp(-((doy - 180) / 45.0) ** 2)
        ln = np.zeros(days)
        e = 0.0
        for d in range(days):
            e = 0.85 * e + 0.25 * rng.standard_normal()
            ln[d] = e
        self.turb = turb_scale * seas * np.exp(ln - 0.5 * np.var(ln))

    def hour_index(self, t_s):
        """Return (day index, hour of day) with day boundary at 06:00 (season starts 06:00)."""
        h = t_s / 3600.0
        return int(h // 24), int(h % 24)

    def scheduled(self, t_s):
        d, _ = self.hour_index(t_s)
        d = min(d, self.days - 1)
        return self.sched[d]

    def actual(self, t_s):
        d, hr = self.hour_index(t_s)
        d = min(d, self.days - 1)
        f = np.where(hr >= self.unsched_hour[d], self.unsched[d], 1.0)
        return self.sched[d] * f

    def turbidity(self, t_s):
        d, _ = self.hour_index(t_s)
        return self.turb[min(d, self.days - 1)]


# ------------------------------------------------------------------------------------------------
class Plant:
    def __init__(self, seed=0, sp: SedimentParams = None, pp: PumpParams = None, cp: CanalParams = None,
                 sediment_on=True):
        self.canal = PumpedCanal(cp)
        self.sp = sp or SedimentParams()
        self.pp = pp or PumpParams()
        self.sed = Sediment(self.canal, self.sp) if sediment_on else None
        self.pump = PumpStation(self.pp)
        self.rng = np.random.default_rng(10_000 + seed)
        self.y_ref_base = self.canal.zb0[self.canal.pool_end_node] + self.canal.par.h_ref


def measure(plant: Plant, sigma_y=0.02, sigma_c=0.10):
    c = plant.canal
    y = c.pool_end_levels() + sigma_y * plant.rng.standard_normal(len(c.pool_end_node))
    return y


def run(plant: Plant, scen: Scenario, controller, days=None, record_every=1, verbose=False,
        survey_every_days=30):
    """Closed-loop season.  controller.act(info) -> dict(Qp, gate_Q (array), tail_Q)."""
    c = plant.canal
    par = c.par
    days = days or scen.days
    dtc = 3600.0
    n_sub = int(dtc / par.dt)
    nctl = int(days * 24)
    rec = []
    # initialise to steady flow with the controller's initial command
    controller.reset(plant, scen)
    t = 0.0
    c.h[:] = par.h_ref
    c.Q[:] = scen.sched[0].sum() + 6.0
    # warm start (48 h, no sediment)
    for k in range(72):
        info = controller.observe(plant, scen, t_override=0.0)
        cmd = controller.act(info)
        _apply(plant, cmd, 0.0, dtc, n_sub, scen.actual(0.0), sediment=False, c_total=0.0)
    controller.after_warmup(plant, scen)
    c.t = 0.0
    totals = dict(spill=0.0, short=0.0, deliv_abs_err=0.0, intended=0.0)
    for k in range(nctl):
        t = k * dtc
        info = controller.observe(plant, scen)
        cmd = controller.act(info)
        c_tot = scen.turbidity(t)
        act = scen.actual(t)
        out = _apply(plant, cmd, t, dtc, n_sub, act, sediment=True, c_total=c_tot)
        controller.post_step(plant, scen, info, cmd, out)
        # bookkeeping
        intended = act
        deliv = out["delivered"]
        totals["spill"] += max(out["Qtail_mean"] - intended[-1], 0.0) * dtc
        totals["short"] += float(np.sum(np.maximum(intended - deliv, 0.0))) * dtc
        totals["deliv_abs_err"] += float(np.sum(np.abs(intended - deliv))) * dtc
        totals["intended"] += float(np.sum(intended)) * dtc
        if k % record_every == 0:
            rec.append(dict(k=k, t=t, Qp=cmd["Qp"], power=out["power"], eta=out["eta"], Yhead=out["Yhead"],
                            h_end=c.pool_end_depth().copy(), Qtail=out["Qtail_mean"], spill=max(out["Qtail_mean"] - intended[-1], 0.0),
                            short=float(np.sum(np.maximum(intended - deliv, 0.0))), gate_q=out["gate_q"].copy(),
                            turb=c_tot, dz_max=float(plant.sed.dz.max()) if plant.sed else 0.0,
                            dz_mean=float(plant.sed.dz.mean()) if plant.sed else 0.0,
                            hmax=float(np.max(c.h)), Ymax_margin=float(np.min((c.par.h_bank + c.zb0) - (c.zb + c.h)))))
        if verbose and k % 240 == 239:
            print("day %d  Qp %.1f  h_end %s  dz_max %.3f  spill %.2f Mm3" % (k // 24, cmd["Qp"], np.round(c.pool_end_depth(), 2),
                                                                             plant.sed.dz.max() if plant.sed else 0, totals["spill"] / 1e6))
    return rec, totals


def _apply(plant, cmd, t, dtc, n_sub, actual_off, sediment, c_total):
    c = plant.canal
    par = c.par
    # gate openings from commanded flows (controller uses its mu_model, plant uses mu_plant) with rate limit
    a_new = np.array([c.gate_opening_for_flow(k, q) for k, q in enumerate(cmd["gate_Q"])])
    da = np.clip(a_new - c.gate_a, -0.4, 0.4)
    a_g = c.gate_a + da
    at_new = c.tail_opening_for_flow(cmd["tail_Q"])
    a_t = c.tail_a + np.clip(at_new - c.tail_a, -0.4, 0.4)
    Qp = float(np.clip(cmd["Qp"], 0.0, plant.pp.Q_max))
    deliv_acc = np.zeros(len(actual_off))
    qt_acc = 0.0
    p_acc = 0.0
    c_bed = (1 - plant.sp.trap) * plant.sp.f_bed * c_total
    for s in range(n_sub):
        c.step(Qp, a_g, a_t, actual_off[:-1])
        qt = c.tail_flow()
        qt_acc += qt
        deliv_acc[:-1] += c.last_off_q
        deliv_acc[-1] += qt           # tail gate flow counted as delivery to tail users (excess is spill)
        if sediment and plant.sed is not None:
            plant.sed.step(par.dt, c_bed)
        p_acc += plant.pump.step(Qp, c.zb[0] + c.h[0], c_total, par.dt)
    n = n_sub
    return dict(delivered=deliv_acc / n, Qtail_mean=qt_acc / n, power=p_acc / n, eta=plant.pump.eta(Qp),
                Yhead=c.zb[0] + c.h[0], gate_q=c.gate_flows())
