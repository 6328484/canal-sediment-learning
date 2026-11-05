"""
Morphology-aware slow layer on top of the economic MPC.

Digital twin: K copies of the plant (one per sediment-capacity hypothesis, multiplier on Engelund-Hansen),
each with its own local water-balance control (with the same level references as the plant), driven by the ACTUAL
pump flow, the nominal schedule and the measured turbidity.  Sparse bed surveys (every 30 days, 14 stations, 2 cm
noise) update a Bayesian weight over the hypotheses (prior centred on the nominal multiplier, forgetting factor).

Planner (MorphMPC): after every survey, each candidate shift of the pool level references (m of depth reduction) is
evaluated for the REST OF THE SEASON on the MAP twin: the hydraulics of the season under the candidate are stored in a
bank (forecast.py), and only the sediment of the twin is advanced with them.  The candidate with the smallest predicted
    J = energy cost + c_short*shortage + c_spill*spill + c_dep*deposit volume + lam * L5(end of season)
is passed to the MPC as new level references.  (MorphWeekly is the first design, which compared the candidates over
the next week; it was rejected on the design seed, see the paper.)
"""
import copy
import numpy as np
from mpc import MPC
from controllers import FFP
from sim import Plant, _apply
from sediment import SedimentParams, RHO_S, POROSITY
from forecast import Bank, Climatology

DEFAULT_MULTS = (0.35, 0.6, 1.0, 1.7, 3.0)


class Twin:
    def __init__(self, plant, scen, mults=DEFAULT_MULTS, mode="learn", fixed_mult=1.0, sigma_survey=0.04, twin_d50=None, survey_noise=0.02, forget=0.9, temper=4.0, twin_dt=None, prior_sigma=0.8):
        self.mode = mode
        self.survey_noise, self.forget, self.temper = survey_noise, forget, temper
        self.sigma = sigma_survey
        if mode == "learn":
            self.mults = np.array(mults, float)
        else:
            self.mults = np.array([fixed_mult], float)
        self.twins, self.ctls = [], []
        for i, m in enumerate(self.mults):
            sp = copy.deepcopy(plant.sp)
            sp.mult = float(m)
            sp.theta_shift = 0.0          # the twins always assume the nominal capacity law
            if twin_d50 is not None:
                sp.d50 = twin_d50
            pl = Plant(seed=500 + i, sp=sp, pp=plant.pp, cp=copy.deepcopy(plant.canal.par))
            if twin_dt:
                pl.canal.par.dt = float(twin_dt)
            ctl = FFP()
            ctl.reset(pl, scen)
            pl.canal.h[:] = plant.canal.h
            pl.canal.Q[:] = plant.canal.Q
            ctl.after_warmup(pl, scen)
            ctl.Qp_prev = scen.sched[0].sum() + 6.0
            self.twins.append(pl)
            self.ctls.append(ctl)
        # prior weights: Gaussian in ln(mu) around the nominal multiplier 1 (a mild preference, overcome by surveys); flat if prior_sigma is falsy
        self.loglik = -0.5 * (np.log(self.mults) / prior_sigma) ** 2 if (mode == "learn" and prior_sigma) else np.zeros(len(self.mults))
        self.map = int(np.argmax(self.loglik)) if (mode == "learn" and prior_sigma) else int(np.argmin(np.abs(np.log(self.mults))))
        ns = len(plant.sed.dz)
        # 14 survey stations at fixed physical positions (the reach containing x = (2.5 + 4 i) L / 57), whatever the grid
        self.stations = np.floor((2.5 + 4 * np.arange(14)) * ns / 57.0).astype(int)
        self.n_surveys = 0
        self.mult_history = []
        self.meas_hist, self.pred_hist, self.loglik_hist = [], [], []

    def advance(self, t, Qp_actual, scen, turb_meas):
        for pl, ctl in zip(self.twins, self.ctls):
            info = ctl.observe(pl, scen, t_override=t)
            cmd = ctl.act(info)
            cmd["Qp"] = Qp_actual
            out = _apply(pl, cmd, t, 3600.0, int(round(3600.0 / pl.canal.par.dt)), scen.scheduled(t), sediment=True, c_total=turb_meas)
            ctl.post_step(pl, scen, info, cmd, out)
            ctl.Qp_prev = Qp_actual

    def survey_update(self, plant, rng):
        meas = plant.sed.dz[self.stations] + self.survey_noise * rng.standard_normal(len(self.stations))
        self.n_surveys += 1
        self.meas_hist.append(meas.copy())
        self.pred_hist.append(np.array([pl.sed.dz[self.stations] for pl in self.twins]))
        if self.mode != "learn":
            return
        for i, pl in enumerate(self.twins):
            r = (meas - pl.sed.dz[self.stations]) / self.sigma
            # tempered likelihood: surveys are correlated through the shared hydraulics
            self.loglik[i] = self.forget * self.loglik[i] - 0.5 * float(np.sum(r ** 2)) / self.temper
        self.map = int(np.argmax(self.loglik))
        self.loglik_hist.append(self.loglik.copy())
        w = np.exp(self.loglik - self.loglik.max())
        w /= w.sum()
        self.mult_hat = float(np.exp(np.sum(w * np.log(self.mults))))
        self.mult_history.append((self.n_surveys, self.mult_hat, self.map))

    @property
    def best(self):
        return self.twins[self.map], self.ctls[self.map]


class MorphMPC(MPC):
    """Seasonal planner: after every survey the candidate shifts are compared on the MAP twin over the rest of the season."""
    name = "MorphMPC"
    GRID = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]

    def __init__(self, lam=2e7, cap_mode="learn", mult_nominal=1.0, plan_every=None, candidates=None, twin_d50=None, survey_noise=0.02, forget=0.9, temper=4.0,
                 twin_dt=None, c_short=0.10, c_spill=0.02, c_dep=3.0, survey_every=720, prior_sigma=0.8, switch_margin=2e4, name=None, **kw):
        super().__init__(**kw)
        self.lam, self.cap_mode, self.mult_nominal = lam, cap_mode, mult_nominal
        self.plan_every = plan_every or survey_every              # re-plan after every survey (and at the start of the season)
        self.c_short, self.c_spill, self.c_dep = c_short, c_spill, c_dep
        self.survey_every = survey_every
        self.twin_d50 = twin_d50
        self.switch_margin = switch_margin                        # USD: the shift is changed only if the predicted seasonal cost falls by more than this
        self.tw_kw = dict(survey_noise=survey_noise, forget=forget, temper=temper, twin_dt=twin_dt, prior_sigma=prior_sigma)
        self.candidates = [tuple(float(v) for v in c) for c in (candidates if candidates is not None else self.GRID)]
        if name:
            self.name = name
        self.kidx = 0
        self.log = []
        self.diag, self.keep_diag, self.plant_ref = [], False, None
        self.turb_hist = []

    def reset(self, plant, scen):
        super().reset(plant, scen)
        self.twin = None
        self.turb_hist = []

    def after_warmup(self, plant, scen):
        super().after_warmup(plant, scen)
        if self.cap_mode == "oracle":
            self.twin = Twin(plant, scen, mode="fixed", fixed_mult=plant.sp.mult, twin_d50=self.twin_d50, **self.tw_kw)
        elif self.cap_mode == "nominal":
            self.twin = Twin(plant, scen, mode="fixed", fixed_mult=self.mult_nominal, twin_d50=self.twin_d50, **self.tw_kw)
        else:
            self.twin = Twin(plant, scen, mode="learn", twin_d50=self.twin_d50, **self.tw_kw)
        self.kidx = 0
        dt = self.tw_kw.get('twin_dt') or plant.canal.par.dt
        self.bank = Bank.get(plant.canal.par, plant.pp, self.candidates, days=scen.days, doy0=scen.doy0, dt=dt)
        self.clim = Climatology(plant.canal, scen.doy0, scen.days)

    # ------------------------------------------------------------------ planning
    def turb_forecast(self, k0):
        """Turbidity of the rest of the season: the seasonal expectation scaled by the ratio of the measured turbidity of the last 30 days to the expectation."""
        seas = self.clim.seas
        K = self.bank.hours
        scale, n = 1.0, len(self.turb_hist)
        if n >= 24:
            m = min(n, 720)
            scale = float(np.mean(self.turb_hist[n - m:])) / float(np.mean(seas[np.minimum(np.arange(n - m, n) // 24, len(seas) - 1)]))
        return scale * seas[np.minimum(np.arange(k0, K) // 24, len(seas) - 1)], scale

    def plan(self, scen, t0):
        k0 = int(round(t0 / 3600.0))
        pl0 = self.twin.best[0]
        tf, scale = self.turb_forecast(k0)
        c_bed = (1.0 - pl0.sp.trap) * pl0.sp.f_bed * tf
        tab = []
        for cand in self.candidates:
            J, d = self.bank.evaluate(self.bank.index(cand), pl0, k0, c_bed, self.price, self.c_short, self.c_spill, self.c_dep, self.lam)
            tab.append((list(cand), float(J), d))
        Js = np.array([t[1] for t in tab])
        jb = int(np.argmin(Js))
        cur = [j for j, t in enumerate(tab) if np.allclose(t[0], self.setpoint_shift)]
        if cur and Js[cur[0]] - Js[jb] <= self.switch_margin:
            jb = cur[0]
        best = tuple(tab[jb][0])
        self.setpoint_shift = np.array(best, float)
        for pl, ctl in zip(self.twin.twins, self.twin.ctls):
            ctl.y_ref = pl.y_ref_base - self.setpoint_shift      # the twins follow the same level references as the plant
        self.log.append((t0, best, float(Js[jb])))
        if self.keep_diag:
            plt_, pln = self.plant_ref.sed, pl0.sed
            self.diag.append(dict(t=float(t0), best=list(best), cands=tab, map=int(self.twin.map), turb_scale=float(scale),
                                  mult_hat=float(getattr(self.twin, 'mult_hat', np.nan)),
                                  plant_dzw=[round(float(v), 5) for v in plt_.dz_window(5000.0)], twin_dzw=[round(float(v), 5) for v in pln.dz_window(5000.0)],
                                  plant_L5=float(plt_.capacity_loss()[0]), twin_L5=float(pln.capacity_loss()[0])))

    # ------------------------------------------------------------------ hooks
    def observe(self, plant, scen, t_override=None):
        info = super().observe(plant, scen, t_override)
        self.plant_ref = plant
        if not self.freeze:
            self.turb_hist.append(info["turb"])
        return info

    def act(self, info):
        t = info["t"]
        k = int(round(t / 3600.0))
        if self.twin is not None and not self.freeze and k % self.plan_every == 0:
            self.plan(info["scen"], t)
        return super().act(info)

    def post_step(self, plant, scen, info, cmd, out):
        super().post_step(plant, scen, info, cmd, out)
        if self.twin is None:
            return
        t = info["t"]
        self.twin.advance(t, cmd["Qp"], scen, info["turb"])
        self.kidx += 1
        if self.kidx % self.survey_every == 0:
            self.twin.survey_update(plant, plant.rng)



class MorphWeekly(MorphMPC):
    """First design of the planner, kept for the record: every plan_every hours the candidates are compared over the next `horizon` hours on the MAP twin
    (full coupled simulation with the local controller), with J = energy + shortage + spill + deposit volume + lam * increase of L5 over the horizon.
    Rejected on the design seed: the first days after a change of the references are dominated by the filling or emptying of the pools, and a week gives no
    view of the aggradation at the end of the season (see the paper)."""
    name = "MorphWeekly"

    def __init__(self, plan_every=168, horizon=168, **kw):
        super().__init__(plan_every=plan_every, **kw)
        self.horizon = horizon

    def after_warmup(self, plant, scen):
        MPC.after_warmup(self, plant, scen)
        if self.cap_mode == "oracle":
            self.twin = Twin(plant, scen, mode="fixed", fixed_mult=plant.sp.mult, twin_d50=self.twin_d50, **self.tw_kw)
        elif self.cap_mode == "nominal":
            self.twin = Twin(plant, scen, mode="fixed", fixed_mult=self.mult_nominal, twin_d50=self.twin_d50, **self.tw_kw)
        else:
            self.twin = Twin(plant, scen, mode="learn", twin_d50=self.twin_d50, **self.tw_kw)
        self.kidx = 0

    def predict(self, scen, t0, shift):
        pl0, ctl0 = self.twin.best
        pl = copy.deepcopy(pl0)
        ctl = copy.deepcopy(ctl0)
        ctl.y_ref = pl.y_ref_base - np.asarray(shift, float)
        cl0 = float(pl.sed.capacity_loss()[0])
        m0 = pl.sed.mass_dep
        e0 = pl.pump.energy_kwh
        short = spill = 0.0
        turb = self.turb_last
        for h in range(self.horizon):
            t = t0 + h * 3600.0
            info = ctl.observe(pl, scen, t_override=t)
            cmd = ctl.act(info)
            sc = scen.scheduled(t)
            out = _apply(pl, cmd, t, 3600.0, int(round(3600.0 / pl.canal.par.dt)), sc, sediment=True, c_total=turb)
            ctl.post_step(pl, scen, info, cmd, out)
            short += float(np.sum(np.maximum(sc - out["delivered"], 0.0))) * 3600.0
            spill += max(out["Qtail_mean"] - sc[-1], 0.0) * 3600.0
        dep_vol = (pl.sed.mass_dep - m0) / (RHO_S * (1 - POROSITY))
        dE = (pl.pump.energy_kwh - e0) * self.price
        agg = float(pl.sed.capacity_loss()[0]) - cl0
        J = dE + self.c_short * short + self.c_spill * spill + self.c_dep * dep_vol + self.lam * agg
        return J, dict(dE=dE, short=short, spill=spill, dep_vol=dep_vol, agg=agg)

    def plan(self, scen, t0):
        best, bestJ = None, np.inf
        for cand in self.candidates:
            J, d = self.predict(scen, t0, cand)
            if J < bestJ:
                best, bestJ = cand, J
        self.setpoint_shift = np.array(best, float)
        for pl, ctl in zip(self.twin.twins, self.twin.ctls):
            ctl.y_ref = pl.y_ref_base - self.setpoint_shift
        self.log.append((t0, best, bestJ))

    def observe(self, plant, scen, t_override=None):
        info = MPC.observe(self, plant, scen, t_override)
        self.turb_last = info["turb"]
        self.plant_ref = plant
        return info

    def act(self, info):
        t = info["t"]
        k = int(round(t / 3600.0))
        if self.twin is not None and not self.freeze and k % self.plan_every == 0 and k > 0:
            self.plan(info["scen"], t)
        return MPC.act(self, info)


class FixedShift(MPC):
    """Predictive controller with a constant shift of the pool-level references (benchmark without learning)."""
    name = "FixedShift"

    def __init__(self, shift=(0, 0, 0, 0), **kw):
        super().__init__(**kw)
        self.shift = np.array(shift, float)

    def reset(self, plant, scen):
        super().reset(plant, scen)
        self.setpoint_shift = self.shift.copy()
