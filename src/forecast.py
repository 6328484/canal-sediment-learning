"""Forecast bank and seasonal rollouts of the planner (Section 3.4 of the paper).

The hydraulics of the canal do not depend on the sediment capacity, and to first order not on the bed.  For every candidate shift of the level references the
hydraulics of the whole season are therefore computed once, with the local controller, the clean bed and the climatological demand (the expected
schedule, without the daily anomalies and the unscheduled changes), and stored hourly: depth and discharge at all nodes, energy, shortage and spill.
A rollout then advances only the sediment of a twin through the rest of the season, with the stored hydraulics (depth reduced by the aggradation at
constant water level) and the forecast turbidity, which costs about one tenth of a coupled simulation.
Banks are cached in ../results/bank/ (they are independent of the seed, of the multiplier and of the controller that runs the plant)."""
import os, sys, json, copy, hashlib, time
import numpy as np
from sim import Scenario, Plant, _apply, profile_q
from sediment import SedimentParams, RHO_S, POROSITY
from canal import CanalParams
from pump import PumpParams
from controllers import FFP

CACHE = os.environ.get('BANK_DIR', '../results/bank')


class Climatology(Scenario):
    """Expected demand and turbidity of the season: the deterministic part of Scenario (no anomalies, no unscheduled changes)."""

    def __init__(self, canal, doy_start=60, days=245):
        self.days, self.doy0 = days, doy_start
        self.share = np.concatenate([canal.off_share, [canal.tail_share]])
        self.D_total = np.array([max(profile_q(doy_start + d + 0.25) - 6.0, 5.0) for d in range(days)])
        self.sched = self.D_total[:, None] * self.share[None, :]
        self.unsched = np.ones_like(self.sched)
        self.unsched_hour = np.zeros(self.sched.shape, int)
        doy = doy_start + np.arange(days)
        self.seas = 0.45 + 1.4 * np.exp(-((doy - 180) / 45.0) ** 2)
        self.turb = self.seas.copy()


class _ZeroRNG:
    def standard_normal(self, size=None):
        return 0.0 if size is None else np.zeros(size)


def _key(cp, pp, cands, days, doy0, dt):
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(cp).items()}
    d.update({'pp_' + k: v for k, v in vars(pp).items()})
    d.update(cands=[list(map(float, c)) for c in cands], days=days, doy0=doy0, dt=dt, ctl='mpc-v1')
    return hashlib.md5(json.dumps(d, sort_keys=True, default=float).encode()).hexdigest()[:12]


class Bank:
    def __init__(self, h, Q, e, short, spill, cands, days):
        self.h, self.Q, self.e, self.short, self.spill = h, Q, e, short, spill     # (S, K, M), (S, K, M), (S, K), (S, K), (S, K)
        self.cands, self.days = [tuple(float(v) for v in c) for c in cands], days
        self.hours = h.shape[1]
        # cumulative sums from the end: remaining energy, shortage and spill from hour k to the end of the season
        self.rem = {n: np.concatenate([np.cumsum(a[:, ::-1], axis=1)[:, ::-1], np.zeros((a.shape[0], 1))], axis=1) for n, a in (('e', e), ('short', short), ('spill', spill))}

    # ------------------------------------------------------------------ construction and cache
    @classmethod
    def get(cls, cp, pp, cands, days=245, doy0=60, dt=3600.0, verbose=False):
        cp = copy.deepcopy(cp)
        key = _key(cp, pp, cands, days, doy0, dt)
        f = os.path.join(CACHE, 'bank_%s.npz' % key)
        if os.path.exists(f):
            z = np.load(f)
            return cls(z['h'], z['Q'], z['e'], z['short'], z['spill'], cands, days)
        t0 = time.time()
        h, Q, e, short, spill = [], [], [], [], []
        for s in cands:
            r = cls._simulate(cp, pp, s, days, doy0, dt, verbose)
            for lst, a in zip((h, Q, e, short, spill), r):
                lst.append(a)
            if verbose:
                print('  bank shift', s, '%.0f s' % (time.time() - t0), flush=True)
        h, Q, e, short, spill = (np.array(a) for a in (h, Q, e, short, spill))
        os.makedirs(CACHE, exist_ok=True)
        tmp = f + '.tmp.npz'
        np.savez(tmp, h=h.astype(np.float32), Q=Q.astype(np.float32), e=e, short=short, spill=spill)
        os.replace(tmp, f)
        z = np.load(f)
        return cls(z['h'], z['Q'], z['e'], z['short'], z['spill'], cands, days)

    @staticmethod
    def _simulate(cp, pp, shift, days, doy0, dt, verbose=False):
        """One season of the plant with the predictive controller and a constant shift, expected demand, clean bed, no measurement noise (as sim.run)."""
        from mpc import MPC
        pl = Plant(seed=0, sp=SedimentParams(), pp=copy.deepcopy(pp), cp=copy.deepcopy(cp), sediment_on=False)
        pl.rng = _ZeroRNG()
        c = pl.canal
        clim = Climatology(c, doy0, days)
        ctl = MPC()
        ctl.reset(pl, clim)
        n_sub = int(round(3600.0 / c.par.dt))
        c.h[:] = c.par.h_ref
        c.Q[:] = clim.sched[0].sum() + 6.0
        for _ in range(72):                                             # warm start, as in sim.run
            info = ctl.observe(pl, clim, t_override=0.0)
            cmd = ctl.act(info)
            _apply(pl, cmd, 0.0, 3600.0, n_sub, clim.actual(0.0), sediment=False, c_total=0.0)
        ctl.after_warmup(pl, clim)
        c.t = 0.0
        ctl.setpoint_shift = np.asarray(shift, float).copy()
        K = days * 24
        h, Q = np.zeros((K, c.M)), np.zeros((K, c.M))
        e, short, spill = np.zeros(K), np.zeros(K), np.zeros(K)
        for k in range(K):
            t = k * 3600.0
            info = ctl.observe(pl, clim)
            cmd = ctl.act(info)
            sc = clim.actual(t)
            e0 = pl.pump.energy_kwh
            out = _apply(pl, cmd, t, 3600.0, n_sub, sc, sediment=False, c_total=0.0)
            ctl.post_step(pl, clim, info, cmd, out)
            h[k], Q[k] = c.h, c.Q
            e[k] = pl.pump.energy_kwh - e0
            short[k] = float(np.sum(np.maximum(sc - out['delivered'], 0.0))) * 3600.0
            spill[k] = max(out['Qtail_mean'] - sc[-1], 0.0) * 3600.0
        return h, Q, e, short, spill

    # ------------------------------------------------------------------ rollout
    def index(self, shift):
        return self.cands.index(tuple(float(v) for v in shift))

    def rollout(self, i, plant0, k0, c_bed):
        """Advance the sediment of a copy of the twin `plant0` from hour k0 to the end of the season under candidate i.
        c_bed[k - k0]: bed-material concentration entering the canal in hour k (kg/m3).  Returns (plant, deposit volume m3)."""
        pl = copy.deepcopy(plant0)
        c, sed = pl.canal, pl.sed
        m0 = sed.mass_dep
        dt = c.par.dt
        hb, Qb = self.h[i], self.Q[i]
        for k in range(k0, self.hours):
            c.h[:] = np.maximum(hb[k] - sed.node_dz(), 0.05)
            c.Q[:] = Qb[k]
            sed.step(dt, c_bed[k - k0])
        return pl, (sed.mass_dep - m0) / (RHO_S * (1 - POROSITY))

    def evaluate(self, i, plant0, k0, c_bed, price, c_sh, c_sp, c_dep, lam):
        pl, dep_vol = self.rollout(i, plant0, k0, c_bed)
        L5 = float(pl.sed.capacity_loss()[0])
        E, sh, sp = (float(self.rem[n][i, k0]) for n in ('e', 'short', 'spill'))
        J = price * E + c_sh * sh + c_sp * sp + c_dep * dep_vol + lam * L5
        return J, dict(E=E, short=sh, spill=sp, dep_vol=dep_vol, L5=L5, dzmax=float(pl.sed.dz.max()))


def _build_one(args):
    n, i = args
    GRID = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]
    cp, pp = CanalParams(N=n), PumpParams()
    f = os.path.join(CACHE, 'part_N%d_%d.npz' % (n, i))
    if not os.path.exists(f):
        r = Bank._simulate(cp, pp, GRID[i], 245, 60, 3600.0)
        np.savez(f, h=r[0].astype(np.float32), Q=r[1].astype(np.float32), e=r[2], short=r[3], spill=r[4])
    return n, i


if __name__ == '__main__':
    # python3 forecast.py N [N ...]: build and cache the banks for these numbers of reaches (the nine candidates are simulated two at a time)
    from multiprocessing import Pool
    GRID = [(a, a, b, 0) for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]
    os.makedirs(CACHE, exist_ok=True)
    for n in [int(a) for a in sys.argv[1:]] or [114]:
        t0 = time.time()
        with Pool(2) as p:
            for n_, i in p.imap_unordered(_build_one, [(n, i) for i in range(len(GRID))]):
                print('N', n_, 'shift', GRID[i], 'done %.0f s' % (time.time() - t0), flush=True)
        parts = [np.load(os.path.join(CACHE, 'part_N%d_%d.npz' % (n, i))) for i in range(len(GRID))]
        cp, pp = CanalParams(N=n), PumpParams()
        key = _key(copy.deepcopy(cp), pp, GRID, 245, 60, 3600.0)
        out = os.path.join(CACHE, 'bank_%s.npz' % key)
        tmp = out + '.tmp.npz'
        np.savez(tmp, h=np.array([q['h'] for q in parts]), Q=np.array([q['Q'] for q in parts]), e=np.array([q['e'] for q in parts]),
                 short=np.array([q['short'] for q in parts]), spill=np.array([q['spill'] for q in parts]))
        os.replace(tmp, out)
        print('N', n, 'bank ready', out, '%.0f s' % (time.time() - t0), flush=True)
