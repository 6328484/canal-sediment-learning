"""Unified resumable experiment runner for the revised study.

Sediment scheme: exponentially fitted exchange (FIT), reach-mean (WIN) conveyance indicator, steady-flow capacity check.
Usage: python3 run_all.py STAGE [STAGE ...]   with stages  main  struct  sens  gridcl
Results: ../results/main.json (append, keyed by configuration)."""
import json, os, sys, time, traceback, numpy as np
from multiprocessing import Pool
from sim import *
from sediment import SedimentParams, RHO_S, POROSITY
from canal import CanalParams
from controllers import FFP
from mpc import MPC
from morph import MorphMPC, FixedShift
import gvf

F = os.environ.get('RESULTS', '../results/main.json')
FIT = int(os.environ.get('FIT', '1'))
NGRID = int(os.environ.get('NGRID', '114'))
WIN = float(os.environ.get('WIN', '5000'))
TWIN_DT = float(os.environ.get('TWIN_DT', '3600')) or None
DAYS = int(os.environ.get('DAYS', '0')) or None                  # shortened season for smoke tests
LAM = float(os.environ.get('LAM', '2e7'))
CANDS = json.loads(os.environ.get('CANDS', 'null'))          # planner candidate set (list of 4-vectors) or null for the default
FIXED = json.loads(os.environ.get('FIXED', '[0, 0, 0.6, 0.6]'))   # shift of the 'Fixed' benchmark (best single shift over the true multipliers)
BEST = json.loads(os.environ.get('BEST', '{"0.4": [0.3, 0.3, 0, 0], "1.0": [0, 0, 0.3, 0], "2.5": [0, 0, 0, 0]}'))   # 'FixedBest': best constant shift for each true multiplier
SEEDS = (1, 2, 3, 4, 5)                                       # seed 0 was used only to design the planner (landscape.py)
MULTS = (0.4, 1.0, 2.5)                                       # true capacity multipliers (none coincides with a twin hypothesis except 1)
Q0 = gvf.capacity(CanalParams(), np.zeros(10), tol=1e-5)


def build(cfg):
    kind, kw = cfg['ctl'], dict(cfg.get('kw', {}))
    if kind == 'FFP':
        return FFP()
    if kind == 'MPC':
        return MPC()
    if kind in ('Fixed', 'FixedBest'):
        return FixedShift(shift=kw['shift'])
    kw.setdefault('lam', LAM)
    if CANDS is not None:
        kw.setdefault('candidates', [tuple(c) for c in CANDS])
    kw.setdefault('twin_dt', TWIN_DT)
    ctl = MorphMPC(cap_mode=kind.split('-')[1], **kw)
    ctl.keep_diag = bool(os.environ.get('DIAG'))
    return ctl


def one(cfg):
    try:
        sp = SedimentParams(mult=cfg.get('mult', 1.0))
        sp.exp_fit = FIT
        sp.win = cfg.get('win', WIN)
        if cfg.get('d50'):
            sp.d50 = cfg['d50']
        sp.theta_shift = cfg.get('tshift', 0.0)
        cp = CanalParams(**{'N': NGRID, **cfg.get('cp', {})})
        plant = Plant(seed=cfg['seed'], sp=sp, cp=cp)
        scen = Scenario(plant.canal, seed=cfg['seed'], turb_scale=cfg.get('turb', 1.0))
        ctl = build(cfg)
        if cfg.get('d50') and hasattr(ctl, 'twin_d50'):
            ctl.twin_d50 = 5e-5
        series = []
        orig = ctl.post_step

        def post(pl, sc, info, cmd, out):
            orig(pl, sc, info, cmd, out)
            k = int(round(info['t'] / 3600.0)) + 1
            if k % 240 == 0:
                series.append((k / 24.0, float(pl.sed.capacity_loss()[0]), float(pl.sed.mass_dep / 1e6)))
        ctl.post_step = post
        t0 = time.time()
        rec, tot = run(plant, scen, ctl, days=DAYS)
        he = np.array([r['h_end'] for r in rec])
        s = plant.sed
        Lsys = gvf.capacity_loss(plant.canal.par, s.dz, s.sp.kappa_n, q0=Q0, dxf=100.0)[0]
        d = dict(cfg=cfg, L5=float(s.capacity_loss()[0]), L2500=float(s.capacity_loss(window=2500.0)[0]),
                 L10k=float(s.capacity_loss(window=10000.0)[0]), Lsys=float(Lsys),
                 Lpeak=float(s.capacity_loss(window=None)[0]), dz_max=float(s.dz.max()),
                 dep_kt=float(s.mass_dep / 1e6), dep_vol_Mm3=float(s.mass_dep / (RHO_S * (1 - POROSITY)) / 1e6),
                 short=tot['short'] / tot['intended'], spill=tot['spill'] / tot['intended'],
                 abs_err=tot['deliv_abs_err'] / tot['intended'], energy=float(plant.pump.energy_kwh / 1e6),
                 pumped=float(plant.pump.water_m3 / 1e6), hmin=float(he.min()), secs=time.time() - t0,
                 mass_in_kt=float(s.mass_in / 1e6), mass_out_kt=float(s.mass_out / 1e6),
                 dz_prof=[float(v) for v in s.dz], series=series)
        tw = getattr(ctl, 'twin', None)
        if tw is not None and getattr(tw, 'meas_hist', None):
            d['mult_hist'] = [(n, m) for n, m, k in tw.mult_history]
            d['loglik_hist'] = [[float(v) for v in l] for l in tw.loglik_hist]
            d['meas_hist'] = [[float(v) for v in m] for m in tw.meas_hist]
            d['pred_hist'] = [[[float(v) for v in row] for row in P] for P in tw.pred_hist]
        if hasattr(ctl, 'log') and ctl.log:
            d['plan_hist'] = [list(map(float, b)) for (t, b, J) in ctl.log]
        if getattr(ctl, 'diag', None):
            d['diag'] = ctl.diag
        return d
    except Exception as e:
        return dict(cfg=cfg, error=repr(e), tb=traceback.format_exc())


def key(c):
    return json.dumps(c, sort_keys=True)


def jobs(stage):
    J = []
    if stage == 'main':
        for s in SEEDS:
            for m in MULTS:
                for c in ('Morph-learn', 'Morph-oracle', 'Morph-nominal', 'MPC', 'FFP', 'Fixed', 'FixedBest'):
                    j = dict(ctl=c, seed=s, mult=m, tag='main')
                    if c == 'Morph-oracle' and m == 1.0:
                        continue                         # identical to Morph-nominal (the nominal multiplier is the true one); copied in the analysis
                    if c == 'Fixed':
                        j['kw'] = dict(shift=FIXED)
                    if c == 'FixedBest':
                        j['kw'] = dict(shift=BEST['%g' % m] if '%g' % m in BEST else BEST[str(m)])
                        if list(j['kw']['shift']) == list(FIXED) or not any(j['kw']['shift']):
                            continue                     # identical to Fixed or to MPC; copied in the analysis
                    J.append(j)
    elif stage == 'struct':
        for s in SEEDS[:3]:
            for var in (dict(d50=2e-5), dict(d50=1.2e-4), dict(tshift=-1.0), dict(tshift=1.0)):
                for c in ('Morph-learn', 'Morph-nominal', 'MPC'):
                    J.append(dict(ctl=c, seed=s, mult=1.0, tag='struct', **var))
    elif stage == 'sens':
        m0 = 0.4
        for s in SEEDS[:3]:
            for lam in (2e6, 7e6, 6e7, 2e8):
                J.append(dict(ctl='Morph-learn', seed=s, mult=m0, kw=dict(lam=lam), tag='lam'))
            for lam in (2e6, 2e8):          # does learning matter at other weights? nominal twin at the extreme weights
                J.append(dict(ctl='Morph-nominal', seed=s, mult=m0, kw=dict(lam=lam), tag='lam_nom'))
            for ev in (360, 1440):
                J.append(dict(ctl='Morph-learn', seed=s, mult=m0, kw=dict(survey_every=ev), tag='survey_every'))
            for sn in (0.01, 0.05):
                J.append(dict(ctl='Morph-learn', seed=s, mult=m0, kw=dict(survey_noise=sn), tag='survey_noise'))
            for fg, tp in ((1.0, 4.0), (0.9, 2.0), (0.9, 8.0)):
                J.append(dict(ctl='Morph-learn', seed=s, mult=m0, kw=dict(forget=fg, temper=tp), tag='likelihood'))
            for tb in (0.5, 2.0):
                J.append(dict(ctl='MPC', seed=s, mult=m0, turb=tb, tag='turb'))
                J.append(dict(ctl='Morph-learn', seed=s, mult=m0, turb=tb, tag='turb'))
    elif stage == 'gridcl':
        for n in (57, 228):
            for m in (0.4, 1.0):
                for c in ('MPC', 'Morph-learn'):
                    J.append(dict(ctl=c, seed=SEEDS[0], mult=m, cp=dict(N=n), tag='gridcl'))
    elif stage == 'base':          # all jobs that do not depend on the planner design (local and predictive control)
        J = [j for st in ('main', 'struct', 'sens', 'gridcl') for j in jobs(st) if j['ctl'] in ('FFP', 'MPC')]
    return J


if __name__ == "__main__":
    stages = sys.argv[1:] or ['main']
    out = json.load(open(F)) if os.path.exists(F) else []
    done = {key(r['cfg']) for r in out}
    for other in [p for p in os.environ.get('SKIP_FROM', '../results/base.json').split(',') if p and p != F and os.path.exists(p)]:
        done |= {key(r['cfg']) for r in json.load(open(other))}      # jobs already stored in another results file
    J = []
    for st in stages:
        J += [j for j in jobs(st) if key(j) not in done]
    print('to run', len(J), '| FIT', FIT, 'WIN', WIN, 'TWIN_DT', TWIN_DT, 'LAM', LAM, flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(one, J):
            if 'error' in r:
                print('ERROR', r['cfg'], r['error'], flush=True); print(r['tb'], flush=True); continue
            out.append(r); json.dump(out, open(F, 'w'))
            print(len(out), r['cfg'].get('tag'), r['cfg']['ctl'], 's', r['cfg']['seed'], 'm', r['cfg'].get('mult'), 'L5=%.4f Lsys=%.4f dep=%.1f short=%.4f' % (r['L5'], r['Lsys'], r['dep_kt'], r['short']), '%ds' % round(r['secs']), flush=True)
    print('ALL DONE', len(out), flush=True)
