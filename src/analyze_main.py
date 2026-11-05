"""Analysis of the final study (results/main.json): main (E1), struct (E2), sens (E3), gridcl (E4), identifiability.

Usage: python3 analyze_main.py [results.json]
Writes ../results/{table_main,paired_main,ident,struct,sens,gridcl}.csv and prints the tables."""
import json, os, sys, numpy as np, pandas as pd
from scipy import stats
from scipy.special import ndtr

FILES = sys.argv[1:] or ['../results/main.json', '../results/base.json']
R, _seen = [], set()
for _f in FILES:
    if not os.path.exists(_f): continue
    for r in json.load(open(_f)):
        k = json.dumps(r['cfg'], sort_keys=True)
        if 'error' in r or k in _seen: continue
        _seen.add(k); R.append(r)
OUT = os.environ.get('OUT', '../results/')
METRICS = ['L5', 'Lsys', 'dep', 'short', 'spill', 'energy', 'J']
PRICE, C_SH, C_SP, C_DEP, LAM0 = 0.03, 0.10, 0.02, 3.0, 2e7        # weights of the planner (Eq. planner) at the default capacity-loss weight
_INT = {}


def intended(seed):
    """Intended delivery volume of a season (m3) of the scenario with this seed."""
    if seed not in _INT:
        from sim import Scenario, Plant
        from canal import CanalParams
        sc = Scenario(Plant(seed=seed, cp=CanalParams(N=114)).canal, seed=seed)
        _INT[seed] = sum(float(np.sum(sc.actual(k * 3600.0))) * 3600.0 for k in range(245 * 24))
    return _INT[seed]


def seasonal_cost(r, seed):
    """Seasonal cost (M USD) with the terms and weights of the planner: energy, shortage, spill, deposit volume and capacity loss."""
    v = intended(seed)
    return (PRICE * r['energy'] * 1e6 + C_SH * r['short'] * v + C_SP * r['spill'] * v + C_DEP * r['dep_vol_Mm3'] * 1e6 + LAM0 * r['L5']) / 1e6
OMEGA, GAMMA, SIG_L, SIG_S = 0.9, 4.0, 0.04, 0.02
MULTS_H = np.array([0.35, 0.6, 1.0, 1.7, 3.0])


def rows():
    out = []
    for r in R:
        c = r['cfg']; kw = c.get('kw', {})
        out.append(dict(tag=c.get('tag'), ctl=c['ctl'], seed=c['seed'], mult=c.get('mult', 1.0), d50=c.get('d50'), tshift=c.get('tshift', 0.0),
                        turb=c.get('turb', 1.0), N=c.get('cp', {}).get('N', 114), kw=json.dumps(kw, sort_keys=True),
                        lam=kw.get('lam'), survey_every=kw.get('survey_every'), survey_noise=kw.get('survey_noise'),
                        forget=kw.get('forget'), temper=kw.get('temper'),
                        L5=100 * r['L5'], Lsys=100 * r['Lsys'], L2500=100 * r['L2500'], L10k=100 * r['L10k'], Lpeak=100 * r['Lpeak'],
                        dep=r['dep_kt'], dzmax=r['dz_max'], short=100 * r['short'], spill=100 * r['spill'], energy=r['energy'],
                        hmin=r['hmin'], secs=r['secs'], mass_in=r.get('mass_in_kt', np.nan), mass_out=r.get('mass_out_kt', np.nan), J=seasonal_cost(r, c['seed'])))
    return pd.DataFrame(out)


D = rows()


def add_copies(d):
    """Runs that were not repeated because they are identical by construction: Morph-oracle at the nominal multiplier equals Morph-nominal;
    FixedBest equals Fixed or MPC when the best shift is the fixed shift or no shift (design.json)."""
    try:
        des = json.load(open('../results/design.json'))
    except Exception:
        des = None
    add = []
    m = d[d.tag == 'main']
    for (sd, mu), g in m.groupby(['seed', 'mult']):
        have = set(g.ctl)
        if mu == 1.0 and 'Morph-oracle' not in have and 'Morph-nominal' in have:
            add.append(g[g.ctl == 'Morph-nominal'].iloc[0].to_dict() | dict(ctl='Morph-oracle'))
        if des and 'FixedBest' not in have:
            b = des['best'].get('%g' % mu)
            if b is not None and list(b) == list(des['fixed']) and 'Fixed' in have:
                add.append(g[g.ctl == 'Fixed'].iloc[0].to_dict() | dict(ctl='FixedBest'))
            elif b is not None and not any(b) and 'MPC' in have:
                add.append(g[g.ctl == 'MPC'].iloc[0].to_dict() | dict(ctl='FixedBest'))
    return pd.concat([d, pd.DataFrame(add)], ignore_index=True) if add else d


D = add_copies(D)


def tci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return x.mean(), np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return x.mean(), x.mean() - t * se, x.mean() + t * se


def summary(d, by=('mult', 'ctl')):
    out = []
    for gk, g in d.groupby(list(by)):
        row = dict(zip(by, gk), n=len(g))
        for k in METRICS:
            row[k] = g[k].mean(); row[k + '_sd'] = g[k].std(ddof=1) if len(g) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def paired(d, a, b, by=('mult',), keys=('seed',)):
    """Paired difference a - b between controllers a and b (same seed and same other factors)."""
    out = []
    for gk, g in d.groupby(list(by)):
        A = g[g.ctl == a].set_index(list(keys)); B = g[g.ctl == b].set_index(list(keys))
        idx = A.index.intersection(B.index)
        if len(idx) < 2: continue
        row = dict(zip(by, gk if isinstance(gk, tuple) else (gk,)), a=a, b=b, n=len(idx))
        for k in METRICS:
            dl = (A.loc[idx, k] - B.loc[idx, k]).values
            m, lo, hi = tci(dl)
            row[k + '_d'], row[k + '_lo'], row[k + '_hi'] = m, lo, hi
            row[k + '_rel'] = 100 * float(np.mean(dl / B.loc[idx, k].values)); row[k + '_k'] = int((dl < 0).sum())
        out.append(row)
    return pd.DataFrame(out)


def fmt(P, mets=('L5', 'Lsys', 'dep', 'short', 'energy', 'J')):
    for _, r in P.iterrows():
        print('  ' + ' '.join('%s=%s' % (k, r[k]) for k in P.columns if k in ('mult', 'N', 'var', 'variant', 'tag')) + ' n=%d | ' % r['n'] +
              ' | '.join('%s %+.3f [%+.3f,%+.3f] (%+.1f%%, %d/%d)' % (k, r[k + '_d'], r[k + '_lo'], r[k + '_hi'], r[k + '_rel'], r[k + '_k'], r['n']) for k in mets))


# ---------------------------------------------------------------------------------------------------- identifiability (Proposition 3)
LPSI = -np.log(np.array([0.35, 0.6, 1.0, 1.7, 3.0])) ** 2 / (2 * 0.8 ** 2)


def ident_rows():
    out = []
    for r in R:
        c = r['cfg']
        if c['ctl'] != 'Morph-learn' or 'pred_hist' not in r or c.get('tag') != 'main': continue
        P = np.array(r['pred_hist'])           # (M, K, 14) twin predictions at the stations
        Z = np.array(r['meas_hist'])           # (M, 14) noisy measurements
        L = np.array(r['loglik_hist'])         # (M, K)
        M = P.shape[0]
        # closest hypothesis to the plant: smallest accumulated (bias-corrected) squared distance of the twins to the measurements
        dist2 = np.array([[np.sum((Z[i] - P[i, k]) ** 2) - Z.shape[1] * SIG_S ** 2 for k in range(P.shape[1])] for i in range(M)])
        kstar = int(np.argmin(dist2.sum(axis=0)))
        for m in range(1, M + 1):
            d = np.array([[np.linalg.norm(P[i, kstar] - P[i, k]) for k in range(P.shape[1])] for i in range(m)])   # (m, K)
            w = OMEGA ** (m - 1 - np.arange(m))
            EG = (w[:, None] * d ** 2).sum(0) / (2 * GAMMA * SIG_L ** 2) + OMEGA ** m * (LPSI[kstar] - LPSI)      # data term + prior term of Proposition 3
            SD = np.sqrt(((w ** 2)[:, None] * SIG_S ** 2 * d ** 2).sum(0)) / (GAMMA * SIG_L ** 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = np.where(SD > 0, EG / SD, np.inf)
            z[kstar] = np.inf
            comp = int(np.argmin(z))                      # nearest competitor in the sense of the smallest separation
            out.append(dict(mult=c['mult'], seed=c['seed'], m=m, kstar=kstar, comp=comp, z=float(z[comp]), p_wrong=float(ndtr(-z[comp])),
                            d_comp=float(d[-1, comp]), G_obs=float(L[m - 1, kstar] - L[m - 1, comp]), mapk=int(np.argmax(L[m - 1])),
                            w_kstar=float(np.exp(L[m - 1, kstar] - L[m - 1].max()) / np.exp(L[m - 1] - L[m - 1].max()).sum())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------- main
if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
    main = D[D.tag == 'main']
    if len(main):
        S = summary(main); S.to_csv(OUT + 'table_main.csv', index=False)
        order = ['FFP', 'MPC', 'Fixed', 'FixedBest', 'Morph-nominal', 'Morph-learn', 'Morph-oracle']
        print('=== E1: mean (SD) over seeds')
        for m in sorted(main.mult.unique()):
            print(' true multiplier', m)
            for c in order:
                x = S[(S.mult == m) & (S.ctl == c)]
                if len(x):
                    x = x.iloc[0]
                    print('   %-14s n=%d ' % (c, x['n']) + ' '.join('%s=%.3f(%.3f)' % (k, x[k], x[k + '_sd']) for k in ('L5', 'Lsys', 'dep', 'short', 'energy', 'J')))
        pairs = [('Morph-learn', 'MPC'), ('Morph-learn', 'Fixed'), ('Morph-learn', 'FixedBest'), ('Morph-oracle', 'MPC'), ('Morph-oracle', 'FixedBest'), ('Morph-nominal', 'MPC'), ('Fixed', 'MPC'), ('FixedBest', 'MPC'),
                 ('MPC', 'FFP'), ('Morph-learn', 'Morph-nominal'), ('Morph-learn', 'Morph-oracle')]
        allp = []
        for a, b in pairs:
            P = paired(main, a, b)
            if len(P):
                print('\n=== paired %s - %s (mean [95%% CI] (relative %%, k/n with negative difference))' % (a, b)); fmt(P); allp.append(P)
        if allp:
            PP = pd.concat(allp); PP.to_csv(OUT + 'paired_main.csv', index=False)
            print('\n=== share of the gain of the true-capacity twin (L5 reduction relative to predictive control)')
            for m in sorted(PP.mult.unique()):
                g = {a_: -PP[(PP.mult == m) & (PP.a == a_) & (PP.b == 'MPC')].L5_d.iloc[0] for a_ in ('Fixed', 'FixedBest', 'Morph-nominal', 'Morph-learn', 'Morph-oracle') if len(PP[(PP.mult == m) & (PP.a == a_) & (PP.b == 'MPC')])}
                print('  mu=%g: ' % m + ', '.join('%s %.2f pp' % (k, v) for k, v in g.items()) + ('; learned/true = %.0f%%, nominal/true = %.0f%%, fixed/true = %.0f%%' % tuple(100 * g.get(k, np.nan) / g['Morph-oracle'] for k in ('Morph-learn', 'Morph-nominal', 'Fixed')) if 'Morph-oracle' in g and g['Morph-oracle'] > 0 else ''))
        # statistics of the weekly decisions of the planner
        ps = []
        for r in R:
            c = r['cfg']
            if c.get('tag') == 'main' and c['ctl'].startswith('Morph') and 'plan_hist' in r:
                P = np.array(r['plan_hist'])
                ps.append(dict(ctl=c['ctl'], mult=c['mult'], seed=c['seed'], weeks=len(P), a_mean=P[:, 0].mean(), b_mean=P[:, 2].mean(), head_weeks=int((P[:, 0] > 0).sum()), pool3_weeks=int((P[:, 2] > 0).sum()),
                               none_weeks=int(((P[:, 0] == 0) & (P[:, 2] == 0)).sum()), changes=int((np.abs(np.diff(P, axis=0)).sum(axis=1) > 0).sum())))
        if ps:
            PS = pd.DataFrame(ps); PS.to_csv(OUT + 'plan_stats.csv', index=False)
            print('\n=== planner decisions (mean over seeds): weeks with a head shift, with a pool-3 shift, with no shift, number of changes of the shift')
            print(PS.groupby(['mult', 'ctl'])[['a_mean', 'b_mean', 'head_weeks', 'pool3_weeks', 'none_weeks', 'changes']].mean().round(2).to_string())
        I = ident_rows()
        if len(I):
            I.to_csv(OUT + 'ident.csv', index=False)
            print('\n=== identifiability: Proposition 3 versus the simulated surveys (nearest competitor)')
            for (m_, s_), g in I.groupby(['mult', 'seed']):
                g = g.set_index('m')
                print(' mult %.1f seed %d k*=%d comp=%d  z(m)=%s  p_wrong(8)=%.2g  MAP=%s' % (m_, s_, g.kstar.iloc[0], g.comp.iloc[-1], np.round(g.z.values, 1).tolist(), g.p_wrong.iloc[-1], g.mapk.tolist()))
            late = I[I.m >= 4]
            print(' MAP equals nearest hypothesis in %.0f%% of (run, survey>=4); mean predicted P(wrong) there %.3f' % (100 * (late.mapk == late.kstar).mean(), late.p_wrong.mean()))
    st = D[D.tag == 'struct']
    if len(st):
        st = st.assign(variant=[('d50=%.2f mm' % (1e3 * d)) if d == d and d else ('Shields exponent %+.0f' % t if t else 'nominal') for d, t in zip(st.d50, st.tshift)])
        out = []
        for v, g in st.groupby('variant'):
            for a, b in (('MPC', None), ('Morph-nominal', 'MPC'), ('Morph-learn', 'MPC'), ('Morph-learn', 'Morph-nominal')):
                if b is None:
                    x = g[g.ctl == a]
                    if len(x): out.append(dict(variant=v, ctl=a, n=len(x), **{k: x[k].mean() for k in METRICS}))
                elif a == 'Morph-learn' and b == 'MPC':
                    x = g[g.ctl == a]
                    if len(x): out.append(dict(variant=v, ctl=a, n=len(x), **{k: x[k].mean() for k in METRICS}))
        T = pd.DataFrame(out)
        mu_by = {}
        for r in R:
            c = r['cfg']
            if c.get('tag') == 'struct' and c['ctl'] == 'Morph-learn' and 'mult_hist' in r:
                v = ('d50=%.2f mm' % (1e3 * c['d50'])) if c.get('d50') else ('Shields exponent %+.0f' % c['tshift'] if c.get('tshift') else 'nominal')
                mu_by.setdefault(v, []).append(r['mult_hist'][-1][1])
        T['muhat'] = [float(np.mean(mu_by[v])) if (c == 'Morph-learn' and v in mu_by) else np.nan for v, c in zip(T.variant, T.ctl)]
        T.to_csv(OUT + 'struct.csv', index=False)
        print('\n=== E2: structural errors, mean over seeds (MPC reference level)'); print(T.round(3).to_string(index=False))
        for a, b in (('Morph-nominal', 'MPC'), ('Morph-learn', 'MPC'), ('Morph-learn', 'Morph-nominal')):
            P = paired(st, a, b, by=('variant',)); print('\n  paired %s - %s' % (a, b)); fmt(P) if len(P) else None
            if len(P): P.to_csv(OUT + 'struct_%s_%s.csv' % (a, b), index=False)
        # learned multiplier under structural error
        for r in R:
            c = r['cfg']
            if c.get('tag') == 'struct' and c['ctl'] == 'Morph-learn' and 'mult_hist' in r:
                print('   struct run', {k: c.get(k) for k in ('seed', 'd50', 'tshift')}, 'final mu_hat=%.2f' % r['mult_hist'][-1][1])
    sn = D[D.tag.isin(['lam', 'lam_nom', 'survey_every', 'survey_noise', 'likelihood', 'turb'])]
    if len(sn):
        def label(r):
            if r.tag == 'lam': return 'lambda=%.0e' % r.lam
            if r.tag == 'lam_nom': return 'nominal twin, lambda=%.0e' % r.lam
            if r.tag == 'survey_every': return 'survey every %d d' % (r.survey_every / 24)
            if r.tag == 'survey_noise': return 'survey noise %.2f m' % r.survey_noise
            if r.tag == 'likelihood': return 'omega=%.1f, gamma=%.0f' % (r.forget, r.temper)
            return 'turbidity x%.1f' % r.turb
        sn = sn.assign(variant=[label(r) for r in sn.itertuples()])
        ref = D[(D.tag == 'main') & (D.mult == 0.4) & (D.ctl.isin(['MPC', 'Morph-learn', 'Morph-nominal']))].assign(variant='reference (E1)', turb=1.0)
        rows_ = []
        for v, g in sn.groupby('variant'):
            ml = g[g.ctl.isin(['Morph-learn', 'Morph-nominal'])].set_index('seed')
            mp = g[g.ctl == 'MPC'].set_index('seed')
            if mp.empty:
                mp = ref[ref.ctl == 'MPC'].set_index('seed')
            idx = ml.index.intersection(mp.index)
            if len(idx) < 2: continue
            row = dict(variant=v, n=len(idx))
            for k in ('L5', 'Lsys', 'dep', 'short', 'energy'):
                dl = (ml.loc[idx, k] - mp.loc[idx, k]).values; m_, lo, hi = tci(dl)
                row[k + '_d'], row[k + '_lo'], row[k + '_hi'] = m_, lo, hi; row[k + '_k'] = int((dl < 0).sum())
            row['L5_abs'] = ml.loc[idx, 'L5'].mean(); row['L5_mpc'] = mp.loc[idx, 'L5'].mean()
            rows_.append(row)
        # the default configuration on the same seeds
        for lab_, cname in (('default (E1, seeds 1-3)', 'Morph-learn'), ('nominal twin, default lambda (E1, seeds 1-3)', 'Morph-nominal')):
            ml = ref[ref.ctl == cname].set_index('seed'); mp = ref[ref.ctl == 'MPC'].set_index('seed')
            idx = [s for s in (1, 2, 3) if s in ml.index and s in mp.index]
            if len(idx) >= 2:
                row = dict(variant=lab_, n=len(idx))
                for k in ('L5', 'Lsys', 'dep', 'short', 'energy'):
                    dl = (ml.loc[idx, k] - mp.loc[idx, k]).values; m_, lo, hi = tci(dl)
                    row[k + '_d'], row[k + '_lo'], row[k + '_hi'] = m_, lo, hi; row[k + '_k'] = int((dl < 0).sum())
                row['L5_abs'] = ml.loc[idx, 'L5'].mean(); row['L5_mpc'] = mp.loc[idx, 'L5'].mean(); rows_.append(row)
        T = pd.DataFrame(rows_); T.to_csv(OUT + 'sens.csv', index=False)
        print('\n=== E3: sensitivity at mu=0.4 (Morph-learn minus MPC, seeds 1-3; mean [95% CI], k/n improved)')
        for _, r in T.iterrows():
            print('  %-26s n=%d L5 %+.2f [%+.2f,%+.2f] %d/%d | short %+.3f | energy %+.2f | L5 abs %.2f (MPC %.2f)' % (r.variant, r.n, r.L5_d, r.L5_lo, r.L5_hi, r.L5_k, r.n, r.short_d, r.energy_d, r.L5_abs, r.L5_mpc))
    gc = D[(D.tag == 'gridcl') | ((D.tag == 'main') & (D.seed == 1) & (D.mult.isin([0.4, 1.0])) & (D.ctl.isin(['MPC', 'Morph-learn'])))]
    if (gc.tag == 'gridcl').any():
        rows_ = []
        for (m_, N), g in gc.groupby(['mult', 'N']):
            a = g[g.ctl == 'Morph-learn']; b = g[g.ctl == 'MPC']
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                rows_.append(dict(mult=m_, N=N, L5_mpc=b.L5, L5_learn=a.L5, dL5=a.L5 - b.L5, Lsys_mpc=b.Lsys, Lsys_learn=a.Lsys, dLsys=a.Lsys - b.Lsys,
                                  dep_mpc=b.dep, dep_learn=a.dep, ddep=a.dep - b.dep, short_mpc=b.short, short_learn=a.short))
        T = pd.DataFrame(rows_).sort_values(['mult', 'N']); T.to_csv(OUT + 'gridcl.csv', index=False)
        print('\n=== E4: closed-loop grid check (seed 1)'); print(T.round(3).to_string(index=False))
