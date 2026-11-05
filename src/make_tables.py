"""Markdown tables of Section 5 from the analysis CSVs (results/*.csv) -> ../jhr/md/scratch/tables_final.md"""
import os, json, numpy as np, pandas as pd
R = os.environ.get('OUT', '../results/')
M = lambda x: ('%s' % x).replace('-', '−')
def f(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '\u2013'
    t = ('%.' + str(nd) + 'f') % x
    return t.lstrip('-') if float(t) == 0 else t.replace('-', '\u2212')
def pm(m, s, nd=1): return '%s ± %s' % (f(m, nd), ('%.' + str(nd) + 'f') % s)
def ci(d, lo, hi, nd=1): return '%s [%s, %s]' % (f(d, nd), f(lo, nd), f(hi, nd))
LAB = {'FFP': 'Local', 'MPC': 'Predictive', 'Fixed': 'Fixed shift', 'FixedBest': 'Best fixed shift', 'Morph-nominal': 'Morphology, nominal', 'Morph-learn': 'Morphology, learned', 'Morph-oracle': 'Morphology, true'}
VN = {'d50=0.02 mm': 'Grain size 0.02 mm', 'd50=0.12 mm': 'Grain size 0.12 mm', 'Shields exponent -1': 'Shields exponent 3/2 (instead of 5/2)', 'Shields exponent +1': 'Shields exponent 7/2 (instead of 5/2)'}
out = []
# ---- main table
if os.path.exists(R + 'table_main.csv'):
    T = pd.read_csv(R + 'table_main.csv')
    out += ['| True *μ* | Controller | *L*~5~ (%) | *L*~sys~ (%) | Deposit (kt) | Shortage (%) | Energy (GWh) | Cost *J* (M USD) |', '|:--|:--|--:|--:|--:|--:|--:|--:|']
    for m in sorted(T.mult.unique()):
        for c in ['FFP', 'MPC', 'Fixed', 'FixedBest', 'Morph-nominal', 'Morph-learn', 'Morph-oracle']:
            x = T[(T.mult == m) & (T.ctl == c)]
            if x.empty: continue
            x = x.iloc[0]
            out.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % ('%g' % m if c == 'FFP' else '', LAB[c], pm(x.L5, x.L5_sd, 1), pm(x.Lsys, x.Lsys_sd, 2), pm(x.dep, x.dep_sd, 1), pm(x.short, x.short_sd, 2), pm(x.energy, x.energy_sd, 1), pm(x.J, x.J_sd, 2)))
    out.append('')
    out.append('<!--END main-->')
# ---- paired table
if os.path.exists(R + 'paired_main.csv'):
    P = pd.read_csv(R + 'paired_main.csv')
    out += ['', '| True *μ* | Contrast | Δ*L*~5~ (pp) | Seeds | Δ*L*~sys~ (pp) | Δ shortage (pp) | Δ energy (GWh) | Δ*J* (M USD) |', '|:--|:--|:--|:--:|:--|:--|:--|:--|']
    for m in sorted(P.mult.unique()):
        first = True
        for a, b in (('Fixed', 'MPC'), ('FixedBest', 'MPC'), ('Morph-nominal', 'MPC'), ('Morph-learn', 'MPC'), ('Morph-oracle', 'MPC'), ('Morph-learn', 'Fixed'), ('Morph-learn', 'FixedBest'), ('Morph-oracle', 'FixedBest'), ('Morph-learn', 'Morph-nominal'), ('Morph-learn', 'Morph-oracle')):
            x = P[(P.mult == m) & (P.a == a) & (P.b == b)]
            if x.empty: continue
            x = x.iloc[0]
            out.append('| %s | %s − %s | %s | %d/%d | %s | %s | %s | %s |' % ('%g' % m if first else '', LAB[a], LAB[b].lower() if b != 'MPC' else 'predictive', ci(x.L5_d, x.L5_lo, x.L5_hi, 2), x.L5_k, x.n, ci(x.Lsys_d, x.Lsys_lo, x.Lsys_hi, 2), ci(x.short_d, x.short_lo, x.short_hi, 2), ci(x.energy_d, x.energy_lo, x.energy_hi, 1), ci(x.J_d, x.J_lo, x.J_hi, 2)))
            first = False
    out.append('')
    out.append('<!--END paired-->')
# ---- structural
if os.path.exists(R + 'struct.csv'):
    S = pd.read_csv(R + 'struct.csv')
    pn = pd.read_csv(R + 'struct_Morph-nominal_MPC.csv') if os.path.exists(R + 'struct_Morph-nominal_MPC.csv') else None
    pl = pd.read_csv(R + 'struct_Morph-learn_MPC.csv') if os.path.exists(R + 'struct_Morph-learn_MPC.csv') else None
    pd_ = pd.read_csv(R + 'struct_Morph-learn_Morph-nominal.csv') if os.path.exists(R + 'struct_Morph-learn_Morph-nominal.csv') else None
    out += ['', '| Structural error in the plant | *L*~5~ predictive (%) | Δ*L*~5~ nominal (pp) | Δ*L*~5~ learned (pp) | Learned − nominal (pp) | Final μ̂ |', '|:--|--:|:--|:--|:--|--:|']
    for v in S.variant.unique():
        row = S[(S.variant == v) & (S.ctl == 'MPC')]
        if row.empty: continue
        g = lambda D: (D[D.variant == v].iloc[0] if D is not None and len(D[D.variant == v]) else None)
        a, b, c = g(pn), g(pl), g(pd_)
        out.append('| %s | %s | %s | %s | %s | %s |' % (VN.get(v, v), f(row.iloc[0].L5, 1), ci(a.L5_d, a.L5_lo, a.L5_hi, 2) if a is not None else '–', ci(b.L5_d, b.L5_lo, b.L5_hi, 2) if b is not None else '–', ci(c.L5_d, c.L5_lo, c.L5_hi, 2) if c is not None else '–', (f(float(S[(S.variant == v) & (S.ctl == 'Morph-learn')].muhat.iloc[0]), 2) if len(S[(S.variant == v) & (S.ctl == 'Morph-learn')]) else '\u2013')))
    out.append('')
    out.append('<!--END struct-->')
# ---- sensitivity
if os.path.exists(R + 'sens.csv'):
    S = pd.read_csv(R + 'sens.csv')
    out += ['', '| Variant | Δ*L*~5~ (pp) | Seeds | Δ shortage (pp) | Δ energy (GWh) | *L*~5~ learned (%) | *L*~5~ predictive (%) |', '|:--|:--|:--:|:--|:--|--:|--:|']
    for _, x in S.iterrows():
        out.append('| %s | %s | %d/%d | %s | %s | %s | %s |' % (x.variant, ci(x.L5_d, x.L5_lo, x.L5_hi, 2), x.L5_k, x.n, f(x.short_d, 2), f(x.energy_d, 1), f(x.L5_abs, 1), f(x.L5_mpc, 1)))
    out.append('')
    out.append('<!--END sens-->')
# ---- closed-loop grid check
if os.path.exists(R + 'gridcl.csv'):
    G = pd.read_csv(R + 'gridcl.csv')
    out += ['', '| True *μ* | *N* | *L*~5~ predictive (%) | *L*~5~ learned (%) | Δ*L*~5~ (pp) | Δ*L*~sys~ (pp) | Δ deposit (kt) |', '|:--|--:|--:|--:|--:|--:|--:|']
    for _, x in G.iterrows():
        out.append('| %g | %d | %s | %s | %s | %s | %s |' % (x.mult, x.N, f(x.L5_mpc, 2), f(x.L5_learn, 2), f(x.dL5, 2), f(x.dLsys, 2), f(x.ddep, 1)))
    out.append('')
    out.append('<!--END gridcl-->')
open('../jhr/md/scratch/tables_final.md', 'w').write('\n'.join(out))
print('\n'.join(out))
