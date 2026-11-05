"""Inspect the weekly decisions of the planner in the design check: where the worst 5-km window of the plant and of the MAP twin is, what the planner predicted for
the zero shift and for the chosen shift (components of J_p), and how the twin's bed differs from the plant's."""
import json, sys, numpy as np
F = sys.argv[1] if len(sys.argv) > 1 else '../results/design_check.json'
want = (sys.argv[2], float(sys.argv[3])) if len(sys.argv) > 3 else None
R = json.load(open(F))
L, N = 56.6, 114
xc = (np.arange(N) + 0.5) * L / N
for r in R:
    c = r['cfg']
    if want and (c['ctl'] != want[0] or c['mult'] != want[1]):
        continue
    D = r.get('diag')
    if not D:
        continue
    print('\n==== %s mu=%g seed %d | final L5 %.2f%% Lsys %.2f%% short %.2f%% dep %.1f kt' % (c['ctl'], c['mult'], c['seed'], 100 * r['L5'], 100 * r['Lsys'], 100 * r['short'], r['dep_kt']))
    print(' wk  best          plantL5 (x_km)  twinL5 (x_km)   | zero: dE  short  dep   dL5%   | best: dE  short  dep   dL5%   | dJ(best-zero) k$')
    for k, e in enumerate(D):
        pw, tw = np.array(e['plant_dzw']), np.array(e['twin_dzw'])
        cands = {tuple(cc[0]): cc for cc in e['cands']}
        z = cands[(0.0, 0.0, 0.0, 0.0)]
        b = cands[tuple(e['best'])]
        print('%3d  %-14s %5.2f (%4.1f)   %5.2f (%4.1f)   | %6.0f %6.0f %5.2f %6.3f | %6.0f %6.0f %5.2f %6.3f | %7.1f' % (
            k + 1, ','.join('%g' % v for v in e['best'][:3]), 100 * e['plant_L5'], xc[np.argmax(pw)], 100 * e['twin_L5'], xc[np.argmax(tw)],
            z[2]['dE'], z[2]['short'] / 1e3, z[2]['dep_vol'] / 1e3, 100 * z[2]['agg'], b[2]['dE'], b[2]['short'] / 1e3, b[2]['dep_vol'] / 1e3, 100 * b[2]['agg'], (b[1] - z[1]) / 1e3))
    ph = r.get('pred_hist')
    if ph and r.get('meas_hist'):
        for i, (m, P) in enumerate(zip(r['meas_hist'], ph)):
            m = np.array(m); P = np.array(P)
            print(' survey %d: rms(meas - twin) per twin (mm):' % (i + 1), np.round(1000 * np.sqrt(np.mean((P - m) ** 2, axis=1)), 1))
