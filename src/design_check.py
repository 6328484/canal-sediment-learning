"""Design check on the design seed (seed 0): the planner with the true-capacity twin (oracle) and with the learning twin, N = 114,
full season, candidate set = grid of pool-1/2 shifts a and pool-3 shifts b, (a, a, b, 0), a, b in {0, 0.3, 0.6} m.
Results: ../results/design_check.json (compare with the open-loop landscape of the same seed)."""
import json, os, sys
GRID = [[a, a, b, 0] for a in (0, 0.3, 0.6) for b in (0, 0.3, 0.6)]
os.environ['CANDS'] = json.dumps(GRID)
os.environ['DIAG'] = '1'
os.environ.setdefault('RESULTS', '../results/design_check.json')
import run_all as ra
from multiprocessing import Pool

if __name__ == '__main__':
    F = '../results/design_check.json'
    out = json.load(open(F)) if os.path.exists(F) else []
    done = {ra.key(r['cfg']) for r in out}
    ctls = sys.argv[1:] or ['Morph-oracle', 'Morph-learn']
    J = [dict(ctl=c, seed=0, mult=m, tag='design') for c in ctls for m in (0.4, 1.0, 2.5)]
    J = [j for j in J if ra.key(j) not in done]
    print('to run', len(J), flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(ra.one, J):
            if 'error' in r:
                print('ERROR', r['cfg'], r['error'], flush=True); print(r['tb'], flush=True); continue
            out.append(r); json.dump(out, open(F, 'w'))
            ph = r.get('plan_hist', [])
            print(r['cfg']['ctl'], 'mu', r['cfg']['mult'], 'L5=%.4f Lsys=%.4f dep=%.1f short=%.4f E=%.1f' % (r['L5'], r['Lsys'], r['dep_kt'], r['short'], r['energy']), '%ds' % round(r['secs']),
                  '| shifts after each survey:', ' '.join(','.join('%g' % v for v in b[:3]) for b in ph), flush=True)
    print('DESIGN CHECK DONE', flush=True)
