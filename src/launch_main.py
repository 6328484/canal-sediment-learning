"""Launch the planner-dependent experiments with the design stored in ../results/design.json (candidate set, fixed shifts, weight)."""
import json, os, sys, subprocess
D = json.load(open('../results/design.json'))
env = dict(os.environ, CANDS=json.dumps(D['cands']), FIXED=json.dumps(D['fixed']), BEST=json.dumps(D['best']), LAM=str(D['lam']), RESULTS='../results/main.json')
print('design:', {k: D[k] for k in ('cands', 'fixed', 'best', 'lam')}, flush=True)
sys.exit(subprocess.call([sys.executable, 'run_all.py'] + (sys.argv[1:] or ['main', 'gridcl', 'struct', 'sens']), env=env))
