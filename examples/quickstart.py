"""Quickstart: a 10-day run of the pumped-canal simulation with the local controller (FFP).

Run from the repository root:   python3 examples/quickstart.py
Takes a few seconds. It builds the synthetic Amu-Zang-type plant, runs 10 days and prints
delivery, energy and the capacity-loss indicators. It is a smoke test of the installation,
not an experiment of the paper."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('DAYS', '10')
import run_all

res = run_all.one(dict(ctl='FFP', seed=1, mult=1.0, tag='quickstart'))
if 'error' in res:
    print(res['tb']); sys.exit(1)
for k in ('short', 'spill', 'energy', 'pumped', 'dep_kt', 'L5', 'Lsys', 'dz_max', 'secs'):
    print('%-8s %10.5g' % (k, res[k]))
print('OK')
