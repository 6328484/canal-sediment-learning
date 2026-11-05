# Code and results for "Learning sediment transport capacity online to limit siltation in pumped canals"

Authors: Z. A. Abdujabborov, A. J. Seytov, N. Q. Choriyorov, M. N. Esonturdiyev.
Corresponding author: Z. A. Abdujabborov (abdujabborov_za@nuu.uz).

This repository contains the simulation code, the raw result files, and the scripts that produce every number, table and figure of the paper.
All inputs are synthetic or taken from the sources cited in the paper (see `docs/parameters.md` for the provenance of every parameter).

## Requirements

Python 3.13 with numpy 2.5, scipy 1.18, pandas 3.0 and matplotlib 3.11 (the versions used; earlier versions are likely to work).
Computation: two CPU cores; the complete study takes about one day on two cores (see "Reproducing the study"; most of it is the 133 seasonal runs of 4 to 12 minutes each and the forecast banks).

## Layout

| Path | Content |
|---|---|
| `src/canal.py` | one-dimensional Saint-Venant solver (Preissmann scheme, Newton iteration with analytic Jacobian), trapezoidal canal, check structures, offtakes, tail gate |
| `src/sediment.py` | suspended-load transport with fitted exchange coefficient, Exner bed evolution, roughness feedback, capacity indicators |
| `src/gvf.py` | steady gradually varied flow, system capacity indicator |
| `src/pump.py` | pump station (head, efficiency, energy) |
| `src/sim.py` | plant, demand and turbidity scenarios, measurements, simulation loop |
| `src/controllers.py`, `src/model_id.py` | local controller, identification of pool storage areas |
| `src/mpc.py` | economic predictive controller (adjoint gradient, offset-free observer) |
| `src/morph.py` | ensemble of digital twins, Bayesian weights with forgetting and prior, seasonal planner (`MorphMPC`), the rejected weekly planner (`MorphWeekly`), fixed-shift controller |
| `src/forecast.py` | bank of precomputed flows (predictive controller, expected demand, clean bed) for the nine candidate shifts and sediment-only rollouts of a twin; banks are cached in `results/bank/` and rebuilt with `python3 forecast.py 114 57 228` (about 20 minutes per grid on two cores) |
| `src/run_all.py` | experiment runner (stages `base`, `main`, `struct`, `sens`, `gridcl`) |
| `src/landscape.py`, `src/design_check.py`, `src/check_bank.py`, `src/make_design.py`, `src/launch_main.py` | design of the planner on the design seed (landscape, forecast check, closed-loop check) and launch of the experiments |
| `src/analyze_main.py`, `src/analyze_landscape.py`, `src/make_tables.py` | statistics and tables |
| `src/mech.py` | mechanism diagnostics (profiles of depth, capacity, concentration) |
| `verification/` | code verification (steady relaxation, volume and sediment balances), grid and time-step convergence |
| `figs/` | figure scripts |
| `results/` | raw result files (`*.json`) and summary tables (`*.csv`) |
| `docs/parameters.md` | parameter provenance (source, derived, assumption) |

## Reproducing the study

```
cd src
python3 forecast.py 114 57 228      # banks of precomputed flows for the planner (cached, one-off)
python3 landscape.py                 # open-loop landscape of constant level shifts, design seed 0
python3 design_check.py              # planner check on the design seed
RESULTS=../results/base.json python3 run_all.py base      # local and predictive control (planner-independent jobs)
python3 make_design.py               # fixed-shift benchmarks from the landscape -> results/design.json
python3 launch_main.py               # main (E1), grid check (E4), structural errors (E2), sensitivity (E3)
python3 analyze_main.py              # statistics -> results/*.csv
python3 make_tables.py
cd ../figs && for f in fig_*.py; do python3 $f; done
```

All runs are resumable: results are appended to the json files keyed by configuration. A single seasonal run takes 0.5 to 12 minutes.

## Notes

* The plant is synthetic. Everything marked A in `docs/parameters.md` is an assumption, not a measurement.
* `results/design_check_weekly_v2.json` and `results/design_check_v1_twin_without_shift.json` are the records of two rejected earlier versions (weekly planner; twins that ignored the shifts) and are not used in the paper's tables.
* Random seeds: seed 0 is the design seed, seeds 1 to 5 are the evaluation seeds.
* The result files contain the shifts chosen by the planner after each survey (`plan_hist`), the survey measurements and the twin predictions (`meas_hist`, `pred_hist`), and the end-of-season bed profile (`dz_prof`) of every run.

## License

Code: MIT. Results and documentation: CC BY 4.0. (The authors may change this before the deposit.)
