# Supplemental data S1. Parameter provenance

Status codes: **S** = taken from a published or official source; **D** = derived from sourced values; **A** = assumption (no data available to the authors); **N** = numerical choice; **P** = policy or design choice of the study.
Everything marked **A** must be replaced by measured data before any quantitative statement about the real Amu-Zang canal is made. The study uses a synthetic plant of Amu-Zang type; its results are not a forecast of the real canal.

## 1. Canal, hydraulics and numerics (code: `canal.py`, `gvf.py`)

| Symbol | Meaning | Value | Status | Basis / comment |
|:--|:--|:--|:-:|:--|
| L | Canal length | 56.6 km | S | Samiev et al. (2024), E3S Web Conf. 508, 07006 |
| b, B, h_ref | Bottom width, top width, depth | 17 m, 34 m, 5 m | S | same |
| m | Side slope (horizontal : vertical) | 1.7 | D | (B - b) / (2 h_ref) |
| S_0 | Bed slope | 6e-5 | S | same |
| n_0 | Manning roughness, clean bed | 0.030 s m^-1/3 | D | uniform flow of 75 m3/s, depth 5 m, slope 6e-5 gives v = 0.59 m/s, the reported v of about 0.6 m/s |
| Q_design | Design discharge of the canal | 120 m3/s | S | same; the steady capacity of the model canal at the freeboard of 0.5 m is Q_s(0) = 124.0 m3/s |
| h_bank | Bank height above the original bed | 7 m | A | not published |
| e_f | Freeboard used in the capacity indicators | 0.5 m | A | indicator definition, not a measured value |
| Checks | Three gated check structures (four pools) | 14.2, 28.3, 42.5 km | A | equal pool lengths assumed; positions not published |
| B_g, B_t | Gate width, tail-gate width | 30 m, 10 m | A | not published |
| C_d,plant, C_d,model | Gate discharge coefficients | 0.676, 0.65 | A | controllers use a coefficient 4 % below the plant to create a realistic model error |
| Offtakes | Positions (km) | 10, 22, 35, 44, 50, tail | A | not published; the large offtake at 35 km is a structural assumption of the study |
| sigma_j | Offtake shares | 0.04, 0.05, 0.50, 0.12, 0.24, 0.05 | A | not published |
| H_min | Head over the sill for full offtake discharge | 3 m | A | not published |
| kappa_s | Seepage rate | 3e-6 m/s (about 6 m3/s in total) | A | order of magnitude only |
| N, dx | Number of reaches, reach length | 114, 497 m | N | grid convergence: Section 2.6 of the paper |
| dt, theta | Time step, Preissmann weighting | 1800 s, 0.65 | N | time-step independence: Section 2.6 |

## 2. Sediment (code: `sediment.py`)

| Symbol | Meaning | Value | Status | Basis / comment |
|:--|:--|:--|:-:|:--|
| d | Grain size of the bed-material fraction | 0.05 mm (0.02 and 0.12 mm in E2) | A | Samiev et al. (2024) used 0.05 mm in their formula comparison; no measurement of the bed material |
| f_b | Bed-material fraction of the turbidity | 0.12 | A | no measurement |
| tau | Fraction trapped in the approach basin | 0.27 | D | Samiev et al. (2024): concentration falls from 1.470 to 1.067 kg/m3 |
| p, rho_s | Porosity, sediment density | 0.4, 2650 kg/m3 | A | textbook values |
| kappa_n | Roughness growth with aggradation | 0.5 | A | no data |
| mu | Capacity multiplier on Engelund-Hansen | true values 0.4, 1, 2.5 | P | four formulae differ by a factor of 3 to 9 on this canal (Samiev et al., 2024), which motivates treating the multiplier as the unknown; the three values are a design choice |
| W | Window of the reach-scale indicator | 5 km | P | |

## 3. Pump station (code: `pump.py`)

| Symbol | Meaning | Value | Status | Basis / comment |
|:--|:--|:--|:-:|:--|
| H_total | Total pump head at design flow | 77 m (33 + 44) | S | Asian Development Bank project completion report for the pumping stations |
| k_f | Friction head coefficient | 6 m at 100 m3/s | A | |
| eta_0, Q_bep, k_q | Best efficiency, best-efficiency discharge, droop | 0.88, 90 m3/s, 0.25 | A | efficiency between the specified 86 % and the measured 92 % reported in the ADB documents |
| Q_max | Maximum pump discharge | 105 m3/s | A | |
| Price | Electricity price | 0.03 USD/kWh | S | economic price in ADB documents (2003); the present tariff is unknown |
| k_w | Wear coefficient | 0 | - | not calibrated; no wear result is reported |

## 4. Demand, turbidity, measurements (code: `sim.py`)

| Item | Value | Status | Basis / comment |
|:--|:--|:-:|:--|
| Monthly pumping profile (Jan..Dec) | 6, 12, 38, 50, 52, 57, 77, 72, 53, 43, 30, 10 m3/s | S | read from a figure in Ikramova et al. (2019); reading uncertainty about +-3 m3/s |
| Daily schedule | steps at 06:00 | A | |
| Demand anomaly | AR(1), coefficient 0.7, standard deviation of the innovation 0.04 | A | |
| Offtake noise | 4 % | A | |
| Unscheduled changes | probability 0.15 per offtake and day, factor 1 + 0.2(2u - 1) from a random hour | A | |
| Turbidity | seasonal bump (peak at day 180, width 45 d) times AR(1) lognormal noise (0.85, 0.25), scale 1 | A | synthetic; the level is of the order reported by Samiev et al. (2024) |
| Level measurement noise | 0.02 m | A | |
| Turbidity measurement noise | 10 % lognormal | A | |
| Season | 245 days from day 60, 72 h warm start | P | |

## 5. Controllers, learning and planner (code: `controllers.py`, `mpc.py`, `morph.py`)

| Symbol | Meaning | Value | Status |
|:--|:--|:--|:-:|
| tau_c, T_I | Local controller: pool time constant, tail integrator | 6 h, 12 h | P |
| N_c, dt_c | Predictive horizon, sampling | 24, 3600 s | P |
| w_y, w_b, band | Level tracking, band penalty, dead band | 5000, 2e5, 0.12 m | P |
| w_t, w_u, w_e | Tail, move, energy weights | 30, 4, 1 | P |
| L_o | Observer gain | 0.25 | P |
| K, mu_k | Number of twins, hypothesis grid | 5; 0.35, 0.6, 1, 1.7, 3 | P |
| dt_twin | Twin time step | 3600 s | N |
| Survey | Stations, interval, noise | 14 at x = (2.5 + 4i) L/57, 30 d, 0.02 m | P |
| omega, gamma, sigma_l | Forgetting, tempering, likelihood scale | 0.9, 4, 0.04 m | P |
| Planner | Decision times, horizon | start of season and after each survey, rest of the season (bank of MPC flows + sediment rollouts of the MAP twin) | P |
| psi_k, sigma_mu, margin | Prior weights (Gaussian in ln mu), prior width, switching margin | 0.8, 2e4 USD | P |
| Candidate shifts | (a, a, b, 0) m, a, b in {0, 0.3, 0.6} | P |
| c_sh, c_sp, c_dep | Shortage, spill, deposit-volume weights | 0.10, 0.02, 3.0 USD/m3 | P |
| lambda | Weight of the reach-scale capacity loss | 2e7 USD per unit L_5 | P |

The weights are policy parameters, not estimated quantities. The value of lambda was set so that one percentage point of L_5 corresponds to the value of 0.2 percentage points of the seasonal deliveries at the shortage weight; Section 5 of the paper maps the response to lambda.

## 6. Experiments

| Experiment | Content |
|:--|:--|
| Design seed | seed 0 only: open-loop landscape of constant shifts, candidate set, fixed benchmarks, planner check |
| E1 | seeds 1-5, true multipliers 0.4, 1, 2.5, seven controllers |
| E2 | seeds 1-3, structural errors in grain size and Shields exponent |
| E3 | seeds 1-3, sensitivity to lambda, survey interval, survey noise, likelihood parameters, turbidity scale |
| E4 | seed 1, closed-loop check on 57 and 228 reaches |
