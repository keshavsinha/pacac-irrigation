# PAC-AC: Pest-Aware Constrained Actor–Critic for AIoT Precision Irrigation

Code, trained models and results for the revised manuscript

> **Pest-Aware Constrained Multi-Objective Optimization Using Deep Reinforcement Learning for AIoT-Enabled Precision Irrigation**
> K. Sinha, S. Choudhury, W. El-Shafai, A. T. Azar, M. Tounsi. *Discover Applied Sciences* (under revision).

This repository contains everything needed to reproduce every number, table and figure in the paper:

- the fully specified simulation testbed
- the PAC-AC learner and all comparators (threshold rules, MPC, TD3, TD3-Lag)
- the ablation, sensitivity and robustness experiments
- the statistical analysis

All results are from **simulation**. The testbed is synthetic and is **not calibrated** to a specific field.

---

## 1. What the code does

Irrigation is formulated as a **finite-horizon constrained Markov decision process (CMDP)**:

- **Control:** every 30 min over a 14-day period (672 steps).
- **Action:** irrigation depth `u_t` in [0, 0.9] mm per step.
- **Reward:** crop soil-moisture proxy − water − grid energy − disease-favourability index.
- **Constraints (undiscounted, per 14 days):** water ≤ 38 mm, grid energy ≤ 6 kWh, cumulative soil-limit violation ≤ 0.15 VWC·step.
- **Safety supervisor:** identical for every controller. It applies actuator bounds, a saturation guard and a deficit guard.

### Simulator (`sim.py`)

| Component | Model |
|---|---|
| Soil water | Lumped FAO-56 root-zone balance, water-stress coefficient Ks, drainage above field capacity, runoff (sandy loam, Zr = 400 mm) |
| Crop ET | Hargreaves–Samani ET0 × Kc (tomato, mid-season, 1.15) × Ks |
| Weather | Markov-chain / Gamma rainfall, diurnal temperature, humidity and wind; noisy 12-h rain forecast |
| Canopy | Canopy humidity with separate background, soil, sprinkler and rainfall terms; leaf-wetness dynamics |
| Disease index | Generic foliar-fungal infection favourability, using the Magarey et al. (2005) temperature response, leaf wetness and Poisson evidence. It is an **index, not a probability** |
| Vegetation | NDVI driven by water stress and disease index |
| Energy | Pump energy (30 m head, η = 0.5, 0.5 ha) minus 1.5 kWp PV gives **grid energy**, which depends on irrigation timing |
| Sensing | Optional noise, bias and dropout, with a Kalman filter on soil moisture |

All exogenous randomness is pre-generated per scenario and is independent of the controller's actions, so every controller sees exactly the same disturbances (paired evaluation).

### Controllers

| Name | File / config | Description |
|---|---|---|
| `thr_classic` | `controllers.Threshold` | Hysteresis 0.20/0.30, 0.9 mm per step |
| `thr_tuned` | `tune_baselines.py` | Hysteresis tuned on development scenarios (120 combinations) |
| `mpc` | `controllers.MPC` | 12-h receding horizon, cross-entropy optimization, true model structure |
| `td3` | `configs.py` | Unconstrained TD3, same reward |
| `td3lag` | `configs.py` | Lagrangian TD3 with discounted cost critics and a replay-minibatch dual update (the original formulation) |
| `pacac` | `configs.py` | **PAC-AC:** time/budget-augmented state, disease-aware reward, Lagrangian-penalized reward, multipliers from Monte-Carlo **undiscounted** episode constraint returns |
| `pacac_*`, `sens_*` | `configs.py` | Ablations and sensitivity variants (see the paper, Tables 9–10) |

Learned actors decide every 2 h (action repeat 4); the supervisor acts every 30 min.

---

## 2. Repository structure

```
sim.py              Simulator, reward, supervisor, sensing/estimator, observation builder
controllers.py      Threshold, MPC and learned-policy wrappers
agents.py           Vectorised TD3 / TD3-Lag / PAC-AC trainer (PyTorch)
configs.py          Named training configurations
train_run.py        Train one configuration:  python train_run.py <config> <seed>
run_queue.sh        Run a list of training jobs, 2 in parallel
tune_baselines.py   Tune threshold rule and MPC on development scenarios
evaluate.py         Evaluate a method on the 20 test scenarios under any robustness condition
layered.py          4-layer cascade soil model (structural-mismatch test)
eval_layered.py     Evaluate controllers on the layered model with probe depths 0-40 cm
validate.py         Simulator verification (mass balance, irrigation-pulse response, weather statistics)
deploy.py           Actor size, MACs, int8 quantisation error, inference latency
analysis.py         Statistics (paired t, CI, Holm, Cohen's dz, Wilcoxon) and plotting helpers
make_results.py     Builds all tables (results/tables) and figures (figs/) and results/numbers.json
figs_arch.py        Redraws architecture figures (Fig. 1, Fig. 2)
eval_all.sh, final_eval.sh   Evaluation pipelines

results/
  models/           Trained actors (*_actor.npz) and training histories (*_hist.json) for every run
  eval/             Per-scenario, per-seed, per-condition metrics for every method (CSV)
  eval_layered.csv  Layered-soil results
  tables/           LaTeX tables and the full paired-test table (tab_paired.csv)
  numbers.json      Every number quoted in the paper
  tune_threshold.json, tune_mpc.json   Baseline tuning grids and selections
  validation.json   Simulator verification results
  deploy.json       Deployment-design estimates
figs/               Figures used in the paper
```

---

## 3. Installation

Tested with Python 3.13, PyTorch 2.15 (CPU), NumPy 2.5, SciPy, pandas and matplotlib.

```bash
git clone https://github.com/keshavsinha/pacac-irrigation.git
cd pacac-irrigation
pip install torch numpy scipy pandas matplotlib
```

No GPU is needed; each PAC-AC training run takes about 8 minutes on one CPU core.

> **Note:** `run_queue.sh`, `eval_all.sh` and `final_eval.sh` start with `cd /home/claude/pacac` (the original working directory).
> Replace that line with `cd "$(dirname "$0")"`, or delete it and run the scripts from the repository root.

---

## 4. Reproducing the paper

The trained models and all evaluation outputs are already included, so you can jump straight to step 5.

**Step 1: verify the simulator**
```bash
python validate.py          # writes results/validation.json
```

**Step 2: tune the baselines** on development scenarios 5000–5009
```bash
python tune_baselines.py threshold
python tune_baselines.py mpc
```

**Step 3: train the learned controllers**
```bash
python train_run.py pacac 0          # one run (config, seed)
```
The full set used in the paper is 32 runs. Put the following lines into `jobs_all.txt` and run `./run_queue.sh jobs_all.txt`:
```
pacac 0 … pacac 4         td3 0 … td3 4         td3lag 0 … td3lag 4
pacac_nopstate 0,1   pacac_noprew 0,1   pacac_nopest 0,1
pacac_noaug 0,1      pacac_cgrad 0,1    pacac_penalty 0,1
sens_a4x2 0   sens_a4x0.5 0   sens_Wx0.85 0   sens_Wx1.15 0
```
(One `name seed` pair per line.) Existing models are skipped automatically.

**Step 4: evaluate** on the 20 held-out test scenarios (seeds 1–20)
```bash
python evaluate.py pacac nominal,noisy_kf,couple0     # any comma-separated list of conditions
python eval_layered.py
python deploy.py
```
Available conditions (see `CONDITIONS` in `evaluate.py`): `nominal`, `noisy_kf`, `noisy_raw`, `bias+0.02`, `bias-0.02`, `dropout30`, `pest_misspec+3C`, `couple0`, `couple0.5`, `couple2`, `soil_Zr300`, `soil_clayloam`, `rain_x0.5`, `rain_x1.5`, `hot+2C`, `Topt+3C_true`.

**Step 5: build all tables, figures and statistics**
```bash
python make_results.py
```

### Scenario seeds (disjoint pools)

| Pool | Seeds |
|---|---|
| Training | 1,000,000 + 10,000·seed + 8·round + k |
| Dual-update rollouts | 2,000,000 + 10,000·seed + 4·round + k |
| Development (tuning and checkpoint selection) | 5000–5009 |
| **Test (never used for training, tuning or selection)** | **1–20** |

---

## 5. Main results (20 test scenarios; learned methods averaged over 5 training seeds)

| Controller | Water (mm/14 d) | Grid energy (kWh) | Mean disease index | Return |
|---|---|---|---|---|
| Threshold (classical) | 61.5 ± 28.9 | 35.8 | 0.250 | 370 |
| Threshold (tuned) | 39.2 ± 18.6 | 5.8 | 0.208 | 386 |
| MPC | 42.6 ± 18.4 | 0.2 | 0.188 | 436 |
| TD3 | 53.1 ± 21.2 | 6.0 | 0.242 | 440 |
| TD3-Lag | 22.4 ± 13.1 | 1.2 | 0.221 | 304 |
| **PAC-AC** | **44.4 ± 17.8** | **5.5** | **0.224** | **390** |

- PAC-AC uses **27.8% less water** than the classical threshold rule (paired difference −17.1 mm, 95% CI [−25.5, −8.7], p = 0.0004), with 85% less grid energy.
- It performs **on par with the tuned threshold rule** and **below MPC**.
- The disease term lowers the modelled index by shifting irrigation to midday, not by using less water.
- The discounted dual update of TD3-Lag never activates. The undiscounted Monte-Carlo estimator of PAC-AC activates only the binding water constraint.
- Expected-value budgets selected on the (wetter) development scenarios were exceeded in many of the drier test realizations.

See the paper for full statistics (Tables 5–12), learning curves and robustness results.

---

## 6. Limitations

- Synthetic, uncalibrated testbed. Crop performance is a soil-moisture/transpiration proxy, not yield.
- The disease index is a modelled favourability measure, not observed disease incidence.
- Each rollout covers 14 days, not a full season.
- The AIoT figures (model size, communication load) are design-level estimates. No hardware measurements were made.

---

## 7. Citation

If you use this code, please cite the paper (citation details will be updated after publication):

```bibtex
@article{sinha2026pacac,
  title   = {Pest-Aware Constrained Multi-Objective Optimization Using Deep Reinforcement Learning for AIoT-Enabled Precision Irrigation},
  author  = {Sinha, Keshav and Choudhury, Sushabhan and El-Shafai, Walid and Azar, Ahmad Taher and Tounsi, Mohamed},
  journal = {Discover Applied Sciences},
  year    = {2026},
  note    = {Under revision}
}
```

## 8. Contact

Dr. Keshav Sinha, School of Computer Science, UPES, Dehradun, India. keshav.sinha@ddn.upes.ac.in
