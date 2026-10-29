# Fault tolerant control of a redundant cable driven parallel robot

Code, configurations and stored results of the paper

> M. J. Tachia, A. V. Maloletov, *Fault tolerant control of a redundant cable driven parallel robot with model based fault isolation and precomputed recovery plans* (submitted to the Chinese Journal of Mechanical Engineering).

The robot is IPAnema 2 (Pott et al. 2013), a fully constrained cable driven parallel robot with eight cables and a platform with six degrees of freedom, with the geometry, mass and inertia of its CASPR model. The simulator has elastic, unilateral cables, geared winches with drive limits and the sensors of the real robot: motor encoders and motor currents, without load cells. Three faults are injected: a cable break, a loss of actuator effectiveness and a torque bias. The controller combines extended state observers at the winches, a 17 mode interacting multiple model (IMM) estimator that isolates the fault, a model predictive controller (MPC) with the tension bounds as constraints, and recovery plans computed in advance along the task. The evaluation compares it with ablations, a threshold baseline and a learned (LSTM) fault classifier on the same randomised runs. All results are from simulation.

Every stored result is written by a script in `scripts/`, kept in `results/`, and can be rerun and compared with the stored file by `reproduce.py` (`REPRODUCE.md`).

## Quick start

Linux, or Windows through WSL2. Python 3.11 or newer and gcc.

```bash
git clone https://github.com/Tachia/cdpr-fault-tolerant.git
cd cdpr-fault-tolerant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-learning.txt   # only for the learned classifier (torch, optuna)

python reproduce.py check     # Python, packages, C compiler, cores
python reproduce.py tests     # the verification tests (about 17 min the first time, with the compilation)
python reproduce.py quick     # the frozen controller on 8 calibration breaks, compared run by run (2.5 min)
python reproduce.py phase6    # the figures and tables of the paper from the stored results
python reproduce.py phase7    # every checked number of the paper against the stored results
```

The first command that needs the estimators or the MPC compiles them to C and caches them in `build/` (about 15 minutes on two cores, once). `python reproduce.py list` shows every step with its run time.

## How the results were produced

| Step | Content | Documentation |
|---|---|---|
| 1 | Model (Lagrangian derivation), workspace, formulation of estimators, MPC and planner | `docs/phase1/methods.pdf` |
| 2 | Compiled simulator and its verification | `docs/phase2/simulator.pdf` |
| 3 | Estimators, controller and recovery library; settings fixed on calibration runs and frozen | `docs/phase3/controller.pdf`, `configs/controller_phase3.yaml` |
| 4 | Learned fault classifier: data from the frozen controller, search, training, thresholds on validation runs, one test | `docs/phase4/classifier.pdf`, `configs/classifier_phase4*.yaml` |
| 5 | Evaluation: nine methods on the same randomised runs, protocol fixed before the runs | `docs/phase5/evaluation.pdf`, `configs/evaluation_phase5.yaml`, `results/phase5/` |
| 6, 7 | Figures, tables and the manuscript, drawn and checked from the stored results | `paper/` |

Settings were chosen on calibration runs (seeds from 100000) and the classifier was trained on its own runs (seeds from 300000); the evaluation uses seeds from 200000 and shares no parameter draw, noise sequence or fault onset with them. In the paper the ablation with the planner solved online is called A3; in the evaluation files it is A5.

## Hardware

| | |
|---|---|
| Processor | CPU only. The simulator, the estimators and the MPC run on NumPy, SciPy and CasADi with IPOPT, with the models compiled to C. A GPU speeds up none of the steps of Phases 1 to 3; on a GPU cloud instance only its CPU cores are used. The networks of Phase 4 were also trained on the CPU, with two threads and deterministic algorithms, so that a rerun gives the same weights; training them on a GPU is faster but gives slightly different weights. |
| Cores | Closed-loop runs are spread over worker processes (`--jobs N`). One 16 s run of the full controller costs 60 to 80 s of one core. Two cores are enough for the reviewer check; the full rerun of Phase 3 took about 4.5 h on two cores and that of Phase 4 about 3.4 h (1.3 h from the stored dataset, without simulating its runs again), and the evaluation campaign of Phase 5 (5,212 runs) took 32 h on two cores, about 64 core hours; on 8 cores it would take about 8 h. |
| Memory | Under 1 GB per worker process. |
| Disk | About 190 MB for the repository (about 110 MB of it the kept traces of the Phase 5 campaign, 52 MB the classifier datasets), 200 MB for the compiled solvers, 100 MB for each calibration set whose records are regenerated and 0.6 GB for the records of the Phase 4 runs. The CUDA build of torch that pip installs by default takes several GB; the CPU build is enough (`requirements-learning.txt`). |
| System | Tested on Linux x86_64 (Ubuntu 24.04, Python 3.13.17, gcc 13.3). On Windows use WSL2: the runs use the `fork` start method and the solvers are compiled with gcc. macOS is not tested. |

## Layout

| Path | Content |
|---|---|
| `reproduce.py` | Runs the tests, redraws the figures, and reruns any step and compares its output with the stored reference |
| `requirements.txt` | Library versions with which every stored result was produced |
| `requirements-learning.txt` | The two further libraries of Phase 4 (torch, optuna) |
| `configs/robot_ipanema2_caspr.yaml` | Robot geometry, inertia, limits, cable (1.5 mm Dyneema) and winch (drum radius 50 mm, gearbox 12, Rexroth MSK030C), each value with its source |
| `configs/controller_phase3.yaml` | Frozen controller configuration, with the source of every group of values |
| `configs/learning_phase4.yaml` | Protocol of the learned classifier, written before its first run (two amendments recorded in it) |
| `configs/classifier_phase4.yaml` | Frozen classifier: hyperparameters, the five models with their hashes, the deployed model and its thresholds |
| `configs/learning_phase4b.yaml`, `configs/classifier_phase4b.yaml` | Protocol and frozen file of the classifier trained on both paths (baseline B3 of Phase 5) |
| `configs/evaluation_phase5.yaml`, `configs/b2_phase5.yaml` | Protocol of the evaluation campaign (methods, runs, metrics, analysis plan); settings of baseline B2 |
| `cdpr_ft/model/` | Kinematics and wrench matrix; platform dynamics (Newton-Euler) and the symbolic Lagrangian derivation (SymPy); reference plant (NumPy) and compiled plant (CasADi); rigid-cable prediction model for the estimators and the MPC; C compilation of CasADi functions |
| `cdpr_ft/sim/` | Sensors (encoder, motor current), initial equilibrium, multirate closed loop (1 kHz, one-sample delay), one run with randomised plant, fault and termination rule, parallel batches |
| `cdpr_ft/estimation/` | IMM over 17 modes with 28 states per filter, log-domain probabilities, arming and innovation gate; offline replay of a recorded run |
| `cdpr_ft/control/` | Winch observers (ESO) and tension loops; probability-aware scenario MPC; post-fault planner and recovery problem (Hermite-Simpson collocation); tension distribution; the fault tolerant controller with its baseline and ablation variants; configuration loading |
| `cdpr_ft/analysis/` | Reference trajectories T1 (circle) and T2 (helix); static wrench feasibility, drum windows, workspace grid |
| `cdpr_ft/learning/` | Classifier inputs computed online on the IMM's time line, the protocol and its run list, data windows, the networks (causal CNN, LSTM), training, and the declaration rule shared with the controller |
| `cdpr_ft/estimation/alternatives.py` | The diagnoses that replace the IMM's probabilities in the baselines: the trained classifier (B3) and the threshold rule (B2) |
| `cdpr_ft/eval/` | The evaluation campaign: methods, run list, one run with its metrics and its trace |
| `cdpr_ft/provenance.py` | Code digest and library versions stamped into every result |
| `scripts/` | One script per analysis, calibration, training or evaluation step (`phase1_*` to `phase6_*`); `paper_figures.py`, `paper_tables.py`, `paper_check_numbers.py` and `paper_package.py` (manuscript); `compare_results.py` |
| `tests/` | Verification gates: Jacobian against finite differences, mass and gravity relations, Lagrange against Newton-Euler, skew symmetry, equilibrium, energy, step halving, faults, geared winch, sourced parameters, drive requirements, drum windows, compiled plant and prediction model, sensors, loop timing, observer poles, IMM probabilities, time scaling, recovery plan, MPC, replay of a run; classifier inputs against the diagnosis, declaration rule, networks, deterministic training |
| `results/` | Stored outputs of every script; `results/README.md` lists them with the script and code version that produced each |
| `figures/verification/`, `figures/phase4/` | Verification and calibration figures, and the figures of the classifier, drawn from `results/` (PDF and PNG); `figures/README.md` |
| `paper/` | The manuscript (`paper/manuscript/`, LaTeX source, bibliography and PDF), its figures (`paper/figures/`, with a manifest of the files each one read) and its tables (`paper/tables/`); `paper/README.md` |
| `docs/phase1/` to `docs/phase5/` | Methods, simulator and verification, estimators and controller with their calibration, learned classifier, baselines and evaluation (LaTeX sources and PDF) |
| `data/pott2013_ipanema1_stiffness.json` | Compliance data of IPAnema 1 (Pott et al. 2013, Tables 1 and 3), transcribed; not used for the cable stiffness because the measured compliance includes drives and frame |

## Provenance

Every result file of Phase 2 and later carries the code digest (a SHA-256 prefix of the package sources and the robot configuration), the library versions and, from Phase 3 on, the hash of the controller configuration. The frozen state of Phase 3 is:

| Item | Hash |
|---|---|
| Code digest | `fe7722d2d710c537` |
| `configs/controller_phase3.yaml` | `71c8e8640f29365f` |
| `results/phase3_recovery_library.npz` | `a8749c838681c808` |
| `results/phase3_safe_poses.npz` | `994fb7d2aaa623a3` |

Phase 4 added the package `cdpr_ft/learning` and changed no file of the controller: the digest of the code without that package is still `fe7722d2d710c537` (`cdpr_ft.learning.protocol.controller_digest`). The frozen state of Phase 4 is:

| Item | Hash |
|---|---|
| `configs/learning_phase4.yaml` (protocol) | `1167c90253d950f9` |
| `configs/classifier_phase4.yaml` | `ed16c7014aa26fcc` |
| `results/phase4_data/dataset.npz` | `00cb48553bc96097` |
| Deployed model `results/phase4_models/seed2.pt` | `e2b4d6691e630b0f` |
| Digest of `cdpr_ft/learning` | `8aa62569041d0e4b` |

Phase 5 added the baselines B2 and B3 to the controller code (variants in `cdpr_ft/control/ftc.py`, `cdpr_ft/estimation/alternatives.py`, `cdpr_ft/eval/`) without changing the variants used before: the quick check of the proposed controller and two Phase 4 runs simulated again with the new code are identical to the stored ones. The frozen state of the campaign (`results/phase5_freeze.json`, after amendment 1) is:

| Item | Hash |
|---|---|
| `configs/evaluation_phase5.yaml` (protocol) | `10331039ba40774c` |
| `configs/b2_phase5.yaml` | `aee6e9a39eb377fa` |
| `configs/classifier_phase4b.yaml` | `c1a51875fc7b0e23` |
| Deployed B3 model `results/phase4b_models/seed2.pt` | `813b05ed8607c087` |
| Controller code with its configuration files (`controller_digest`) | `2e03873280e4d102` |
| Condensed rows `results/phase5/rows.jsonl.gz` | `82282551f395f21f` |

`.gitattributes` keeps LF line endings on every platform, because the hashes are taken on the file bytes.

## Citation

See `CITATION.cff`.

## Licence

The code is released under the MIT licence (`LICENSE`). The documents, figures and stored results are released under the Creative Commons Attribution 4.0 International licence (`LICENSE-docs`).
