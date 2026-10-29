# Stored results

Every file here was written by a script in `scripts/` and is the reference that `reproduce.py` compares a new run with. The JSON files of Phase 2 and later carry a provenance stamp: the code digest (SHA-256 prefix of the package sources and the robot configuration, `cdpr_ft/provenance.py`), the library versions and the time of writing; from Phase 3 on also the hash of the controller configuration. Phase 1 files predate the stamp.

The code changed during Phase 3, as the controller was built and calibrated, so the files carry different digests. The frozen code has digest `fe7722d2d710c537` and the frozen configuration `configs/controller_phase3.yaml` has hash `71c8e8640f29365f`. Which files the code of this repository reproduces when the scripts are run again, and how closely, is recorded in `REPRODUCE.md`, Section 6. Files marked "record" are kept as the record of a decision and cannot be rerun as they were (the reason is given).

None of these files is a study result. Development runs used seeds below 100, calibration runs seeds from 100000. The evaluation (Phase 5) will use seeds from 200000 and the learned classifier (Phase 4) seeds from 300000.

## Phase 1: formulation and workspace

| File | Content | Script |
|---|---|---|
| `phase1_drum_windows.json` | Installed length window of each cable (6 m drum stroke) | `phase1_drum_windows.py` |
| `phase1_cable_analysis.json` | Sag, elastic modes, stiffness sweep, motor load for t_min = 10/30/40 N | `phase1_cable_analysis.py` |
| `phase1_workspace.json` | Nominal and post-fault static workspace, two orientation sets, three t_min | `phase1_workspace.py` |
| `phase1_postfault_orientation.json`, `_tmin30`, `_tmin40` | Post-fault feasibility when the platform may tilt up to 20 deg | `phase1_postfault_orientation.py [30 or 40]` |
| `phase1_trajectory_design.json` | Admissible circle radius by period; references T1 and T2; post-fault coverage along them | `phase1_trajectory_design.py` |
| `phase1_verification.txt` | Numerical values of the verification appendix of the methods | `phase1_verification.py` |
| `cable.log`, `tilt10.log`, `tilt30.log`, `tilt40.log`, `trajectory.log`, `verification.log`, `workspace.log` | Console output of the scripts above | |

## Phase 2: simulator and verification (code digest `c77663b917040fde`)

| File | Content | Script |
|---|---|---|
| `phase2_verification.json` | Energy drift, skew symmetry, equilibrium, compiled against NumPy plant | `phase2_verification.py` |
| `phase2_step_convergence.json` | Step halving 1 ms to 31.25 us in seven closed-loop scenarios | `phase2_step_convergence.py` |
| `phase2_reference_solution.json` | RK4 plant against Radau IIA (rtol 1e-9) | `phase2_reference_solution.py` |
| `phase2_mujoco_crosscheck.json` | Independent cross-check in MuJoCo, taut and slack cables | `phase2_mujoco_crosscheck.py` |
| `phase2_damping_onset_effect.json` | Peak tension after a break with and without the damping onset | `phase2_damping_onset_effect.py` |
| `phase2_timing.json` | Computation times on the machine used | `phase2_timing.py` |
| `phase2_convergence.log`, `phase2_mujoco.log`, `phase2_reference.log`, `phase2_verification.log` | Console output | |

## Phase 3: estimators, controller, calibration

| File | Content | Script | Code digest |
|---|---|---|---|
| `phase3_safe_poses.json`, `.npz` | Poses held with a 10 N margin after each cable loss | `phase3_safe_poses.py` | `807f4e7853808503` |
| `phase3_recovery_library.json`, `.npz` | Planner along T1 and T2 for every cable and break instant (1776 plans); the `.npz` is the library the controller loads | `phase3_recovery_library.py` | `4b9f4ab5b239495b` |
| `phase3_postfault_modes.json` | Elastic modes at the end poses of the recovery plans | `phase3_postfault_modes.py` | `7c6995d399ff5f48` |
| `phase3_calib/c1/runs.json.gz` | Calibration set 1 (56 runs, declarations disabled): first MPC weights | `phase3_calibration_runs.py c1` | `ee266f291185d0f7` |
| `phase3_mpc_weights.json` | MPC weight study on the 16 fault-free calibration runs | `phase3_mpc_weights.py` | `2b833249d3db311e` |
| `phase3_calib/c2/runs.json.gz` | Calibration set 2: chosen MPC weights, first estimator setting | `phase3_calibration_runs.py c2` | `0fe72283f1854aa8` |
| `phase3_replay_c2.json` | Replay of the diagnosis on the records of set 2, one-at-a-time variations | `phase3_replay.py c2` | `0fe72283f1854aa8` |
| `phase3_replay_stage2_settings.json` | Settings of the second replay round (input, written before that round) | | |
| `phase3_replay_c2_stage2_settings.json` | Second replay round | `phase3_replay.py c2 2 results/phase3_replay_stage2_settings.json` | `0fe72283f1854aa8` |
| `phase3_eso_check_c2.json` | Accuracy and lag of the winch tension estimates | `phase3_eso_check.py c2` | `292597f515559964` |
| `phase3_roc_c2.json` | Declaration thresholds on set 2 (superseded by set 3) | `phase3_roc.py c2` | `292597f515559964` |
| `phase3_calib/c3/runs.json.gz` | Calibration set 3: final estimator setting | `phase3_calibration_runs.py c3` | `292597f515559964` |
| `phase3_roc_c3.json` | Declaration thresholds on set 3 (the chosen ones) | `phase3_roc.py c3` | `292597f515559964` |
| `phase3_check_post_break_scaled.json`, `_none.json` | Post-break weight scaling against none, on the 8 calibration breaks. Record: run with the configuration file as it was before the thresholds were fixed (hash `ad805665ba8907c8`) | `phase3_closed_loop_check.py` | `a2d8362340df584d` |
| `phase3_check_final_run1.json` | Frozen controller on the calibration set, first run. Record: agrees with the next file in every value except the hash of the configuration file, whose header changed between the two runs | `phase3_closed_loop_check.py` | `d12e13462012e503` |
| `phase3_check_final.json` | Frozen controller on the calibration set (56 runs, declarations on) | `phase3_closed_loop_check.py final controller_phase3.yaml 2 all` | `fe7722d2d710c537` |
| `phase3_*.log` | Console output | | |

`phase3_calib/<set>/runs.json.gz` holds the specification, the metrics and the mode probabilities of every run. The records of the readings and commands (`run*.npz`, about 100 MB per set) are not kept; `phase3_calibration_runs.py` writes them again from the seeds. They are needed only by `phase3_replay.py` and `phase3_eso_check.py`.

## Phase 4: learned fault classifier (baseline B3)

Protocol: `configs/learning_phase4.yaml`; frozen classifier: `configs/classifier_phase4.yaml`. Seeds from 300000. The runs use the frozen controller with the variant "no_hedging" and declarations disabled. Every file carries the hash of the protocol (`1167c90253d950f9`). The code of the controller is that of the Phase 3 freeze: its digest, taken without `cdpr_ft/learning` (`cdpr_ft.learning.protocol.controller_digest`), is still `fe7722d2d710c537`. The general code digest of the stamp covers the new package as well and is `08704d34d393de8c` for the search, training and threshold files and `81f12fa8652acd71` for the test file; the only difference between the two is `configs/classifier_phase4.yaml`, which the freeze wrote between them (the digest includes the configuration files other than `controller_*`).

| File | Content | Script |
|---|---|---|
| `phase4_data/dataset.npz` | For every estimator cycle of the 528 runs: run index, time, the 24 classifier inputs (float16), true mode, and the most probable IMM mode with its probability | `phase4_generate.py`, then `phase4_dataset.py` |
| `phase4_data/runs.json` | The runs: split, trajectory, seed, noise factor, fault, position in the arrays, outcome | `phase4_dataset.py` |
| `phase4_search.json` | Hyperparameter search on the training and validation runs: every trial with its settings, state and validation history | `phase4_search.py` |
| `phase4_models/seed<s>.pt` | Final models, seeds 0-4 (PyTorch state, hyperparameters, input standardisation) | `phase4_train.py` |
| `phase4_training.json` | Training histories of the five seeds, the deployed seed, and the training-size curve | `phase4_train.py` |
| `phase4_thresholds.json` | Declaration thresholds of every seed on the validation runs, and the chosen pair | `phase4_thresholds.py` |
| `phase4_test.json` | The single evaluation on the T1 test runs and the T2 runs: window and run metrics of the five seeds, of the deployed model and of the IMM, bootstrap intervals | `phase4_test.py` |
| `phase4_*.log` | Console output | |

The runs themselves (`results/phase4_work/`: records of the readings and commands, per-run metrics and IMM probabilities, about 0.6 GB) are not kept; `phase4_generate.py` writes them again from the seeds.

## Phase 4b: the classifier trained on both paths

Protocol: `configs/learning_phase4b.yaml`; frozen classifier: `configs/classifier_phase4b.yaml` (baseline B3 of Phase 5). Seeds 340000 and up. The runs are the Phase 4 runs on the helix T2 (training and validation), generated with the controller code of the Phase 3 freeze (digest `fe7722d2d710c537`, recorded in `phase4b_data/runs.json`).

| File | Content | Script |
|---|---|---|
| `phase4b_data/dataset.npz`, `runs.json` | Inputs, true mode and IMM mode of every estimator cycle of the 352 new T2 runs; appended to the Phase 4 dataset by `scripts/phase4b_common.load_combined` | `phase4b_generate.py`, then `phase4b_dataset.py` |
| `phase4b_models/seed<s>.pt` | Five models with the Phase 4 hyperparameters, trained on both paths | `phase4b_train.py` |
| `phase4b_training.json` | Training histories | `phase4b_train.py` |
| `phase4b_thresholds.json` | Temperature (calibration), threshold table and chosen pair of every seed on the validation runs of both paths; the deployed seed | `phase4b_calibrate.py` |
| `phase4b_test.json` | The single offline test on the Phase 4 test runs (T1 and T2), with the IMM and the Phase 4 classifier on the same runs | `phase4b_test.py` |
| `phase4b_*.log` | Console output | |

## Phase 5: settings fixed before the campaign

Protocol: `configs/evaluation_phase5.yaml` (amendment 1 recorded in its header). Every B2 setting was chosen on calibration runs of B2's own controller (seeds 140000 and 150000 and up).

| File | Content | Script |
|---|---|---|
| `phase5_b2_gains.json` | Gains of B2's computed-torque law on 8 fault-free calibration runs (seeds 140000 and up), seven settings, and the chosen one (rule: no lost run, tension bounds kept, then the smallest error) | `phase5_b2_gains.py` |
| `phase5_b2_data.npz`, `phase5_b2_runs.json` | B2 calibration runs with B2's own controller and the chosen gains (seeds 150000 and up): inputs of the threshold rule per estimator cycle | `phase5_b2_runs.py` |
| `phase5_b2_thresholds.json` | Threshold table and chosen thresholds of B2 (also written to `configs/b2_phase5.yaml`) | `phase5_b2_calibrate.py` |
| `phase5_b2_postbreak.json` | B2 run as a whole on its 32 calibration breaks with five gains after a break declaration, and the chosen gains (also written to `configs/b2_phase5.yaml`) | `phase5_b2_postbreak.py` |
| `phase5_freeze.json` | Hashes of everything the campaign depends on; the runner and its workers refuse to run if any changed | `phase5_freeze.py` |
| `phase5_b2_v1/` | Record: the first B2 settings (gains 20 rad/s, chosen without weighing the tension bounds), their threshold study and the first freeze, replaced by amendment 1 | |
| `phase5_first_start/` | Record: the 75 rows of the first start of the campaign (stopped for amendment 1); they enter no result. The traces are not kept | `phase5_run.py` |

## Phase 5: the evaluation campaign

The campaign (`scripts/phase5_run.py`) writes every row and trace to `results/phase5_work/` (not kept: 1.7 GB). `scripts/phase5_condense.py` writes what is kept:

| File | Content | Script |
|---|---|---|
| `phase5/rows.jsonl.gz` | Every (method, run) row of the protocol, with the recovery time of every fault run added from its trace | `phase5_condense.py` |
| `phase5/traces/<method>/run<run>.npz` | The traces fixed by the protocol: repetition 0 of every path, fault type and cable, and fault-free repetitions 0 and 1, for every method that ran them (200 Hz: pose, active reference, errors, tensions and commands, winch estimates, IMM and classifier probabilities, deficit estimates, MPC solve times) | `phase5_condense.py` |
| `phase5/subset_rows.json` | The rows of the 28 jobs that `phase5_check.py` simulates again, as the campaign wrote them | `phase5_condense.py` |
| `phase5/condense.json` | Counts, bands of the recovery time, hashes | `phase5_condense.py` |
| `phase5/reproduce_check.log` | Console output of `reproduce.py phase5 --only campaign_subset,analysis` after the campaign (28 jobs simulated again; rows, traces and analysis identical) | `reproduce.py` |
| `phase5/analysis.json` | Statistics of the analysis plan: summaries by method, path, fault type, cable group and noise; the primary endpoints with Holm-adjusted p-values; the secondary comparisons | `phase5_analyse.py` |

Pilot runs (seeds 900000 and up) went to `results/phase5_pilot/` and are not results.

## Phase 6: for the manuscript figures

| File | Content | Script |
|---|---|---|
| `phase6_workspace_slices.npz`, `.json` | Tension margin and drum-window flag of the static wrench-feasible poses (zero orientation) on the planes z = 3 m and y = 0, with all cables and without cable 1 or cable 5 (figure 3) | `phase6_workspace_slices.py` |
| `phase6_workspace_slices.log` | Console output | |

