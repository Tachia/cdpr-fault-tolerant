# Figures

Drawn by `scripts/make_figures.py` from the stored files in `results/`; `python reproduce.py figures` redraws them in seconds. Every figure is written as PDF (vector, for LaTeX) and PNG (300 dpi, for viewing). The figures of the paper will be drawn from the Phase 5 evaluation logs in Phase 6.

## verification/

Verification and calibration figures of Phases 1 to 3. None of them is a study result: calibration runs use seeds from 100000 and are kept apart from the evaluation.

| Figure | Content | Source in `results/` |
|---|---|---|
| `verification/p1_circle_radius` | Largest admissible circle radius against the period of one turn, with the limit that binds; the selected references T1 and T2 use 0.5 m and 4 s. | `phase1_trajectory_design.json` |
| `verification/p1_postfault_workspace` | Share of the nominal static workspace that the seven remaining cables still hold after the loss of one cable, for three lower tension bounds and two orientation sets. | `phase1_workspace.json` |
| `verification/p2_step_convergence` | Step halving of the plant integrator in closed loop: largest relative change of an evaluation metric between step h and h/2, for the seven Phase 2 scenarios. | `phase2_step_convergence.json` |
| `verification/p3_library_drop` | Recovery library: planned drop below the height at the break (recovery path, branch 2) against the break instant, for every cable on T1 and T2. | `phase3_recovery_library.json` |
| `verification/p3_library_branch` | Recovery library: planner branch chosen for every cable and break instant. | `phase3_recovery_library.json` |
| `verification/p3_eso_lag` | Winch observers on the fault-free runs of calibration set c2: RMS error of the tension estimate against the lag of the plant tension it is compared with. | `phase3_eso_check_c2.json` |
| `verification/p3_mpc_weights` | MPC weight study: RMS position error of each fault-free calibration run (first 8 s) for the four weight settings; open circles mark lost platforms. | `phase3_mpc_weights.json` |
| `verification/p3_threshold_grid` | Declaration thresholds on calibration set c3: median declaration delay for every pair (mu_iso, N_c), with the pairs that fail the selection rule marked. | `phase3_roc_c3.json` |
| `verification/p3_mode_probability` | Probability of the true fault mode after the fault in every fault run of calibration set c3 (recorded with declarations disabled, so without accommodation). | `phase3_calib/c3/runs.json.gz` |
| `verification/p3_fault_free_probability` | Largest fault-mode probability over time in the 16 fault-free runs of calibration set c3. | `phase3_calib/c3/runs.json.gz` |
| `verification/p3_final_check` | Frozen controller on the calibration set, declarations on: declaration delay and share of the post-fault time with a tension out of bounds, per run (40 fault runs). | `phase3_check_final.json` |

## phase4/

The learned fault classifier (baseline B3), Phase 4 runs (seeds from 300000). The search, learning-curve, size-curve and threshold figures use the training and validation runs only; the two figures marked as test results show the single evaluation on the test runs.

| Figure | Content | Source in `results/` |
|---|---|---|
| `phase4/p4_search` | Hyperparameter search of the classifier: best validation macro-F1 of every trial (pruned trials: last value before pruning), by architecture. | `phase4_search.json` |
| `phase4/p4_learning_curves` | Training and validation loss and validation macro-F1 against the epoch for the five seeds of the final configuration; the kept epoch of each seed is marked. | `phase4_training.json` |
| `phase4/p4_size_curve` | Validation macro-F1 against the number of training runs (seeds 0-2). | `phase4_training.json` |
| `phase4/p4_thresholds` | Declaration thresholds of the deployed classifier on the validation runs: median delay for every pair (mu_iso, N_c), the pairs that fail marked, the chosen pair boxed. | `phase4_thresholds.json` |
| `phase4/p4_test_confusion` | Test result: confusion matrices of the deployed classifier and of the IMM, per estimator cycle, on the T1 test runs and on the T2 runs. | `phase4_test.json` |
| `phase4/p4_test_delays` | Test result: delays of the correct declarations of the classifier and of the IMM, by fault type, on the T1 test runs and on the T2 runs. | `phase4_test.json` |
