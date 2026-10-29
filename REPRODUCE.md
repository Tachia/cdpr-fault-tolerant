# Reproducing the results

This file describes how to set up the environment, what each level of checking covers, how a step is compared with the stored reference, and which differences to expect on another machine. Section 6 records the check made on the reference machine from a fresh clone.

## 1. Environment

The code runs on the CPU. A GPU, if the machine has one, is not used: the stored networks of Phase 4 were also trained on the CPU, so that a rerun gives the same weights. What matters on a cloud machine is the number of CPU cores (see `README.md`, Hardware).

Phase 4 needs two more libraries, listed in `requirements-learning.txt`. The default torch wheel for Linux brings the CUDA libraries (several GB); the CPU build is enough and much smaller:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install optuna==5.0.0
```

The stored results were produced with the default PyPI wheel (`2.14.1+cu130`) running on the CPU, and the check of Section 6 used that wheel; the CPU build could not be downloaded on the reference machine, so whether it gives bit-identical weights has not been checked (Section 4). Without the two libraries everything else runs; `reproduce.py check` reports them as needed for Phase 4 only, and the four tests that need torch are skipped.

**A. Linux or WSL2 with a virtual environment (tested).**

```bash
sudo apt-get install -y python3-venv gcc git      # Debian or Ubuntu; gcc is needed for the solvers
git clone <repository URL> cdpr_ft && cd cdpr_ft
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py check
```

On Windows, install WSL2 with Ubuntu and run the same commands inside it, with the repository on the Linux file system (for example `~/cdpr_ft`, not `/mnt/c/...`, which is much slower).

**B. Any cloud machine.** A fresh Linux virtual machine of any provider is set up as in A. For the full rerun of Phase 3 choose 8 or more cores and pass `--jobs` with that number. A long rerun should be started so that it survives a closed connection, for example `nohup python reproduce.py phase3 --jobs 8 > phase3.log 2>&1 &`.

**C. Docker (the `Dockerfile` was not tested on the reference machine, which could not reach a container registry).**

```bash
docker build -t cdpr_ft .
docker run --rm -it -v "$PWD/reproduction:/cdpr_ft/reproduction" cdpr_ft python reproduce.py quick
```

**D. Google Colab or a similar notebook service (not tested).** In a notebook cell:

```
!git clone <repository URL> cdpr_ft
%cd cdpr_ft
!pip install -r requirements.txt
!python reproduce.py check
!python reproduce.py tests
!python reproduce.py quick --jobs 2
```

If the notebook runs a Python version for which a pinned package has no wheel, install the same packages without the pins; the results may then differ in the last digits (Section 4).

## 2. Levels of checking

Times are those measured on the reference machine (2 cores, Section 6).

| Command | What it checks | Time on 2 cores |
|---|---|---|
| `python reproduce.py check` | Python, library versions against `requirements.txt`, gcc, cores | seconds |
| `python reproduce.py tests` | The 56 verification tests of the model, simulator, estimators, controller, classifier and campaign | 17 min the first time (with the compilation), under a minute after |
| `python reproduce.py figures` | Redraws `figures/verification/` and `figures/phase4/` from `results/` | 15 s |
| `python reproduce.py quick` | The frozen controller on the 8 cable breaks of the calibration set, every run compared with the stored final check | 2.5 min after the compilation |
| `python reproduce.py phase1` | Every Phase 1 analysis | about 2.5 h |
| `python reproduce.py phase2` | Every Phase 2 verification | about 10 min |
| `python reproduce.py phase3` | The Phase 3 chain: safe poses, recovery library, the three calibration sets, the MPC weight study, the replays, the threshold study, the freeze and the final check | about 4.5 h |
| `python reproduce.py phase3 --only final_check` | The frozen controller on the whole calibration set (56 runs) | 21 min |
| `python reproduce.py phase4 --only test` | The single test evaluation of the classifier, from the stored models and dataset | under 1 min |
| `python reproduce.py phase4 --only search,train,thresholds,freeze,test` | Search, training, thresholds, freeze and test from the stored dataset, without simulating the runs again | about 1.3 h |
| `python reproduce.py phase4` | The whole Phase 4 chain, including the 528 closed-loop runs (`generate`) | about 3.4 h |
| `python reproduce.py phase4b` | The classifier trained on both paths: 352 closed-loop runs, training, calibration, freeze and the offline test | about 2.1 h |
| `python reproduce.py phase5 --only campaign_subset,analysis` | 28 jobs of the evaluation campaign simulated again (every method on four fixed runs) and compared with the stored rows and traces; the statistics recomputed from the stored rows | about 11 min |
| `python reproduce.py phase5` | The settings of baseline B2 (gain studies, its calibration runs, thresholds), the freeze, the campaign subset and the statistics | about 1.1 h |
| `python reproduce.py phase6` | The workspace slices of figure 2, the eleven figures and the three tables of the paper, compared byte for byte | about 2.5 min |
| `python reproduce.py phase7` | Every checked number of the manuscript against the stored results (fails if any disagrees) | 15 s |
| `python reproduce.py all` | All of the above | about 14 h |

The compilation of the solvers (about 15 minutes on two cores) happens once, in whichever command first needs them, and is cached in `build/`.

## 3. How a step is checked

For every step, `reproduce.py`

1. copies the stored output files to `reproduction/<time>/reference/`,
2. runs the script on the stored inputs,
3. compares each new file with its reference (`scripts/compare_results.py`),
4. moves the new files to `reproduction/<time>/new/` and puts the references back, so that the repository is left unchanged and each step is checked on its own inputs (`--keep` leaves the new files in place instead),
5. writes the outcome to `reproduction/<time>/report.md` and the console output of the script to `reproduction/<time>/<step>.log`.

The comparison leaves out the provenance stamp and every value whose key contains `seconds`, because these depend on the machine and the date. Integers, booleans and strings (declared mode, correct isolation, lost platform, planner branch, solver status) must match exactly; other numbers must agree to a relative tolerance of 1e-9. A step can also leave out named values, and the report then lists them: the calibration sets c1 and c2 store three settings of the controller that act only after a fault is declared (`plan_latency`, `post_break_weights`, `mpc_weight_scale`). These runs disable declarations, so the settings have no effect on them, but they were changed after the two sets were recorded and the stored copies keep the old values. A file is reported as IDENTICAL, AGREES WITHIN TOLERANCE, or DIFFERENT with the list of differing values. Any two files can also be compared directly:

```bash
python scripts/compare_results.py results/phase3_check_final.json reproduction/<time>/new/results/phase3_check_final.json
```

## 4. Differences: what to expect

**On Linux x86_64 with the pinned versions** the reruns reproduce the stored files exactly (Section 6). The calibration runs, the replays and the final check are deterministic: every random draw comes from the seed of the run.

**On another processor, compiler or library build** the last digits of floating point results can change: IPOPT and its linear solver, the BLAS library and the C code compiled by gcc do not promise identical rounding across platforms. Open-loop analyses are insensitive to this. A closed-loop run after a cable break is not: the snap load of a cable that catches its load again depends on where the integration step falls, so a peak tension or a declaration delay can move by a sample while the outcomes (declared mode, correct isolation, lost platform, planner branch) stay the same. The report lists every such value. Differences in the outcomes would be a reason for concern and we would be glad to hear of them.

**Figures.** The stored figures were drawn in the environment of `requirements.txt`. Redrawn there, they come out byte for byte the same. A different build of matplotlib or of its dependencies can change the bytes of the PNG files without changing what they show: in an environment with pillow 12.3.0 and fonttools 4.66.1 instead of 13.0.0 and 4.67.0 (and a different python-dateutil), the PNG files came out a few pixels wider while the PDF files were byte for byte the same. The two packages are therefore pinned in `requirements.txt`. The PNG files kept in the repository may also carry metadata chunks that a redrawn file does not have (the copies delivered from the cloud workspace in which they were drawn carry an embedded content-credentials manifest); the image data are the same, and the PDF files, which the documents use, are not affected.

**Files produced by earlier versions of the code.** The code changed while the controller was built and calibrated. `results/README.md` lists the code digest of every stored file; Section 6 shows that the code of this repository reproduces all of them that a script of the repository writes. Two kinds of files are kept as records only: the post-break weight check (`phase3_check_post_break_*.json`), run with the configuration file as it was before the thresholds were fixed, and the first run of the final check (`phase3_check_final_run1.json`), which agrees with `phase3_check_final.json` in every value and differs only in the hash of the configuration file, whose header was changed between the two runs.

**Phase 4.** The networks are trained with deterministic algorithms and two threads, as the protocol fixes, so on the reference platform a rerun gives the same weights and the same search. On another processor the matrix kernels of torch can round differently. The search and the training are sensitive to this: a different last digit in an early step changes the path of the optimiser, so a trial can be pruned at another epoch and a model can reach a slightly different validation score. The comparison then reports differences in `phase4_search.json`, `phase4_training.json` and the model files, and the steps after them inherit the new models. To check the evaluation itself independently of training, run `phase4 --only test`: it reads the stored models and dataset and differs at most in the last digits of the probabilities, which can move a declaration by one estimator cycle near a threshold. The scripts use the CPU only; moving the networks to a GPU would not give the stored weights.

## 5. The chains of Phases 3 to 5

The steps depend on each other as follows; `reproduce.py` runs each on the stored inputs, so any of them can be run alone with `--only`.

**Phase 3.**

| Step | Needs |
|---|---|
| `safe_poses` | |
| `recovery_library` | `phase3_safe_poses.npz` |
| `postfault_modes` | `phase3_recovery_library.npz` |
| `calibration_c1` (first MPC weights) | |
| `mpc_weights` | |
| `calibration_c2` (chosen weights, first estimator setting) | |
| `replay_c2`, `replay_c2_stage2`, `eso_check_c2` | the records of set 2, written by `calibration_c2` |
| `roc_c2` | `phase3_calib/c2/runs.json.gz` |
| `calibration_c3` (final estimator setting) | |
| `roc_c3` | `phase3_calib/c3/runs.json.gz` |
| `freeze` | `phase3_roc_c3.json` |
| `final_check` | `controller_phase3.yaml`, `phase3_recovery_library.npz`, `phase3_safe_poses.npz` |

The records of the calibration runs (`results/phase3_calib/<set>/run*.npz`) are not stored in the repository. They are written again by the calibration step of that set and are needed only by the replays and the observer check.

**Phase 4.**

| Step | Needs |
|---|---|
| `generate` (528 closed-loop runs, seeds from 300000) | `controller_phase3.yaml`, `phase3_recovery_library.npz`, `phase3_safe_poses.npz` |
| `dataset` | the runs of `generate` in `results/phase4_work/` |
| `search` | `phase4_data/dataset.npz`, `runs.json` |
| `train` | the dataset, `phase4_search.json` |
| `thresholds` | the dataset, the models of `train` |
| `freeze` | `phase4_training.json`, `phase4_thresholds.json`, the models |
| `test` | `classifier_phase4.yaml` and every file whose hash it records |

The runs of `generate` (`results/phase4_work/`, about 0.6 GB) are not stored in the repository; `generate` itself writes no stored file and is checked through `dataset`, whose output is compared value by value. All later steps start from the stored dataset, so `--only search,train,thresholds,freeze,test` reproduces the classifier without simulating the runs again. The test step refuses to run if any frozen file has changed and repeats the single test evaluation only because `reproduce.py` passes it `--rerun`.

**Phase 4b.** `generate` (352 runs on T2, seeds from 340000) feeds `dataset`; `train` needs the Phase 4 dataset and `phase4b_data/dataset.npz`; `calibrate` the models; `freeze` the training and threshold files; `test` the frozen file and the stored Phase 4 test runs. The runs of `generate` (`results/phase4b_work/`, about 0.4 GB) are not stored.

**Phase 5.**

| Step | Needs |
|---|---|
| `b2_gains` (56 fault-free calibration runs, seeds from 140000) | `controller_phase3.yaml` |
| `b2_runs` (176 calibration runs of B2's controller, seeds from 150000) | `phase5_b2_gains.json` |
| `b2_thresholds` | `phase5_b2_data.npz`, `phase5_b2_gains.json` |
| `b2_postbreak` (160 runs: five gains on B2's 32 calibration breaks) | `configs/b2_phase5.yaml` as `b2_thresholds` writes it |
| `freeze` | the protocol and every file it names |
| `campaign_subset` | the frozen state; compared with `results/phase5/subset_rows.json` and the kept traces |
| `analysis` | `results/phase5/rows.jsonl.gz` |

The campaign itself (`scripts/phase5_run.py`, 5,212 jobs, about 27 h on two cores, under 8 h on eight) writes every row and trace to `results/phase5_work/` (not stored); `scripts/phase5_condense.py` keeps all rows, with the recovery time of every fault run computed from its trace, and the traces of a subset fixed in the protocol (repetition 0 of every path, fault type and cable, and fault-free repetitions 0 and 1) in `results/phase5/`. Any single run can be simulated again from its seed with `phase5_run.py`, and the check of Section 6 shows that rerun jobs reproduce the stored rows and traces exactly. `b2_thresholds` writes `configs/b2_phase5.yaml` without the gains after a break declaration, which `b2_postbreak` adds; run alone, it is compared without them and the stored file is put back afterwards.

## 6. Check on the reference machine

On 26 October 2026 the repository was cloned into an empty folder, a new virtual environment was made with `pip install -r requirements.txt` (2 minutes), and every step of `reproduce.py` was run from that copy: Linux x86_64, 2 cores, 7 GB of memory, Python 3.13.17, gcc 13.3. The first block of steps (tests to final check) ran alone; the later steps ran two at a time to save time, so their times are longer than on an idle machine of the same size.

| Step | Time [min] | Outcome |
|---|---|---|
| `tests` | 17.2, with the compilation | 44 passed |
| `figures` | 0.1 | 11 figures drawn |
| `quick` | 2.4 | identical, 8 runs and their summary (432 values) |
| phase1 `drum_windows` | 0.3 | identical |
| phase1 `cable_analysis` | 0.5 | identical |
| phase1 `workspace` | 10.0 | identical |
| phase1 `postfault_orientation` (t_min 10, 30, 40 N) | 52.8, 39.9, 34.1 | identical |
| phase1 `trajectory_design` | 3.0 | identical |
| phase1 `verification` | 0.6 | identical |
| phase2 `verification` | 0.2 | identical |
| phase2 `step_convergence` | 4.3 | identical |
| phase2 `reference_solution` | 2.0 | identical |
| phase2 `mujoco_crosscheck` | 2.5 | identical |
| phase2 `damping_onset_effect` | 0.2 | identical |
| phase2 `timing` | 0.9 | written; machine dependent, not compared |
| phase3 `safe_poses` | 37.5 | identical (summary and the 49,308 values of the pose sets) |
| phase3 `recovery_library` | 13.6, one worker | identical (statistics, and the 573,652 values of the 1776 plans) |
| phase3 `postfault_modes` | 0.5 | identical |
| phase3 `calibration_c1` | 36.6 | identical in every run (1,741,156 values); three stored settings that act only after a declaration are left out (Section 3) |
| phase3 `mpc_weights` | 32.6 | identical |
| phase3 `calibration_c2` | 35.4 | identical in every run (1,889,274 values); the same three settings left out |
| phase3 `replay_c2` | 47.3 | identical (19 estimator settings replayed on 56 runs) |
| phase3 `replay_c2_stage2` | 13.4 | identical |
| phase3 `eso_check_c2` | 0.3 | identical |
| phase3 `roc_c2` | 0.2 | identical |
| phase3 `calibration_c3` | 20.6 | identical in every run (1,903,590 values) |
| phase3 `roc_c3` | 0.2 | identical |
| phase3 `freeze` | 0.0 | identical configuration values (90) |
| phase3 `final_check` | 21.4 | identical, 56 runs (2111 values) |

"Identical" means that every compared value agreed to the last bit; no step needed the tolerance. The calibration runs, the replays and the closed-loop check are therefore deterministic on this platform, and every stored result that a script writes is exactly what the code of this repository produces. The two kinds of records named in Section 4 are the only stored files that are not rerun.

Two steps needed a change to their script so that they reproduce the stored file with the code as it is now. Neither change touches the package, so the code digest is unchanged.

- `phase3_mpc_weights.py` applies its scalings to the weights and the estimator setting in force when the study was run (velocity weights 400, tension-change weight 0.01, `sigma_acc_ang` = 0.5), now written in the script. It used to take them from the defaults of the configuration classes, which have since been set to the outcome of the calibration.
- `phase3_calibration_runs.py` takes the MPC weights as an optional argument, so that set c1 can be run again with the first weights.

The calibration scripts also now write and read `runs.json.gz` instead of the uncompressed `runs.json` (20 MB per set), which is the file kept in the repository.

**Phase 4 (27 October 2026).** The Phase 4 steps were checked in the working copy of the repository on the same machine, not in a fresh clone, with the system Python (Python 3.13.17, torch 2.14.1+cu130 on the CPU, optuna 5.0.0). The runs of `generate` were in `results/phase4_work/` from the original generation, so the check started at `dataset`. Four runs (indices 5 and 300, training; 430, a fault-free validation run; 500, T2) were also simulated again and compared with the stored ones: the records, the IMM probabilities and the metrics (1,386,951 array values) were identical.

| Step | Time [min] | Outcome |
|---|---|---|
| phase4 `generate` | 123 (original generation, 528 runs, 4.1 core hours) | four runs simulated again: identical |
| phase4 `dataset` | 6.2 | identical (18,741,219 values of the dataset and 8470 of the run list) |
| phase4 `search` | 55.9 | identical (30 trials, 987 values) |
| phase4 `train` | 17.2 | identical: training histories and size curve (586 values) and the five models, 57,483 values each; the model files are also identical byte for byte |
| phase4 `thresholds` | 0.3 | identical (4676 values) |
| phase4 `freeze` | 0.0 | identical (40 values; the file is identical byte for byte) |
| phase4 `test` | 0.7 | identical (7880 values) |

The search and the training ran while other work used the machine at times, so their times are a little longer than on an idle machine (the original search took 52 min).

**Phase 5 (29 October 2026).** In the working copy, on the machine of the campaign, after the campaign had finished. The settings of B2 were checked while they were made: the second gain study ran the settings 10, 20 and 30 rad/s of the first study again and reproduced its runs value by value. The campaign was started twice (amendment 1 of the protocol); the 67 rows of the first start that belong to methods other than B2 came out identical, value for value, when the second start ran them. `reproduce.py phase5 --only campaign_subset,analysis` then simulated 28 jobs again (every method on four fixed runs) and recomputed the statistics: the rows (1,437 values) and traces (3,885,579 values) of the 28 jobs and the analysis file (37,758 values) were identical to the stored ones, in 7.4 and 0.4 min. The steps `b2_gains`, `b2_runs`, `b2_thresholds`, `b2_postbreak` and `freeze` were not rerun through `reproduce.py`.

**Phase 6 (29 October 2026).** `reproduce.py phase6` in the working copy: the workspace slices (68,835 values), the eleven figures (22 PDF and PNG files, compared byte for byte), their manifest and the six tables came out identical, in 2.2 min.

**Phase 7 (29 October 2026).** After the manuscript was written and revised, the paper keeps three tables (parameters, methods, results) and prints the online-planner ablation as A3; `reproduce.py phase6` then reproduced every figure and table byte for byte, and `reproduce.py phase7` confirmed the 80 checked passages of the manuscript against the stored results.

**Phase 8 (29 October 2026).** After the internal review the figure files were renamed in the order in which the manuscript shows them (`fig01` to `fig11`), and one label of the block diagram (figure 3) was corrected; `reproduce.py phase6` then reproduced every figure and table byte for byte, `reproduce.py phase7` confirmed the 82 checked passages, and the 56 tests pass.

