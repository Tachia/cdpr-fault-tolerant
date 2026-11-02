# Replication on other machines

This protocol checks that the committed code, run on a machine other than the one that produced the results, gives the stored results and the numbers printed in the paper. It is fixed here before any replication is run, so that the outcome cannot be adjusted afterwards. `REPRODUCE.md` describes the reproduction steps themselves; this file describes how several machines are compared.

What a replication shows is computational: the same code and inputs give the same results elsewhere. It does not test whether the model describes the physical robot. That question is addressed by the verification of the model (Appendix A of the paper, `tests/`) and, in the end, only by tests on hardware.

## 1. What is fixed in advance

- **The code.** Every machine runs the tag `v1.0-replication`. The report records the commit and the tree of the code, and reports made from different trees are not compared.
- **The environment.** Linux on x86_64, Python 3.13, the versions pinned in `requirements.txt` and `requirements-learning.txt`, and gcc. `replication/setup_env.sh` builds it in `.venv/` on any such machine; on Windows, use WSL2 (the runs use the `fork` start method).
- **No changes to the code.** `scripts/replicate.py` refuses to start in a working tree with changes and checks again at the end. A replication that only works after a change is reported as not reproduced, together with the change; it is not patched.
- **The sample of tier 2.** For every path, fault type and cable, the fault run whose repetition equals the cable index (48 runs), and the first five fault free runs of each path (10 runs), each with every method that the campaign ran on it: 394 jobs (`sample_jobs()` in `scripts/replicate.py`).
- **The comparison.** Values that describe an outcome (declared mode, correct isolation, lost platform, missed fault, false alarm, planner branch, and every other integer, boolean or string) must match exactly. Floating point values are compared by their relative difference. Timing values are left out, since they depend on the machine.
- **The judgement** of Section 5.

## 2. Tiers

| Tier | What it checks | Time on the reference machine (2 cores) |
|---|---|---|
| 1 | The 56 tests; the statistics of the evaluation recomputed from the stored rows; the workspace slices and the tables; the figures drawn from data, byte for byte; every checked number of the manuscript (82 passages) | about 5 min, plus about 20 min the first time the solvers are compiled |
| 2 | Tier 1, then simulation again: the reviewer check of the frozen controller (8 calibration breaks), the 28 jobs whose rows and traces the repository keeps, and the sample of 394 jobs, compared row by row with the stored campaign | about 3 h (4.6 core hours for the sample) |
| 3 | Tier 2, then the whole campaign of 5,212 jobs, condensed and analysed again; the table of results and the checked numbers of the manuscript recomputed from the new campaign | about 64 core hours more |

Tier 2 is the recommended level for every machine. Tier 3 is worth running on one machine with enough cores.

Figure 3 (the block diagram) is compiled by pdflatex, and its bytes depend on the TeX installation, so it is left out of the byte comparison. The other ten figures are compared as PDF and PNG files.

## 3. Running a replication

On any Linux machine with gcc and internet access to PyPI and GitHub:

```bash
git clone --branch v1.0-replication https://github.com/Tachia/cdpr-fault-tolerant.git
cd cdpr-fault-tolerant
bash replication/setup_env.sh                  # or --torch-cpu for the smaller CPU build of torch
.venv/bin/python scripts/replicate.py --label <name> --tier 2
```

`<name>` identifies the machine in the reports, for example `kaggle` or `workstation`. The report goes to `replication/runs/<name>/`: `report.md` to read, `report.json` for the comparison, the logs of every step and the simulated rows. An interrupted tier 2 or 3 continues where it stopped when the same command is run again. Tier 3 accepts `--max-campaign-jobs N`, which stops after N new jobs of the campaign so that a session with a time limit can end cleanly; the report then says UNFINISHED until the campaign is complete.

The torch wheel matters only for the learned classifier (methods B3 and B3T). The stored results were produced with the PyPI wheel running on the CPU; the CPU build of the same version can differ in the last digits.

**Hosted Linux environments.** In a cloud development environment, run `bash replication/setup_env.sh` in the setup stage, where internet access is available, and the replication command in the working stage. A tier 2 run needs a few hours of wall time; if the environment ends tasks sooner, run the same command again, and the replication continues from its saved rows.

**Kaggle.** Import `replication/kaggle_replication.ipynb` (File, Import Notebook, from the GitHub link), set the accelerator to None and turn internet access on, then run it with Save Version, Save & Run All, so that it runs in the background and its output is kept. The notebook clones the tag outside `/kaggle/working`, builds the environment, links the report, the compiled solvers and the campaign folder into `/kaggle/working`, and prints the report at the end. For tier 3, a session that ends with UNFINISHED is continued by a new version of the notebook with the output of the previous version attached as input.

## 4. Comparing the machines

Collect the report folders in one place and run

```bash
python scripts/compare_replications.py replication/runs/reference replication/runs/kaggle replication/runs/<other> \
       --out replication/comparison.md
```

The table lists, per machine, the hardware, the Python and package versions, and the verdict of every step; then, for every pair of machines, how many sample jobs are identical, differ only in floating point values (with the largest relative difference), or differ in an outcome; and finally the verdict of the protocol. The report of the reference machine is kept in `replication/reference/`.

## 5. How the result is judged

| Verdict | Meaning | Consequence |
|---|---|---|
| REPRODUCED IDENTICALLY | Every compared value agrees to the last bit | None |
| REPRODUCED WITHIN NUMERICAL PRECISION | Every outcome agrees; some floating point values differ in their last digits | None. This is what to expect on another processor, compiler or BLAS library (`REPRODUCE.md`, Section 4) |
| DIFFERENCES TO REVIEW | Some jobs differ in an outcome | Each listed job is examined: a declaration one estimator cycle earlier or later near the threshold, or a platform lost or kept with a tilt near the 30° limit, is a run near a decision boundary. If tier 3 shows that no checked number of the paper changes, the result stands. If a number of the paper changes, the paper reports the variation across machines |
| NOT REPRODUCED | A test or a step fails, a tracked file changes, or the code had to be changed | The cause is found and fixed in the code before the results are used, and every machine runs the fixed version again |

A difference is never removed by changing the comparison or the sample after the replications have been run.
