# Manuscript, figures and tables

The manuscript is now prepared for the **International Journal of Control** (Taylor & Francis), in `ijc/`. The journal reviews double anonymously and accepts LaTeX: `ijc/main.tex` gives both versions, `ijc/main_anonymous.pdf` ("Manuscript - anonymous": no author details, funding, contributions or repository address) and `ijc/main_full.pdf` ("Manuscript - with author details"), built by `ijc/build.sh` or by `scripts/paper_package_ijc.py`, which also writes `ijc/ijc_latex_source.zip`, the single zip of LaTeX source files the submission asks for (both versions as separate files, their .bbl files, the Taylor & Francis class and style, the figures as PDF and EPS, the tables). It uses the Interact layout of Taylor & Francis (`interact.cls`, author-date references with `tfcad.bst`); the journal accepts any consistent layout and reference style at submission. Its figures are those of `figures/` redrawn at the page width of that layout (`scripts/paper_figures.py --layout ijc`, `ijc/figures/`), its tables are written in the Taylor & Francis form (`scripts/paper_tables.py --layout ijc`, `ijc/tables/`), and `scripts/paper_check_numbers.py --ijc` checks its numbers (`ijc/number_check.txt`). The abstract is shortened to the 150 words of the journal, the keywords to six, and the highlights of the earlier version are dropped.

The version for the first target journal follows. The manuscript is `manuscript/main.tex` (`main.pdf`), written in the Elsevier CAS double column class named by the author guide of the Chinese Journal of Mechanical Engineering. `manuscript/build.sh` compiles it (pdflatex and BibTeX; the class files are in the folder), and `scripts/paper_package.py` writes `manuscript/cjme_manuscript.zip`, a self-contained LaTeX project for the journal or Overleaf. `scripts/paper_check_numbers.py` checks 109 passages of the text against the stored files (the results of the evaluation, the example run, the classifier test, the model and workspace analyses, the recovery library, the frozen settings, and the replay of the lost runs and the latency study made after the review), writes `manuscript/number_check.txt` and fails if any number disagrees (`reproduce.py phase7`). The few remaining numbers, from development runs and the appendix table, are those of the technical reports in `docs/`.

Apart from the text of the manuscript, everything in this folder is written by a script from stored results; nothing is drawn or typed by hand, except the block diagram of figure 3 (TikZ source in `figures/src/`, whose blocks are modules of `cdpr_ft/`).

| Path | Content | Written by |
|---|---|---|
| `figures/fig01_robot_task` to `fig11_computation` (`.pdf` vector, `.png` 300 dpi) | The eleven figures of the manuscript | `scripts/paper_figures.py` |
| `figures/manifest.json` | For every figure: the stored files it read with their SHA-256 prefixes, and the rule of any choice | `scripts/paper_figures.py` |
| `figures/src/fig03_architecture.tex` | TikZ source of the block diagram | by hand; compiled by `scripts/paper_figures.py` |
| `tables/tab_*.tex` | The tables of the manuscript (LaTeX) | `scripts/paper_tables.py` |
| `ijc/` | The version for the International Journal of Control: `main.tex` (both versions), `main_anonymous.pdf`, `main_full.pdf`, `build.sh`, the Taylor & Francis template files, `figures/`, `tables/`, the record of its number check | LaTeX; `scripts/paper_figures.py --layout ijc`, `scripts/paper_tables.py --layout ijc`, `scripts/paper_package_ijc.py`, `scripts/paper_check_numbers.py --ijc` |
| `manuscript/` | The version for the first target journal (`main.tex`, `main.pdf`), its bibliography (`references.bib`, every entry checked against its Crossref record), its build script, the class files and the record of the number check | LaTeX; `scripts/paper_check_numbers.py` |

## Figure standard

The figures follow the usual conventions of robotics and control journals (IEEE and Springer author guidelines):

- **Size and type.** Width of one column (3.5 in) or of the page (7.16 in), drawn at final size. Serif type (STIX, close to Times) at 8 pt, never below 7 pt; lines of at least 0.75 pt. Vector PDF, with a 300 dpi PNG for previews.
- **Text.** No titles and no sentences inside a figure. Panels are labelled (a), (b), ..., and short column headings are used only where panels differ by a single condition (fault type, lost cable, diagnosis). What a line or a mark means, every count and every statistic go in the caption or in a table. Axis labels name the quantity and its unit.
- **Methods and colours.** Methods appear by their codes (P, A1, ..., B3T), defined in the methods table. Every method keeps one colour in all figures; the colours drawn together stay distinguishable under the common forms of colour vision deficiency, and line style or marker fill carries the method or the path as well, so a grey print stays readable. Reference values (periods, tension bounds, the nominal path) are thin grey lines.
- **Data.** Every curve and mark comes from a stored result (rows, traces, test files, analyses), never from a drawn or assumed curve. A figure that shows one run uses a run chosen by a rule written before drawing.

## Evidence that the figures show the simulations

- **Inputs.** `figures/manifest.json` lists the stored files each figure read, with their SHA-256 prefixes.
- **Figure 4 rerun.** The run shown in figure 4 (run 1) was simulated again from its seed for P, B3, B2 and B0 on 29 October 2026. Its 494,716 trace values were identical to the stored ones, and so were its metrics.
- **Campaign rows and traces.** `reproduce.py phase5 --only campaign_subset` reran 28 jobs of the campaign, and their rows and traces were identical. The 67 non-B2 rows of the first start of the campaign also came out identical when rerun. The campaign statistics behind figures 5 to 9 and 11 therefore come from runs that this code reproduces.
- **Classifier tests.** The offline points of figure 7 come from the single tests of Phases 4 and 4b, which `reproduce.py phase4 --only test` and `phase4b --only test` recompute identically.
- **Workspace slices.** Figure 2 rests on `results/phase6_workspace_slices.npz`, which `reproduce.py phase6` recomputes identically.

## The figures and what each one shows

| Figure | Content | Data |
|---|---|---|
| 1 | The robot and the two reference paths | `configs/robot_ipanema2_caspr.yaml`, `cdpr_ft/analysis/trajectories.py` |
| 2 | Architecture of the controller with its rates | block diagram |
| 3 | Static wrench-feasible poses before and after the loss of an upper and of a lower cable, with the planner's safe poses | `results/phase6_workspace_slices.npz` (`scripts/phase6_workspace_slices.py`), `results/phase3_safe_poses.npz` |
| 4 | One break in closed loop: detection, height with the recovery plan, tilt, cable tensions | Phase 5 traces of run 1 (P, B3, B2, B0), chosen by the rule in `pick_example()` |
| 5 | Delay of correct declarations, by fault type | `results/phase5/rows.jsonl.gz` |
| 6 | False alarms in fault-free runs | same |
| 7 | Platform lost and time outside the tension bounds after breaks | `results/phase5/analysis.json` |
| 8 | Effect of the probability-aware tightening: margin with against without it, run by run | rows |
| 9 | Recovery after breaks | rows (recovery times computed from the traces by `scripts/phase5_condense.py`) |
| 10 | Learned against model-based diagnosis, offline and in closed loop | `results/phase4_test.json`, `results/phase4b_test.json`, rows |
| 11 | Computation time against the periods | rows and kept traces |

## Rebuilding

```bash
python scripts/phase6_workspace_slices.py   # 2 min, only for figure 2
python scripts/paper_figures.py             # seconds; or name figures: fig04 fig07
python scripts/paper_tables.py
python scripts/paper_figures.py --layout ijc   # the same, for the International Journal of Control version
python scripts/paper_tables.py --layout ijc
python reproduce.py phase6                  # all of them, compared with the stored files byte for byte
```

The figures use the STIX fonts shipped with matplotlib, so they look the same on every machine; figure 3 needs `pdflatex` with the `mathptmx` package. Redrawn in the environment of `requirements.txt`, every PDF and PNG is identical byte for byte to the stored one.
