"""Reproduce the stored results of the study and compare them with the references in results/.

Every step runs one script of scripts/ on the stored inputs, compares what it writes with the
stored reference files (scripts/compare_results.py), moves the new files to
reproduction/<time stamp>/new/ and puts the references back, so the repository is left as it
was and each step is checked on its own. The report is reproduction/<time stamp>/report.md.

  python reproduce.py check            Python, packages, C compiler, cores
  python reproduce.py list             all steps with the time they took on the reference machine
  python reproduce.py tests            automated verification gates (56 tests; those that need torch
                                       are skipped when requirements-learning.txt is not installed)
  python reproduce.py figures          redraw the verification figures from the stored results
  python reproduce.py quick            reviewer check: frozen controller on the 8 calibration breaks
  python reproduce.py phase1|phase2|phase3|phase4|phase4b|phase5|phase6|phase7 [--only NAME,...]
  python reproduce.py all              everything above, in order

Options: --jobs N (worker processes for the closed-loop steps, default: number of cores, at most
8), --keep (leave the regenerated files in results/ instead of restoring the references).
Times are those measured on the reference machine (2 cores, x86_64), often with a second step
running on the other core (REPRODUCE.md, Section 6); the closed-loop steps scale with --jobs.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_results import compare, describe, verdict  # noqa: E402

JOBS = "{jobs}"


@dataclass
class Step:
    name: str
    args: list[str]                      # script and arguments; "{jobs}" is replaced
    outputs: list[str]                   # files the step writes, relative to the repository
    minutes: float                       # wall time on the reference machine
    note: str = ""
    compare_with: dict[str, str] = field(default_factory=dict)  # output -> reference, if not itself
    rtol: float = 1e-9
    timing_only: bool = False            # machine-dependent output: kept, not compared
    ignore: list[str] = field(default_factory=list)  # values left out of the comparison, with the reason in note


R = "results/"
# Settings of the controller that act only after a declaration. The calibration runs disable
# declarations, so these settings do not affect them; they were changed after sets c1 and c2
# were recorded, and the settings stored with those sets keep the old values.
AFTER_DECLARATION = ["$.meta.ftc.plan_latency", "$.meta.ftc.post_break_weights", "$.meta.ftc.mpc_weight_scale"]
AFTER_NOTE = ("the stored settings that act only after a declaration (disabled in these runs) "
              "were changed later and are left out of the comparison")
# Phase 5 added the baselines B2 and B3 to the controller code (cdpr_ft/control/ftc.py,
# cdpr_ft/estimation/alternatives.py, cdpr_ft/sim/episode.py), so the digest of the controller code
# that the Phase 4 files record (fe7722d2d710c537, the Phase 3 freeze) is no longer that of the
# package. The variants used before Phase 5 are unchanged: the quick check and a Phase 4 run
# simulated again are identical (REPRODUCE.md, Section 6).
PHASE5_CODE_NOTE = ("the recorded digest of the controller code is left out: Phase 5 added the B2 and B3 variants "
                    "to that code without changing the variants used here")


def _phase5_subset_outputs() -> list[str]:
    """Rows and traces that scripts/phase5_check.py writes (the jobs of its fixed subset)."""
    try:
        from phase5_check import PROTOCOL, subset
        sys.path.insert(0, str(ROOT))
        from cdpr_ft.eval.campaign import load_protocol
        jobs, _ = subset(load_protocol(PROTOCOL)[0])
    except Exception:                    # noqa: BLE001 - listing the steps must not need the package
        jobs = []
    return [R + "phase5/subset_rows.json"] + [R + f"phase5/traces/{m}/run{r:05d}.npz" for m, r in jobs]


PAPER_FIGURES = ["fig01_robot_task", "fig02_architecture", "fig03_postfault_workspace", "fig04_example_break",
                 "fig05_detection_delay", "fig06_false_alarms", "fig07_safety_after_break", "fig08_tightening",
                 "fig09_recovery", "fig10_learned_vs_model", "fig11_computation"]
PAPER_TABLES = ["tab_parameters", "tab_methods", "tab_results"]


PHASES: dict[str, list[Step]] = {
    "phase1": [
        Step("drum_windows", ["phase1_drum_windows.py"], [R + "phase1_drum_windows.json"], 0.3),
        Step("cable_analysis", ["phase1_cable_analysis.py"], [R + "phase1_cable_analysis.json"], 0.5),
        Step("workspace", ["phase1_workspace.py"], [R + "phase1_workspace.json"], 10),
        Step("postfault_orientation", ["phase1_postfault_orientation.py"], [R + "phase1_postfault_orientation.json"], 53),
        Step("postfault_orientation_30", ["phase1_postfault_orientation.py", "30"],
             [R + "phase1_postfault_orientation_tmin30.json"], 40),
        Step("postfault_orientation_40", ["phase1_postfault_orientation.py", "40"],
             [R + "phase1_postfault_orientation_tmin40.json"], 34),
        Step("trajectory_design", ["phase1_trajectory_design.py"], [R + "phase1_trajectory_design.json"], 3),
        Step("verification", ["phase1_verification.py"], [R + "phase1_verification.txt"], 0.6),
    ],
    "phase2": [
        Step("verification", ["phase2_verification.py"], [R + "phase2_verification.json"], 0.2),
        Step("step_convergence", ["phase2_step_convergence.py"], [R + "phase2_step_convergence.json"], 4.3),
        Step("reference_solution", ["phase2_reference_solution.py"], [R + "phase2_reference_solution.json"], 2),
        Step("mujoco_crosscheck", ["phase2_mujoco_crosscheck.py"], [R + "phase2_mujoco_crosscheck.json"], 2.5),
        Step("damping_onset_effect", ["phase2_damping_onset_effect.py"], [R + "phase2_damping_onset_effect.json"], 0.2),
        Step("timing", ["phase2_timing.py"], [R + "phase2_timing.json"], 0.9, timing_only=True,
             note="computation times of this machine; kept for information, not compared"),
    ],
    "phase3": [
        Step("safe_poses", ["phase3_safe_poses.py"], [R + "phase3_safe_poses.json", R + "phase3_safe_poses.npz"], 38),
        Step("recovery_library", ["phase3_recovery_library.py", JOBS],
             [R + "phase3_recovery_library.json", R + "phase3_recovery_library.npz"], 9),
        Step("postfault_modes", ["phase3_postfault_modes.py"], [R + "phase3_postfault_modes.json"], 0.5),
        Step("calibration_c1", ["phase3_calibration_runs.py", "c1", JOBS, '{"sigma_acc_ang": 0.5}',
                                '{"q_vel": 400.0, "q_angvel": 400.0, "r_rate": 0.01}'],
             [R + "phase3_calib/c1/runs.json.gz"], 37, ignore=AFTER_DECLARATION,
             note="first MPC weights and estimator setting; " + AFTER_NOTE),
        Step("mpc_weights", ["phase3_mpc_weights.py", JOBS], [R + "phase3_mpc_weights.json"], 33),
        Step("calibration_c2", ["phase3_calibration_runs.py", "c2", JOBS, '{"sigma_acc_ang": 0.5}'],
             [R + "phase3_calib/c2/runs.json.gz"], 35, ignore=AFTER_DECLARATION,
             note="chosen MPC weights, first estimator setting; " + AFTER_NOTE),
        Step("replay_c2", ["phase3_replay.py", "c2", JOBS], [R + "phase3_replay_c2.json"], 47,
             note="needs the records of calibration_c2"),
        Step("replay_c2_stage2", ["phase3_replay.py", "c2", JOBS, R + "phase3_replay_stage2_settings.json"],
             [R + "phase3_replay_c2_stage2_settings.json"], 13, note="needs the records of calibration_c2"),
        Step("eso_check_c2", ["phase3_eso_check.py", "c2"], [R + "phase3_eso_check_c2.json"], 0.3,
             note="needs the records of calibration_c2"),
        Step("roc_c2", ["phase3_roc.py", "c2"], [R + "phase3_roc_c2.json"], 0.2),
        Step("calibration_c3", ["phase3_calibration_runs.py", "c3", JOBS], [R + "phase3_calib/c3/runs.json.gz"], 21,
             note="final estimator setting"),
        Step("roc_c3", ["phase3_roc.py", "c3"], [R + "phase3_roc_c3.json"], 0.2),
        Step("freeze", ["phase3_freeze.py", "c3"], ["configs/controller_phase3.yaml"], 0.1,
             note="the values are compared; the header carries the date of writing"),
        Step("final_check", ["phase3_closed_loop_check.py", "final", "controller_phase3.yaml", JOBS, "all"],
             [R + "phase3_check_final.json"], 21),
    ],
    "phase4": [
        Step("generate", ["phase4_generate.py", JOBS], [], 123,
             note="528 runs into results/phase4_work (not stored); checked through the dataset"),
        Step("dataset", ["phase4_dataset.py"], [R + "phase4_data/dataset.npz", R + "phase4_data/runs.json"], 6.2,
             note="needs the runs of generate; " + PHASE5_CODE_NOTE, ignore=["$.controller_digest"]),
        Step("search", ["phase4_search.py"], [R + "phase4_search.json"], 56),
        Step("train", ["phase4_train.py"], [R + "phase4_training.json"] + [R + f"phase4_models/seed{s}.pt" for s in range(5)],
             17.2),
        Step("thresholds", ["phase4_thresholds.py"], [R + "phase4_thresholds.json"], 0.3),
        Step("freeze", ["phase4_freeze.py"], ["configs/classifier_phase4.yaml"], 0.1,
             note="the values are compared; the header carries the date of writing; " + PHASE5_CODE_NOTE,
             ignore=["$.hashes.controller_digest"]),
        Step("test", ["phase4_test.py", "--rerun"], [R + "phase4_test.json"], 0.7),
    ],
    "phase4b": [
        Step("generate", ["phase4b_generate.py", JOBS], [], 96,
             note="352 runs into results/phase4b_work (not stored); checked through the dataset"),
        Step("dataset", ["phase4b_dataset.py"], [R + "phase4b_data/dataset.npz", R + "phase4b_data/runs.json"], 2.5,
             note="needs the runs of generate"),
        Step("train", ["phase4b_train.py"], [R + "phase4b_training.json"] + [R + f"phase4b_models/seed{s}.pt" for s in range(5)],
             25),
        Step("calibrate", ["phase4b_calibrate.py"], [R + "phase4b_thresholds.json"], 2),
        Step("freeze", ["phase4b_freeze.py"], ["configs/classifier_phase4b.yaml"], 0.1,
             note="the values are compared; the header carries the date of writing"),
        Step("test", ["phase4b_test.py", "--rerun"], [R + "phase4b_test.json"], 2),
    ],
    "phase5": [
        Step("b2_gains", ["phase5_b2_gains.py", JOBS], [R + "phase5_b2_gains.json"], 13),
        Step("b2_runs", ["phase5_b2_runs.py", JOBS], [R + "phase5_b2_data.npz", R + "phase5_b2_runs.json"], 20),
        Step("b2_thresholds", ["phase5_b2_calibrate.py"], [R + "phase5_b2_thresholds.json", "configs/b2_phase5.yaml"], 1,
             note="the script writes configs/b2_phase5.yaml without the post-break gains, which b2_postbreak adds",
             ignore=["$.ct_kp_post", "$.ct_kd_post", "$.source"]),
        Step("b2_postbreak", ["phase5_b2_postbreak.py", JOBS], [R + "phase5_b2_postbreak.json", "configs/b2_phase5.yaml"],
             20),
        Step("freeze", ["phase5_freeze.py", "--force"], [R + "phase5_freeze.json"], 0.1),
        Step("campaign_subset", ["phase5_check.py", JOBS], _phase5_subset_outputs(), 7.4,
             note="28 jobs of the campaign simulated again (every method on four fixed runs) and compared, "
                  "row and trace, with what the campaign stored; the full campaign is about 27 h on two cores"),
        Step("analysis", ["phase5_analyse.py"], [R + "phase5/analysis.json"], 0.4,
             note="statistics from the stored rows (results/phase5/rows.jsonl.gz)"),
    ],
    "phase6": [
        Step("workspace_slices", ["phase6_workspace_slices.py"],
             [R + "phase6_workspace_slices.npz", R + "phase6_workspace_slices.json"], 2.2),
        Step("figures", ["paper_figures.py"],
             [f"paper/figures/{n}.{e}" for n in PAPER_FIGURES for e in ("pdf", "png")] + ["paper/figures/manifest.json"],
             0.1, note="the manuscript figures from the stored results; PDF and PNG files compared byte for byte"),
        Step("tables", ["paper_tables.py"], [f"paper/tables/{n}.tex" for n in PAPER_TABLES], 0.1),
    ],
    "phase7": [
        Step("numbers", ["paper_check_numbers.py"], ["paper/manuscript/number_check.txt"], 0.2,
             note="every checked number of the manuscript against the stored results; fails if any disagrees"),
    ],
}
QUICK = Step("quick", ["phase3_closed_loop_check.py", "quick", "controller_phase3.yaml", JOBS, "breaks"],
             [R + "phase3_check_quick.json"], 3,
             note="frozen controller on the 8 calibration breaks; compared run by run with the final check",
             compare_with={R + "phase3_check_quick.json": R + "phase3_check_final.json"})


def out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ROOT / "reproduction").mkdir(exist_ok=True)
    for k in range(100):  # two runs started in the same second get separate folders
        d = ROOT / "reproduction" / (stamp if k == 0 else f"{stamp}_{k}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError("no free folder name under reproduction/")


def log(report: Path, text: str):
    print(text, flush=True)
    with report.open("a") as f:
        f.write(text + "\n")


def run_step(step: Step, jobs: int, work: Path, report: Path, keep: bool) -> bool:
    ref_dir, new_dir = work / "reference", work / "new"
    refs = [ROOT / step.compare_with.get(o, o) for o in step.outputs]
    for o in set(step.outputs) | set(step.compare_with.values()):
        src = ROOT / o
        if src.exists():
            dst = ref_dir / o
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    cmd = [sys.executable, str(ROOT / "scripts" / step.args[0])] + [a.replace(JOBS, str(jobs)) for a in step.args[1:]]
    log(report, f"\n### {step.name}\n\n`{' '.join(cmd[1:]).replace(str(ROOT) + os.sep, '')}`"
                + (f" ({step.note})" if step.note else "") + f", about {step.minutes:g} min on the reference machine\n")
    t0 = time.time()
    with (work / f"{step.name}.log").open("w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - t0) / 60
    if rc != 0:
        log(report, f"FAILED with exit status {rc} after {minutes:.1f} min; see {step.name}.log\n")
        _restore(step, ref_dir, new_dir, keep)
        return False
    ok = True
    lines = [f"finished in {minutes:.1f} min"]
    for o, ref in zip(step.outputs, refs):
        new = ROOT / o
        ref_copy = ref_dir / ref.relative_to(ROOT)
        if step.timing_only:
            lines.append(f"{o}: written (machine-dependent timings, not compared)")
        elif not ref_copy.exists():
            lines.append(f"{o}: no stored reference")
        elif not new.exists():
            lines.append(f"{o}: NOT WRITTEN")
            ok = False
        else:
            t = compare(ref_copy, new, rtol=step.rtol, ignore=step.ignore)
            ok &= verdict(t) != "DIFFERENT"
            lines.append(describe(f"{o}", t))
    log(report, "\n".join(lines) + "\n")
    _restore(step, ref_dir, new_dir, keep)
    return ok


def _restore(step: Step, ref_dir: Path, new_dir: Path, keep: bool):
    for o in step.outputs:
        src = ROOT / o
        if src.exists():
            dst = new_dir / o
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        ref = ref_dir / o
        if keep:
            continue
        if ref.exists():
            shutil.copy2(ref, src)
        elif src.exists():
            src.unlink()


def check_environment() -> bool:
    import importlib
    ok = True
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
          f"{os.cpu_count()} cores")
    if platform.system() != "Linux":
        print("  note: tested on Linux only (the runs use the 'fork' start method); on Windows use WSL2")
    want = {}
    for name, needed in (("requirements.txt", True), ("requirements-learning.txt", False)):
        if not (ROOT / name).exists():
            continue
        for line in (ROOT / name).read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                k, v = line.split("==")
                want[k.strip()] = (v.strip(), name, needed)
    modules = {"PyYAML": "yaml", "pillow": "PIL", "fonttools": "fontTools"}
    for pkg, (ver, name, needed) in want.items():
        try:
            m = importlib.import_module(modules.get(pkg, pkg.lower()))
            have = getattr(m, "__version__", "?")
            flag = "" if have.split("+")[0] == ver else f"   (reference: {ver})"
            print(f"  {pkg:12s} {have}{flag}")
        except ImportError:
            if needed:
                print(f"  {pkg:12s} MISSING (pip install -r {name})")
                ok = False
            else:
                print(f"  {pkg:12s} not installed: needed for Phase 4 only (pip install -r {name})")
    gcc = shutil.which("gcc")
    if gcc:
        v = subprocess.run([gcc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        print(f"  gcc          {v}")
    else:
        print("  gcc          MISSING: the compiled estimators and MPC need a C compiler")
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["check", "list", "tests", "figures", "quick", "phase1", "phase2", "phase3", "phase4", "phase4b", "phase5", "phase6", "phase7", "all"])
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    ap.add_argument("--only", help="comma separated step names within a phase")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if args.target == "check":
        sys.exit(0 if check_environment() else 1)
    if args.target == "list":
        for ph, steps in PHASES.items():
            print(f"{ph}: about {sum(s.minutes for s in steps) / 60:.1f} h on the reference machine")
            for s in steps:
                print(f"  {s.name:26s} {s.minutes:6.1f} min  {' '.join(s.args)}" + (f"   [{s.note}]" if s.note else ""))
        print(f"quick: about {QUICK.minutes:g} min plus the first compilation of the solvers (about 15 min)")
        return
    if args.target in ("tests", "all"):
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT).returncode
        if rc != 0 or args.target == "tests":
            sys.exit(rc)
    if args.target in ("figures", "all"):
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py")], cwd=ROOT).returncode
        if rc != 0 or args.target == "figures":
            sys.exit(rc)

    if args.target == "quick":
        steps = [QUICK]
    elif args.target == "all":
        steps = [QUICK] + [s for ph in PHASES.values() for s in ph]
    else:
        steps = PHASES[args.target]
        if args.only:
            names = args.only.split(",")
            unknown = set(names) - {s.name for s in steps}
            if unknown:
                sys.exit(f"unknown step(s) {sorted(unknown)}; see: python reproduce.py list")
            steps = [s for s in steps if s.name in names]
    work = out_dir()
    report = work / "report.md"
    log(report, f"# Reproduction report\n\nTarget: {args.target}; {len(steps)} step(s); --jobs {args.jobs}; "
                f"Python {platform.python_version()}, {platform.system()} {platform.machine()}, {os.cpu_count()} cores; "
                f"started {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    results = {s.name: run_step(s, args.jobs, work, report, args.keep) for s in steps}
    bad = [k for k, v in results.items() if not v]
    log(report, "\n## Summary\n\n" + ("All steps reproduced the stored results within tolerance." if not bad else
                                      "Steps with differences or failures: " + ", ".join(bad)
                                      + ". See the step sections above and REPRODUCE.md, section 'Differences'."))
    print(f"\nreport: {report.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
