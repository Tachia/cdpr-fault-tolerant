"""Verification and calibration figures of Phases 1 to 3, drawn from the stored result files.

Every figure here is read from a file in results/ (no simulation is run), so the figures can be
redrawn in seconds on any machine that has the repository. They document the verification of the
model and simulator and the calibration of the controller; none of them is a study result. The
figures of the manuscript come from the Phase 5 evaluation runs and are made in Phase 6.

Output: figures/verification/<name>.pdf (vector) and .png (300 dpi), and figures/README.md with
the source file of every figure.
Usage: python scripts/make_figures.py [name ...]      (no argument: all figures)
"""
from __future__ import annotations

import gzip
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures" / "verification"

# Categorical slots in fixed order (checked for colour differences under simulated colour vision
# deficiency: adjacent pairs for lines and bars, all pairs for the first three slots).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ONE_COL, TWO_COL = 3.5, 7.16  # inches, IEEE column widths

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})


def load(name):
    p = RES / name
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return json.loads(p.read_text())


def folder(name):
    return ROOT / "figures" / ("phase4" if name.startswith("p4_") else "verification")


def save(fig, name):
    out = folder(name)
    out.mkdir(parents=True, exist_ok=True)
    # no creation date in the files, so that redrawing an unchanged figure leaves it unchanged
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Phase 1


def fig_p1_circle_radius():
    d = load("phase1_trajectory_design.json")
    tab = d["largest_radius_by_period"]
    per = np.array([float(k[1:]) for k in tab])
    rad = np.array([v["largest_radius_m"] for v in tab.values()])
    lim = [v["limited_by"] for v in tab.values()]
    sel = d["selected"]["T1_circle"]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    ax.plot(per, rad, color=MUTED, lw=0.8, zorder=1)
    for lab, col, mk in (("speed or acceleration", C[0], "o"), ("tension", C[1], "s")):
        m = np.array([x == lab for x in lim])
        ax.scatter(per[m], rad[m], s=26, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3,
                   label=f"limited by {lab}")
    ax.scatter([sel["period_s"]], [sel["radius_m"]], s=46, marker="*", color=INK, zorder=4,
               label="selected T1, T2 (0.5 m, 4 s)")
    ax.set_xlabel("Period of one turn (s)")
    ax.set_ylabel("Largest admissible radius (m)")
    ax.set_ylim(0, 2.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=2, columnspacing=1.0)
    save(fig, "p1_circle_radius")


def fig_p1_postfault_workspace():
    d = load("phase1_workspace.json")["cases"]
    tmins = [10, 30, 40]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.2), sharey=True)
    for ax, (ori, title) in zip(axes, (("zero", "Zero orientation"),
                                       ("pm10deg_single_axis", "Zero and +/-10 deg about each axis in turn"))):
        x = np.arange(len(tmins))
        for k, (lab, idx) in enumerate((("upper cable lost (1-4)", 0), ("lower cable lost (5-8)", 4))):
            v = [100 * d[f"tmin{t}_{ori}"]["post_fault_fraction"][idx] for t in tmins]
            b = ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=C[k], label=lab, edgecolor="white", lw=1)
            ax.bar_label(b, fmt="%.1f", fontsize=7, color=INK2, padding=1.5)
        ax.set_xticks(x, [f"{t} N" for t in tmins])
        ax.set_xlabel("Lower tension bound t_min")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of the nominal\nworkspace still held (%)")
    axes[0].set_ylim(0, 50)
    axes[1].legend(loc="upper right")
    save(fig, "p1_postfault_workspace")


# ---------------------------------------------------------------------------------------------
# Phase 2


def fig_p2_step_convergence():
    d = load("phase2_step_convergence.json")
    sc = d["scenarios"]
    names = [("nominal", "no fault"), ("break_lower", "break, cable 5"), ("break_upper", "break, cable 1"),
             ("loe", "loss of effectiveness 0.6, winch 3"), ("stiff_break", "break, cable 5, stiff cable")]
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.4))
    for k, (key, lab) in enumerate(names):
        for suffix, ls in (("", "-"), ("_noise_free", "--")):
            if key + suffix not in sc:
                continue
            h = sc[key + suffix]["halving"]
            dt = [float(s.split("->")[0]) for s in h]
            v = [100 * h[s]["worst_metric_change"] for s in h]
            ax.plot(1e6 * np.array(dt), v, ls=ls, color=C[k], marker="o", ms=3.2, mec="white", mew=0.5,
                    label=lab if not suffix else None)
    ax.axhline(0.1, color=INK2, lw=0.8, ls=":")
    ax.text(1.05e3, 0.13, "0.1 % (specification)", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.axvline(62.5, color=MUTED, lw=0.8)
    ax.text(67, 1.3, "plant step\nof the study", fontsize=7, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [62.5, 125, 250, 500, 1000]
    ax.set_xticks(ticks, ["62.5", "125", "250", "500", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Plant step h (us), compared with h/2")
    ax.set_ylabel("Largest change of a metric (%)")
    ax.set_ylim(1e-6, 10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK2, ls="--"))
    labels.append("dashed: same run without sensor noise")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=7, ncol=2,
               handlelength=1.8, columnspacing=1.0)
    save(fig, "p2_step_convergence")


# ---------------------------------------------------------------------------------------------
# Phase 3


def fig_p3_library_drop():
    d = load("phase3_recovery_library.json")["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, 3.8), sharex=True)
    for i, traj in enumerate(("T1", "T2")):
        for j, (lo, title) in enumerate(((0, "upper cables"), (4, "lower cables"))):
            ax = axes[i, j]
            for k in range(4):
                cab = lo + k
                c = sorted((x for x in d if x["trajectory"] == traj and x["cable"] == cab), key=lambda x: x["onset"])
                scale = 1.0 if lo == 0 else 100.0
                # branch 1 (time scaling) has no recovery path: left as a gap
                y = [scale * x["drop_m"] if "drop_m" in x else np.nan for x in c]
                ax.plot([x["onset"] for x in c], y, color=C[k], lw=1.0, label=f"cable {cab + 1}")
            ax.set_title(f"{traj}, break of one of the {title}")
            ax.set_ylabel("Planned drop (m)" if lo == 0 else "Planned drop (cm)")
            ax.set_ylim(0, 1.3 * np.nanmax([ln.get_ydata() for ln in ax.get_lines()]))
            if i == 0:
                ax.legend(loc="upper center", ncol=4, fontsize=7, columnspacing=1.0, handlelength=1.5)
    for ax in axes[1]:
        ax.set_xlabel("Break instant on the reference (s)")
    fig.text(0.01, -0.03, "Planned drop: largest fall of the reference point below its height at the break, along the "
             "recovery path (branch 2). Gaps: instants answered by time scaling (branch 1), which keeps the path.",
             fontsize=7, color=INK2, ha="left")
    save(fig, "p3_library_drop")


def fig_p3_library_branch():
    d = load("phase3_recovery_library.json")
    onsets = np.array(d["onsets"])
    cmap = LinearSegmentedColormap.from_list("br", [C[0], C[1]], N=2)
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.0), sharey=True)
    for ax, traj in zip(axes, ("T1", "T2")):
        grid = np.full((8, len(onsets)), np.nan)
        for x in d["cases"]:
            if x["trajectory"] == traj:
                grid[x["cable"], int(np.argmin(np.abs(onsets - x["onset"])))] = x["branch"]
        assert not np.isnan(grid).any() and set(np.unique(grid)) <= {1.0, 2.0}, "unexpected branch values"
        step = onsets[1] - onsets[0]
        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=1, vmax=2, interpolation="nearest",
                  extent=(onsets[0] - step / 2, onsets[-1] + step / 2, 8.5, 0.5))
        ax.set_yticks(range(1, 9))
        ax.set_title(traj)
        ax.set_xlabel("Break instant (s)")
        ax.grid(False)
    axes[0].set_ylabel("Broken cable")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[0]), plt.Rectangle((0, 0), 1, 1, color=C[1])]
    axes[1].legend(handles, ["branch 1: time scaling", "branch 2: recovery to a held pose"],
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    save(fig, "p3_library_branch")


def fig_p3_eso_lag():
    d = load("phase3_eso_check_c2.json")["bandwidths"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.2), sharey=True)
    for ax, (key, title) in zip(axes, (("noise_x1", "Nominal sensor noise"), ("noise_x2", "Sensor noise doubled"))):
        for k, bw in enumerate(("25", "50", "100")):
            r = d[bw][key]["rms_N_at_lag"]
            lag = np.array([int(s) for s in r])
            ax.plot(lag, [r[s] for s in r], color=C[k], marker="o", ms=3, mec="white", mew=0.5,
                    label=f"observer bandwidth {bw} Hz" + (" (chosen)" if bw == "50" else ""))
        ax.set_title(title)
        ax.set_xlabel("Lag of the plant tension compared (ms)")
    axes[0].set_ylim(0, 1.15 * max(max(ln.get_ydata()) for a in axes for ln in a.get_lines()))
    axes[0].set_ylabel("RMS error of the tension estimate (N)")
    axes[0].legend(loc="lower left", fontsize=7)
    save(fig, "p3_eso_lag")


def fig_p3_mpc_weights():
    d = load("phase3_mpc_weights.json")
    sets = ["first", "A", "B", "C"]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.5))
    rng = np.random.default_rng(0)  # horizontal jitter of the dots only
    for i, s in enumerate(sets):
        runs = d["settings"][s]["runs"]
        for traj, col in (("T1", C[0]), ("T2", C[1])):
            rr = [r for r in runs if r["trajectory"] == traj]
            x = i + (-0.12 if traj == "T1" else 0.12) + rng.uniform(-0.05, 0.05, len(rr))
            y = [r["rms_mm"] for r in rr]
            lost = np.array([r["lost"] is not None for r in rr])
            ax.scatter(x[~lost], np.array(y)[~lost], s=14, color=col, edgecolor="white", lw=0.5, zorder=3,
                       label=traj if i == 0 else None)
            if lost.any():
                ax.scatter(x[lost], np.array(y)[lost], s=30, facecolor="none", edgecolor=col, lw=1.0,
                           zorder=3)
                for xx, yy in zip(x[lost], np.array(y)[lost]):
                    ax.annotate("lost", (xx, yy), xytext=(5, -3), textcoords="offset points", fontsize=7, color=INK2)
        med = d["settings"][s]["rms_mm_median"]
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=1.0, zorder=4)
    ax.set_xticks(range(len(sets)), ["first", "A", "B (chosen)", "C"])
    ax.set_yscale("log")
    ax.set_ylabel("RMS position error, first 8 s (mm)")
    ax.set_xlabel("MPC weight setting (16 fault-free calibration runs each)")
    ax.grid(axis="x", visible=False)
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=1.0))
    lab.append("median")
    ax.legend(h, lab, loc="upper right", ncol=3, fontsize=7)
    save(fig, "p3_mpc_weights")


def fig_p3_threshold_grid():
    d = load("phase3_roc_c3.json")
    tab = d["table"]
    mus = sorted({t["mu_iso"] for t in tab})
    ncs = sorted({t["n_c"] for t in tab})
    ch = d["chosen"]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.9), sharey=True)
    kinds = (("break", "Breaks (8 runs)"), ("loe", "Loss of effectiveness (16)"), ("bias", "Bias (16)"))
    for ax, (kind, title) in zip(axes, kinds):
        grid = np.full((len(ncs), len(mus)), np.nan)
        flag = np.zeros_like(grid, bool)
        for t in tab:
            i, j = ncs.index(t["n_c"]), mus.index(t["mu_iso"])
            e = t[kind]
            grid[i, j] = 1e3 * e["median_delay"] if e["median_delay"] is not None else np.nan
            flag[i, j] = t["false_alarm_runs"] > 0 or e["correct"] < e["n"] or e["declared"] < e["n"]
        vmax = np.nanmax(grid)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        for i in range(len(ncs)):
            for j in range(len(mus)):
                v = grid[i, j]
                txt = f"{int(np.floor(round(v, 3) + 0.5))}" if np.isfinite(v) else "-"  # half up, 41.5 -> 42
                col = "white" if np.isfinite(v) and v > 0.55 * vmax else INK
                ax.text(j, i, txt + ("*" if flag[i, j] else ""), ha="center", va="center", fontsize=6.5, color=col)
        jc, ic = mus.index(ch["mu_iso"]), ncs.index(ch["n_c"])
        ax.add_patch(plt.Rectangle((jc - 0.5, ic - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
        ax.set_xticks(range(len(mus)), [f"{m:g}" for m in mus], rotation=90)
        ax.set_yticks(range(len(ncs)), [str(n) for n in ncs])
        ax.set_title(title)
        ax.set_xlabel("Isolation threshold mu_iso")
        ax.grid(False)
    axes[0].set_ylabel("Consecutive cycles N_c")
    fig.text(0.01, -0.14, "Cells: median declaration delay (ms). * : a false alarm in the fault-free runs, a missed "
             "declaration or a wrong isolation. Box: chosen setting (mu_iso 0.9999, N_c 2).",
             fontsize=7, color=INK2, ha="left")
    save(fig, "p3_threshold_grid")


def _runs_c3():
    return load("phase3_calib/c3/runs.json.gz")["runs"]


def fig_p3_mode_probability():
    runs = _runs_c3()
    n = 8
    kinds = (("break", "Breaks"), ("loe", "Loss of effectiveness"), ("bias", "Bias"))
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.5), sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        sel = [r for r in runs if r["spec"]["fault"] and r["spec"]["fault"][0] == kind]
        for r in sorted(sel, key=lambda r: r["lost"]):
            kind_f, cab, onset, _ = r["spec"]["fault"]
            j = 1 + cab if kind_f == "break" else 1 + n + cab
            t = np.array(r["t_imm"]) - onset
            mu = np.array(r["mu"])
            m = (t >= -0.05) & (t <= 0.3)
            ax.plot(1e3 * t[m], mu[m, j], color=C[1] if r["lost"] else C[0], lw=0.7, alpha=0.75)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{title} ({len(sel)})")
        ax.set_xlabel("Time after the fault (ms)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Probability of the true mode")
    fig.legend([Line2D([], [], color=C[0]), Line2D([], [], color=C[1])],
               ["platform kept", "platform lost later (these runs have no accommodation)"],
               loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    save(fig, "p3_mode_probability")


def fig_p3_fault_free_probability():
    d = load("phase3_calib/c3/runs.json.gz")
    t_arm = d["meta"]["imm"]["t_arm"]
    runs = [r for r in d["runs"] if not r["spec"]["fault"]]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.4))
    for r in sorted(runs, key=lambda r: r["trajectory"]):
        t = np.array(r["t_imm"])
        mu = np.array(r["mu"])
        top = mu[:, 1:].max(axis=1)
        m = (t >= t_arm) & (top > 0)  # fault modes are held at probability zero until armed
        col = C[0] if r["trajectory"] == "T1" else C[1]
        ax.plot(t[m], top[m], color=col, lw=0.6, alpha=0.7)
    ax.axhline(0.9999, color=INK2, lw=0.8, ls=":")
    ax.text(16, 0.25, "declaration threshold 0.9999", fontsize=7, color=INK2, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 3)
    ax.set_xlim(0, 16.2)
    ax.axvspan(0, t_arm, color=GRID, alpha=0.6, lw=0)
    ax.text(t_arm / 2, 1e-3, "fault modes\nnot armed", fontsize=7, color=INK2, ha="center")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Largest fault-mode probability")
    ax.legend([Line2D([], [], color=C[0]), Line2D([], [], color=C[1])], ["T1 (8 runs)", "T2 (8 runs)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "p3_fault_free_probability")


def fig_p3_final_check():
    d = load("phase3_check_final.json")["runs"]
    kinds = (("break", "Breaks"), ("loe", "Loss of\neffectiveness"), ("bias", "Bias"))
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.3), gridspec_kw={"wspace": 0.3})
    rng = np.random.default_rng(0)  # horizontal jitter of the dots only
    for i, (kind, _) in enumerate(kinds):
        sel = [r for r in d if r["fault"] and r["fault"][0] == kind]
        x = i + rng.uniform(-0.15, 0.15, len(sel))
        delay = np.array([1e3 * r["delay_s"] for r in sel])
        out = np.array([r.get("post_outside_bounds_pct", 0.0) for r in sel])
        for ax, y in zip(axes, (delay, out)):
            ax.scatter(x, y, s=14, color=C[0], edgecolor="white", lw=0.5, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color=INK, lw=1.0, zorder=4)
    for ax, lab in zip(axes, ("Declaration delay (ms)", "Post-fault time out of bounds (%)")):
        ax.set_xticks(range(3), [k[1] for k in kinds])
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=-0.5 if "%" in lab else 0)
        ax.grid(axis="x", visible=False)
    axes[0].legend([Line2D([], [], color=INK, lw=1.0)], ["median"], loc="upper left")
    save(fig, "p3_final_check")


# ---------------------------------------------------------------------------------------------
# Phase 4: learned fault classifier (training and validation; the test figures are marked)

MODE_LABELS = ["H"] + [f"B{i}" for i in range(1, 9)] + [f"A{i}" for i in range(1, 9)]


def fig_p4_search():
    d = load("phase4_search.json")
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.4))
    for arch, col in (("cnn", C[0]), ("lstm", C[1])):
        done = [t for t in d["trials"] if t["params"]["arch"] == arch and t["state"] == "complete"]
        pruned = [t for t in d["trials"] if t["params"]["arch"] == arch and t["state"] == "pruned"]
        if done:
            ax.scatter([t["number"] for t in done], [t["value"] for t in done], s=18, color=col, edgecolor="white",
                       lw=0.5, zorder=3, label=f"{arch.upper()}, completed")
        if pruned:
            ax.scatter([t["number"] for t in pruned], [t["last_value"] for t in pruned], s=18, facecolor="none",
                       edgecolor=col, lw=0.9, zorder=3, label=f"{arch.upper()}, pruned (last value)")
    b = d["best"]
    ax.scatter([b["number"]], [b["value"]], s=60, marker="*", color=INK, zorder=4, label="best trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation macro-F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
    save(fig, "p4_search")


def fig_p4_learning_curves():
    d = load("phase4_training.json")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.3), gridspec_kw={"wspace": 0.28})
    for seed, info in d["seeds"].items():
        h = info["history"]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], color=C[0], lw=0.9, alpha=0.8)
        axes[0].plot(ep, [r["val_loss"] for r in h], color=C[1], lw=0.9, alpha=0.8)
        axes[1].plot(ep, [r["val_macro_f1"] for r in h], color=C[0], lw=0.9, alpha=0.8)
        axes[1].scatter([info["best_epoch"]], [info["best_val_macro_f1"]], s=16, color=INK, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Class-weighted cross-entropy")
    axes[0].legend([Line2D([], [], color=C[0]), Line2D([], [], color=C[1])], ["training", "validation"], loc="upper right")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro-F1")
    axes[1].legend([Line2D([], [], color=C[0]), Line2D([], [], ls="", marker="o", color=INK, ms=4)],
                   ["five seeds", "best epoch (kept)"], loc="lower right")
    save(fig, "p4_learning_curves")


def fig_p4_size_curve():
    d = load("phase4_training.json")["size_curve"]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    runs = sorted({r["runs"] for r in d})
    for n in runs:
        v = [r["best_val_macro_f1"] for r in d if r["runs"] == n]
        ax.scatter([n] * len(v), v, s=16, color=C[0], edgecolor="white", lw=0.5, zorder=3)
    ax.plot(runs, [np.mean([r["best_val_macro_f1"] for r in d if r["runs"] == n]) for n in runs], color=INK, lw=1.0)
    ax.set_xlabel("Training runs")
    ax.set_ylabel("Validation macro-F1")
    ax.legend([Line2D([], [], ls="", marker="o", color=C[0], ms=4), Line2D([], [], color=INK)],
              ["seeds 0-2", "mean"], loc="lower right")
    save(fig, "p4_size_curve")


def fig_p4_thresholds():
    d = load("phase4_thresholds.json")
    seed = str(d["deployed_seed"])
    tab, ch = d["seeds"][seed]["table"], d["seeds"][seed]["chosen"]
    mus = sorted({t["mu_iso"] for t in tab})
    ncs = sorted({t["n_c"] for t in tab})
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.9), sharey=True)
    for ax, (kind, title) in zip(axes, (("break", "Breaks"), ("loe", "Loss of effectiveness"), ("bias", "Bias"))):
        grid = np.full((len(ncs), len(mus)), np.nan)
        flag = np.zeros_like(grid, bool)
        n = None
        for t in tab:
            i, j = ncs.index(t["n_c"]), mus.index(t["mu_iso"])
            e = t[kind]
            n = e["n"]
            grid[i, j] = 1e3 * e["median_delay"] if e["median_delay"] is not None else np.nan
            flag[i, j] = t["false_alarm_runs"] > 0 or e["correct"] < e["n"]
        vmax = np.nanmax(grid)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        for i in range(len(ncs)):
            for j in range(len(mus)):
                v = grid[i, j]
                txt = f"{int(np.floor(round(v, 3) + 0.5))}" if np.isfinite(v) else "-"
                col = "white" if np.isfinite(v) and v > 0.55 * vmax else INK
                ax.text(j, i, txt + ("*" if flag[i, j] else ""), ha="center", va="center", fontsize=5.6, color=col)
        jc, ic = mus.index(ch["mu_iso"]), ncs.index(ch["n_c"])
        ax.add_patch(plt.Rectangle((jc - 0.5, ic - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
        ax.set_xticks(range(len(mus)), [f"{m:g}" for m in mus], rotation=90)
        ax.set_yticks(range(len(ncs)), [str(x) for x in ncs])
        ax.set_title(f"{title} ({n})")
        ax.set_xlabel("Isolation threshold mu_iso")
        ax.grid(False)
    axes[0].set_ylabel("Consecutive cycles N_c")
    fig.text(0.01, -0.2, textwrap.fill("Validation runs (number of fault runs in brackets), deployed model. Cells: median declaration delay (ms). "
             f"* : a false alarm, a missed declaration or a wrong isolation. Box: chosen setting (mu_iso {ch['mu_iso']:g}, N_c {ch['n_c']}).", 140),
             fontsize=7, color=INK2, ha="left", va="top")
    save(fig, "p4_thresholds")


def _confusion(ax, cm, title):
    cm = np.asarray(cm, float)
    rows = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(rows, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(17), MODE_LABELS, rotation=90, fontsize=6)
    ax.set_yticks(range(17), MODE_LABELS, fontsize=6)
    ax.set_xlabel("Most probable mode")
    ax.set_title(title)
    ax.grid(False)
    return im


def fig_p4_test_confusion():
    d = load("phase4_test.json")
    seed = str(d["deployed_seed"])
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, 6.6), gridspec_kw={"wspace": 0.25, "hspace": 0.45})
    for row, (key, name) in enumerate((("test", "T1 test runs"), ("test_T2", "T2 runs"))):
        s = d["splits"][key]
        _confusion(axes[row, 0], s["seeds"][seed]["window"]["confusion"], f"Classifier, {name}")
        im = _confusion(axes[row, 1], s["imm_window"]["confusion"], f"IMM, {name}")
        axes[row, 0].set_ylabel("True mode")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.03)
    cb.set_label("Share of the cycles of the true mode")
    cb.outline.set_visible(False)
    fig.text(0.01, 0.0, textwrap.fill("TEST RESULT. Share of the estimator cycles of each true mode (rows sum to 1), cycles from 2 s "
             "to the end of each run. T2: path not seen in training. H: healthy; Bi: break of cable i; Ai: actuator fault of winch i.", 140),
             fontsize=7, color=INK2, ha="left", va="top")
    save(fig, "p4_test_confusion")


def fig_p4_test_delays():
    d = load("phase4_test.json")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.5), sharey=True)
    rng = np.random.default_rng(0)  # horizontal jitter of the dots only
    for ax, (key, name) in zip(axes, (("test", "T1 test runs"), ("test_T2", "T2 runs"))):
        s = d["splits"][key]
        for i, kind in enumerate(("break", "loe", "bias")):
            for off, rows, col, lab in ((-0.17, s["deployed_rows"], C[0], "classifier"), (0.17, s["imm_rows"], C[1], "IMM")):
                v = [1e3 * r["delay"] for r in rows if r["fault"] and r["fault"][0] == kind and r.get("correct")]
                x = i + off + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(x, v, s=9, color=col, edgecolor="white", lw=0.3, zorder=3, label=lab if i == 0 else None)
                if v:
                    ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color=INK, lw=1.0, zorder=4)
        ax.set_xticks(range(3), ["Breaks", "Loss of\neffectiveness", "Bias"])
        ax.set_title(name)
        ax.set_yscale("log")
        ax.set_yticks([10, 20, 50, 100, 200, 500], ["10", "20", "50", "100", "200", "500"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Delay of correct declarations (ms)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=1.0))
    lab.append("median")
    axes[1].legend(h, lab, loc="upper right", fontsize=7)
    fig.text(0.01, -0.1, textwrap.fill("TEST RESULT. Correct declarations only. Each model with its own thresholds (classifier: "
             "validation runs; IMM: frozen Phase 3 values).", 140), fontsize=7, color=INK2, ha="left", va="top")
    save(fig, "p4_test_delays")


FIGURES = {
    "p1_circle_radius": (fig_p1_circle_radius, ["phase1_trajectory_design.json"],
                         "Largest admissible circle radius against the period of one turn, with the limit that "
                         "binds; the selected references T1 and T2 use 0.5 m and 4 s."),
    "p1_postfault_workspace": (fig_p1_postfault_workspace, ["phase1_workspace.json"],
                               "Share of the nominal static workspace that the seven remaining cables still hold "
                               "after the loss of one cable, for three lower tension bounds and two orientation sets."),
    "p2_step_convergence": (fig_p2_step_convergence, ["phase2_step_convergence.json"],
                            "Step halving of the plant integrator in closed loop: largest relative change of an "
                            "evaluation metric between step h and h/2, for the seven Phase 2 scenarios."),
    "p3_library_drop": (fig_p3_library_drop, ["phase3_recovery_library.json"],
                        "Recovery library: planned drop below the height at the break (recovery path, branch 2) "
                        "against the break instant, for every cable on T1 and T2."),
    "p3_library_branch": (fig_p3_library_branch, ["phase3_recovery_library.json"],
                          "Recovery library: planner branch chosen for every cable and break instant."),
    "p3_eso_lag": (fig_p3_eso_lag, ["phase3_eso_check_c2.json"],
                   "Winch observers on the fault-free runs of calibration set c2: RMS error of the tension "
                   "estimate against the lag of the plant tension it is compared with."),
    "p3_mpc_weights": (fig_p3_mpc_weights, ["phase3_mpc_weights.json"],
                       "MPC weight study: RMS position error of each fault-free calibration run (first 8 s) for "
                       "the four weight settings; open circles mark lost platforms."),
    "p3_threshold_grid": (fig_p3_threshold_grid, ["phase3_roc_c3.json"],
                          "Declaration thresholds on calibration set c3: median declaration delay for every pair "
                          "(mu_iso, N_c), with the pairs that fail the selection rule marked."),
    "p3_mode_probability": (fig_p3_mode_probability, ["phase3_calib/c3/runs.json.gz"],
                            "Probability of the true fault mode after the fault in every fault run of calibration "
                            "set c3 (recorded with declarations disabled, so without accommodation)."),
    "p3_fault_free_probability": (fig_p3_fault_free_probability, ["phase3_calib/c3/runs.json.gz"],
                                  "Largest fault-mode probability over time in the 16 fault-free runs of "
                                  "calibration set c3."),
    "p3_final_check": (fig_p3_final_check, ["phase3_check_final.json"],
                       "Frozen controller on the calibration set, declarations on: declaration delay and share of "
                       "the post-fault time with a tension out of bounds, per run (40 fault runs)."),
    "p4_search": (fig_p4_search, ["phase4_search.json"],
                  "Hyperparameter search of the classifier: best validation macro-F1 of every trial (pruned "
                  "trials: last value before pruning), by architecture."),
    "p4_learning_curves": (fig_p4_learning_curves, ["phase4_training.json"],
                           "Training and validation loss and validation macro-F1 against the epoch for the five "
                           "seeds of the final configuration; the kept epoch of each seed is marked."),
    "p4_size_curve": (fig_p4_size_curve, ["phase4_training.json"],
                      "Validation macro-F1 against the number of training runs (seeds 0-2)."),
    "p4_thresholds": (fig_p4_thresholds, ["phase4_thresholds.json"],
                      "Declaration thresholds of the deployed classifier on the validation runs: median delay for "
                      "every pair (mu_iso, N_c), the pairs that fail marked, the chosen pair boxed."),
    "p4_test_confusion": (fig_p4_test_confusion, ["phase4_test.json"],
                          "Test result: confusion matrices of the deployed classifier and of the IMM, per estimator "
                          "cycle, on the T1 test runs and on the T2 runs."),
    "p4_test_delays": (fig_p4_test_delays, ["phase4_test.json"],
                       "Test result: delays of the correct declarations of the classifier and of the IMM, by fault "
                       "type, on the T1 test runs and on the T2 runs."),
}


def write_readme():
    lines = [
        "# Figures",
        "",
        "Drawn by `scripts/make_figures.py` from the stored files in `results/`; `python reproduce.py figures` redraws "
        "them in seconds. Every figure is written as PDF (vector, for LaTeX) and PNG (300 dpi, for viewing). The "
        "figures of the paper will be drawn from the Phase 5 evaluation logs in Phase 6.",
        "",
        "## verification/",
        "",
        "Verification and calibration figures of Phases 1 to 3. None of them is a study result: calibration runs use "
        "seeds from 100000 and are kept apart from the evaluation.",
        "",
        "| Figure | Content | Source in `results/` |",
        "|---|---|---|",
    ]
    rows = lambda pre: [f"| `{folder(n).name}/{n}` | {cap} | {', '.join(f'`{x}`' for x in src)} |"
                        for n, (_, src, cap) in FIGURES.items() if n.startswith(pre)]
    lines += rows(("p1", "p2", "p3"))
    lines += ["", "## phase4/", "",
              "The learned fault classifier (baseline B3), Phase 4 runs (seeds from 300000). The search, learning-curve, "
              "size-curve and threshold figures use the training and validation runs only; the two figures marked as test "
              "results show the single evaluation on the test runs.", "",
              "| Figure | Content | Source in `results/` |", "|---|---|---|"]
    lines += rows(("p4",))
    (ROOT / "figures" / "README.md").write_text("\n".join(lines) + "\n")


def main(names):
    for name in names or FIGURES:
        FIGURES[name][0]()
        print("written", folder(name) / f"{name}.pdf")
    if not names:
        write_readme()


if __name__ == "__main__":
    main(sys.argv[1:])
