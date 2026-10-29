"""Phase 6: the figures of the manuscript, drawn only from stored results.

Every figure reads files of results/ (the condensed campaign of Phase 5, its traces, the tests of
the classifiers, the workspace slices) and the robot configuration; no simulation is run here, and
nothing in a figure is drawn by hand except the block diagram of figure 3 (TikZ). Where a figure
shows a single run, the run is chosen by a rule written in this file before the figure was drawn
(pick_example). Each figure is written as vector PDF and as PNG (300 dpi) to paper/figures/, and
paper/figures/manifest.json records, for every figure, the files it read with their SHA-256
prefixes and the rule of any choice.

Figure standard (paper/README.md):
  - widths of a journal column (3.5 in) or of the page (7.16 in); serif type (STIX, close to Times)
    at 8 pt, never below 7 pt; lines at least 0.75 pt; vector PDF;
  - no titles and no sentences inside a figure: panels are labelled (a), (b), ... and short column
    headings are used only where panels differ by one condition; what a line or a mark means, and
    every count, goes into the caption or a table;
  - axis labels name the quantity and its unit; methods appear by their code (P, A1, ..., B3T),
    defined in the methods table; one colour per method in every figure, chosen so that the colours
    drawn together stay distinguishable under the common forms of colour vision deficiency, and line
    styles or marker fills carry the method or the path as well, so that a grey print stays
    readable; reference values (periods, tension bounds) are thin grey lines.
Usage: python scripts/paper_figures.py [fig01 fig04 ...]      (no argument: all figures)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"

ONE, TWO = 3.5, 7.16
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e6e5df", "#b9b8ae"
REF = "#8a8984"                     # reference lines (periods, bounds)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STYLE = {
    "P": dict(color="#2a78d6", ls="-"),
    "A1": dict(color="#2a78d6", ls="--"),
    "A2": dict(color="#e87ba4", ls="-."),
    "A5": dict(color="#008300", ls="--"),
    "B0": dict(color="#8a8984", ls=":"),
    "B1": dict(color="#3d3d3a", ls=(0, (5, 2))),
    "B2": dict(color="#1baf7a", ls="-."),
    "B3": dict(color="#eb6834", ls="--"),
    "B3T": dict(color="#eb6834", ls=":"),
}
# Codes printed in the manuscript. The online-planner ablation is A5 in the evaluation files (the protocol
# numbered A3 and A4 for two ablations that were not built) and A3 in the paper.
NAME = {"A5": "A3"}
KINDS = [("break", "Cable break"), ("loe", "Loss of effectiveness"), ("bias", "Torque bias")]

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "legend.frameon": False, "legend.handlelength": 2.2, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.titlelocation": "center", "axes.titlepad": 3,
    "figure.constrained_layout.use": True,
})

_MANIFEST: dict = {}
_INPUTS: list = []


# --------------------------------------------------------------------------------------- helpers
def sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def use(path):
    """Register an input file of the figure being drawn and return its path."""
    p = Path(path)
    _INPUTS.append({"file": str(p.relative_to(ROOT)), "sha256_16": sha16(p)})
    return p


def rows():
    with gzip.open(use(RES / "phase5" / "rows.jsonl.gz"), "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def analysis():
    return json.loads(use(RES / "phase5" / "analysis.json").read_text())


def trace(method, run):
    return np.load(use(RES / "phase5" / "traces" / method / f"run{run:05d}.npz"))


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ecdf(values, total=None):
    """Steps of the empirical distribution; with `total`, values missing from the list count as
    never reached (the curve then ends below 1)."""
    v = np.sort(np.asarray(values, float))
    n = len(v) if total is None else total
    x = np.r_[v[0] if len(v) else 0.0, v]
    y = np.r_[0.0, np.arange(1, len(v) + 1) / n]
    return x, y


def label(ax, text, x=-0.02, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def line(ax, m, x, y, **kw):
    kw = {**dict(color=STYLE[m]["color"], ls=STYLE[m]["ls"], label=NAME.get(m, m)), **kw}
    return ax.plot(x, y, **kw)


def save(fig, name, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(OUT / f"{name}.png", metadata={"Software": None})
    plt.close(fig)
    seen, inputs = set(), []
    for i in _INPUTS:
        if i["file"] not in seen:
            seen.add(i["file"])
            inputs.append(i)
    _MANIFEST[name] = {"inputs": inputs, "note": note}
    _INPUTS.clear()
    print("written", name, flush=True)


def rot(q):
    from cdpr_ft.model.kinematics import rot_zyx
    return rot_zyx(np.asarray(q[3:6], float))


def tilt_deg(q):
    return np.array([np.rad2deg(np.arccos(np.clip(rot(x)[2, 2], -1, 1))) for x in q])


def box_edges(b):
    """Edges of the box spanned by the attachment points b (platform frame)."""
    lo, hi = b.min(0), b.max(0)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs(c[i] - c[j]) > 1e-9) == 1]


def draw_platform(ax, q, b, **kw):
    R, p = rot(q), np.asarray(q[:3], float)
    for u, v in box_edges(b):
        w = p + np.c_[u, v].T @ R.T
        ax.plot(w[:, 0], w[:, 1], w[:, 2], **kw)
    # the reference point P, which follows the path, and the post that joins it to the platform
    top = p + R @ np.array([0.0, 0.0, b[:, 2].min()])
    ax.plot([p[0], top[0]], [p[1], top[1]], [p[2], top[2]], **kw)
    ax.scatter(*p, s=5, color=kw.get("color", INK), depthshade=False)


def style_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis._axinfo["grid"]["color"] = GRID
    ax.tick_params(pad=-3, labelsize=7)


# ------------------------------------------------------------------------------- figure 1: task
def fig01():
    from cdpr_ft.analysis.trajectories import reference
    from cdpr_ft.params import load_robot
    use(ROOT / "configs" / "robot_ipanema2_caspr.yaml")
    r = load_robot()
    a, b = r.anchors, r.attachments
    q0 = np.asarray(r.home_pose, float)
    pts = q0[:3] + b @ rot(q0).T
    fig = plt.figure(figsize=(ONE, 2.75), layout="none")
    ax = fig.add_axes([-0.04, 0.06, 1.04, 1.0], projection="3d")
    ax.set_proj_type("ortho")
    lo, hi = a.min(0), a.max(0)
    xs, ys = (lo[0], hi[0]), (lo[1], hi[1])
    for z in (lo[2], hi[2]):
        ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]], [z] * 5, color=AXIS, lw=0.7)
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], [0, hi[2]], color=AXIS, lw=0.7)
    for i in range(r.n):
        up = i < 4
        ax.plot([a[i, 0], pts[i, 0]], [a[i, 1], pts[i, 1]], [a[i, 2], pts[i, 2]],
                color=INK if up else INK2, lw=0.6, ls="-" if up else (0, (3, 1.5)))
        ax.scatter(*a[i], s=6, color=INK, depthshade=False)
        ax.text(a[i, 0] * 1.07, a[i, 1] * 1.07, a[i, 2] + (0.35 if up else -0.35), str(i + 1), fontsize=7,
                ha="center", va="center", color=INK2)
    draw_platform(ax, q0, b, color=INK, lw=0.8)
    for name, ls, col in (("T1", "-", "#256abf"), ("T2", (0, (4, 1.5)), "#184f95")):
        p = reference(name, home=tuple(q0[:3]))
        xyz = p.sample(np.linspace(0, p.duration, 600))[0]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=col, lw=1.0, ls=ls)
    ax.set_xlim(-4.2, 4.2), ax.set_ylim(-3.2, 3.2), ax.set_zlim(0, 5.2)
    ax.set_box_aspect((8.4, 6.4, 5.2))
    ax.view_init(elev=18, azim=-58)
    ax.set_xticks([-4, 0, 4]), ax.set_yticks([-3, 0, 3]), ax.set_zticks([0, 1, 3, 5])
    ax.set_xlabel("$x$ (m)", labelpad=-6), ax.set_ylabel("$y$ (m)", labelpad=-6), ax.set_zlabel("$z$ (m)", labelpad=-7)
    style_3d(ax)
    handles = [Line2D([], [], color=INK, lw=0.6, label="cables 1–4"),
               Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 1.5)), label="cables 5–8"),
               Line2D([], [], color="#256abf", lw=1.0, label="T1"),
               Line2D([], [], color="#184f95", lw=1.0, ls=(0, (4, 1.5)), label="T2")]
    fig.legend(handles=handles, loc="lower center", ncol=4, handlelength=1.8, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig01_robot_task", "geometry of configs/robot_ipanema2_caspr.yaml at the home pose; paths of the "
         "reference point P from cdpr_ft.analysis.trajectories.reference")


# ------------------------------------------------------------------ figure 2: what a fault leaves
def fig02():
    d = np.load(use(RES / "phase6_workspace_slices.npz"))
    safe = np.load(use(RES / "phase3_safe_poses.npz"))
    use(ROOT / "configs" / "robot_ipanema2_caspr.yaml")
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    fig, axes = plt.subplots(2, 3, figsize=(TWO, 3.7), gridspec_kw={"height_ratios": [5, 3]})
    cases = [("all", "All cables", None), ("without_1", "Cable 1 lost", 0), ("without_5", "Cable 5 lost", 4)]
    vmax = 60.0
    im = None
    t = np.linspace(0, 2 * np.pi, 200)
    for col, (case, title, cable) in enumerate(cases):
        for row, (plane, u, v, ul, vl) in enumerate([("z3", d["x"], d["y"], "$x$ (m)", "$y$ (m)"),
                                                    ("y0", d["x"], d["z"], "$x$ (m)", "$z$ (m)")]):
            ax = axes[row, col]
            m = d[f"{plane}_{case}_margin"]
            ok = (m >= 0) & d[f"{plane}_{case}_window"]
            shown = np.where(ok, np.clip(m, 0, vmax), np.nan)
            im = ax.pcolormesh(u, v, shown.T, cmap=cmap, vmin=0, vmax=vmax, shading="nearest", rasterized=True)
            ax.contour(u, v, ok.T.astype(float), levels=[0.5], colors=[SEQ[5]], linewidths=0.6)
            if plane == "z3":
                ax.plot(0.5 * np.cos(t), 0.5 * np.sin(t), color=INK, lw=1.0)
            else:
                ax.plot([-0.5, 0.5], [3.0, 3.0], color=INK, lw=1.0)
                ax.plot([-0.5, -0.5, 0.5, 0.5], [2.5, 3.5, 3.5, 2.5], color=INK, lw=0.6, ls=(0, (3, 1.5)))
            if cable is not None:
                pos = safe[f"pos{cable}"]
                sel = np.isclose(pos[:, 2], 3.0) if plane == "z3" else np.isclose(pos[:, 1], 0.0)
                pp = pos[sel]
                ax.scatter(pp[:, 0], pp[:, 1] if plane == "z3" else pp[:, 2], s=5, facecolors="none",
                           edgecolors=INK, linewidths=0.5, zorder=3)
            ax.set_aspect("equal")
            ax.set_xlim(u[0], u[-1]), ax.set_ylim(v[0], v[-1])
            ax.grid(False)
            ax.set_xlabel(ul)
            if col == 0:
                ax.set_ylabel(vl)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(title)
            label(ax, "(%s)" % "abcdef"[3 * row + col])
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("tension margin (N)")
    cb.outline.set_linewidth(0.4)
    handles = [Line2D([], [], color=INK, lw=1.0, label="T1"),
               Line2D([], [], color=INK, lw=0.6, ls=(0, (3, 1.5)), label="T2"),
               Line2D([], [], marker="o", ls="none", mfc="none", mec=INK, mew=0.5, ms=3, label="safe pose")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "fig02_postfault_workspace", "static wrench-feasible poses at zero orientation in the planes z = 3 m "
         "and y = 0 (scripts/phase6_workspace_slices.py); circles: poses of the planner's safe set "
         "(results/phase3_safe_poses.npz, margin >= 10 N with tilt up to 20 deg) lying in the plane")


# ------------------------------------------------------------- figure 3: architecture (TikZ)
def fig03():
    """The block diagram is TikZ (paper/figures/src/fig03_architecture.tex), compiled here."""
    import os
    import shutil
    import subprocess
    src = OUT / "src"
    tex = use(src / "fig03_architecture.tex")
    use(ROOT / "configs" / "controller_phase3.yaml")
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")      # reproducible PDF bytes
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name], cwd=src, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copy(src / "fig03_architecture.pdf", OUT / "fig03_architecture.pdf")
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(OUT / "fig03_architecture.pdf"),
                    str(OUT / "fig03_architecture")], check=True)
    seen, inputs = set(), []
    for i in _INPUTS:
        if i["file"] not in seen:
            seen.add(i["file"])
            inputs.append(i)
    _MANIFEST["fig03_architecture"] = {"inputs": inputs, "note": "block diagram; every block is a module of the "
                                       "repository, rates from configs/controller_phase3.yaml"}
    _INPUTS.clear()
    print("written fig03_architecture", flush=True)


# ---------------------------------------------------------------- figure 4: one break in closed loop
def pick_example(rs):
    """Rule fixed before drawing: among the break runs of an upper cable whose traces are kept
    (repetition 0: cables 1 to 4 on T1 and T2, eight runs), the run whose declaration delay with P
    is the median, the lower of the two middle runs when ordered by (delay, run index)."""
    cand = sorted([r for r in rs if r["method"] == "P" and r["group"] == "fault" and r["rep"] == 0
                   and r["fault"][0] == "break" and r["fault"][1] < 4], key=lambda r: (r["delay_s"], r["run"]))
    return cand[(len(cand) - 1) // 2]


def fig04():
    from cdpr_ft.params import load_robot
    use(ROOT / "configs" / "robot_ipanema2_caspr.yaml")
    robot = load_robot()
    rs = rows()
    sel = pick_example(rs)
    run, onset, cable = sel["run"], sel["fault"][2], sel["fault"][1]
    true_mode = 1 + cable
    by = {(r["method"], r["run"]): r for r in rs}
    tr = {m: trace(m, run) for m in ("P", "B3", "B2", "B0")}
    delay = {m: by[(m, run)]["delay_s"] for m in ("P", "B3", "B2")}
    t0, t1 = -0.05, 1.5
    fig, axes = plt.subplots(2, 2, figsize=(TWO, 3.9))
    # (a) probability of the true mode until each declaration; vertical lines: declarations
    ax = axes[0, 0]
    for m, src, name in (("P", "mu", "IMM"), ("B3", "mu_alt", "classifier")):
        z = tr[m]
        mu = z[src][:, true_mode]
        tt = z["t_imm"][-len(mu):] - onset
        k = (tt >= -0.01) & (tt <= delay[m] + 1e-9)
        ax.plot(1e3 * tt[k], mu[k], color=STYLE[m]["color"], ls=STYLE[m]["ls"], marker=".", ms=3, label=name)
    for m in ("P", "B2", "B3"):
        ax.axvline(1e3 * delay[m], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=0.7)
    ax.set_xlim(-10, 35), ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("time from the break (ms)"), ax.set_ylabel("probability of the break")
    ax.legend(loc="center left", handlelength=1.8)
    label(ax, "(a)")
    # (b) height of the platform and the reference P follows (the recovery plan after its declaration)
    ax = axes[0, 1]
    zp = tr["P"]
    tt = zp["t"] - onset
    k = (tt >= t0) & (tt <= t1)
    ax.plot(tt[k], zp["q_ref"][k, 2], color=REF, lw=0.9, ls=(0, (4, 2)), label="plan")
    for m in ("B0", "B2", "B3", "P"):
        z = tr[m]
        tt = z["t"] - onset
        k = (tt >= t0) & (tt <= t1)
        line(ax, m, tt[k], z["q"][k, 2], lw=1.1 if m == "P" else 0.9)
        if by[(m, run)]["lost"]:
            ax.plot(tt[k][-1], z["q"][k, 2][-1], marker="x", color=STYLE[m]["color"], ms=5, mew=1.0)
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(x) for x in ("P", "B3", "B2", "B0", "plan")]
    ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper right", handlelength=1.8)
    ax.set_xlim(t0, t1), ax.set_xlabel("time from the break (s)"), ax.set_ylabel("height $z$ (m)")
    label(ax, "(b)")
    # (c) tilt
    ax = axes[1, 0]
    for m in ("B0", "B2", "B3", "P"):
        z = tr[m]
        tt = z["t"] - onset
        k = (tt >= t0) & (tt <= t1)
        y = tilt_deg(z["q"][k])
        line(ax, m, tt[k], y, lw=1.1 if m == "P" else 0.9)
        if by[(m, run)]["lost"]:
            ax.plot(tt[k][-1], y[-1], marker="x", color=STYLE[m]["color"], ms=5, mew=1.0)
    ax.set_xlim(t0, t1), ax.set_xlabel("time from the break (s)"), ax.set_ylabel("tilt (deg)")
    label(ax, "(c)")
    # (d) cable tensions under P
    ax = axes[1, 1]
    tt = zp["t"] - onset
    k = (tt >= t0) & (tt <= t1)
    for i in range(robot.n):
        if i != cable:
            ax.plot(tt[k], zp["tension"][k, i], color="#6da7ec", lw=0.7)
    ax.plot(tt[k], zp["tension"][k, cable], color=INK, lw=1.2)
    for v in (robot.t_min, robot.t_max):
        ax.axhline(v, color=REF, lw=0.7, ls=(0, (4, 2)))
    ax.set_xlim(t0, t1), ax.set_ylim(-5, 220)
    ax.set_xlabel("time from the break (s)"), ax.set_ylabel("cable tension under P (N)")
    handles = [Line2D([], [], color=INK, lw=1.2, label=f"cable {cable + 1}"),
               Line2D([], [], color="#6da7ec", lw=0.7, label="intact cables"),
               Line2D([], [], color=REF, lw=0.7, ls=(0, (4, 2)), label="bounds")]
    ax.legend(handles=handles, loc="center right", handlelength=1.8)
    label(ax, "(d)")
    save(fig, "fig04_example_break", f"run {run} ({sel['trajectory']}, break of cable {cable + 1} at {onset:.3f} s), "
         "chosen by pick_example(): median declaration delay of P among the kept upper-cable break runs")


# ------------------------------------------------------------- figure 5: detection and isolation
def fig05():
    rs = rows()
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 2.2), sharey=True)
    for ax, (kind, title), lab in zip(axes, KINDS, "abc"):
        for m in ("P", "B2", "B3"):
            g = [r for r in rs if r["method"] == m and r["fault"] and r["fault"][0] == kind]
            d = [1e3 * r["delay_s"] for r in g if r["correct"]]
            if d:
                x, y = ecdf(d, total=len(g))
                ax.step(x, y, where="post", color=STYLE[m]["color"], ls=STYLE[m]["ls"], label=m)
            else:
                ax.plot([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], label=m)
        ax.set_xscale("log")
        ax.set_xlim(8, 4000), ax.set_ylim(0, 1.02)
        ax.set_xlabel("declaration delay (ms)")
        ax.set_title(title)
        label(ax, f"({lab})")
    axes[0].set_ylabel("fault runs declared correctly")
    axes[0].legend(loc="lower right", handlelength=1.8)
    save(fig, "fig05_detection_delay", "empirical distribution of the delay of correct declarations over all fault "
         "runs of the campaign (both paths); runs without a correct declaration never reach the curve")


# --------------------------------------------------------------------------- figure 6: false alarms
def dotplot(ax, items, xmax=100):
    """items: (y, method or colour, share, (lo, hi), filled)."""
    for y, col, share, (lo, hi), filled in items:
        c = STYLE[col]["color"] if col in STYLE else col
        ax.plot([100 * lo, 100 * hi], [y, y], color=c, lw=1.0)
        ax.plot(100 * share, y, marker="o" if filled else "s", ms=3.6, color=c, mfc=c if filled else "white",
                mew=0.9, clip_on=False, zorder=3)
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)


def path_legend(fig, a="T1", b="T2"):
    handles = [Line2D([], [], marker="o", ls="none", color=INK2, ms=3.6, label=a),
               Line2D([], [], marker="s", ls="none", color=INK2, mfc="white", ms=3.6, label=b)]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)


def fig06():
    rs = rows()
    methods = ["P", "A1", "B2", "B3", "B3T"]
    fig, ax = plt.subplots(figsize=(ONE, 2.0))
    ypos = np.arange(len(methods))[::-1]
    items = []
    for y, m in zip(ypos, methods):
        for traj, dy in (("T1", 0.17), ("T2", -0.17)):
            g = [r for r in rs if r["method"] == m and r["group"] == "fault_free" and r["trajectory"] == traj]
            k, n = sum(r["false_alarm"] for r in g), len(g)
            items.append((y + dy, m, k / n, wilson(k, n), traj == "T1"))
    dotplot(ax, items)
    ax.set_yticks(ypos), ax.set_yticklabels(methods)
    ax.set_xlabel("fault free runs with a false alarm (%)")
    path_legend(fig)
    save(fig, "fig06_false_alarms", "fault-free runs of the campaign (100 per path; 30 per path for B3T), share with "
         "a false alarm and 95 % Wilson interval")


# ------------------------------------------------- figure 7: learned against model-based diagnosis
def fig07():
    p4 = json.loads(use(RES / "phase4_test.json").read_text())["splits"]
    p4b = json.loads(use(RES / "phase4b_test.json").read_text())["splits"]
    rs = rows()
    groups = [("Classifier, T1 (offline)", "#b0afa8", lambda sp: p4[sp]["deployed_rows"]),
              ("Classifier, T1+T2 (offline)", "B3", lambda sp: p4b[sp]["deployed_rows"]),
              ("IMM (offline)", "P", lambda sp: p4[sp]["imm_rows"])]
    stats = []
    for name, col, get in groups:
        for sp, traj in (("test", "T1"), ("test_T2", "T2")):
            rr = get(sp)
            free = [r for r in rr if r["fault"] is None]
            fault = [r for r in rr if r["fault"] is not None]
            stats.append((name, col, traj, sum(r["false_alarm"] for r in free), len(free),
                          sum(bool(r.get("correct")) for r in fault), len(fault)))
    for name, m in (("B3 (closed loop)", "B3"), ("P (closed loop)", "P")):
        for traj in ("T1", "T2"):
            free = [r for r in rs if r["method"] == m and r["group"] == "fault_free" and r["trajectory"] == traj]
            fault = [r for r in rs if r["method"] == m and r["group"] == "fault" and r["trajectory"] == traj]
            stats.append((name, m, traj, sum(r["false_alarm"] for r in free), len(free),
                          sum(bool(r.get("correct")) for r in fault), len(fault)))
    names = list(dict.fromkeys(s[0] for s in stats))
    ypos = {n: len(names) - 1 - i for i, n in enumerate(names)}
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.3), sharey=True)
    for ax, idx, xmin in ((axes[0], (3, 4), 0), (axes[1], (5, 6), 40)):
        items = [(ypos[n] + (0.17 if tr == "T1" else -0.17), c, s[idx[0]] / s[idx[1]], wilson(s[idx[0]], s[idx[1]]),
                  tr == "T1") for s in stats for n, c, tr in [(s[0], s[1], s[2])]]
        dotplot(ax, items)
        ax.set_xlim(xmin, 100)
    axes[0].set_yticks(list(ypos.values())), axes[0].set_yticklabels(list(ypos.keys()))
    axes[0].set_xlabel("fault free runs with a false alarm (%)")
    axes[1].set_xlabel("fault runs declared correctly (%)")
    for ax, lab in zip(axes, "ab"):
        label(ax, f"({lab})")
    path_legend(fig)
    save(fig, "fig07_learned_vs_model", "offline: the single test of Phase 4 (results/phase4_test.json: T1-trained "
         "classifier and IMM) and of Phase 4b (results/phase4b_test.json), 8 fault-free and 80 fault runs per path; "
         "closed loop: the campaign, 100 fault-free and 312 fault runs per path; 95 % Wilson intervals")


# ---------------------------------------------------------------------- figure 8: safety after a break
def fig08():
    s = analysis()["summary"]
    methods = ["P", "B1", "B3", "A5", "A2", "B2", "B0"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.5), sharey=True)
    ypos = np.arange(len(methods))[::-1]
    items = []
    for y, m in zip(ypos, methods):
        for grp, dy in (("upper", 0.17), ("lower", -0.17)):
            g = s[f"{m}|all|break|{grp}"]
            items.append((y + dy, m, g["lost"][0] / g["n"], tuple(g["lost"][1]), grp == "upper"))
    dotplot(axes[0], items)
    axes[0].set_yticks(ypos), axes[0].set_yticklabels([NAME.get(m, m) for m in methods])
    axes[0].set_xlabel("break runs with the platform lost (%)")
    label(axes[0], "(a)")
    items = []
    for y, m in zip(ypos, methods):
        for grp, dy in (("upper", 0.17), ("lower", -0.17)):
            med, (lo, hi), n = s[f"{m}|all|break|{grp}"]["transient_outside_pct"]
            items.append((y + dy, m, med / 100, (lo / 100, hi / 100), grp == "upper"))
    dotplot(axes[1], items)
    axes[1].set_xlabel("time outside the tension bounds (%)")
    label(axes[1], "(b)")
    path_legend(fig, "upper cable broken", "lower cable broken")
    save(fig, "fig08_safety_after_break", "(a) share of the 104 upper- and 104 lower-cable break runs (both paths) in "
         "which the platform was lost, 95 % Wilson intervals; (b) median share of the 2 s after the break with an "
         "intact cable outside [t_min, t_max], 95 % bootstrap intervals")


# -------------------------------------------------------------- figure 9: recovery and re-planning
def fig09():
    rs = rows()
    fig, ax = plt.subplots(figsize=(ONE, 2.1))
    for m in ("B1", "P", "A5", "A2"):
        g = [r for r in rs if r["method"] == m and r["fault"] and r["fault"][0] == "break"]
        rec = [r["recovery_time_s"] for r in g if r.get("recovery_time_s") is not None]
        x, y = ecdf(rec, total=len(g))
        ax.step(np.r_[x, 3.0], np.r_[y, y[-1]], where="post", color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                label=NAME.get(m, m))
    ax.set_xlim(0, 3.0), ax.set_ylim(0, 1.0)
    ax.set_xlabel("time from the break (s)")
    ax.set_ylabel("break runs recovered")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), handlelength=1.8)
    save(fig, "fig09_recovery", "recovery: first time after the onset from which the position error stays for 1 s "
         "within twice the 95th percentile of the fault-free runs of the same path (protocol, amendment 1); lost "
         "runs and runs that did not recover before their end (4 s after the onset) never reach the curve")


# -------------------------------------------------------------------- figure 10: the tightening
def fig10():
    rs = rows()
    by = {(r["method"], r["run"]): r for r in rs}
    key = "transient_min_margin_N"
    fig, axes = plt.subplots(1, 2, figsize=(ONE, 1.95), sharey=True)
    lim = (-160, 60)
    for ax, (on, off, title), lab in zip(axes, [("P", "A1", "IMM"), ("B3T", "B3", "Classifier")], "ab"):
        runs = sorted(k for (m, k) in by if m == on and (off, k) in by and by[(on, k)]["fault"]
                      and by[(on, k)].get(key) is not None and by[(off, k)].get(key) is not None)
        x = np.clip([by[(off, k)][key] for k in runs], *lim)
        y = np.clip([by[(on, k)][key] for k in runs], *lim)
        ax.plot(lim, lim, color=REF, lw=0.7)
        ax.scatter(x, y, s=5, color=STYLE[on]["color"], alpha=0.45, linewidths=0, clip_on=False)
        ax.set_xlim(*lim), ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("without tightening (N)")
        label(ax, f"({lab})")
    axes[0].set_ylabel("with tightening (N)")
    save(fig, "fig10_tightening", "smallest tension margin of the intact cables in the transient window of every "
         "fault run, with against without the probability-aware tightening (P against A1, B3T against B3, paired "
         "runs); values below -160 N drawn at -160 N")


# ------------------------------------------------------------------------ figure 11: computation
def fig11():
    rs = rows()
    fig, axes = plt.subplots(1, 2, figsize=(ONE, 1.95))
    ax = axes[0]
    runs = sorted(r["run"] for r in rs if r["method"] == "P" and r["group"] == "fault" and r["rep"] == 0)
    sec = np.concatenate([trace("P", k)["mpc_seconds"] for k in runs])
    x, y = ecdf(1e3 * sec)
    ax.step(x, y, where="post", color=STYLE["P"]["color"], ls=STYLE["P"]["ls"])
    ax.axvline(20, color=REF, lw=0.7, ls=(0, (4, 2)))
    ax.set_xscale("log"), ax.set_xlim(10, 500), ax.set_ylim(0, 1.0)
    ax.set_xlabel("MPC solve time (ms)")
    ax.set_ylabel("share of solves")
    label(ax, "(a)")
    ax = axes[1]
    for m in ("P", "B3"):
        v = [1e3 * r["imm_seconds"]["p99"] for r in rs if r["method"] == m and r.get("imm_seconds")]
        x, y = ecdf(v)
        ax.step(x, y, where="post", color=STYLE[m]["color"], ls=STYLE[m]["ls"], label=m)
    ax.axvline(5, color=REF, lw=0.7, ls=(0, (4, 2)))
    ax.set_xlim(2, 12), ax.set_ylim(0, 1.0)
    ax.set_xlabel("estimator cycle (ms)")
    ax.set_ylabel("share of runs")
    ax.legend(loc="lower right", handlelength=1.8)
    label(ax, "(b)")
    save(fig, "fig11_computation", "(a) every MPC solve of the kept fault-run traces of P (repetition 0, 48 runs); "
         "(b) the 99th percentile of the estimator cycle (observers, IMM and, for B3, the classifier) of every run "
         "of P and B3; two runs shared the two cores of the machine throughout the campaign")


FIGURES = {"fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04, "fig05": fig05, "fig06": fig06,
           "fig07": fig07, "fig08": fig08, "fig09": fig09, "fig10": fig10, "fig11": fig11}


def main():
    from cdpr_ft.provenance import stamp
    names = sys.argv[1:] or list(FIGURES)
    path = OUT / "manifest.json"
    old = json.loads(path.read_text())["figures"] if path.exists() else {}
    for n in names:
        FIGURES[n]()
    old.update(_MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"figures": dict(sorted(old.items())), "provenance": stamp()}, indent=1))


if __name__ == "__main__":
    main()
