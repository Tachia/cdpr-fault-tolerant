"""Phase 7: check the numbers of the manuscript against the stored results.

Every claim below pairs a passage of paper/manuscript/main.tex with the values that the stored results give for
it. The check passes when the passage is found in the manuscript and every value, formatted as the manuscript
prints it, appears in the passage. A passage that is edited without updating this list fails, so the list and
the text move together. Sources: results/phase5/analysis.json and rows.jsonl.gz (the evaluation), the kept
traces of run 1 (the example of figure 4), the Phase 1 and Phase 3 result files (model, workspace, library),
the Phase 4 test file (classifier), results/phase6_workspace_slices.json, the robot configuration, and the
frozen controller configuration for the table of settings.
Writes paper/manuscript/number_check.txt (every passage checked). Usage: python scripts/paper_check_numbers.py
With --ijc the version for the International Journal of Control (paper/ijc/main.tex) is checked and the record
is written to paper/ijc/number_check.txt.
(exit status 1 if any claim fails)
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"


def load(name):
    return json.loads((RES / name).read_text())


def normalise(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    for a, b in (("~", " "), ("\\,", ""), ("\\%", "%"), ("$", ""), ("{", ""), ("}", ""), ("\\textwidth", "")):
        tex = tex.replace(a, b)
    return re.sub(r"\s+", " ", tex)


# the manuscript checked: the Elsevier CAS version (default) or the International Journal of Control
# version (--ijc); passages that exist in one of them only (abstract, highlights) are marked with MS
MS = "ijc" if "--ijc" in sys.argv[1:] else "cas"
MSDIR = ROOT / "paper" / ("ijc" if MS == "ijc" else "manuscript")
TEXT = normalise((MSDIR / "main.tex").read_text())
A = load("phase5/analysis.json")
S = A["summary"]
ROWS = [json.loads(line) for line in gzip.open(RES / "phase5" / "rows.jsonl.gz", "rt")]
PRIMARY = {e["endpoint"]: {c["comparison"]: c for c in e["comparisons"]} for e in A["primary"]}
SEC = A["secondary"]
FAILED, PASSED, LOG = [], 0, []


def LA_ALL_LOST_BEFORE():
    la = load("latency/analysis.json")["conditions"]
    return all(la[d]["lost"] == la[d]["lost_before_onset"] == la[d]["runs"] for d in ("20", "40", "60"))


def claim(passage, *values):
    global PASSED
    passage = re.sub(r"\s+", " ", passage)
    if passage not in TEXT:
        FAILED.append(f"passage not in the manuscript: {passage!r}")
        return
    for v in values:
        if isinstance(v, (bool, np.bool_)):
            if not v:
                FAILED.append(f"condition from the results not met for: {passage!r}")
                return
            continue
        if str(v) not in passage:
            FAILED.append(f"value {v!r} (from the results) not in: {passage!r}")
            return
    PASSED += 1
    LOG.append(passage)


def n(x, nd=0):
    return f"{x:,.{nd}f}" if abs(x) >= 1e4 else f"{x:.{nd}f}"


def pct(k, m, nd=1):
    return f"{100 * k / m:.{nd}f}"


def s(method, path="all", kind="break", group="all"):
    return S[f"{method}|{path}|{kind}|{group}"]


def rows(method, group="fault", kind=None):
    return [r for r in ROWS if r["method"] == method and r["group"] == group and (kind is None or r["fault"][0] == kind)]


# ------------------------------------------------------------------ evaluation: counts that recur
lost = {m: s(m)["lost"][0] for m in ("P", "A1", "A2", "A5", "B0", "B1", "B2", "B3")}
upper = {m: s(m, group="upper")["lost"][0] for m in lost}
lower = {m: s(m, group="lower")["lost"][0] for m in lost}
nbreak = s("P")["n"]
correct = {m: s(m, kind="all_faults")["correct"][0] for m in ("P", "A1", "B2", "B3")}
nfault = s("P", kind="all_faults")["n"]
fa = PRIMARY["false alarms per hour of armed fault-free operation"]
hours = fa["P vs A1"]["a"][1]
fa_P, fa_B2, fa_B3 = fa["P vs A1"]["a"][0], fa["P vs B2"]["b"][0], fa["P vs B3"]["b"][0]
delay_break = {m: round(s(m)["delay_ms_median"][0]) for m in ("P", "B2", "B3")}
max_break = round(s("P")["delay_ms_max"])
x2 = SEC["diagnosis_x_tightening"]["table"]
assert A["rows"] == 5212

nff = sum(1 for r in ROWS if r["method"] == "P" and r["group"] == "fault_free")
if MS == "cas":                     # abstract or highlights of the CAS version
    claim(f"a paired simulation study of {nfault} fault cases and {nff} fault free runs")
if MS == "cas":                     # abstract or highlights of the CAS version
    claim(f"It identified {correct['P']} of the {nfault} faults correctly, declared every cable break within {max_break} ms and lost "
          f"the platform after {lost['P']} of {nbreak} cable breaks, against {lost['B0']} without fault tolerance and {lost['B2']} with "
          f"a classical threshold based pipeline.")
if MS == "cas":                     # abstract or highlights of the CAS version
    claim(f"when its probabilities also weighted the fault scenarios of the controller, the platform was lost in "
          f"{x2['B3T']['lost_after_break'][0]} of {x2['B3T']['lost_after_break'][1]} break runs.")
if MS == "ijc":                     # abstract of the International Journal of Control version
    claim(f"In {nfault} simulated fault runs, the controller declared every break within {max_break} ms and lost the platform "
          f"after {lost['P']} of {nbreak} breaks, against {lost['B0']} without fault tolerance.")
    claim("delayed by one period, the platform fell in every run tested.", LA_ALL_LOST_BEFORE())
claim(f"the platform was lost after every one of {upper['B0']} simulated breaks of an upper cable",
      upper["B0"] == s("B0", group="upper")["n"])
if MS == "cas":                     # abstract or highlights of the CAS version
    claim(f"Recovery plans computed offline keep the platform after {nbreak - lost['P']} of {nbreak} cable breaks")
if MS == "cas":                     # abstract or highlights of the CAS version
    claim(f"isolates every cable break within {max_break} ms")
claim(f"It identified {correct['P']} of {nfault} faults correctly and declared every break within {max_break} ms.")
claim(f"With it the platform was lost after {lost['P']} of {nbreak} breaks, against {lost['A5']} with the same planner solved "
      f"online and {lost['B0']} without fault tolerance.")
claim(f"the weighting came with false alarms in {x2['B3T']['false_alarm_runs']} of "
      f"{x2['B3T']['fault_runs'] + x2['B3T']['fault_free_runs']} runs and with unsuccessful solves of the MPC, and the platform "
      f"was lost in {x2['B3T']['lost_after_break'][0]} of {x2['B3T']['lost_after_break'][1]} break runs.")
claim(f"Counting every method on every run it applies to, B3T on a subset fixed in advance, the evaluation comprises "
      f"{A['rows']:,} simulations.")

# ------------------------------------------------------------------ results: detection
claim(f"The IMM declared {correct['P']} of the {nfault} faults with the correct mode")
claim(f"Its median delay was {delay_break['P']} ms for a break, {round(s('P', kind='loe')['delay_ms_median'][0])} ms for a "
      f"loss of effectiveness and {round(s('P', kind='bias')['delay_ms_median'][0])} ms for a bias, and no break took longer "
      f"than {max_break} ms.")
miss = [r for r in rows("P") if not r["correct"]]
assert len(miss) == 1
mr = miss[0]
fa_lead = mr["fault"][2] - mr["t_declare"]
assert mr["fault"][0] == "loe" and mr["false_alarm"] and mr["declared"] == 9 + mr["fault"][1] and fa_P == 1
claim(f"Its one false alarm, {fa_lead:.1f} s before a loss of effectiveness of the same winch, is also the one fault it did not "
      f"declare correctly.")
dP = PRIMARY["declaration delay of correct declarations"]
claim(f"On the same runs P declared {-dP['P vs B2']['median_diff_ms']:.0f} ms earlier than B2 and {-dP['P vs B3']['median_diff_ms']:.0f} "
      f"ms earlier than B3")
b2_wrong_break = [r for r in rows("B2", kind="loe") if r["declared"] is not None and 1 <= r["declared"] <= 8
                  and not r["false_alarm"]]
assert all(r["fault"][1] < 4 and r["declared"] >= 5 for r in b2_wrong_break)
claim(f"B2 declared every break correctly and no actuator fault, and in {len(b2_wrong_break)} runs with a loss of effectiveness "
      f"of an upper winch it declared a break of a healthy lower cable", s("B2")["correct"][0] == nbreak,
      correct["B2"] == nbreak)
b3 = {k: s("B3", kind=k)["correct"][0] for k in ("break", "loe", "bias")}
claim(f"The classifier of B3 declared all {b3['break']} breaks correctly, {b3['loe']} of the 208 losses of effectiveness and "
      f"{b3['bias']} of the 208 biases.", b3["break"] == 208)
claim(f"The classifier raised {fa_B3} false alarms in the {hours:.2f} h of operation in which P raised one, and its fault free runs "
      f"carried a false alarm in {s('B3', 'T1', 'fault_free')['false_alarm_runs']} of 100 on T1 and in "
      f"{s('B3', 'T2', 'fault_free')['false_alarm_runs']} of 100 on T2")
p4 = load("phase4_test.json")["splits"]["test_T2"]
ff4 = [r for r in p4["deployed_rows"] if r["fault"] is None]
c4 = sum(bool(r.get("correct")) for r in p4["deployed_rows"] if r["fault"] is not None)
claim(f"it raised a false alarm in {sum(r['false_alarm'] for r in ff4)} of 8 fault free test runs and declared {c4} of 80 faults "
      f"correctly", len(ff4) == 8)

# ------------------------------------------------------------------ results: safety and recovery
pl = PRIMARY["platform lost after a break"]
claim(f"lost the platform after {lost['P']} of the {nbreak} breaks ({pct(lost['P'], nbreak)}%), against {lost['B0']} without "
      f"fault tolerance ({pct(lost['B0'], nbreak)}%), {lost['B2']} for B2 ({pct(lost['B2'], nbreak)}%), {lost['B3']} for B3 "
      f"({pct(lost['B3'], nbreak)}%) and {lost['A2']} without replanning ({pct(lost['A2'], nbreak)}%)",
      all(c["p_holm"] < 0.05 for c in pl.values()))
claim(f"All {lost['P']} losses of P followed the break of an upper cable, {s('P', 'T1', group='upper')['lost'][0]} of "
      f"{s('P', 'T1', group='upper')['n']} on T1 and {s('P', 'T2', group='upper')['lost'][0]} of {s('P', 'T2', group='upper')['n']} on T2",
      lower["P"] == 0, upper["P"] == lost["P"])
claim(f"The oracle B1 lost {lost['B1']}, three of them after upper cable breaks", upper["B1"] == 3)
claim(f"a cable was outside its bounds for {s('P')['transient_outside_pct'][0]:.1f}% of the time under P and "
      f"{s('B1')['transient_outside_pct'][0]:.1f}% under the oracle")
claim(f"lost {lost['B2']} platforms, {lower['B2']} of them after a lower cable break")
rp = SEC["replanning"]
claim(f"Without it (A2), {lost['A2']} platforms were lost, {rp['P vs A2']['transient_outside_pct']['median_b']:.1f}% of the "
      f"transient was spent outside the bounds in the median, and the error after the transient was "
      f"{s('A2')['post_rmse_p_mm'][0]:.1f} mm against {s('P')['post_rmse_p_mm'][0]:.1f} mm for P")
claim(f"lost {lost['A5']} platforms against {lost['P']} (paired, p={rp['P vs A5']['lost']['p']:.2f}, a secondary comparison) and "
      f"raised the time outside the bounds in the transient from {s('P')['transient_outside_pct'][0]:.1f} to "
      f"{s('A5')['transient_outside_pct'][0]:.1f}%")
rec = {m: (s(m)["recovered"][0], s(m)["recovery_time_s"][0]) for m in ("P", "B1", "A5", "A2")}
claim(f"in {rec['P'][0]} of the {nbreak} break runs, after {rec['P'][1]:.2f} s in the median")
claim(f"the oracle recovered {rec['B1'][0]} runs after {rec['B1'][1]:.2f} s, A3 {rec['A5'][0]} after {rec['A5'][1]:.2f} s and "
      f"A2 {rec['A2'][0]} after {rec['A2'][1]:.2f} s")
claim(f"With the 0.4 s of latency, nine more platforms were lost, a difference that is suggestive but not significant on its "
      f"own (p={rp['P vs A5']['lost']['p']:.2f}), while the share of the transient outside the bounds quadrupled", lost["A5"] - lost["P"] == 9,
      round(s("A5")["transient_outside_pct"][0] / s("P")["transient_outside_pct"][0]) == 4)
claim(f"the classifier cost {lost['B3'] - lost['P']} platforms more than the IMM, {lost['B3']} against {lost['P']}; used also to "
      f"weight the scenarios of the MPC, it lost the platform in {x2['B3T']['lost_after_break'][0]} of "
      f"{x2['B3T']['lost_after_break'][1]} break runs")

# ------------------------------------------------------------------ results: actuator faults
def lost_split(method, kinds):
    """Platforms lost in the runs with a fault of the given kinds, after and before the onset."""
    rr = [r for r in ROWS if r["method"] == method and r["lost"] and r["fault"] and r["fault"][0] in kinds]
    after = [r for r in rr if r["t_end"] >= r["fault"][2]]
    before = [r for r in rr if r["t_end"] < r["fault"][2]]
    return after, before


b2_after, b2_before = lost_split("B2", ("loe", "bias"))
b3t_after, b3t_before = lost_split("B3T", ("loe", "bias"))
claim(f"except B3T, which lost {len(b3t_after)} after the onset of an actuator fault and {len(b3t_before)} more in such runs "
      f"before the onset")
claim(f"B2, which cannot isolate them, lost {sum(r['fault'][0] == 'loe' for r in b2_after)} platforms after a loss of effectiveness, "
      f"and in {['', 'one', 'two', 'three'][len(b2_before)]} other runs with an actuator fault it lost the platform before the onset, "
      f"after a false alarm.", all(r["false_alarm"] and r["t_declare"] < r["fault"][2] for r in b2_before),
      all(r["fault"][0] == "loe" for r in b2_after))
loeP = rows("P", kind="loe")
by_rho = {rho: [r for r in loeP if r["fault"][3] == rho] for rho in (0.3, 0.6)}
med = lambda xs: float(np.median([x for x in xs if x is not None]))
claim(f"an RMS error of {s('P', kind='loe')['deficit_rmse_N'][0]:.0f} N in the median, "
      f"{med([r['deficit_rmse_N'] for r in by_rho[0.3]]):.0f} N for ρ=0.3 and {med([r['deficit_rmse_N'] for r in by_rho[0.6]]):.0f} N"
      .replace("ρ=", "\\rho="))
claim(f"the error after the transient was {s('P', kind='loe')['post_rmse_p_mm'][0]:.1f} mm, "
      f"{med([r['post_rmse_p_mm'] for r in by_rho[0.3]]):.0f} and {med([r['post_rmse_p_mm'] for r in by_rho[0.6]]):.0f} mm for the two "
      f"magnitudes, so that only {s('P', kind='loe')['recovered'][0]} of the {s('P', kind='loe')['n']} runs returned")
claim(f"After a bias the deficit error was {s('P', kind='bias')['deficit_rmse_N'][0]:.1f} N and {s('P', kind='bias')['recovered'][0]} "
      f"of {s('P', kind='bias')['n']} runs returned to the band.")

# ------------------------------------------------------------------ results: the tightening
pa = {(r["method"], r["run"]): r for r in ROWS}
pairs = [(pa[("P", r["run"])], pa[("A1", r["run"])]) for r in rows("P", kind="break")]
p_only = sum(a["lost"] and not b["lost"] for a, b in pairs)
a_only = sum(b["lost"] and not a["lost"] for a, b in pairs)
claim(f"P and A1 lost the same number of platforms after a break, {lost['P']} each, with the outcome differing in "
      f"{['', '', '', '', '', '', '', '', '', '', 'ten'][p_only + a_only]} runs, {['', '', '', '', '', 'five'][p_only]} each way",
      lost["P"] == lost["A1"], p_only == a_only)
mg = PRIMARY["smallest tension margin in the transient window"]["P vs A1"]
claim(f"(paired median difference {mg['median_diff']:.0f} N, p={mg['p']:.2f})")
allp = [(pa[("P", r["run"])], pa[("A1", r["run"])]) for r in ROWS if r["method"] == "P"]
same_pre = sum(a["pre_rmse_p_mm"] == b["pre_rmse_p_mm"] for a, b in allp)
claim(f"the position error before the fault was identical in {same_pre} of the {len(allp)} paired runs")
# declarations of P and A1, run by run (a comparison made after the evaluation, outside its analysis plan)
fa_p = [r for r in rows("P") if r["false_alarm"]]
fa_a1 = pa[("A1", fa_p[0]["run"])] if len(fa_p) == 1 else None
claim(f"P raised one false alarm, {fa_p[0]['fault'][2] - fa_p[0]['t_declare']:.1f} s before a loss of effectiveness that A1 declared correctly",
      len(fa_p) == 1, fa_p[0]["fault"][0] == "loe", bool(fa_a1 and fa_a1["correct"]))
dd = {k: [round(1e3 * (pa[("P", r["run"])]["delay_s"] - pa[("A1", r["run"])]["delay_s"])) for r in rows("P", kind=k)
          if pa[("P", r["run"])]["correct"] and pa[("A1", r["run"])]["correct"]] for k in ("break", "loe", "bias")}
early = [-x for x in dd["bias"] if x < 0]
claim(f"in {sum(x != 0 for x in dd['bias'])} of the {len(rows('P', kind='bias'))} runs with a torque bias the two declared at different "
      f"times, P earlier in {len(early)} of them, by {min(early)} to {max(early)} ms, and later in two",
      sum(x > 0 for x in dd["bias"]) == 2, not any(dd["break"]), not any(dd["loe"]))
mb = PRIMARY["smallest tension margin in the transient window"]["B3T vs B3"]
claim(f"a false alarm was raised in {x2['B3T']['false_alarm_runs']} of {x2['B3T']['fault_runs'] + x2['B3T']['fault_free_runs']} "
      f"runs against {x2['B3']['false_alarm_runs']} for B3 on the same runs, the correct declarations fell from "
      f"{x2['B3']['correct'][0]} to {x2['B3T']['correct'][0]} of {x2['B3']['correct'][1]}, the platform was lost in "
      f"{x2['B3T']['lost_after_break'][0]} of {x2['B3T']['lost_after_break'][1]} break runs against "
      f"{x2['B3']['lost_after_break'][0]}, in {len(lost_split('B3T', ('break',))[1])} of them before the cable broke, and the "
      f"margin in the transient fell by {-mb['median_diff']:.0f} N (paired median; 95% interval {mb['median_diff_ci'][0]:.1f} to "
      f"{mb['median_diff_ci'][1]:.1f} N)")
# unsuccessful solves: counts stored by the campaign, details recorded by the replay of the lost runs (post-review audit)
AUD = load("audit/lost_runs.json")
fsm = AUD["failed_solves_by_method"]
b3t_rows = [r for r in ROWS if r["method"] == "B3T"]
p_fail = [r for r in ROWS if r["method"] == "P" and r["base_metrics"]["mpc_failures"]]
claim(f"IPOPT returned without success in {sum(r['base_metrics']['mpc_failures'] for r in b3t_rows)} solves in "
      f"{sum(r['base_metrics']['mpc_failures'] > 0 for r in b3t_rows)} of the {len(b3t_rows)} runs of B3T, against one solve in one "
      f"run for P.", len(p_fail) == 1 and p_fail[0]["base_metrics"]["mpc_failures"] == 1)
fb = fsm["B3T"]
claim(f"all unsuccessful ones stopped at the limit of 30 iterations, {fb['scenarios']['3']} of the {fb['failed_solves']} with the "
      f"healthy and two fault scenarios in the programme, and the iterates they returned violated the equality constraints of the "
      f"prediction by {fb['inf_pr_of_returned_iterate']['median']:.2f} in the median and by up to "
      f"{fb['inf_pr_of_returned_iterate']['max']:.0f} (in metres, radians or their rates), against at most "
      f"{fsm['A2']['inf_pr_of_returned_iterate']['max']:.3f} for the {['', '', '', '', '', '', '', '', 'eight'][fsm['A2']['failed_solves']]} "
      f"unsuccessful solves in the lost runs of A2.",
      AUD["row_check"] == {"identical": AUD["runs"]}, AUD["failed_solve_count_equals_stored"],
      all(set(v["statuses"]) <= {"Maximum_Iterations_Exceeded"} and set(v["iterations"]) <= {"30"} for v in fsm.values()))
tdf = fb["by_timing_declaration_failure"]
pre = {k: v for k, v in tdf.items() if k.startswith("lost_before_onset")}
lb = AUD["lost_before_onset_by_method"]["B3T"]
claim(f"Of the {fb['lost_runs']} platforms B3T lost, {sum(pre.values())} were lost before any fault, {lb['fault_free']} of them in "
      f"fault free runs; {sum(v for k, v in pre.items() if k.endswith('|failed_solve'))} of these {sum(pre.values())} runs contained "
      f"an unsuccessful solve, {sum(v for k, v in pre.items() if '|before_onset|' in k)} a false declaration, and "
      f"{sum(v for k, v in pre.items() if '|none|' in k)} no declaration at all.",
      sum(pre.values()) == lb["fault_free"] + sum(lb["fault_runs_lost_before_onset"].values()))
lead = fb["loss_after_first_failure_s"]
claim(f"In each of the {fb['lost_runs_with_failed_solve']} lost runs with an unsuccessful solve the first one came before the loss, "
      f"by {lead['median']:.2f} s in the median, and in {fb['first_failure_with_attitude_below_10_deg']} of them the attitude was "
      f"still below 10^\\circ when it came; in "
      f"{sum(r['base_metrics']['mpc_failures'] > 0 for r in b3t_rows) - fb['lost_runs_with_failed_solve']} other runs the solver "
      f"failed and the platform was kept.", fb["first_failure_before_loss"] == fb["lost_runs_with_failed_solve"], lead["min"] > 0,
      sum(1 for r in b3t_rows if r["base_metrics"]["mpc_failures"] > 0 and not r["lost"])
      == sum(r["base_metrics"]["mpc_failures"] > 0 for r in b3t_rows) - fb["lost_runs_with_failed_solve"])

# ------------------------------------------------------------------ replay of the lost runs (after the review)
trig = {}
for v in AUD["trigger_by_method"].values():
    for k, c in v.items():
        trig[k] = trig.get(k, 0) + c
YAW = load("audit/yaw_runs.json")
yr = YAW["runs"]
kept = [x for x in yr if not x["lost_roll_pitch_rule"]]
cont = [x for x in yr if x["lost_roll_pitch_rule"]]
lead_yaw = max(x["t_roll_or_pitch"] - x["t_stored_stop"] for x in cont)
cmp3 = {c["comparison"]: c for c in YAW["primary_platform_lost_after_a_break"]["three_angles"]}
cmp2 = {c["comparison"]: c for c in YAW["primary_platform_lost_after_a_break"]["roll_or_pitch"]}
p2 = cmp2["P vs B2"]["p_holm"]
claim(f"After the evaluation we simulated again every one of the {AUD['runs']} runs in which a platform was lost",
      AUD["runs"] == sum(r["lost"] for r in ROWS))
claim(f"All {AUD['runs']} rows came out identical to the stored ones.", AUD["row_check"] == {"identical": AUD["runs"]})
claim(f"Roll ended {trig['roll']} of the runs, pitch {trig['pitch']} and yaw {trig['yaw']}. In one of the eight, pitch had passed "
      f"30^\\circ as well; in the other seven, runs of A2 and B2, yaw alone had.",
      trig["yaw"] == 8, len(yr) == 7, {x["method"] for x in yr} == {"A2", "B2"}, YAW["summary"]["all_same_as_stored"])
claim(f"In six of them roll or pitch passed 30^\\circ within {lead_yaw:.2f} s of the stored end, and so did the tilt of the "
      f"platform axis; in the seventh, a break of an upper cable under B2, roll, pitch and tilt stayed below 30^\\circ until the "
      f"run ended.", len(cont) == 6, all(x["t_physical_tilt"] is not None for x in cont), len(kept) == 1,
      kept[0]["method"] == "B2" and kept[0]["fault"][0] == "break" and kept[0]["fault"][1] < 4,
      kept[0]["t_physical_tilt"] is None and kept[0]["other_stop"] is None
      and abs(kept[0]["t_last"] - (kept[0]["fault"][2] + 4.0)) < 1e-9)
claim(f"B2 then loses {cmp2['P vs B2']['b']} platforms after a break instead of {cmp3['P vs B2']['b']}, and its difference from P "
      f"remains after the Holm correction (p={p2 / 10 ** np.floor(np.log10(p2)):.1f}\\times10^{int(np.floor(np.log10(p2)))})",
      cmp3["P vs B2"]["p_holm"] == PRIMARY["platform lost after a break"]["P vs B2"]["p_holm"],
      all(cmp2[k]["b"] == cmp3[k]["b"] and cmp2[k]["a"] == cmp3[k]["a"] for k in cmp2 if k != "P vs B2"))
claim("No other loss changes, and every loss of P ended on roll.", AUD["trigger_by_method"]["P"] == {"roll": lost["P"]})
FS = load("audit/failed_solve_runs.json")
fa = FS["all_runs_with_failed_solve"]
nfs = sum(r["base_metrics"]["mpc_failures"] for r in ROWS)
fsp = [json.loads(x) for x in (RES / "audit" / "failed_solve_runs.jsonl").read_text().splitlines()]
fsp = [x for x in fsp if x["method"] == "P"]
pf = fsp[0]["failures"][0] if len(fsp) == 1 and fsp[0]["failed_solves"] == 1 else None
claim(f"We also replayed the {FS['runs']} runs in which a solve was unsuccessful but the platform was kept, again with identical rows.",
      FS["row_check"] == {"identical": FS["runs"]},
      FS["runs"] == sum(1 for r in ROWS if not r["lost"] and r["base_metrics"]["mpc_failures"] > 0))
claim(f"All {fa['failed_solves']} unsuccessful solves of the evaluation, the number the campaign had stored, stopped at the limit of "
      f"30 iterations. B3T had {fa['by_method']['B3T']['failed_solves']} of them; P had one, {pf['t'] - fsp[0]['fault'][2]:.1f} s after "
      f"the break of an upper cable, in a run in which the platform was kept.",
      fa["failed_solves"] == nfs == FS["stored_failed_solves_total"], fa["count_equals_stored"],
      fa["statuses"] == {"Maximum_Iterations_Exceeded": nfs}, fa["iterations"] == {"30": nfs},
      fsp[0]["fault"][0] == "break" and fsp[0]["fault"][1] < 4)
claim(f"All {fa['failed_solves']} unsuccessful solves of the evaluation stopped at the limit of 30 iterations")

# ------------------------------------------------------------------ latency study (after the review)
LA = load("latency/analysis.json")
LC = LA["conditions"]
LROWS = [json.loads(x) for x in (RES / "latency" / "rows.jsonl").read_text().splitlines()]
LM = load("latency/motion_20ms.json")["summary"]
l20 = [r for r in LROWS if r["delay_ms"] == 20]
from collections import Counter
dec20 = Counter(r["declared"] for r in l20)
lead20 = [r["t_end"] - r["t_declare"] for r in l20]
onsets = [r["fault"][2] for r in l20]
claim(f"With D=0 all {LA['validation']['identical']} rows were identical to those of the evaluation.",
      LA["validation"]["identical"] == LA["validation"]["runs"] == 104, not LA["missing"])
claim(f"With the output one period late, D=20 ms, the platform was lost in all {LC['20']['lost']} runs, and every loss came before "
      f"the cable broke, between {min(r['t_end'] for r in l20):.2f} and {max(r['t_end'] for r in l20):.2f} s after the start, while "
      f"the breaks were due between 3 and 7 s.", LC["20"]["lost"] == LC["20"]["lost_before_onset"] == len(l20) == 104,
      3.0 <= min(onsets) and max(onsets) <= 7.0)
claim(f"The roll of the platform oscillated at about {LM['zero_crossing_hz']['median']:.0f} Hz with an amplitude that grew by a "
      f"factor of {LM['growth']['median']:.0f} in the median over half a second, and every run ended on roll.",
      LM["angle"]["roll"] == LM["lost"] == 104, LM["row_check"] == {"identical": 104},
      abs(LM["zero_crossing_hz"]["q25"] - LM["zero_crossing_hz"]["median"]) < 0.5)
claim(f"In every run the IMM, not yet armed, then declared a fault of an intact cable or winch, a break of an upper cable in "
      f"{sum(v for k, v in dec20.items() if k is not None and 1 <= k <= 4)} of them, and the platform was lost {min(lead20):.2f} to "
      f"{max(lead20):.2f} s later.", all(r["declared"] is not None and r["t_declare"] < 2.0 for r in l20))
claim(f"since in {LM['above_5_deg_before_declaration']} runs roll or pitch had already passed 5^\\circ")
fs40, fs60 = LC["40"]["runs_with_failed_solve"], LC["60"]["runs_with_failed_solve"]
claim(f"With 40 and 60 ms the platform was lost in every run as well, again before the break, and unsuccessful solves appeared in "
      f"{fs40} and {fs60} of the runs. The largest computation time tested without significantly more losses than at D=0 is "
      f"therefore {LA['largest_delay_without_significantly_more_losses_ms']} ms.",
      all(LC[d]["lost"] == LC[d]["lost_before_onset"] == 104 for d in ("40", "60")),
      all(LA["against_0_ms"][d]["lost"]["p_holm"] < 0.05 for d in ("20", "40", "60")))
if MS == "cas":                     # abstract or highlights of the CAS version
    claim("an output delayed by one period of 20 ms made the platform fall before any fault in all 104 runs tested",
          LC["20"]["lost_before_onset"] == 104)
early_eval = sum(1 for r in ROWS if r["t_declare"] is not None and r["t_declare"] < 2.0)
early_lat = sum(1 for r in LROWS if r["delay_ms"] > 0 and r["t_declare"] is not None and r["t_declare"] < 2.0)
claim("this happened in no run of the evaluation and in most runs of the latency study",
      early_eval == 0, early_lat > sum(1 for r in LROWS if r["delay_ms"] > 0) / 2)
claim("with the output of the MPC delayed by one period the platform fell in every run tested")

# ------------------------------------------------------------------ results: computation
cP = S["P|all|all_faults|all"]
claim(f"took {cP['imm_ms']['mean_median']:.1f} ms on average and {cP['imm_ms']['p99_median']:.1f} ms at the 99th percentile, "
      f"against its period of 5 ms.")
claim(f"The MPC took {cP['mpc_ms']['mean_median']:.0f} ms on average and {cP['mpc_ms']['p99_median']:.0f} ms at the 99th percentile")
claim(f"The MPC needed {cP['mpc_ms']['mean_median']:.0f} ms on average against its period of 20 ms")

# ------------------------------------------------------------------ the example run (figure 4)
import paper_figures as pf  # noqa: E402

sel = pf.pick_example(ROWS)
run, onset, cable = sel["run"], sel["fault"][2], sel["fault"][1]
by = {(r["method"], r["run"]): r for r in ROWS}
dl = {m: round(1e3 * by[(m, run)]["delay_s"]) for m in ("P", "B2", "B3")}
claim(f"Cable {cable + 1} breaks at {onset:.3f} s on {sel['trajectory']}.")
zP, zB3, zB2 = pf.trace("P", run), pf.trace("B3", run), pf.trace("B2", run)
mu = zP["mu"][:, 1 + cable]
ti = zP["t_imm"][-len(mu):] - onset
mu3 = zB3["mu_alt"][:, 1 + cable]
t3 = zB3["t_imm"][-len(mu3):] - onset
first = lambda t, m, thr: round(1e3 * t[np.argmax((m >= thr) & (t >= 0))])
claim(f"The IMM gives the break mode a probability above 0.99 in the cycle {first(ti, mu, 0.99)} ms after the break and "
      f"declares it after {dl['P']} ms.")
claim(f"The classifier's probability rises earlier, to 0.99 at {round(1e3 * t3[np.argmax(mu3 >= 0.98)])} ms, but both reach "
      f"the threshold of 0.9999 in the same cycle, {first(ti, mu, 0.9999)} ms after the break; the IMM needs two such cycles and "
      f"declares after {dl['P']} ms, the classifier needs four and declares after {dl['B3']} ms, and the threshold rule of B2 "
      f"declares at {dl['B2']} ms.", first(ti, mu, 0.9999) == first(t3, mu3, 0.9999))
tt = zP["t"] - onset
k = (tt >= 0) & (tt <= 1.5)
tilt = pf.tilt_deg(zP["q"][k])
drop = zP["q_ref"][tt >= 0][0, 2] - zP["q_ref"][k][-1, 2]
claim(f"Under P it descends by {drop:.1f} m along the recovery plan to a pose the seven remaining cables can hold, tilting by "
      f"at most {tilt.max():.1f}° and settling at {tilt[-1]:.1f}°".replace("°", "^\\circ"))
intact = [i for i in range(8) if i != cable]
w = (tt >= 0) & (tt < 2.0)
T = zP["tension"][w][:, intact]
out = ((T < 10) | (T > 200)).any(axis=1)
T3 = zP["tension"][tt >= 0.3][:, intact]
claim(f"an intact cable leaves its bounds for {100 * out.mean():.0f}% of the first 2 s, all of it within "
      f"{tt[w][out].max():.2f} s of the break, and from 0.3 s on every intact tension lies between {T3.min():.0f} and "
      f"{T3.max():.0f} N")
claim(f"Without fault tolerance the platform turns past 30^\\circ and is lost {by[('B0', run)]['t_end'] - onset:.2f} s after the break.",
      by[("B0", run)]["lost"])
# ------------------------------------------------------------------ model, workspace and paths (Phases 1, 3 and 6)
c = yaml.safe_load((ROOT / "configs" / "robot_ipanema2_caspr.yaml").read_text())
anch, att = np.array(c["robot"]["anchors"]), np.array(c["robot"]["attachments"])
sx, sy = np.sign(anch[:, 0]), np.sign(anch[:, 1])
assert np.allclose(anch[:, 0], 4 * sx) and np.allclose(anch[:, 1], 3 * sy) and np.allclose(sx, np.sign(att[:, 0]))
assert np.allclose(anch[:4, 2], 5) and np.allclose(anch[4:, 2], 1)
assert np.allclose(np.abs(att[:4]), [0.65, 0.125, 0.25]) and np.allclose(np.abs(att[4:]), [0.75, 0.10, 0.75])
assert list(zip(sx[:4], sy[:4])) == [(-1, 1), (1, 1), (1, -1), (-1, -1)] == list(zip(sx[4:], sy[4:]))
claim("(0.65σx,i,0.125σy,i,0.25) m for the upper and (0.75σx,i,0.10σy,i,0.75) m for the lower cables".replace("σ", "\\sigma_")
      .replace("\\sigma_x,i", "\\sigma_x,i").replace(",0.125", ",0.125"), c["robot"]["mass"] == 10.0)
cab = load("phase1_cable_analysis.json")
claim(f"which gives EA={cab['cases']['tempel2015_iftomm_scaled']['EA_N'] / 1e3:.1f} kN", c["cable"]["EA"] == cab["cases"]["tempel2015_iftomm_scaled"]["EA_N"])
claim(f"acts like a mass of I_m/r_d^2={cab['winch']['rotor_equivalent_cable_mass_kg']:.1f} kg")
ws = load("phase1_workspace.json")["cases"]["tmin10_zero"]
claim(f"the eight cables hold the level platform at {ws['n_nominal']:,} positions within the drum windows. After the loss of one "
      f"upper cable {100 * ws['post_fault_fraction'][0]:.1f}% of these positions remain, and after the loss of a lower cable "
      f"{100 * ws['post_fault_fraction'][4]:.1f}%.")
sl = load("phase6_workspace_slices.json")["summary"]
claim(f"{sl['z3_all']['feasible']:,} of {sl['z3_all']['points']:,} points of a 0.05 m grid are feasible with all cables, "
      f"{sl['z3_without_1']['feasible']} after the loss of the upper cable 1 and {sl['z3_without_5']['feasible']:,} after the loss of the lower cable 5")
po = load("phase1_postfault_orientation.json")
claim(f"{100 * po['fraction_feasible_with_tilt'][0]:.1f}% of the positions can be held after an upper cable breaks and "
      f"{100 * po['fraction_feasible_with_tilt'][4]:.1f}% after a lower one, with a median tilt of {po['median_tilt_deg_when_needed'][0]:.0f}^\\circ")
tr = load("phase1_trajectory_design.json")["selected"]["T1_circle"]
claim(f"The peak speed of {tr['max_speed_m_s']:.2f} m/s is a third of the speed limit", tr["radius_m"] == 0.5, tr["period_s"] == 4.0)
lib = load("phase3_recovery_library.json")
cases = lib["cases"]
sol = [x["solver_seconds"] for x in cases if x["branch"] == 2]
claim(f"Solving (eq:recovery) took {np.median(sol):.2f} s in the median and at most {max(sol):.2f} s on one core"
      .replace("(eq:recovery)", "\\eqrefeq:recovery"))
claim(f"applies after a latency of 0.4 s, which covers the 99th percentile of the solve times", np.percentile(sol, 99) <= 0.4)
pc = lib["per_cable"]
dmed = [np.median([x["drop_m"] for x in cases if x["cable"] == i]) for i in range(4)]
b1 = [pc[str(i)]["branch_counts"]["1"] for i in range(5, 9)]
dlow = max(x["drop_m"] for x in cases if x["cable"] >= 4 and x.get("drop_m") is not None)
assert all(pc[str(i)]["branch_counts"]["1"] == 0 for i in range(1, 5)) and all(v["branch_counts"]["3"] == 0 for v in pc.values())
claim(f"with a median drop of the reference point of {min(dmed):.2f} to {max(dmed):.2f} m depending on the cable, whereas after a "
      f"lower cable break the time scaling sufficed for {min(b1)} to {max(b1)} of the 222 instants of both paths and the recovery plans dropped "
      f"by at most {dlow:.2f} m")
drift = max(x["drift_end_m"] for x in cases if x["branch"] == 2 and x["cable"] < 4)
claim(f"drifts from a recovery path after an upper cable break by up to {drift:.2f} m by its end")
rec2 = [x for x in cases if x["branch"] == 2]
st = [x["solver_status"] for x in rec2]
claim(f"All {len(rec2):,} recovery problems of the library met the termination criteria of IPOPT, {st.count('Solve_Succeeded'):,} at its "
      f"optimal and {st.count('Solved_To_Acceptable_Level')} at its acceptable level, with every slack zero",
      st.count("Solve_Succeeded") + st.count("Solved_To_Acceptable_Level") == len(rec2), max(x["max_slack"] for x in rec2) <= 1e-6)
claim(f"the plans deviate from their collocation solution by at most {1e3 * max(v['local_error_max_m'] for v in pc.values()):.2f} mm")
pm = load("phase3_postfault_modes.json")["per_cable"]
claim(f"the lowest elastic mode of the platform falls to {min(v['lowest_Hz_min'] for v in pm.values()):.1f} to "
      f"{max(v['lowest_Hz_max'] for v in pm.values()):.1f} Hz")
eso = load("phase3_eso_check_c2.json")["bandwidths"]["50"]
claim(f"RMS error of {eso['noise_x1']['rms_N_at_lag']['0']:.2f} N at the nominal current noise and "
      f"{eso['noise_x2']['rms_N_at_lag']['0']:.2f} N at twice that noise, at the price of a lag of 3/ω_o=9.5 ms"
      .replace("ω_o", "\\omega_o"), abs(3 / (2 * np.pi * 50) * 1e3 - 9.5) < 0.05)

case = cab["cases"]["tempel2015_iftomm_scaled"]["tmin10"]
claim(f"At the home pose the six elastic modes of the platform with locked winches then lie between "
      f"{min(case['home_modes_straight_Hz']):.1f} and {max(case['home_modes_straight_Hz']):.1f} Hz.")
claim(f"never falls below {100 * case['workspace_sag']['min_EAeq_over_EA']:.1f}% of EA")
trs = load("phase1_trajectory_design.json")["selected"]
cov = [x for k in ("T1_circle", "T2_helix") for x in trs[k]["post_fault_static_coverage"][4:]]
assert all(x == 0 for k in ("T1_circle", "T2_helix") for x in trs[k]["post_fault_static_coverage"][:4])
claim(f"while after a lower cable breaks it can on {100 * min(cov):.0f} to {100 * max(cov):.0f}% of the path")
claim(f"rotor of {1e4 * c['winch']['inertia']:.1f} kg cm^2 acts like a mass".replace("kg cm^2", "kgcm^2"))
claim(f"vertical lines: declarations of P ({dl['P']} ms), B2 ({dl['B2']} ms) and B3 ({dl['B3']} ms)")
claim(f"A break of the upper cable {cable + 1} at {onset:.3f} s on {sel['trajectory']} (run {run} of the evaluation)")
claim(f"declared every break within {max_break} ms and raised one false alarm in {hours:.2f} h. With recovery plans computed in "
      f"advance along the task the platform was lost after {lost['P']} of {nbreak} breaks, all of them upper cable breaks, against "
      f"{lost['B0']} without fault tolerance and {lost['A5']} when the same plans were solved online.")
claim(f"(a) IMM (P against A1, {mg['pairs']} runs): the points lie on the diagonal. (b) Classifier (B3T against B3, {mb['pairs']} runs)")

# ------------------------------------------------------------------ settings table against the frozen configuration
cfg = yaml.safe_load((ROOT / "configs" / "controller_phase3.yaml").read_text())
im, mp, pl_, rc, ft = cfg["imm"], cfg["mpc"], cfg["planner"], cfg["recovery"], cfg["ftc"]
claim(f"bandwidth \\omega_o/2\\pi; k_P, k_I & {ft['eso_bandwidth_hz']:.0f} Hz; {ft['tension_kp']:g}, {ft['tension_ki']:.0f} s^-1")
claim(f"period; arming time & {1e3 * im['dt']:.0f} ms; {im['t_arm']:.0f} s")
claim(f"process noise, linear and angular acceleration & {im['sigma_acc_lin']:g} ms^-2, {im['sigma_acc_ang']:g} rads^-2")
claim(f"measurement noise, lengths and tensions & {1e3 * im['r_length']:g} mm, {im['r_tension']:g} N")
claim(f"deficit random walk; fault and return rates & {im['sigma_deficit']:.0f} Ns^-1/2; 10^-3, 10^-2 s^-1",
      im["fault_rate"] == 1e-3, im["return_rate"] == 1e-2)
claim(f"\\mu_\\mathrmiso, N_c & {ft['mu_declare']:g}, {ft['n_confirm']}")
claim(f"period, horizon & {1e3 * mp['dt']:.0f} ms, {mp['horizon']} steps")
claim(f"weights: pose; velocity; tension change & 4\\times10^4; {mp['q_vel']:.0f}; {mp['r_rate']:g}",
      mp["q_pos"] == mp["q_rot"] == 4e4, mp["q_vel"] == mp["q_angvel"])
pb = dict((k, v) for k, v in ft["post_break_weights"])
claim(f"velocity \\times{pb['q_vel']:g}, tension change \\times{pb['r_rate']:g}", pb["q_vel"] == pb["q_angvel"])
claim(f"scenarios: threshold \\mu_s, at most & 10^-3, {ft['max_fault_scenarios']} fault scenarios", ft["mu_scenario"] == 1e-3)
claim(f"time scaling: horizon, \\kappa_\\min & {pl_['horizon']:.0f} s, {pl_['kappa_min']:g}")
claim(f"T, intervals; s_p, s_\\min & {rc['t_lo']:.0f} s, {rc['steps']}; {rc['s_path']:.0f}, {rc['s_min']:.0f} N", rc["t_lo"] == rc["t_hi"])
claim(f"library: break instants; detection delay & every 50 ms; {1e3 * ft['library_delay']:.0f} ms")
claim(f"the stage cost of a fault scenario is scaled by {mp['fault_track_scale']:g}")
claim(f"a soft tension margin of {rc['s_soft']:.0f} N, penalised quadratically", rc["w_margin"] > 0)

if MS == "ijc":
    # the word count stated on the title page, recounted with detex: Sections 1 to 8, without and with
    # the floats (tables, figures with their captions, the algorithm) and without display equations
    import shutil
    import subprocess
    import tempfile
    if shutil.which("detex"):
        src = (MSDIR / "main.tex").read_text()
        body = src[src.index("\\section{Introduction}"):src.index("\\section*{Disclosure statement}")]
        noflt = body
        for env in ("equation", "align", "gather", "multline", "table", "figure", "algorithm"):
            noflt = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), " ", noflt, flags=re.S)
        withflt = body
        for env in ("equation", "align", "gather", "multline"):
            withflt = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), " ", withflt, flags=re.S)

        def words(tex):
            with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as f:
                f.write(tex)
            return len(subprocess.run(["detex", "-n", f.name], capture_output=True, text=True).stdout.split())
        n0, n1 = words(noflt), words(withflt)
        claim(f"Word count: about {50 * round(n0 / 50):,} words in Sections 1 to 8 (about {50 * round(n1 / 50):,} with the "
              f"tables, figure captions and algorithm)")

print(f"{PASSED} claims checked against the stored results")
(MSDIR / "number_check.txt").write_text(
    f"Passages of {MSDIR.relative_to(ROOT)}/main.tex whose numbers agree with the stored results "
    "(scripts/paper_check_numbers.py):\n" + "".join(f"PASS  {x}\n" for x in LOG)
    + "".join(f"FAIL  {x}\n" for x in FAILED))
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print(" -", f)
    sys.exit(1)
print("all claims agree with the stored results")
