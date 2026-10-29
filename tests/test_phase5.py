"""Phase 5 checks: the B2 rule, the online classifier against the offline windows, the run list."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_threshold_rule_offline_equals_online():
    """The vectorised rule used to choose the B2 thresholds equals the per-cycle rule the controller runs."""
    from cdpr_ft.estimation.alternatives import threshold_mode
    from phase5_b2_calibrate import modes
    rng = np.random.default_rng(0)
    x = np.zeros((4000, 24), np.float32)
    x[:, :8] = rng.normal(0, 1.0, (4000, 8))                  # r_t / 10 N
    x[:, 8:16] = rng.uniform(0.0, 1.2, (4000, 8))             # t_hat / 100 N
    for tb, ta in ((5.0, 10.0), (2.0, np.inf), (8.0, 2.5)):
        vec = modes(x, tb, ta)
        one = np.array([threshold_mode(row.reshape(3, 8), tb, ta) for row in x])
        assert np.array_equal(vec, one)


def test_online_classifier_matches_offline_windows(robot):
    """ClassifierDiagnosis, fed sample by sample, gives the probabilities that the offline
    evaluation computes from the same features (window of the last W cycles, logits / T)."""
    torch = pytest.importorskip("torch")
    from cdpr_ft.estimation.alternatives import ClassifierDiagnosis
    from cdpr_ft.learning.features import FeatureStream
    from cdpr_ft.learning.models import build
    from cdpr_ft.params import inner_rate, load_cable_winch
    from cdpr_ft.sim.episode import EpisodeSpec, controller_start
    from phase4b_common import softmax_t
    torch.manual_seed(0)
    cw = load_cable_winch()
    _, ref, q0, t_nom, l_u0 = controller_start(EpisodeSpec(), robot, cw)
    dt = 1.0 / inner_rate()
    w = 6
    model = build({"arch": "lstm", "window": w, "lstm_hidden": 8, "lstm_layers": 1, "dropout": 0.0}).eval()
    temp = 1.7
    online = ClassifierDiagnosis(FeatureStream(robot, cw, l_u0, ref, 50.0, 5, dt, t_nom), model, w, temp, t_from=0.05)
    offline = FeatureStream(robot, cw, l_u0, ref, 50.0, 5, dt, t_nom)
    rng = np.random.default_rng(2)
    feats, probs, times = [], [], []
    theta = np.zeros(8)
    for k in range(300):
        theta = theta + rng.normal(0, 1e-3, 8)
        y = None if k == 0 else {"theta": theta.copy(), "tau": cw.transmission * t_nom + rng.normal(0, 0.01, 8)}
        td = t_nom + rng.normal(0, 1.0, 8)
        p = online.sample(k, y, td)
        o = offline.sample(k, y, td)
        assert (p is None) == (o is None)
        if o is not None:
            feats.append(o[1].reshape(-1))
            probs.append(p)
            times.append(o[0])
    feats, probs = np.array(feats), np.array(probs)
    for i in range(len(feats)):
        if times[i] < 0.05 or i < w - 1:
            assert probs[i][0] == 1.0
            continue
        x = torch.from_numpy(feats[i - w + 1:i + 1].T[None].astype(np.float32))
        with torch.no_grad():
            logits = model(x).numpy()
        np.testing.assert_allclose(probs[i], softmax_t(logits, temp)[0], rtol=0, atol=1e-6)


def test_phase5_run_list():
    from cdpr_ft.eval.campaign import jobs_for, load_protocol, run_list
    protocol, _ = load_protocol()
    runs = run_list(protocol)
    seeds = [r["seed"] for r in runs]
    assert len(seeds) == len(set(seeds))
    assert all(200000 <= s < 300000 for s in seeds)                       # evaluation range only
    fault = [r for r in runs if r["group"] == "fault"]
    assert len(fault) == 2 * 13 * 3 * 8 and sum(r["group"] == "fault_free" for r in runs) == 200
    for kind in ("break", "loe", "bias"):
        for traj in ("T1", "T2"):
            sel = [r for r in fault if r["fault"][0] == kind and r["trajectory"] == traj]
            assert len(sel) == 104 and {r["fault"][1] for r in sel} == set(range(8))
            assert all(3.0 <= r["fault"][2] <= 7.0 and r["t_end"] <= r["fault"][2] + 4.0 + 1e-9 for r in sel)
    jobs = jobs_for(protocol, runs)
    assert len(jobs) == len(set(jobs))
    by = {}
    for m, r in jobs:
        by.setdefault(m, set()).add(r)
    assert by["P"] >= by["B3"] and by["P"] == by["A1"]                    # every comparison is paired
    assert len(by["A2"]) == 208 and len(by["A5"]) == 208


def test_alt_diagnosis_matches_variant():
    from cdpr_ft.control.ftc import ALTERNATIVE_DIAGNOSIS, COMPUTED_TORQUE, DECLARING, HEDGING
    assert set(ALTERNATIVE_DIAGNOSIS) <= set(DECLARING)
    assert "proposed" in HEDGING and "b3_tight" in HEDGING and "b3" not in HEDGING
    assert set(COMPUTED_TORQUE) == {"tda", "b2"}


def test_b2_settings_reach_the_controller():
    """The B2 settings of configs/b2_phase5.yaml (gains before and after a break declaration,
    thresholds) are the ones the campaign hands the controller."""
    import yaml
    from cdpr_ft.eval import campaign as C
    protocol, _ = C.load_protocol()
    c = yaml.safe_load((C.ROOT / protocol["b2"]["file"]).read_text())
    C._W.pop("b2", None)
    C._W["ftc"] = __import__("cdpr_ft.control.config", fromlist=["load_controller"]).load_controller(
        protocol["controller"]["file"])[0]["ftc"]
    cfg, factory, extra = C.method_config("B2", protocol)
    assert cfg.variant == "b2" and cfg.n_confirm == int(c["n_c"])
    assert extra["ct_gains"] == (tuple(c["ct_kp"]), tuple(c["ct_kd"]))
    assert extra["ct_gains_post"] == (tuple(c["ct_kp_post"]), tuple(c["ct_kd_post"]))
    for name in ("P", "A1", "B0"):
        assert "ct_gains_post" not in C.method_config(name, protocol)[2]
    C._W.pop("b2", None)
