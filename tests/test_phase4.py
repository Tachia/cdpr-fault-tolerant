"""Phase 4 checks: classifier inputs, declaration rule, models.

The tests of the networks need torch (requirements-learning.txt) and are skipped without it."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class _RecordingIMM:
    """Stands in for the IMM inside Diagnosis and records what it is given."""

    def __init__(self):
        self.homed, self.calls = False, []

    def reset(self, q0):
        pass

    def home(self, l_u, t_d):
        self.homed = True

    def step(self, l_u, t_hat, td_avg, u_avg):
        self.calls.append((np.array(l_u), np.array(td_avg), np.array(u_avg)))
        return np.ones(17) / 17


def test_features_follow_the_diagnosis_timing(robot):
    """FeatureStream sees, at the same cycles, exactly the lengths, averaged commands and averaged
    winch estimates that the controller's Diagnosis hands to the IMM."""
    from cdpr_ft.estimation.diagnosis import Diagnosis
    from cdpr_ft.learning.features import SCALES, FeatureStream
    from cdpr_ft.params import inner_rate, load_cable_winch
    from cdpr_ft.sim.episode import EpisodeSpec, controller_start
    cw = load_cable_winch()
    _, ref, q0, t_nom, l_u0 = controller_start(EpisodeSpec(), robot, cw)
    dt = 1.0 / inner_rate()
    rng = np.random.default_rng(1)
    k_n = 400
    theta = np.cumsum(rng.normal(0, 1e-3, (k_n, 8)), axis=0)
    tau = cw.transmission * t_nom + rng.normal(0, 0.01, (k_n, 8))
    td = t_nom + rng.normal(0, 1.0, (k_n, 8))
    theta[0] = np.nan                                   # no reading at the first sample
    imm = _RecordingIMM()
    diag = Diagnosis(cw, l_u0, imm, 50.0, 5, dt, t_nom, q0, imm_input="estimate")
    fs = FeatureStream(robot, cw, l_u0, ref, 50.0, 5, dt, t_nom)
    feats = []
    for k in range(k_n):
        y = None if np.isnan(theta[k, 0]) else {"theta": theta[k], "tau": tau[k]}
        diag.sample(k, y, td[k])
        o = fs.sample(k, y, td[k])
        if o is not None:
            feats.append(o)
    assert len(feats) == len(imm.calls) > 50
    for (t, f), (l_u, td_avg, u_avg) in zip(feats, imm.calls):
        expected = np.vstack([u_avg - td_avg, u_avg]) / SCALES[:2, None]
        np.testing.assert_allclose(f[:2], expected.astype(np.float16).astype(np.float32), rtol=0, atol=0)
        k = int(round(t / dt)) + 1                       # time stamp t = k dt - dt of the reading used at sample k
        np.testing.assert_allclose(l_u, l_u0 - cw.transmission * theta[k], atol=0)


def test_declaration_rule_matches_phase3():
    from cdpr_ft.learning.declare import evaluate, top_and_p
    import phase3_roc as roc
    rng = np.random.default_rng(3)
    runs, mine = [], []
    for i in range(30):
        n = 300
        mu = rng.dirichlet(np.full(17, 0.3), size=n)
        if i % 3:                                        # a fault mode that becomes dominant
            j = 1 + i % 16
            k0 = rng.integers(50, 250)
            mu[k0:, :] *= 0.01
            mu[k0:, j] += 0.99
        mu /= mu.sum(axis=1, keepdims=True)
        t = 2.0 + 0.005 * np.arange(n)
        fault = None if i % 3 == 0 else (("break", "loe", "bias")[i % 3], (i % 16) % 8, float(t[100]), 0.0)
        runs.append({"fault": fault, "t_imm": t, "mu": mu})
        top, p = top_and_p(mu)
        mine.append({"fault": fault, "t": t, "top": top, "p": p})
    for m in roc.MU_ISO:
        for n_c in roc.N_C:
            assert evaluate(mine, m, n_c) == roc.evaluate(runs, m, n_c)


def test_selection_rule_gives_the_phase3_thresholds():
    """On the stored Phase 3 threshold table, the lexicographic rule of the Phase 4 protocol
    chooses what the Phase 3 rule chose (it reduces to it when a pair is free of false alarms and
    isolates every fault)."""
    from cdpr_ft.learning.declare import select
    d = json.loads((ROOT / "results" / "phase3_roc_c3.json").read_text())
    row = select(d["table"])
    assert (row["mu_iso"], row["n_c"]) == (d["chosen"]["mu_iso"], d["chosen"]["n_c"])


@pytest.mark.parametrize("params", [
    {"arch": "cnn", "window": 40, "cnn_layers": 3, "cnn_channels": 16, "cnn_kernel": 5, "dropout": 0.1},
    {"arch": "lstm", "window": 20, "lstm_hidden": 16, "lstm_layers": 2, "dropout": 0.1}])
def test_models_shape_and_input_layer(params):
    torch = pytest.importorskip("torch")
    from cdpr_ft.learning.models import build, compress
    torch.manual_seed(0)
    mean, std = np.linspace(-1, 1, 24), np.linspace(0.5, 2, 24)
    model = build(params, mean, std).eval()
    x = torch.randn(5, 24, params["window"]) * 30
    assert model(x).shape == (5, 17)
    z = model.norm(x).numpy()
    np.testing.assert_allclose(z, (compress(x.numpy()) - mean[None, :, None]) / std[None, :, None], rtol=1e-5, atol=1e-5)


def test_cnn_uses_the_newest_sample():
    """With padding on the past side only, the newest sample of the window enters the output."""
    torch = pytest.importorskip("torch")
    from cdpr_ft.learning.models import build
    torch.manual_seed(0)
    model = build({"arch": "cnn", "window": 40, "cnn_layers": 2, "cnn_channels": 8, "cnn_kernel": 3, "dropout": 0.0}).eval()
    x = torch.randn(1, 24, 40)
    y = x.clone()
    y[0, :, -1] += 1.0
    assert not torch.allclose(model(x), model(y))


def test_training_is_deterministic(tmp_path):
    pytest.importorskip("torch")
    from cdpr_ft.learning.data import Dataset
    from cdpr_ft.learning.train import train
    rng = np.random.default_rng(0)
    runs, xs, ys, ts, rs = [], [], [], [], []
    off = 0
    for i, split in enumerate(["train"] * 6 + ["val"] * 2):
        n = 120
        lab = 1 + i % 16
        x = rng.normal(0, 1, (n, 24)).astype(np.float32)
        y = np.where(np.arange(n) >= 60, lab, 0)
        x[60:, lab % 24] += 3.0
        runs.append({"index": i, "split": split, "fault": ("break", 0, 0.3, 0.0), "offset": off, "length": n})
        xs.append(x), ys.append(y), ts.append(0.005 * np.arange(n)), rs.append(np.full(n, i))
        off += n
    ds = Dataset(np.concatenate(xs), np.concatenate(ys), np.concatenate(rs), np.concatenate(ts),
                 np.zeros(off, np.int8), np.zeros(off), runs)
    p = {"arch": "cnn", "window": 20, "cnn_layers": 2, "cnn_channels": 8, "cnn_kernel": 3, "dropout": 0.1,
         "lr": 1e-3, "weight_decay": 1e-5}
    a = train(p, ds, seed=0, max_epochs=3, patience=3, t_from=0.0, m0_stride=2, threads=1, batch=32, log=None)
    b = train(p, ds, seed=0, max_epochs=3, patience=3, t_from=0.0, m0_stride=2, threads=1, batch=32, log=None)
    for k, v in a["model"].state_dict().items():
        assert np.array_equal(v.numpy(), b["model"].state_dict()[k].numpy())
    assert [h["val_macro_f1"] for h in a["history"]] == [h["val_macro_f1"] for h in b["history"]]
