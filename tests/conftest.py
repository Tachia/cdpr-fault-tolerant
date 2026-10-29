import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdpr_ft.params import CableWinch, load_robot  # noqa: E402

# Verification checks are independent of the numerical values of the cable and
# winch parameters. The values below are used only by the tests; they are not
# study parameters and never enter reported results.
TEST_CABLE_WINCH = CableWinch(EA=1.0e5, damping_time=0.0, drum_radius=0.05,
                              inertia=2.0e-3, viscous=0.0, coulomb=0.0)


@pytest.fixture(scope="session")
def robot():
    return load_robot()


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20261025)


@pytest.fixture(scope="session")
def lagrange():
    from cdpr_ft.model import lagrange_sym
    lagrange_sym.numeric_functions()
    return lagrange_sym
