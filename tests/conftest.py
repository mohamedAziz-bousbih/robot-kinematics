import numpy as np
import pytest

from robokin import Planar3R, Puma560, Scara
from robokin.transforms import wrap_angle


@pytest.fixture
def rng():
    # Fixed seed: the randomised tests are reproducible run to run.
    return np.random.default_rng(20260930)


@pytest.fixture(params=["planar", "scara", "puma"])
def robot(request):
    return {"planar": Planar3R, "scara": Scara, "puma": Puma560}[request.param]()


def joint_distance(robot, q_a, q_b):
    """Largest joint difference, comparing revolute joints modulo 2 pi."""
    d = np.asarray(q_a, float) - np.asarray(q_b, float)
    rev = robot.revolute
    d[rev] = wrap_angle(d[rev])
    return float(np.max(np.abs(d)))


def assert_pose_close(T_a, T_b, atol=1e-9):
    np.testing.assert_allclose(T_a, T_b, atol=atol, rtol=0)
