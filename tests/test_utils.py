# AI Summary: Quick sanity checks for svbmc.utils helpers.
import numpy as np
from svbmc.utils import find_init_bounds


def test_find_init_bounds_basic():
    """Basic LB/UB input should be returned unchanged."""
    lb, ub = find_init_bounds(LB=[-1, -1], UB=[1, 1])
    assert np.allclose(lb, [-1, -1])
    assert np.allclose(ub, [1, 1])


def test_find_init_bounds_plausible():
    """When only PLB/PUB are provided they should define the sampling bounds."""
    lb, ub = find_init_bounds(PLB=[0, 0], PUB=[2, 2])
    assert np.allclose(lb, [0, 0])
    assert np.allclose(ub, [2, 2])
