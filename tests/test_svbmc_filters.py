# AI Summary: Tests the S-VBMC initial filtering (s_max) and acceptance threshold (M_min).
# Covers: happy path, strict inequality boundary, proportion vs integer M_min,
# invalid inputs, and printed warnings for rounding and upper-bounding.

import numpy as np
import pytest

import svbmc


class _IdentityTransformer:
    """Do-nothing transformer with analytical Jacobian (keeps shapes)."""
    def __call__(self, x):  # forward
        return x
    def inverse(self, x):   # inverse
        return x
    def log_abs_det_jacobian(self, x):
        # Preserve element-wise shape if array input is provided
        return np.zeros(x.shape[0]) if isinstance(x, np.ndarray) else 0.0


class _MockVP:
    """
    Minimal VBMC 'VariationalPosterior' stand-in with the attributes/methods
    SVBMC.__init__ expects, allowing isolated testing of constructor logic.
    """
    def __init__(self, d=2, k=1, stable=True, J_value=0.01, elbo=0.0):
        # Transformed-space mixture parameters (shapes must be consistent)
        self.mu = np.zeros((d, k))
        self.sigma = np.ones((d, k))
        self.lambd = np.ones((d, k))
        self.w = np.full(k, 1.0 / k)
        self.parameter_transformer = _IdentityTransformer()
        # Stats SVBMC reads during initialisation
        self.stats = {
            # I_sk: shape (S, K) – only need to exist; values don't matter for constructor
            "I_sk": np.ones((5, k)),
            "elbo": elbo,
            "stable": stable,
            # J_sjk is used via np.sqrt(np.max(J_sjk)) < s_max – supply tiny matrix
            "J_sjk": np.array([[J_value]], dtype=float),
        }

    # SVBMC.sample() isn't exercised in these constructor-focused tests,
    # but we provide a stub to keep the object realistic.
    def sample(self, n):
        d, _ = self.mu.shape
        return np.zeros((n, d)), None


# ---------------------------- s_max filtering ---------------------------------

def test_filters_keep_only_stable_and_below_threshold():
    """
    Given a mix of stable/unstable and small/large J_sjk runs, only the
    stable runs with sqrt(max(J_sjk)) < s_max survive the filter.
    """
    # Choose s_max so that J=0.01 (sqrt=0.1) passes and J=0.09 (sqrt=0.3) fails
    s_max = 0.2
    good = _MockVP(stable=True,  J_value=0.01)   # sqrt=0.1 < 0.2 -> kept
    bad_j = _MockVP(stable=True, J_value=0.09)   # sqrt=0.3 >= 0.2 -> dropped
    bad_stable = _MockVP(stable=False, J_value=0.0)  # unstable -> dropped

    obj = svbmc.SVBMC([good, bad_j, bad_stable], s_max=s_max, M_min=1)

    # Exactly one run should survive
    assert obj.M == 1
    assert obj.K == [1]
    assert obj.D == 2


def test_filters_strict_inequality_at_boundary_raises_if_none_survive():
    """
    The implementation uses a strict '< s_max' comparison.
    A run with sqrt(max(J_sjk)) == s_max must be excluded.
    """
    # Make sqrt(max(J_sjk)) == s_max by setting J_value = s_max**2
    s_max = 0.2
    boundary = _MockVP(stable=True, J_value=s_max ** 2)  # sqrt=0.2 == s_max -> exclude

    with pytest.raises(ValueError) as excinfo:
        svbmc.SVBMC([boundary], s_max=s_max, M_min=1)

    assert "expected at least" in str(excinfo.value)


# ------------------------------ M_min handling --------------------------------

def test_M_min_as_fraction_of_total_runs():
    """
    When M_min <= 1, it's treated as a proportion of all supplied runs.
    Ensure construction succeeds when surviving runs >= len(vp_list) * M_min.
    """
    # Three good runs survive; proportion 2/3 => threshold 2.0
    runs = [_MockVP(), _MockVP(), _MockVP()]
    obj = svbmc.SVBMC(runs, s_max=1.0, M_min=2/3)
    assert obj.M == 3  # all survive


def test_M_min_negative_raises_value_error():
    """M_min must be positive; negative values are invalid."""
    with pytest.raises(ValueError) as excinfo:
        svbmc.SVBMC([_MockVP(), _MockVP()], s_max=1.0, M_min=-0.1)
    assert "`M_min` should be a positive number" in str(excinfo.value)


def test_M_min_integer_threshold_not_met_raises_value_error():
    """
    If fewer than M_min well-converged runs survive filtering,
    constructor raises ValueError with an informative message.
    """
    # Only one survives
    good = _MockVP(J_value=0.0)
    bad = _MockVP(J_value=100.0)  # filtered out
    with pytest.raises(ValueError) as excinfo:
        svbmc.SVBMC([good, bad], s_max=0.2, M_min=2)
    msg = str(excinfo.value)
    assert "expected at least 2 well-converged VBMC runs" in msg
    assert "but got 1" in msg


def test_M_min_greater_than_len_triggers_warning_and_succeeds(capsys):
    """
    For M_min > len(vp_list), code prints a warning and clamps to len(vp_list).
    Construction should still succeed if enough runs survive.
    """
    runs = [_MockVP(), _MockVP()]
    obj = svbmc.SVBMC(runs, s_max=1.0, M_min=10)  # warn & clamp to 2
    captured = capsys.readouterr().out
    assert "Warning: `M_min` should be lower than or equal to `len(vp_list)`" in captured
    assert obj.M == 2


def test_M_min_non_integer_gt_one_rounds_with_warning(capsys):
    """
    Non-integer M_min > 1 prints a rounding warning and rounds to nearest int.
    """
    runs = [_MockVP(), _MockVP(), _MockVP()]
    obj = svbmc.SVBMC(runs, s_max=1.0, M_min=2.4)  # rounds to 2
    captured = capsys.readouterr().out
    assert "Rounding to closest integer, 2" in captured
    assert obj.M == 3  # construction still ok since 3 >= 2


def test_M_min_exact_len_boundary_succeeds():
    """
    If the number of surviving runs equals the (processed) M_min threshold,
    construction should succeed.
    """
    runs = [_MockVP(), _MockVP()]
    obj = svbmc.SVBMC(runs, s_max=1.0, M_min=2)
    assert obj.M == 2
    