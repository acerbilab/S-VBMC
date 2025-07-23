import sys
import types
from types import SimpleNamespace
import numpy as np
import torch
import pytest

# -----------------------------------------------------------------------------
# Create a *very* light stub for the optional ``corner`` dependency so that the
# import of ``svbmc`` does not fail even when the real library is not installed
# in the test environment.  For our unit tests we never call ``SVBMC.plot``
# (the only method that actually needs ``corner``), therefore a no‑op stub is
# perfectly adequate.
# -----------------------------------------------------------------------------
corner_stub = types.ModuleType("corner")
corner_stub.corner = lambda *args, **kwargs: None  # noqa: E731 – simple stub
sys.modules.setdefault("corner", corner_stub)

# Now we can safely import the module under test
import svbmc  # noqa: E402 – must come after the stub setup

# -----------------------------------------------------------------------------
# Helper classes to mimic the minimal public interface that SVBMC expects from a
# PyVBMC "VariationalPosterior" instance.  They deliberately implement only the
# attributes and methods actually accessed by SVBMC so that the tests remain
# lightweight and fully self‑contained.
# -----------------------------------------------------------------------------
class _IdentityTransformer:
    """A do‑nothing transformer used to keep the maths trivial."""

    # Forward transform – identity
    def __call__(self, x):
        return x

    # Inverse transform – identity
    def inverse(self, x):
        return x

    # Log absolute determinant of the Jacobian – always zero
    def log_abs_det_jacobian(self, x):
        # When *x* is an array we need to return an array of the same first‑axis
        # length because SVBMC expects element‑wise corrections.
        if isinstance(x, np.ndarray):
            return np.zeros(x.shape[0])
        # Scalar input (very first correction inside ``stacked_entropy``)
        return 0.0


class _MockVP:
    """A *very* simplified fake of PyVBMC's ``VariationalPosterior``."""

    def __init__(self, d: int, k: int, mu_offset: float = 0.0, elbo: float = 0.0):
        self.mu = np.zeros((d, k)) + mu_offset  # means in transformed space
        self.sigma = np.ones((d, k))            # diagonal std‑devs
        self.lambd = np.ones((d, k))            # scaling factors – keep at 1
        self.w = np.full(k, 1.0 / k)            # uniform mixture weights
        # Statistics returned by VBMC – only the fields actually used by SVBMC
        self.stats = {
            "I_sk": np.tile(np.arange(1, k + 1, dtype=float), (5, 1)),
            "elbo": elbo,
        }
        self.parameter_transformer = _IdentityTransformer()

    # The real VBMC ``sample`` method returns a tuple (samples, log‑probability);
    # SVBMC only needs the samples, so we can safely return ``None`` as the
    # second element.
    def sample(self, n: int):
        d, k = self.mu.shape
        # Draw *n* samples from a very simple isotropic Gaussian centred on the
        # first component's mean – this is *more* than enough for the tests.
        return np.random.randn(n, d) + self.mu[:, 0], None

# -----------------------------------------------------------------------------
# Tests for basic object construction
# -----------------------------------------------------------------------------

def test_initialisation_properties(simple_svbmc):
    obj = simple_svbmc

    # Dimensionality and counts ------------------------------------------------
    assert obj.D == 2
    assert obj.M == 2               # two VBMC runs stacked
    assert obj.K == [2, 2]          # each with 2 mixture components

    # Weights ------------------------------------------------------------------
    total_components = sum(obj.K)
    assert obj.w.shape == (1, total_components)
    # ``obj.w`` is *already* normalised in the constructor
    np.testing.assert_allclose(obj.w.sum(), 1.0, rtol=0, atol=1e-12)

    # Expected log‑joint stats --------------------------------------------------
    assert obj.I.shape == (1, total_components)

# -----------------------------------------------------------------------------
# Tests for the entropy computation – ensures differentiability wrt *w*
# -----------------------------------------------------------------------------

def test_stacked_entropy_gradient(simple_svbmc):
    obj = simple_svbmc
    k_total = sum(obj.K)

    # Start from the internal weights but mark them as requiring gradients
    w_torch = torch.tensor(obj.w.flatten(), dtype=torch.float64, requires_grad=True)

    H, J = obj.stacked_entropy(w_torch)

    # 1.  Function returns the expected types and shapes
    assert isinstance(H, torch.Tensor) and H.ndim == 0  # scalar tensor
    assert isinstance(J, np.ndarray) and J.shape == (k_total,)
    # The identity transformer leads to zero Jacobian corrections everywhere
    np.testing.assert_array_equal(J, np.zeros_like(J))

    # 2.  Check that we can back‑propagate – gradients must not be ``None``
    H.backward()
    assert w_torch.grad is not None, "Entropy must be differentiable wrt the weights"
    # The gradient should have the same shape as *w*
    assert w_torch.grad.shape == w_torch.shape

# -----------------------------------------------------------------------------
# Tests for ELBO consistency and input flexibility
# -----------------------------------------------------------------------------

def test_stacked_elbo_consistency(simple_svbmc):
    obj = simple_svbmc

    # Evaluate once with a PyTorch tensor …
    w_tensor = torch.tensor(obj.w.flatten(), dtype=torch.float32)
    elbo_torch, H_torch = obj.stacked_ELBO(w_tensor)

    # … and once with a *list* to hit the conversion branch
    w_list = w_tensor.tolist()
    elbo_list, H_list = obj.stacked_ELBO(w_list)

    # 1.  The two computations must agree closely (up to numerical error)
    torch.testing.assert_close(elbo_torch, elbo_list, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(H_torch, H_list, rtol=1e-6, atol=1e-6)

    # 2.  Manual cross‑check of the ELBO formula  (G + H)
    #     *I_corrected* reduces to ``obj.I`` here because *J* = 0 for the
    #     identity transform built into the tests.
    I_corrected = obj.I  # shape (1, K_total)
    G_manual = (w_tensor / w_tensor.sum()) @ torch.tensor(I_corrected, dtype=w_tensor.dtype).T
    expected_elbo = G_manual[0] + H_torch

    torch.testing.assert_close(elbo_torch, expected_elbo, rtol=1e-6, atol=1e-6)

# -----------------------------------------------------------------------------
# Tests for the *naïve stacking* branch in ``maximize_ELBO`` – this avoids heavy
# optimisation loops while still exercising the surrounding logic.
# -----------------------------------------------------------------------------

def test_maximize_elbo_naive(simple_svbmc):
    obj = simple_svbmc
    w_final, elbo_best, entropy_best = obj.maximize_ELBO(version="ns", n_samples=10)

    # 1.  Quick sanity checks on the shapes and normalisation
    assert isinstance(w_final, torch.Tensor) and w_final.ndim == 1
    torch.testing.assert_close(w_final.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    # 2.  ELBO reported by the method must match a fresh direct computation
    elbo_direct, entropy_direct = obj.stacked_ELBO(w_final, n_samples=10)
    torch.testing.assert_close(elbo_best, elbo_direct, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy_best, entropy_direct, rtol=1e-6, atol=1e-6)

# -----------------------------------------------------------------------------
# Tests for the public sampling API – verifies that the requested number of
# samples is returned and that their dimensionality matches the problem.
# -----------------------------------------------------------------------------

def test_sampling_shape(simple_svbmc):
    obj = simple_svbmc
    obj.optimize(version="ns", n_samples=5)  # ensures internal state is ready

    n_requested = 123
    samples = obj.sample(n_requested)

    assert isinstance(samples, np.ndarray)
    # The sampler *may* round the per‑posterior allocation, but the overall
    # number of returned samples should be within one of the request.
    assert abs(samples.shape[0] - n_requested) <= 1
    assert samples.shape[1] == obj.D