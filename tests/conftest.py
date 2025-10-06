# AI Summary: Pytest configuration and shared fixtures.
# Sets a non‑interactive matplotlib backend and provides mock VBMC
# objects plus the `simple_svbmc` fixture for use by all test modules.
import matplotlib

matplotlib.use("Agg")  # force headless backend before any pyplot import

import sys
import types
from typing import Tuple

import numpy as np
import pytest
import torch

# -----------------------------------------------------------------------------#
# Optional dependency stubs                                                    #
# -----------------------------------------------------------------------------#
# Many tests monkey‑patch the absence of `corner`.  Providing a stub here
# guarantees that import errors cannot arise even in modules that forget to
# patch on their own.
corner_stub = types.ModuleType("corner")
corner_stub.corner = lambda *_, **__: None  # simple no‑op stub
sys.modules.setdefault("corner", corner_stub)

# -----------------------------------------------------------------------------#
# Lightweight mocks for a PyVBMC VariationalPosterior                          #
# -----------------------------------------------------------------------------#
class _IdentityTransformer:
    """Do‑nothing parameter transformer with analytical Jacobian."""

    def __call__(self, x: np.ndarray) -> np.ndarray:          # forward
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:           # inverse
        return x

    def log_abs_det_jacobian(self, x) -> np.ndarray | float:  # |x| ∂x/∂x = 0
        # Preserve element‑wise shape when *x* is an array so callers can
        # broadcast corrections correctly.
        return np.zeros(x.shape[0]) if isinstance(x, np.ndarray) else 0.0


class _MockVP:
    """
    Extremely thin stand‑in for PyVBMC's ``VariationalPosterior`` sufficient
    for unit‑testing SVBMC logic without the heavy dependency stack.
    """

    def __init__(self, d: int, k: int, mu_offset: float = 0.0, elbo: float = 0.0):
        self.mu = np.zeros((d, k)) + mu_offset
        self.sigma = np.ones((d, k))
        self.lambd = np.ones((d, k))
        self.w = np.full(k, 1.0 / k)
        self.parameter_transformer = _IdentityTransformer()
        # Minimal statistics required by SVBMC
        self.stats = {
            "I_sk": np.tile(np.arange(1, k + 1, dtype=float), (5, 1)),
            "elbo": elbo,
            "stable": True, 
            "J_sjk": 0.1*np.eye(k)
        }

    # The real PyVBMC returns (samples, log‑prob); SVBMC uses only samples.
    def sample(self, n: int) -> Tuple[np.ndarray, None]:
        d, _ = self.mu.shape
        return np.random.randn(n, d) + self.mu[:, 0], None


# -----------------------------------------------------------------------------#
# Shared fixtures                                                              #
# -----------------------------------------------------------------------------#
@pytest.fixture(scope="module")
def simple_svbmc():
    """
    Return an `SVBMC` instance built from two mock variational posteriors.
    Being defined in *conftest.py* makes it available to every test module.
    """
    import svbmc as _svbmc  # local import avoids polluting global namespace

    torch.manual_seed(0)
    np.random.seed(0)

    vp1 = _MockVP(d=2, k=2, mu_offset=0.0, elbo=1.0)
    vp2 = _MockVP(d=2, k=2, mu_offset=1.0, elbo=1.0)

    return _svbmc.SVBMC([vp1, vp2], testing = True) 
