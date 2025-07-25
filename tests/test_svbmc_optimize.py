# AI Summary: Additional tests for SVBMC optimisation paths, ensuring ELBO gains,
# error handling, and correct state updates post‑optimisation.
import numpy as np
import torch
import pytest


def _set_seeds(seed: int = 0):
    """Deterministic helper for NumPy and Torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------#
# maximise_ELBO variants                                                       #
# -----------------------------------------------------------------------------#
def test_maximize_elbo_variants_improve_over_naive(simple_svbmc):
    """
    `all-weights` and `posterior-only` optimisation should never yield a worse
    ELBO than naïve stacking.
    """
    _set_seeds()

    w_ns, elbo_ns, _ = simple_svbmc.maximize_ELBO(version="ns", n_samples=5)

    w_all, elbo_all, _ = simple_svbmc.maximize_ELBO(
        version="all-weights", n_samples=5, lr=0.2, max_steps=20
    )

    w_post, elbo_post, _ = simple_svbmc.maximize_ELBO(
        version="posterior-only", n_samples=5, lr=0.2, max_steps=20
    )

    # Weights remain normalised -------------------------------------------------
    for w in (w_ns, w_all, w_post):
        torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    # ELBO monotonicity ---------------------------------------------------------
    assert elbo_all >= elbo_ns
    assert elbo_post >= elbo_ns


def test_maximize_elbo_invalid_version(simple_svbmc):
    """Unknown optimisation version must raise AttributeError."""
    with pytest.raises(AttributeError):
        simple_svbmc.maximize_ELBO(version="does-not-exist")


# -----------------------------------------------------------------------------#
# optimize() convenience wrapper                                               #
# -----------------------------------------------------------------------------#
def test_optimize_updates_state(simple_svbmc):
    """Calling optimize() should populate weights, entropy, and ELBO dict."""
    _set_seeds()

    simple_svbmc.optimize(n_samples=5, lr=0.2, max_steps=20, version="all-weights")

    np.testing.assert_allclose(simple_svbmc.w.sum(), 1.0, rtol=0, atol=1e-12)
    assert isinstance(simple_svbmc.entropy, float)
    for k in ("estimated", "debiased_I_median", "debiased_E_median"):
        assert k in simple_svbmc.elbo


# -----------------------------------------------------------------------------#
# stacked_ELBO fallback path                                                   #
# -----------------------------------------------------------------------------#
def test_stacked_elbo_fallback_to_internal_weights(simple_svbmc):
    """
    Passing an incompatible *w* should trigger internal fallback without error.
    """
    elbo, H = simple_svbmc.stacked_ELBO(w={"invalid": True})

    assert isinstance(elbo, torch.Tensor) and elbo.ndim == 0
    assert isinstance(H, torch.Tensor) and H.ndim == 0
