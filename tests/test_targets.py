# AI Summary: Validates basic behaviour of GMM and Ring synthetic targets—
# sampling shape, finite log‑densities, and reproducibility.
import numpy as np
import pytest
from svbmc.targets import GMM, Ring


def test_gmm_sampling_and_log_pdf():
    """Samples should have correct shape and finite log‑densities."""
    gmm = GMM()
    samples = gmm.sample(n=5)
    assert samples.shape == (5, 2)

    # Check each sampled point has a finite log probability
    for row in samples:
        assert np.isfinite(gmm.log_pdf(row))


@pytest.mark.parametrize("n", [1, 10])
def test_ring_sampling_and_log_pdf(n):
    """Ring sampler returns expected shape and finite log‑pdf."""
    ring = Ring()
    draws = ring.sample(n=n)
    assert draws.shape == (n, 2)

    # Compute log pdf for first sample and for the centre
    assert np.isfinite(ring.log_pdf(draws[0]))
    assert np.isfinite(ring.log_pdf(np.asarray([1.0, -2.0])))
