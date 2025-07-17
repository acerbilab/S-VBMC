# AI Summary: Covers corner‑plot helper and additional error branches of
# find_init_bounds for improved utils test coverage.
import numpy as np
import pytest
import matplotlib
from matplotlib.figure import Figure

from svbmc.utils import find_init_bounds, overlay_corner_plot


def test_find_init_bounds_infinite_error():
    """Supplying only infinite LB/UB without plausible bounds should error."""
    with pytest.raises(ValueError):
        find_init_bounds(LB=[-np.inf], UB=[np.inf])


def test_overlay_corner_plot_returns_figure():
    """
    overlay_corner_plot should return a matplotlib Figure and not crash even
    under head‑less backend (configured in conftest).
    """
    rng = np.random.default_rng(0)
    samples1 = rng.standard_normal((100, 2))
    samples2 = rng.standard_normal((80, 2)) + np.array([2.0, -1.0])

    fig = overlay_corner_plot(
        [samples1, samples2],
        labels=["A", "B"],
        colors=["C0", "C1"],
        smooth=0.2,
        bins=15,
    )
    assert isinstance(fig, Figure)
    # Each subplot should have been created
    expected_axes = 4  # 2D corner: D(D+1)/2 with D=2
    assert len(fig.axes) == expected_axes
    matplotlib.pyplot.close(fig)
