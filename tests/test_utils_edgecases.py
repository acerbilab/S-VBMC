# AI Summary: Extra edge‑case coverage for utils – validation failures and
# successful figure generation.
import numpy as np
import pytest
import matplotlib

from svbmc.utils import find_init_bounds, overlay_corner_plot


# -----------------------------------------------------------------------------#
# find_init_bounds – error branches                                            #
# -----------------------------------------------------------------------------#
def test_find_init_bounds_length_mismatch():
    """LB/UB with different lengths should raise ValueError."""
    with pytest.raises(ValueError):
        find_init_bounds(LB=[-1, -1], UB=[1])


def test_find_init_bounds_non_finite_plausible():
    """Infinite plausible bounds are invalid when LB/UB are infinite."""
    with pytest.raises(ValueError):
        find_init_bounds(LB=[-np.inf], UB=[np.inf], PLB=[-np.inf], PUB=[np.inf])


def test_find_init_bounds_scalar_broadcast():
    """Scalar bounds should broadcast correctly across dimensions."""
    lb, ub = find_init_bounds(LB=-2, UB=2, PLB=[-1, -1], PUB=[1, 1])
    np.testing.assert_array_equal(lb, [-1, -1])
    np.testing.assert_array_equal(ub, [1, 1])


# -----------------------------------------------------------------------------#
# overlay_corner_plot – validation & success paths                             #
# -----------------------------------------------------------------------------#
def test_overlay_corner_plot_dimensionality_mismatch():
    """Sample arrays with differing dimensionality must error."""
    s2d = np.random.randn(10, 2)
    s3d = np.random.randn(10, 3)
    with pytest.raises(ValueError):
        overlay_corner_plot([s2d, s3d])


def test_overlay_corner_plot_label_mismatch():
    """Number of labels must match number of sample sets."""
    s1 = np.random.randn(20, 2)
    s2 = np.random.randn(20, 2)
    with pytest.raises(ValueError):
        overlay_corner_plot([s1, s2], labels=["only-one"])


def test_overlay_corner_plot_empty_samples():
    """Empty list of samples should raise ValueError."""
    with pytest.raises(ValueError):
        overlay_corner_plot([])


def test_overlay_corner_plot_returns_figure_and_closes():
    """Successful call returns a matplotlib Figure that can be closed."""
    fig = overlay_corner_plot(
        [np.random.randn(30, 2), np.random.randn(30, 2)],
        labels=["A", "B"],
        colors=["C0", "C1"],
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)
