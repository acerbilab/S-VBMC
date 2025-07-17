# AI Summary: Utility helpers for plotting and initialization; moved into package namespace.
# AI Summary: Utility helpers for plotting and initialization; exposes overlay_corner_plot and find_init_bounds.
__all__ = ["overlay_corner_plot", "find_init_bounds"]

import itertools
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner                           


def overlay_corner_plot(
    samples: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    smooth: Optional[float] = None,
    base: float = 2.5,
    **corner_kwargs,
):
    """
    Overlay a corner plot for multiple sample sets.

    Parameters:
    -----------
    samples : list[np.ndarray]
        List of arrays, each of shape (`N_i`,`D`) where `D` is the
        dimensionality.
    labels : list[str], optional
        Legend labels corresponding to each item in `samples`.
        Defaults to "Run 1", "Run 2", ...
    colors : list[str], optional
        Matplotlib colors for each sample set.  Defaults to the rcParams
        color cycle.
    figsize : (float, float) or `None`, optional
        Figure size in inches.  If `None`, uses ``≈ `base`*`D` on each side.
    smooth : float or `None`, optional
        Gaussian kernel smoothing (px) applied by ``corner``.
    base : float, default 2.5
        Inches per variable when auto-sizing.
    **corner_kwargs
        Extra keywords forwarded to :pyfunc:`corner.corner`
        (e.g. ``bins=40``, ``levels=(0.68, 0.95)``).

    Returns:
    --------
    matplotlib.figure.Figure
        The resulting corner-plot figure.
    """
    # Sanity checks & dimensionality
    if not samples:
        raise ValueError("`samples` must contain at least one array.")

    D = samples[0].shape[1]
    if not all(s.shape[1] == D for s in samples):
        raise ValueError("All sample arrays must have the same dimensionality.")

    # Global range so every marginal shares a scale
    mins = np.min(np.vstack(samples), axis=0)
    maxs = np.max(np.vstack(samples), axis=0)
    global_range = [(lo, hi) for lo, hi in zip(mins, maxs)]
    corner_kwargs.setdefault("range", global_range)

    # Colors & labels
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = list(itertools.islice(itertools.cycle(colors), len(samples)))

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(samples))]
    elif len(labels) != len(samples):
        raise ValueError("`labels` length must match `samples` length.")

    # Figure size & axis labels
    if figsize is None:
        figsize = (base * D, base * D)

    axis_labels = [fr"$x_{{{i+1}}}$" for i in range(D)]

    # Build the plot
    fig = plt.figure(figsize=figsize)

    for idx, (samp_arr, col) in enumerate(zip(samples, colors)):
        weights = np.full(samp_arr.shape[0], 1.0 / samp_arr.shape[0])  # area-normalized
        fig = corner.corner(
            samp_arr,
            labels=axis_labels,
            color=col,
            weights=weights,
            smooth=smooth,
            show_titles=(idx == 0), # show stats only once
            fig=fig,
            **corner_kwargs,
        )

    # Legend positioned to the right
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.subplots_adjust(right=0.80)   # reserve space
    ax0 = fig.axes[0]
    ax0.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize="medium",
    )

    plt.show()

    return fig


def find_init_bounds(
        LB : np.ndarray = None, 
        UB : np.ndarray = None, 
        PLB : np.ndarray = None, 
        PUB : np.ndarray = None
    ):
    """
    Function for find the bounds to sample uniformly from when choosing starting point for 
    VBMC. 
    If plausible lower bounds (`PLB`) are specified, it uses those as lower sampling bounds
    (`sample_LB`), otherwise it uses lower bounds (`LB`). 
    Upper sampling bounds (`sample_UB`) are determined in the same way with plausible upper bounds 
    (`PUB`) or upper bounds (`UB`).

    NOTE: At least one of `LB`, `UB`, `PLB` and `PUB` must be specified as an array with the 
    same dimensionality of the inference problem.

    Parameters:
    -----------
    LB, UB : np.ndarray, optional
        Inputs for VBMC. They represent lower (`LB`) and upper (`UB`) bounds 
        for the coordinate vector, `x`, so that the posterior has support on 
        `LB` < `x` < `UB`. If scalars, the bound is replicated in each dimension. 
        Use ``None`` for `LB` and `UB` if no bounds exist. Set `LB` [`d`] = -`inf`
        and `UB` [`d`] = `inf` if the `d`-th coordinate is unbounded (while
        other coordinates may be bounded). Note that if `LB` and `UB` contain
        unbounded variables, the respective values of `PLB` and `PUB` need to
        be specified (see below). If `PLB` and `PUB` are not specified (see below), 
        the lower and upper sampling bounds (`sample_LB` and `sample_UB`, respectively)
        will be determined by `LB` and `UB`. 
        Both are by default `None`.
    PLB, PUB : np.ndarray, optional
        Inputs for VBMC. They represent a set of plausible lower (`PLB`) and upper (`PUB`)
        bounds such that `LB` < `PLB` < `PUB` < `UB`.
        Both `PLB` and `PUB` need to be finite. `PLB` and `PUB` represent a
        "plausible" range, which should denote a region of high posterior
        probability mass. Among other things, the plausible box is used by VBMC to
        draw initial samples and to set priors over hyperparameters of the
        algorithm. If `PLB` and `PUB` are specified, they determine the lower and upper
        sampling bounds (`sample_LB` and `sample_UB`, respectively).
        If they are not, the sampling bounds will be determined by `LB` and `UB`.
        Both are by default `None`.

    Returns:
    --------
    sample_LB, sample_UB : np.ndarray
        Lower (`sample_LB`) and upper (`sample_UB`) bounds to sample uniformly from 
        when initializing VBMC.
    """

    # Infer problem dimensionality from bounds
    def _len_or_zero(x):
        return 0 if x is None else np.asarray(x).size
    D = max(_len_or_zero(LB), _len_or_zero(UB),
            _len_or_zero(PLB), _len_or_zero(PUB))
    if D == 0:
        raise ValueError("Cannot infer dimensionality: provide at least one \
                         bound with the same dimensionality of your inference problem.")


    # Helper to broadcast / validate 
    def _to_array(x, name):
        if x is None:
            return None
        arr = np.asarray(x, dtype=float)
        if arr.size == 1:                         # scalar → broadcast
            return np.full(D, arr.item())
        if arr.size == D:                         # correct length
            return arr
        raise ValueError(f"{name} has length {arr.size}, expected 1 or {D}.")

    # Broadcast everything
    LB  = _to_array(LB,  "LB")
    UB  = _to_array(UB,  "UB")
    PLB = _to_array(PLB, "PLB")
    PUB = _to_array(PUB, "PUB")

    # Build finite sampling box 
    sample_LB, sample_UB = np.empty(D), np.empty(D)
    for d in range(D):
        lo = PLB[d] if PLB is not None and np.isfinite(PLB[d]) else \
             (LB[d]  if LB  is not None and np.isfinite(LB[d])  else -np.inf)
        hi = PUB[d] if PUB is not None and np.isfinite(PUB[d]) else \
             (UB[d]  if UB  is not None and np.isfinite(UB[d])  else  np.inf)

        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(
                f"Dimension {d}: infinite bound. Supply finite PLB/PUB when LB/UB are infinite."
            )
        if lo >= hi:
            raise ValueError(f"Dimension {d}: lower bound {lo} ≥ upper bound {hi}.")

        sample_LB[d], sample_UB[d] = lo, hi

    return sample_LB, sample_UB
