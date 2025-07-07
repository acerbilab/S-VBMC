import itertools
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner                           


def overlay_corner_plot(
    samples: List[np.ndarray],
    label: Optional[List[str]] = None,
    color: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    smooth: Optional[float] = None,
    base: float = 2.5,
    **corner_kwargs,
):
    """
    Overlay a corner plot for multiple sample sets.

    Parameters
    ----------
    samples : list[np.ndarray]
        List of arrays, each of shape ``(N_i, D)`` where *D* is the
        dimensionality.
    label : list[str], optional
        Legend labels corresponding to each item in `samples`.
        Defaults to "Run 1", "Run 2", ...
    color : list[str], optional
        Matplotlib colours for each sample set.  Defaults to the rcParams
        colour cycle.
    figsize : (float, float) or None, optional
        Figure size in inches.  If *None*, uses ``≈ base·D`` on each side.
    smooth : float or None, optional
        Gaussian kernel smoothing (px) applied by *corner*.
    base : float, default 2.5
        Inches per variable when auto-sizing.
    **corner_kwargs
        Extra keywords forwarded to :pyfunc:`corner.corner`
        (e.g. ``bins=40``, ``levels=(0.68, 0.95)``).

    Returns
    -------
    matplotlib.figure.Figure
        The resulting corner-plot figure.
    """
    # ──────────────────────────────────────────────────
    # 1. sanity checks & dimensionality
    # ──────────────────────────────────────────────────
    if not samples:
        raise ValueError("`samples` must contain at least one array.")

    D = samples[0].shape[1]
    if not all(s.shape[1] == D for s in samples):
        raise ValueError("All sample arrays must have the same dimensionality.")

    # ──────────────────────────────────────────────────
    # 2. global range so every marginal shares a scale
    # ──────────────────────────────────────────────────
    mins = np.min(np.vstack(samples), axis=0)
    maxs = np.max(np.vstack(samples), axis=0)
    global_range = [(lo, hi) for lo, hi in zip(mins, maxs)]
    corner_kwargs.setdefault("range", global_range)

    # ──────────────────────────────────────────────────
    # 3. colours & labels
    # ──────────────────────────────────────────────────
    if color is None:
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = list(itertools.islice(itertools.cycle(color), len(samples)))

    if label is None:
        label = [f"Run {i+1}" for i in range(len(samples))]
    elif len(label) != len(samples):
        raise ValueError("`label` length must match `samples` length.")

    # ──────────────────────────────────────────────────
    # 4. figure size & axis labels
    # ──────────────────────────────────────────────────
    if figsize is None:
        figsize = (base * D, base * D)

    axis_labels = [fr"$x_{{{i+1}}}$" for i in range(D)]

    # ──────────────────────────────────────────────────
    # 5. build the plot
    # ──────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)

    for idx, (samp_arr, col) in enumerate(zip(samples, color)):
        weights = np.full(samp_arr.shape[0], 1.0 / samp_arr.shape[0])  # area-normalised
        fig = corner.corner(
            samp_arr,
            labels=axis_labels,
            color=col,
            weights=weights,
            smooth=smooth,
            show_titles=(idx == 0),             # show stats only once
            fig=fig,
            **corner_kwargs,
        )

    # ──────────────────────────────────────────────────
    # 6. legend positioned to the right
    # ──────────────────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(color, label)]
    fig.subplots_adjust(right=0.80)   # reserve space
    ax0 = fig.axes[0]
    ax0.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize="medium",
    )

    plt.plot()

    return fig
