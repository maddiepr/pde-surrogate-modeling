"""
visualization.py

Visualization utilities for inspecting PDE solutions, surrogate model predictions,
and comparative error analyses. Designed for reproducible, publication-quality
plots using Matplotlib.

Current capabilities (planned):
- 2D heatmaps of solution fields over (x, t)
- Line plots of solution snapshots at selected times
- Error maps between surrogate and reference solutions
- Training/validation loss curves

Author: Madeline (Maddie) Preston
"""

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

def plot_solution_heatmap(
        U: np.ndarray,
        x: np.ndarray,
        t: np.ndarray,
        title: str = "PDE Solution",
        cmap: str = "viridis",
        save_path: Optional[str] = None
) -> None:
    """
    Plot a 2D heatmap of the PDE solution over space and time.

    Parameters
    ----------
    U : ndarray, shape (nt + 1, nx)
        Solution array where U[k, :] corresponds to time t[k].
    x : ndarray, shape (nx, )
        Spatial coordinates.
    t : ndarray, shape (nt + 1, )
        Time coordinates.
    title : str, default "PDE Solution"
        Plot title.
    cmap : str, default "viridis"
        Matplotlib colormap for the heatmap.
    save_path : str or None, default None
        If provided, save the plot to this path; otherwise, display interactively.

    Returns
    -------
    None

    Notes
    -----
    - Assumes U is ordered as (time, space).
    - Colorbar shows solution magnitude.
    - Intended for quick diagnostics and report figures.

    Examples
    --------
    >>> plot_solution_heatmap(U, x, t, title="Burgers' Equation Solution")
    """
    if U.ndim != 2:
        raise ValueError(f"U must be 2D (nt + 1, nx), got shape {U.shape}")
    
    # pcolormesh expects 2D coordinate grids; we use imshow with extent for simplicity
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(
        U,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1] + (x[1] - x[0]), t[0], t[-1]],
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="u(x,t)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    


def plot_comparison_snapshots(
        x: np.ndarray,
        reference: np.ndarray,
        prediction: np.ndarray,
        times: Sequence[int],
        labels: tuple[str, str] = ("Reference", "Prediction"),
        title: str = "Snapshot Comparison",
        save_path: Optional[str] = None
) -> None:
    """
    Plot side-by-side line plots of reference vs. predicted solutions
    at selected time indices.

    Parameters
    ----------
    x : ndarray, shape (nx, )
        Spatial coordinates.
    reference : ndarray, shape (nt + 1, nx)
        Ground-truth solution array.
    prediction : ndarray, shape (nt + 1, nx)
        Surrogate-predicted solution array.
    times : sequence of int
        Indices of time steps to plot.
    labels : tuple of str, default ("Reference", "Prediction")
        Legend labels for the two datasets.
    title : str, default "Snapshot Comparison"
        Overall figure title.
    save_path : str or None, default None
        If provided, save the plot; otherwise display.

    Returns
    -------
    None

    Notes
    -----
    - Overlays curves for direct visual comparison.
    - Ensure both arrays share the same grid and time indices.

    Examples
    --------
    >>> plot_comparison_snapshots(x, U_ref, U_pred, times=[0, 50, 100])
    """
    pass