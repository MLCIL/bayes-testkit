"""Barycentric-simplex and shrinkage-forest plots for the hierarchical t-test.

The simplex (Benavoli et al. 2017, Fig. 11) shows the posterior over the
next-dataset three-way partition ``(theta_l, theta_e, theta_r)``; the forest
shows per-dataset shrinkage of the raw means towards the population mean.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Cartesian coordinates of the simplex vertices. The convention matches the
# paper: left model bottom-left, right model bottom-right, rope at the top.
_V_LEFT = np.array([0.0, 0.0])
_V_RIGHT = np.array([1.0, 0.0])
_V_ROPE = np.array([0.5, np.sqrt(3.0) / 2.0])
_CENTROID = (_V_LEFT + _V_RIGHT + _V_ROPE) / 3.0


def _to_cartesian(
    theta_left: np.ndarray, theta_rope: np.ndarray, theta_right: np.ndarray
) -> np.ndarray:
    """Map barycentric ``(left, rope, right)`` weights to Cartesian points."""
    return (
        np.outer(theta_left, _V_LEFT)
        + np.outer(theta_rope, _V_ROPE)
        + np.outer(theta_right, _V_RIGHT)
    )


def plot_hierarchical_simplex(
    theta_left: np.ndarray,
    theta_rope: np.ndarray,
    theta_right: np.ndarray,
    left_model: str,
    right_model: str,
    rope_band: tuple[float, float],
    ax: plt.Axes | None = None,
    point_size: float = 4.0,
    alpha: float = 0.25,
    **kwargs,
) -> plt.Axes:
    """Plot the barycentric simplex of the next-dataset ROPE triple.

    Parameters
    ----------
    theta_left, theta_rope, theta_right : np.ndarray
        Posterior samples of the three partition probabilities (each of length
        ``n_samples``; rows should sum to one).
    left_model, right_model : str
        Names for the two bottom vertices (positive difference favours the left
        model, drawn bottom-right so it matches the paper's ``aode`` corner).
    rope_band : tuple[float, float]
        The ROPE band, used to annotate the rope vertex.
    ax : plt.Axes | None, default None
        Axes to draw on; created if ``None``.
    point_size : float, default 4.0
        Marker size for the posterior cloud.
    alpha : float, default 0.25
        Marker transparency.
    **kwargs
        Forwarded to the scatter call.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    if ax is not None and not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object or None.")
    if ax is None:
        _, ax = plt.subplots()

    # Triangle edges.
    triangle = np.vstack([_V_LEFT, _V_RIGHT, _V_ROPE, _V_LEFT])
    ax.plot(triangle[:, 0], triangle[:, 1], color="#ff7f0e", linewidth=1.5)

    # Region boundaries: medians from the centroid to each edge midpoint. These
    # are the level curves theta_i = theta_j that delimit "region i wins".
    for v_a, v_b in ((_V_LEFT, _V_RIGHT), (_V_RIGHT, _V_ROPE), (_V_ROPE, _V_LEFT)):
        midpoint = (v_a + v_b) / 2.0
        ax.plot(
            [_CENTROID[0], midpoint[0]],
            [_CENTROID[1], midpoint[1]],
            color="#ff7f0e",
            linewidth=0.8,
        )

    # Posterior cloud. Positive difference favours the left model, drawn at the
    # bottom-right vertex; theta_right therefore lands on the right.
    points = _to_cartesian(
        np.asarray(theta_left), np.asarray(theta_rope), np.asarray(theta_right)
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=point_size,
        alpha=alpha,
        color="#1f77b4",
        edgecolors="none",
        **kwargs,
    )

    # Vertex labels: theta_right -> left model (better), theta_left -> right model.
    ax.text(*_V_RIGHT, f"\n{left_model} better", ha="center", va="top", fontsize=9)
    ax.text(*_V_LEFT, f"\n{right_model} better", ha="center", va="top", fontsize=9)
    ax.text(
        *_V_ROPE,
        f"rope\n[{rope_band[0]:.3g}, {rope_band[1]:.3g}]\n",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    margin = 0.12
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, _V_ROPE[1] + margin)
    return ax


def plot_hierarchical_forest(
    dataset_names: list[str],
    raw_means: np.ndarray,
    shrunk_means: np.ndarray,
    shrunk_hdi_low: np.ndarray,
    shrunk_hdi_high: np.ndarray,
    rope_band: tuple[float, float],
    left_model: str,
    right_model: str,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot per-dataset shrinkage of raw means towards the shrunk estimates.

    Parameters
    ----------
    dataset_names : list[str]
        Dataset labels, one per row.
    raw_means : np.ndarray
        Raw per-dataset mean differences ``x_bar_i``.
    shrunk_means : np.ndarray
        Posterior-mean shrunk estimates ``mu_i``.
    shrunk_hdi_low, shrunk_hdi_high : np.ndarray
        HDI bounds of the shrunk per-dataset estimates.
    rope_band : tuple[float, float]
        The ROPE band to shade.
    left_model, right_model : str
        Names of the two algorithms (for the axis label).
    ax : plt.Axes | None, default None
        Axes to draw on; created if ``None``.
    **kwargs
        Forwarded to the shrunk-estimate scatter call.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    if ax is not None and not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object or None.")
    if ax is None:
        _, ax = plt.subplots()

    y = np.arange(len(dataset_names))
    ax.axvspan(rope_band[0], rope_band[1], color="#ff7f0e", alpha=0.12, label="rope")
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle=":")

    ax.hlines(y, shrunk_hdi_low, shrunk_hdi_high, color="#1f77b4", linewidth=1.5)
    ax.scatter(
        shrunk_means, y, color="#1f77b4", s=25, label="shrunk", zorder=3, **kwargs
    )
    ax.scatter(
        raw_means,
        y,
        facecolors="none",
        edgecolors="black",
        s=25,
        label="raw",
        zorder=3,
    )
    # Connect raw to shrunk to make the shrinkage visible.
    for yi, raw, shrunk in zip(y, raw_means, shrunk_means, strict=True):
        ax.plot([raw, shrunk], [yi, yi], color="grey", linewidth=0.6, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(dataset_names, fontsize=8)
    ax.set_xlabel(f"mean difference ({left_model} - {right_model})")
    ax.set_title("Per-dataset shrinkage")
    ax.legend(loc="best", fontsize=8)
    return ax
