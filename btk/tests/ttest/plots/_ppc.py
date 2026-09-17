"""Posterior predictive check plot for the hierarchical correlated t-test."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from btk.tests.common import hdi_from_samples


def plot_hierarchical_ppc(
    dataset_names: list[str],
    replicated_means: np.ndarray,
    observed_means: np.ndarray,
    hdi_prob: float = 0.9,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Draw the replicated mean differences against the observed ones.

    One row per dataset: the horizontal bar is the HDI of the mean difference the
    fitted model replicates for that dataset, and the marker is the mean actually
    observed. A marker outside its bar is a dataset the model cannot reproduce.

    Parameters
    ----------
    dataset_names : list[str]
        Dataset labels, bottom to top.
    replicated_means : np.ndarray
        ``(draws, datasets)`` matrix of replicated per-dataset mean differences.
    observed_means : np.ndarray
        The observed per-dataset mean differences.
    hdi_prob : float, optional
        Probability mass of the drawn intervals. Defaults to 0.9.
    ax : plt.Axes | None, optional
        Axes to draw on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to the marker ``scatter``.

    Returns
    -------
    plt.Axes
        The Axes the figure was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.4 * len(dataset_names) + 2))

    positions = np.arange(len(dataset_names))
    bounds = np.array(
        [hdi_from_samples(replicated_means[:, i], hdi_prob) for i in positions]
    )
    inside = (bounds[:, 0] <= observed_means) & (observed_means <= bounds[:, 1])

    ax.hlines(positions, bounds[:, 0], bounds[:, 1], color="C0", linewidth=3, alpha=0.6)
    ax.scatter(
        observed_means[inside],
        positions[inside],
        color="k",
        zorder=3,
        label="observed (covered)",
        **kwargs,
    )
    if not inside.all():
        ax.scatter(
            observed_means[~inside],
            positions[~inside],
            color="C3",
            marker="X",
            zorder=3,
            label="observed (outside)",
            **kwargs,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(dataset_names)
    ax.set_xlabel("mean difference")
    ax.set_title(
        f"Posterior predictive check: {int(100 * hdi_prob)}% HDI of the "
        f"replicated means\n{int(inside.sum())} of {len(dataset_names)} "
        "observed means covered"
    )
    ax.legend(loc="best", fontsize="small")
    return ax
