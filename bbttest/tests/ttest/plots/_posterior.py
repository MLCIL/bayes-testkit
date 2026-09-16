"""Posterior-density plot for the correlated t-test (Benavoli et al. 2017, Fig. 4-5)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_correlated_posterior(
    df: float,
    loc: float,
    scale: float,
    rope_band: tuple[float, float],
    hdi: tuple[float, float],
    hdi_prob: float,
    left_model: str,
    right_model: str,
    kind: str = "posterior",
    ax: plt.Axes | None = None,
    n_points: int = 512,
    **kwargs,
) -> plt.Axes:
    """Plot the Student posterior of the mean difference with the ROPE and HDI.

    Parameters
    ----------
    df, loc, scale : float
        Degrees of freedom, location and scale of the Student posterior.
    rope_band : tuple[float, float]
        The ROPE interval on the difference scale.
    hdi : tuple[float, float]
        The HDI bounds to mark.
    hdi_prob : float
        Mass of the HDI (for the label).
    left_model, right_model : str
        Names of the two algorithms (positive difference favours ``left_model``).
    kind : {"posterior", "hdi"}, default "posterior"
        Which figure to draw.
    ax : plt.Axes | None, default None
        Axes to draw on; created if ``None``.
    n_points : int, default 512
        Grid resolution for the density curve.
    **kwargs
        Extra keyword arguments forwarded to the density ``plot`` call.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    if ax is not None and not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object or None.")
    if ax is None:
        _, ax = plt.subplots()

    if kind == "hdi":
        return _plot_hdi_fan(df, loc, scale, rope_band, ax)

    lo, hi = rope_band
    if scale > 0:
        span = max(4 * scale, 1.5 * (hi - lo))
        xs = np.linspace(loc - span, loc + span, n_points)
        ys = stats.t.pdf(xs, df=df, loc=loc, scale=scale)
    else:
        # Degenerate point mass: draw a spike at loc.
        span = max(hi - lo, 1e-6)
        xs = np.linspace(loc - span, loc + span, n_points)
        ys = np.zeros_like(xs)
        ys[n_points // 2] = 1.0

    ax.plot(xs, ys, color="#1f77b4", label="posterior", **kwargs)
    ax.fill_between(xs, ys, color="#1f77b4", alpha=0.25)

    # ROPE band
    ax.axvline(lo, color="#ff7f0e", linewidth=1.5)
    ax.axvline(hi, color="#ff7f0e", linewidth=1.5, label="rope")
    ax.axvspan(lo, hi, color="#ff7f0e", alpha=0.10)

    # HDI marker along the baseline
    hdi_low, hdi_high = hdi
    ax.hlines(
        0,
        hdi_low,
        hdi_high,
        color="black",
        linewidth=2.5,
        label=f"{int(hdi_prob * 100)}% HDI",
    )

    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel(f"mean difference ({left_model} - {right_model})")
    ax.set_ylabel("pdf")
    ax.set_title(f"Correlated t-test posterior: {left_model} vs {right_model}")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def _plot_hdi_fan(
    df: float,
    loc: float,
    scale: float,
    rope_band: tuple[float, float],
    ax: plt.Axes,
    probs: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> plt.Axes:
    """Draw HDIs of increasing mass, as in Fig. 5 of the paper."""
    lo, hi = rope_band
    for prob in probs:
        if scale > 0:
            tail = (1.0 + prob) / 2.0
            half = float(stats.t.ppf(tail, df=df)) * scale
            low, high = loc - half, loc + half
        else:
            low, high = loc, loc
        ax.vlines(prob, low, high, color="#1f77b4", linewidth=2)
        ax.plot([prob], [low], marker="_", color="#1f77b4")
        ax.plot([prob], [high], marker="_", color="#1f77b4")

    ax.axhspan(lo, hi, color="#ff7f0e", alpha=0.12, label="rope")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("probability")
    ax.set_ylabel("HDI")
    ax.legend(loc="upper left", fontsize=8)
    return ax
