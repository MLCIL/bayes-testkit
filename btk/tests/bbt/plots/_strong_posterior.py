"""Forest plot of the pairwise posterior under the strong interpretation (Wainer 2023, sec. 8.3)."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

STRONG_VERDICTS = ("better", "equivalent", "no claim", "weaker")
STRONG_COLOURS = {
    "better": "#1b9e77",
    "equivalent": "#7570b3",
    "no claim": "#9e9e9e",
    "weaker": "#d95f02",
}


def strong_verdicts(
    means: np.ndarray, better_threshold: float, equal_threshold: float
) -> np.ndarray:
    """Classify posterior means of ``P(left > right)`` under the strong reading.

    The rule is applied symmetrically around 0.5, so a pair oriented with the
    weaker model on the left is reported as ``weaker`` rather than ``no claim``.
    """
    means = np.asarray(means, dtype=float)
    return np.select(
        [
            means > better_threshold,
            1.0 - means > better_threshold,
            (means <= equal_threshold) & (1.0 - means <= equal_threshold),
        ],
        ["better", "weaker", "equivalent"],
        default="no claim",
    )


def plot_strong_posterior(
    labels: Sequence[str],
    means: np.ndarray,
    hdi_low: np.ndarray,
    hdi_high: np.ndarray,
    better_threshold: float,
    equal_threshold: float,
    hdi_prob: float,
    value_label: str,
    orientation: str = "horizontal",
    subtitle: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Draw ``E[pi]`` with its HDI per comparison, coloured by the strong verdict.

    Parameters
    ----------
    labels : Sequence[str]
        One label per comparison.
    means : np.ndarray
        Posterior means of ``pi = P(left > right)``.
    hdi_low, hdi_high : np.ndarray
        HDI bounds of ``pi``.
    better_threshold : float
        ``E[pi]`` above this is ``better`` (below ``1 - better_threshold`` is ``weaker``).
    equal_threshold : float
        ``E[pi]`` within ``[1 - equal_threshold, equal_threshold]`` is ``equivalent``.
    hdi_prob : float
        Mass of the HDI, used in the title.
    value_label : str
        Label of the probability axis.
    orientation : {"horizontal", "vertical"}, default "horizontal"
        ``horizontal`` lays the comparisons along the x axis, best on the left;
        ``vertical`` lays them along the y axis, best on top.
    subtitle : str | None, default None
        Line drawn under the title, e.g. naming the control model.
    ax : plt.Axes | None, default None
        Axes to draw on; created if ``None``.
    **kwargs
        Extra keyword arguments forwarded to the ``scatter`` call of the means.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    if ax is not None and not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object or None.")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}."
        )
    horizontal = orientation == "horizontal"

    # Horizontal reads best-first left to right; vertical puts the best on top,
    # i.e. last along the y axis.
    means = np.asarray(means, dtype=float)
    order = np.argsort(-means if horizontal else means, kind="stable")
    labels = [labels[i] for i in order]
    means = means[order]
    hdi_low = np.asarray(hdi_low, dtype=float)[order]
    hdi_high = np.asarray(hdi_high, dtype=float)[order]
    verdicts = strong_verdicts(means, better_threshold, equal_threshold)
    n = len(labels)

    if ax is None:
        figsize = (0.45 * n + 3.5, 5.5) if horizontal else (9, 0.32 * n + 1.6)
        _, ax = plt.subplots(figsize=figsize)

    band = ax.axhspan if horizontal else ax.axvspan
    midline = ax.axhline if horizontal else ax.axvline
    for lo, hi, verdict in (
        (0.0, 1.0 - better_threshold, "weaker"),
        (1.0 - equal_threshold, equal_threshold, "equivalent"),
        (better_threshold, 1.0, "better"),
    ):
        band(lo, hi, color=STRONG_COLOURS[verdict], alpha=0.10, lw=0)
    midline(0.5, color="k", ls=":", lw=1.2)

    positions = np.arange(n)
    colours = [STRONG_COLOURS[v] for v in verdicts]
    scatter_kwargs = {"s": 45, "zorder": 3, "edgecolor": "white", **kwargs}
    value_limits = (
        min(1.0 - better_threshold - 0.05, float(hdi_low.min()) - 0.02),
        max(better_threshold + 0.05, float(hdi_high.max()) + 0.02),
    )
    if horizontal:
        ax.vlines(positions, hdi_low, hdi_high, colors=colours, lw=2.2)
        ax.scatter(positions, means, c=colours, **scatter_kwargs)
        ax.set_xticks(positions, labels, rotation=60, ha="right")
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(*value_limits)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(value_label)
    else:
        ax.hlines(positions, hdi_low, hdi_high, colors=colours, lw=2.2)
        ax.scatter(means, positions, c=colours, **scatter_kwargs)
        ax.set_yticks(positions, labels)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(*value_limits)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(value_label)

    _add_title_and_legend(ax, verdicts, hdi_prob, subtitle)
    return ax


def _add_title_and_legend(
    ax: plt.Axes, verdicts: np.ndarray, hdi_prob: float, subtitle: str | None
) -> None:
    title = rf"Strong interpretation: $E[\pi]$ with {hdi_prob:.0%} HDI"
    if subtitle is None:
        ax.set_title(title, loc="left")
    else:
        ax.set_title(title, loc="left", pad=20)
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize="small",
            color="0.35",
        )

    present = set(verdicts.tolist())
    handles = [
        Line2D(
            [], [], marker="o", ls="", color=STRONG_COLOURS[v], markersize=8, label=v
        )
        for v in STRONG_VERDICTS
        if v in present
    ]
    handles += [
        Patch(color=STRONG_COLOURS[v], alpha=0.25, label=f"{v} region")
        for v in ("better", "equivalent", "weaker")
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
        frameon=False,
    )
