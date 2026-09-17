"""Point plots of the pairwise posterior under the weak interpretation (Wainer 2023, sec. 8.2)."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from ._strong_posterior import STRONG_COLOURS as WEAK_COLOURS
from ._strong_posterior import STRONG_VERDICTS as WEAK_VERDICTS


def weak_verdicts(
    above_50: np.ndarray,
    below_50: np.ndarray,
    in_rope: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Classify comparisons of ``pi = P(left > right)`` under the weak reading.

    Equivalence is checked first, as in ``BBTTest.posterior_table``; a pair
    oriented with the weaker model on the left is reported as ``weaker``.
    """
    return np.select(
        [
            np.asarray(in_rope) >= threshold,
            np.asarray(above_50) >= threshold,
            np.asarray(below_50) >= threshold,
        ],
        ["equivalent", "better", "weaker"],
        default="no claim",
    )


def _lighten(colour: str, amount: float = 0.55) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(colour))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_weak_posterior(
    labels: Sequence[str],
    means: np.ndarray,
    above_50: np.ndarray,
    below_50: np.ndarray,
    in_rope: np.ndarray,
    threshold: float,
    rope_value: tuple[float, float],
    value_label: str,
    orientation: str = "horizontal",
    subtitle: str | None = None,
    ax: Sequence[plt.Axes] | None = None,
    **kwargs,
) -> np.ndarray:
    """Draw ``P(pi > 0.5)`` and ``P(pi in ROPE)`` per comparison in two panels.

    Parameters
    ----------
    labels : Sequence[str]
        One label per comparison.
    means : np.ndarray
        Posterior means of ``pi = P(left > right)``, used only to order the comparisons.
    above_50, below_50 : np.ndarray
        Posterior probabilities that ``pi`` is above / below 0.5.
    in_rope : np.ndarray
        Posterior probability that ``pi`` lies in ``rope_value``.
    threshold : float
        Probability a quantity must reach for a claim.
    rope_value : tuple[float, float]
        The ROPE, shown in the axis label.
    value_label : str
        Definition of ``pi``, shown in the title.
    orientation : {"horizontal", "vertical"}, default "horizontal"
        ``horizontal`` stacks the panels and lays the comparisons along the x
        axis, best on the left; ``vertical`` puts the panels side by side and
        lays the comparisons along the y axis, best on top.
    subtitle : str | None, default None
        Line drawn under the title, e.g. naming the control model.
    ax : Sequence[plt.Axes] | None, default None
        Two Axes: ``P(pi > 0.5)`` is drawn on the first, ``P(pi in ROPE)`` on
        the second. Created if ``None``.
    **kwargs
        Extra keyword arguments forwarded to both ``scatter`` calls.

    Returns
    -------
    np.ndarray
        The two Axes drawn on.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}."
        )
    horizontal = orientation == "horizontal"

    means = np.asarray(means, dtype=float)
    order = np.argsort(-means if horizontal else means, kind="stable")
    labels = [labels[i] for i in order]
    above_50 = np.asarray(above_50, dtype=float)[order]
    below_50 = np.asarray(below_50, dtype=float)[order]
    in_rope = np.asarray(in_rope, dtype=float)[order]
    verdicts = weak_verdicts(above_50, below_50, in_rope, threshold)
    n = len(labels)

    axes = _resolve_axes(ax, n, horizontal)
    positions = np.arange(n)
    colours = [WEAK_COLOURS[v] for v in verdicts]
    scatter_kwargs = {"s": 110, "zorder": 3, "edgecolor": "white", **kwargs}
    rope_label = rf"$P(\pi \in$ ROPE$)$, ROPE = [{rope_value[0]}, {rope_value[1]}]"

    bands = (("better", threshold, 1.02), ("weaker", -0.02, 1.0 - threshold))
    _draw_panel(
        axes[0],
        positions,
        above_50,
        colours,
        labels,
        r"$P(\pi > 0.5)$",
        bands,
        horizontal,
        scatter_kwargs,
    )
    bands = (("equivalent", threshold, 1.02),)
    _draw_panel(
        axes[1],
        positions,
        in_rope,
        colours,
        labels,
        rope_label,
        bands,
        horizontal,
        scatter_kwargs,
    )

    if horizontal:
        # Stacked panels share the model axis; only the bottom one names it.
        axes[0].tick_params(axis="x", labelbottom=False)
    else:
        axes[1].tick_params(axis="y", labelleft=False)

    _add_title_and_legend(axes, verdicts, threshold, value_label, subtitle, horizontal)
    return axes


def _resolve_axes(
    ax: Sequence[plt.Axes] | None, n: int, horizontal: bool
) -> np.ndarray:
    if ax is None:
        if horizontal:
            _, axes = plt.subplots(2, 1, figsize=(0.42 * n + 3.5, 8.5), sharex=True)
        else:
            _, axes = plt.subplots(1, 2, figsize=(11, 0.32 * n + 1.6), sharey=True)
        return np.asarray(axes, dtype=object)
    axes = (
        np.array([ax], dtype=object)
        if isinstance(ax, plt.Axes)
        else np.asarray(ax, dtype=object).ravel()
    )
    if len(axes) != 2 or not all(isinstance(a, plt.Axes) for a in axes):
        raise ValueError(
            "The weak-posterior plot draws two panels and needs two Axes: "
            "ax[0] for P(pi > 0.5) and ax[1] for P(pi in ROPE), "
            f"got {len(axes)} object(s). Pass ax=None to create them, or e.g. "
            "fig, ax = plt.subplots(2, 1, sharex=True)."
        )
    return axes


def _draw_panel(
    panel: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    colours: list[str],
    labels: list[str],
    value_label: str,
    bands: Sequence[tuple[str, float, float]],
    horizontal: bool,
    scatter_kwargs: dict,
) -> None:
    """One lollipop panel: a stem from 0 to each value, lighter than its dot."""
    band = panel.axhspan if horizontal else panel.axvspan
    line = panel.axhline if horizontal else panel.axvline
    for verdict, lo, hi in bands:
        band(lo, hi, color=WEAK_COLOURS[verdict], alpha=0.10, lw=0)
        # Dash the claim threshold, i.e. the band edge inside [0, 1].
        line(hi if lo < 0 else lo, color="0.4", ls="--", lw=1)

    stems = [_lighten(c) for c in colours]
    n = len(positions)
    if horizontal:
        panel.vlines(positions, 0.0, values, colors=stems, lw=2.5, zorder=2)
        panel.scatter(positions, values, c=colours, **scatter_kwargs)
        panel.set_xticks(positions, labels, rotation=60, ha="right")
        panel.set_xlim(-0.7, n - 0.3)
        panel.set_ylim(-0.02, 1.02)
        panel.grid(axis="x", visible=False)
        panel.set_ylabel(value_label)
    else:
        panel.hlines(positions, 0.0, values, colors=stems, lw=2.5, zorder=2)
        panel.scatter(values, positions, c=colours, **scatter_kwargs)
        panel.set_yticks(positions, labels)
        panel.set_ylim(-0.7, n - 0.3)
        panel.set_xlim(-0.02, 1.02)
        panel.grid(axis="y", visible=False)
        panel.set_xlabel(value_label)


def _add_title_and_legend(
    axes: np.ndarray,
    verdicts: np.ndarray,
    threshold: float,
    value_label: str,
    subtitle: str | None,
    horizontal: bool,
) -> None:
    title = f"Weak interpretation: claim at {threshold},  {value_label}"
    if subtitle is None:
        axes[0].set_title(title, loc="left")
    else:
        axes[0].set_title(title, loc="left", pad=20)
        axes[0].text(
            0.0,
            1.01,
            subtitle,
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize="small",
            color="0.35",
        )

    present = set(verdicts.tolist())
    handles = [
        Line2D([], [], marker="o", ls="", color=WEAK_COLOURS[v], markersize=8, label=v)
        for v in WEAK_VERDICTS
        if v in present
    ]
    # Beside the top panel when stacked, beside the right one when side by side.
    (axes[0] if horizontal else axes[1]).legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
        frameon=False,
    )
