"""Numerical helpers shared across the test estimators."""

from __future__ import annotations

import numpy as np


def hdi_from_samples(
    samples: np.ndarray, hdi_prob: float = 0.89
) -> tuple[float, float]:
    """Highest-density interval of a 1-D sample array (shortest-interval rule).

    This is the single implementation of the shortest-interval HDI used across
    the package: BBT applies it column-wise over its pairwise samples, while the
    t-tests apply it to a single estimand.

    Parameters
    ----------
    samples : np.ndarray
        1-D array of posterior samples.
    hdi_prob : float, optional
        Target mass of the interval. Defaults to 0.89.

    Returns
    -------
    tuple[float, float]
        The ``(low, high)`` bounds of the interval.
    """
    x = np.sort(np.asarray(samples).ravel())
    n = len(x)
    # Width of the interval in order statistics: the smallest window that still
    # covers ``hdi_prob`` of the draws. ``ceil`` rather than ``floor`` so the
    # interval never covers *less* than the requested mass, and the number of
    # candidate windows is n - span.
    span = int(np.ceil(n * hdi_prob)) - 1
    if span >= n - 1:
        return float(x[0]), float(x[-1])
    low_poss = x[: n - span]
    upp_poss = x[span:]
    best = int(np.argmin(upp_poss - low_poss))
    return float(low_poss[best]), float(upp_poss[best])
