"""Numerical helpers shared across the test estimators."""

from __future__ import annotations

import arviz as az
import numpy as np

# Shared decision label vocabulary. The raw symbols mirror BBT's ``>``/``=``/
# ``?``, with an added ``<`` because the two-model tests compare a *fixed* left
# vs. right pair (BBT orders its pairs strongest-first and so never needs ``<``).
DECISION_LEFT_BETTER = ">"
DECISION_RIGHT_BETTER = "<"
DECISION_EQUIVALENT = "="
DECISION_UNDETERMINED = "?"


def resolve_rope(rope: float | tuple[float, float]) -> tuple[float, float]:
    """Resolve a ROPE argument to an explicit ``(lo, hi)`` band on the difference scale.

    A scalar is interpreted as a half-width ``r`` giving the symmetric band
    ``[-r, r]`` (this is what keeps the CLI's ``--rope`` a consistent
    half-width shape across tests). A 2-tuple is used verbatim.

    Parameters
    ----------
    rope : float | tuple[float, float]
        Half-width ``r`` (scalar) or explicit ``(lo, hi)`` band.

    Returns
    -------
    tuple[float, float]
        The resolved ``(lo, hi)`` interval, with ``lo < hi``.
    """
    if isinstance(rope, tuple):
        lo, hi = rope
        if lo >= hi:
            raise ValueError(f"ROPE band must satisfy lo < hi, got ({lo}, {hi}).")
        return float(lo), float(hi)
    if rope < 0:
        raise ValueError(f"ROPE half-width must be non-negative, got {rope}.")
    return -float(rope), float(rope)


def decision_from_partition(
    p_left: float,
    p_rope: float,
    p_right: float,
    left_model: str,
    right_model: str,
    threshold: float,
) -> tuple[str, str]:
    """Turn a three-way ROPE partition into a decision label and raw symbol.

    The partition is taken over the difference ``left - right`` (positive means
    the left model is better), so ``p_right``, the mass above the ROPE, votes
    for the left model and ``p_left``, the mass below it, votes for the right
    model. A side is declared only if its probability exceeds ``threshold``
    (Benavoli et al. 2017, Section 3.2, uses 0.95); otherwise the decision is
    undetermined.

    Parameters
    ----------
    p_left, p_rope, p_right : float
        The three partition probabilities (should sum to one).
    left_model, right_model : str
        Names of the two algorithms being compared.
    threshold : float
        Probability a region must exceed to be declared.

    Returns
    -------
    tuple[str, str]
        ``(decision, decision_raw)`` where ``decision`` uses the human strings
        (``"<model> better"`` / ``"Equivalent"`` / ``"Unknown"``, matching BBT)
        and ``decision_raw`` is one of ``>`` / ``<`` / ``=`` / ``?``.
    """
    if p_right >= threshold:
        return f"{left_model} better", DECISION_LEFT_BETTER
    if p_left >= threshold:
        return f"{right_model} better", DECISION_RIGHT_BETTER
    if p_rope >= threshold:
        return "Equivalent", DECISION_EQUIVALENT
    return "Unknown", DECISION_UNDETERMINED


def region_probabilities(
    theta_left: np.ndarray, theta_rope: np.ndarray, theta_right: np.ndarray
) -> tuple[float, float, float]:
    """Probability of each simplex region, by counting posterior draws.

    Benavoli et al. (2017), Sections 4.2.2 and 4.3.2 quantify a comparison "by
    counting the number of points that fall in the three regions" delimited by
    ``theta_i >= max(theta_j, theta_k)``.

    Parameters
    ----------
    theta_left, theta_rope, theta_right : np.ndarray
        Posterior draws of the three-way ROPE triple.

    Returns
    -------
    tuple[float, float, float]
        The proportions of draws falling in the left, rope and right regions.
    """
    winner = np.argmax(np.column_stack([theta_left, theta_rope, theta_right]), axis=1)
    return (
        float(np.mean(winner == 0)),
        float(np.mean(winner == 1)),
        float(np.mean(winner == 2)),
    )


def samples_to_inference_data(
    posterior: dict[str, np.ndarray],
) -> az.InferenceData:
    """Wrap named sample arrays into an :class:`arviz.InferenceData`.

    Each array is treated as a single chain of draws, so the tests that draw
    independent Monte Carlo samples expose exactly the same ``idata_`` endpoint
    as the MCMC-sampled ones.

    Parameters
    ----------
    posterior : dict[str, np.ndarray]
        Mapping from variable name to an array of draws along its first axis.

    Returns
    -------
    az.InferenceData
        Inference data with a single chain per variable.
    """
    reshaped = {
        name: np.asarray(draws)[np.newaxis, ...] for name, draws in posterior.items()
    }
    return az.from_dict(posterior=reshaped)


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
